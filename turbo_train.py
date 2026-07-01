#!/usr/bin/env python3
"""Consistency Distillation (CD) training for SAM-Audio-Turbo.

Distills the 16-step flow-matching teacher into a 1--4 step student,
achieving ~4--16x inference speed-up.

Core idea:
    - Student predicts clean endpoint from any noisy state along the
      teacher ODE trajectory.
    - EMA student (same architecture, frozen) from a cleaner state
      provides the regression target.
    - Teacher only runs ODE segments for supervision (no gradient).

Memory (LoRA mode, BF16):
    teacher (~4.5GB) + student (~4.5GB) + EMA shadow (~4.5GB, no opt state)
    vs. DMD: teacher + student + discriminator + 3 optimizers → ~2GB saved.

Usage:
    # Full-weight distillation (4-step student)
    python turbo_train.py --num-steps 4 --epochs 20

    # LoRA-based distillation (smaller GPU footprint)
    python turbo_train.py --num-steps 4 --use-lora --lora-rank 8
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.train import (
    apply_lora,
    freeze_encoders,
    strip_vision_branch,
)
from sam_audio.turbo.consistency import (
    EMAHelper,
    cd_intervals,
    consistency_loss,
    _make_vf_fn,
)


# ── training loop ────────────────────────────────────────────────────────


def turbo_train(
    student: nn.Module,
    processor: SAMAudioProcessor,
    dataloader: DataLoader,
    *,
    teacher: nn.Module | None = None,
    num_steps: int = 4,
    epochs: int = 20,
    lr: float = 1e-5,
    ema_decay: float = 0.999,
    device: str = "cuda",
    save_dir: str = "turbo-checkpoint",
    log_every: int = 5,
):
    """Consistency Distillation training loop.

    Args:
        student: Trainable student SAM-Audio (may be PeftModel).
        processor: SAMAudioProcessor for batch encoding.
        dataloader: DataLoader yielding ``(audios, descriptions)`` tuples.
        teacher: Frozen teacher SAM-Audio. When ``None`` (LoRA mode), the
            student's adapters are disabled for teacher calls, so the frozen
            pretrained base supplies the teacher without a second model copy.
        num_steps: Number of CD intervals (1 / 2 / 4).
        epochs: Training epochs.
        lr: Learning rate for student (single optimizer, no discriminator).
        ema_decay: EMA decay rate for target student.
        device: Training device.
        save_dir: Checkpoint directory.
        log_every: Log metrics every N steps.
    """
    # ── freeze teacher ──────────────────────────────────────────────
    device = torch.device(device)
    amp_enabled = device.type == "cuda"
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    if teacher is not None:
        teacher.eval()
        teacher.to(device)
        for p in teacher.parameters():
            p.requires_grad = False

    # ── setup student ───────────────────────────────────────────────
    student.to(device)
    student.train()
    freeze_encoders(student)

    # These modules only build forward_args and are never traversed by the
    # student's vector-field forward. Freeze them explicitly; eval() alone
    # does not clear requires_grad. With a separate teacher, its encoders build
    # the conditioning, so the duplicate student copies can stay on CPU.
    student_base = getattr(student, "base_model", student)
    for encoder_name in ("audio_codec", "text_encoder", "vision_encoder"):
        encoder = getattr(student_base, encoder_name, None)
        if encoder is None:
            continue
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad_(False)
        if teacher is not None:
            encoder.to("cpu")

    # ── derive teacher from student.base_model when not explicitly provided ──
    shared_lora_teacher = teacher is None
    if teacher is None:
        if not hasattr(student, "disable_adapter"):
            raise ValueError(
                "teacher=None is only supported for adapter training; pass a "
                "separate frozen teacher for full-weight distillation"
            )
        teacher = student

    # ── EMA student (shadow weights, no optimizer) ──────────────────
    trainable_named = [
        (name, param)
        for name, param in student.named_parameters()
        if param.requires_grad
    ]
    if not trainable_named:
        raise ValueError("student has no trainable parameters")
    ema = EMAHelper(trainable_named, decay=ema_decay)

    # ── optimizer (single, no discriminator) ────────────────────────
    optimizer = torch.optim.AdamW(
        [param for _, param in trainable_named], lr=lr, betas=(0.5, 0.9)
    )

    # ── stats ───────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student trainable params: {trainable:,}")
    print(f"CD intervals:             {num_steps}")
    print(f"EMA decay:                {ema_decay}")
    os.makedirs(save_dir, exist_ok=True)

    # ── epoch loop ──────────────────────────────────────────────────
    for epoch in range(epochs):
        epoch_start = time.time()
        losses_cd = []

        for step, batch in enumerate(dataloader):
            # --- unpack / encode ---
            if isinstance(batch, tuple):
                audios, descriptions = batch
                batch = processor(audios=audios, descriptions=descriptions)
            batch = batch.to(device)

            base_t = getattr(teacher, "base_model", teacher)

            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                forward_args = base_t._get_forward_args(batch)
            # The encoded tensors in forward_args are all that the ODE needs;
            # release the much larger waveform batch before model forwards.
            del batch

            audio_features = forward_args["audio_features"]  # z_0: clean latent
            z_1 = torch.randn_like(audio_features)  # noise

            # --- sample a CD interval ---
            intervals = cd_intervals(num_steps)
            # Pick one interval per batch; random or round-robin
            t, s = intervals[torch.randint(len(intervals), ()).item()]

            # --- teacher ODE segment (t → s, no grad) ---
            teacher_vf = _make_vf_fn(
                teacher,
                forward_args,
                disable_adapters=shared_lora_teacher,
                force_eval=shared_lora_teacher,
            )

            # === Consistency loss ===
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                loss, _, _ = consistency_loss(
                    teacher_vf=teacher_vf,
                    student=student,
                    ema_model=ema,
                    z_0=audio_features,
                    z_1=z_1,
                    forward_args=forward_args,
                    t=t,
                    s=s,
                    loss_fn="huber",
                )

            loss.backward()
            optimizer.step()

            # --- update EMA ---
            ema.update()

            # --- logging ---
            losses_cd.append(loss.item())

            if (step + 1) % log_every == 0:
                print(
                    f"Epoch {epoch + 1:3d} | Step {step + 1:4d} | "
                    f"CD: {losses_cd[-1]:.6f}"
                )

        # --- epoch summary ---
        elapsed = time.time() - epoch_start
        if not losses_cd:
            raise ValueError("dataloader produced no batches")
        avg_loss = sum(losses_cd) / len(losses_cd)
        print(
            f"Epoch {epoch + 1:3d} | "
            f"CD_avg: {avg_loss:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # --- checkpoint ---
        ckpt_path = f"{save_dir}/epoch-{epoch + 1}"
        if hasattr(student, "save_pretrained"):
            student.save_pretrained(ckpt_path)
        else:
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(student.state_dict(), f"{ckpt_path}/model.pt")
        state = {
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "num_steps": num_steps,
        }
        torch.save(state, f"{ckpt_path}/training_state.pt")
        print(f"  -> saved checkpoint to {ckpt_path}")

    return student


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consistency Distillation for SAM-Audio-Turbo"
    )
    parser.add_argument(
        "--num-steps", type=int, default=4, help="CD intervals (1/2/4)"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--ema-decay", type=float, default=0.999, help="EMA decay for target student"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Apply LoRA to student for smaller GPU footprint",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--model-id", type=str, default="facebook/sam-audio-small"
    )
    parser.add_argument("--save-dir", type=str, default="turbo-checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/lixing/audiolens/dataset",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"Device: {device}")
    print(f"Mode: {'LoRA' if args.use_lora else 'full-weight'} CD")

    # ── Load model(s) ─────────────────────────────────────────────────
    teacher = None
    if args.use_lora:
        student = SAMAudio.from_pretrained(args.model_id)
        student = student.to(model_dtype)
        strip_vision_branch(student)
        student = apply_lora(student, rank=args.lora_rank, alpha=16)
        freeze_encoders(student)
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student.parameters())
        print(
            f"LoRA trainable: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)"
        )
    else:
        teacher = SAMAudio.from_pretrained(args.model_id)
        teacher = teacher.to(model_dtype)
        strip_vision_branch(teacher)

        student = SAMAudio.from_pretrained(args.model_id)
        student = student.to(model_dtype)
        strip_vision_branch(student)

    # ── Processor ─────────────────────────────────────────────────────
    processor = SAMAudioProcessor.from_pretrained(args.model_id)

    # ── Dataset ───────────────────────────────────────────────────────
    # (preserved from original DMD setup)
    import json
    import os.path as osp

    import pandas as pd
    import torchaudio
    from torch.utils.data import ConcatDataset, Dataset

    SR = 48_000

    class ClothoDS(Dataset):
        def __init__(self, data_root=args.data_dir, sr=SR):
            self.sr = sr
            csv_path = osp.join(data_root, "clotho_captions_development.csv")
            self.audio_dir = osp.join(data_root, "clotho", "development", "audio")
            df = pd.read_csv(csv_path)
            self.items = []
            for _, row in df.iterrows():
                fname = row["file_name"]
                for k in range(1, 6):
                    self.items.append((fname, row[f"caption_{k}"]))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            fname, caption = self.items[idx]
            wav, sr = torchaudio.load(osp.join(self.audio_dir, fname))
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            return wav.mean(0, keepdim=True), caption

    class FSD50KDevDS(Dataset):
        def __init__(self, data_root=args.data_dir, sr=SR):
            self.sr = sr
            json_path = osp.join(data_root, "fsd50k_dev_auto_caption.json")
            self.audio_dir = osp.join(data_root, "FSD50K", "FSD50K.dev_audio")
            with open(json_path) as f:
                entries = json.load(f)["data"]
            self.items = [(e["wav"], e["caption"]) for e in entries]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            fname, caption = self.items[idx]
            wav, sr = torchaudio.load(osp.join(self.audio_dir, fname))
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            return wav.mean(0, keepdim=True), caption

    def _collate(items):
        audios, descriptions = zip(*items)
        return list(audios), list(descriptions)

    clotho = ClothoDS()
    fsd50k = FSD50KDevDS()
    print(f"Clotho: {len(clotho)} items, FSD50K: {len(fsd50k)} items")

    train_data = ConcatDataset([clotho, fsd50k])
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=4,
    )

    # ── Train ─────────────────────────────────────────────────────────
    student = turbo_train(
        teacher=teacher,
        student=student,
        processor=processor,
        dataloader=train_loader,
        num_steps=args.num_steps,
        epochs=args.epochs,
        lr=args.lr,
        ema_decay=args.ema_decay,
        device=device,
        save_dir=args.save_dir,
    )

    print("CD distillation complete.")
    print(f"Checkpoints saved to {args.save_dir}/")
