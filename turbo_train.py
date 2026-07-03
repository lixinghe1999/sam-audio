#!/usr/bin/env python3
"""Few-step SAM-Audio training with consistency distillation or MeanFlow.

Consistency distillation uses teacher ODE segments and an EMA target.
MeanFlow is simulation-free: it learns interval-average velocity using the
MeanFlow identity and a no-grad Jacobian-vector product (JVP).

Usage:
    # Full-weight distillation (4-step student)
    python turbo_train.py --num-steps 4 --epochs 20

    # LoRA-based distillation (smaller GPU footprint)
    python turbo_train.py --num-steps 4 --use-lora --lora-rank 8

    # MeanFlow (one-step objective; no teacher or EMA)
    python turbo_train.py --objective meanflow --use-lora --epochs 20
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
from sam_audio.turbo.meanflow import meanflow_loss, sample_meanflow_times


# ── training loop ────────────────────────────────────────────────────────


def turbo_train(
    student: nn.Module,
    processor: SAMAudioProcessor,
    dataloader: DataLoader,
    *,
    teacher: nn.Module | None = None,
    objective: str = "consistency_distillation",
    num_steps: int = 4,
    epochs: int = 20,
    lr: float = 1e-5,
    ema_decay: float = 0.999,
    device: str = "cuda",
    save_dir: str = "turbo-checkpoint",
    log_every: int = 5,
    meanflow_nonzero_ratio: float = 0.25,
    meanflow_time_distribution: str = "logit_normal",
    meanflow_logit_mean: float = 0.4,
    meanflow_logit_std: float = 1.0,
    meanflow_adaptive_p: float = 1.0,
    meanflow_adaptive_eps: float = 0.01,
):
    """Train SAM-Audio with consistency distillation or MeanFlow.

    Args:
        student: Trainable student SAM-Audio (may be PeftModel).
        processor: SAMAudioProcessor for batch encoding.
        dataloader: DataLoader yielding ``(audios, descriptions)`` tuples.
        teacher: Frozen teacher SAM-Audio. When ``None`` (LoRA mode), the
            student's adapters are disabled for teacher calls, so the frozen
            pretrained base supplies the CD teacher without a second model copy.
            MeanFlow does not use a teacher.
        objective: ``"consistency_distillation"`` or ``"meanflow"``.
        num_steps: Number of CD intervals (1 / 2 / 4).
        epochs: Training epochs.
        lr: Learning rate for student (single optimizer, no discriminator).
        ema_decay: EMA decay rate for target student.
        device: Training device.
        save_dir: Checkpoint directory.
        log_every: Log metrics every N steps.
        meanflow_nonzero_ratio: Fraction of MeanFlow samples with ``s != t``.
            Remaining samples train the instantaneous velocity at ``s=t``.
        meanflow_time_distribution: ``"logit_normal"`` or ``"uniform"``.
        meanflow_logit_mean: Logit-normal mean, adapted to noise-at-zero time.
        meanflow_logit_std: Logit-normal standard deviation.
        meanflow_adaptive_p: Exponent for adaptive MeanFlow loss weighting.
        meanflow_adaptive_eps: Stabilizer for adaptive loss weighting.
    """
    # ── freeze teacher ──────────────────────────────────────────────
    device = torch.device(device)
    amp_enabled = device.type == "cuda"
    valid_objectives = {"consistency_distillation", "meanflow"}
    if objective not in valid_objectives:
        raise ValueError(
            f"objective must be one of {sorted(valid_objectives)}, got {objective!r}"
        )
    use_cd = objective == "consistency_distillation"
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if log_every < 1:
        raise ValueError(f"log_every must be positive, got {log_every}")
    if not use_cd:
        if not 0 <= meanflow_nonzero_ratio <= 1:
            raise ValueError("meanflow_nonzero_ratio must be between 0 and 1")
        if meanflow_logit_std <= 0:
            raise ValueError("meanflow_logit_std must be positive")
        if meanflow_adaptive_p < 0 or meanflow_adaptive_eps <= 0:
            raise ValueError("invalid MeanFlow adaptive-weighting parameters")

    if use_cd and teacher is not None:
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
        if use_cd and teacher is not None:
            encoder.to("cpu")

    # ── derive teacher from student.base_model when not explicitly provided ──
    shared_lora_teacher = use_cd and teacher is None
    if shared_lora_teacher:
        if not hasattr(student, "disable_adapter"):
            raise ValueError(
                "teacher=None is only supported for adapter training; pass a "
                "separate frozen teacher for full-weight distillation"
            )
        teacher = student

    # MeanFlow's JVP must see the same deterministic function as its ordinary
    # prediction. Gradients still work in eval mode, while dropout is disabled.
    if not use_cd:
        student.eval()

    # ── EMA student (shadow weights, no optimizer) ──────────────────
    trainable_named = [
        (name, param)
        for name, param in student.named_parameters()
        if param.requires_grad
    ]
    if not trainable_named:
        raise ValueError("student has no trainable parameters")
    ema = EMAHelper(trainable_named, decay=ema_decay) if use_cd else None

    # ── optimizer (single, no discriminator) ────────────────────────
    optimizer = torch.optim.AdamW(
        [param for _, param in trainable_named], lr=lr, betas=(0.5, 0.9)
    )

    # ── stats ───────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student trainable params: {trainable:,}")
    print(f"Training objective:       {objective}")
    if use_cd:
        print(f"CD intervals:             {num_steps}")
        print(f"EMA decay:                {ema_decay}")
    else:
        print(f"MeanFlow interval ratio:  {meanflow_nonzero_ratio}")
    os.makedirs(save_dir, exist_ok=True)

    # ── epoch loop ──────────────────────────────────────────────────
    for epoch in range(epochs):
        epoch_start = time.time()
        losses = []

        for step, batch in enumerate(dataloader):
            # --- unpack / encode ---
            if isinstance(batch, tuple):
                audios, descriptions = batch
                batch = processor(audios=audios, descriptions=descriptions)
            batch = batch.to(device)

            conditioning_model = teacher if use_cd else student
            base_t = getattr(conditioning_model, "base_model", conditioning_model)

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

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                if use_cd:
                    intervals = cd_intervals(num_steps)
                    t, s = intervals[torch.randint(len(intervals), ()).item()]
                    teacher_vf = _make_vf_fn(
                        teacher,
                        forward_args,
                        disable_adapters=shared_lora_teacher,
                        force_eval=shared_lora_teacher,
                    )
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
                else:
                    t, s = sample_meanflow_times(
                        audio_features.size(0),
                        audio_features.device,
                        nonzero_ratio=meanflow_nonzero_ratio,
                        distribution=meanflow_time_distribution,
                        logit_mean=meanflow_logit_mean,
                        logit_std=meanflow_logit_std,
                    )
                    loss, _, _ = meanflow_loss(
                        student=student,
                        clean=audio_features,
                        noise=z_1,
                        forward_args=forward_args,
                        t=t,
                        s=s,
                        adaptive_p=meanflow_adaptive_p,
                        adaptive_eps=meanflow_adaptive_eps,
                    )

            loss.backward()
            optimizer.step()

            # --- update EMA ---
            if ema is not None:
                ema.update()

            # --- logging ---
            losses.append(loss.item())

            if (step + 1) % log_every == 0:
                print(
                    f"Epoch {epoch + 1:3d} | Step {step + 1:4d} | "
                    f"{objective}: {losses[-1]:.6f}"
                )

        # --- epoch summary ---
        elapsed = time.time() - epoch_start
        if not losses:
            raise ValueError("dataloader produced no batches")
        avg_loss = sum(losses) / len(losses)
        print(
            f"Epoch {epoch + 1:3d} | "
            f"{objective}_avg: {avg_loss:.6f} | "
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
            "objective": objective,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "num_steps": num_steps,
        }
        if ema is not None:
            state["ema"] = ema.state_dict()
        else:
            state["meanflow"] = {
                "nonzero_ratio": meanflow_nonzero_ratio,
                "time_distribution": meanflow_time_distribution,
                "logit_mean": meanflow_logit_mean,
                "logit_std": meanflow_logit_std,
                "adaptive_p": meanflow_adaptive_p,
                "adaptive_eps": meanflow_adaptive_eps,
            }
        torch.save(state, f"{ckpt_path}/training_state.pt")
        print(f"  -> saved checkpoint to {ckpt_path}")

    return student


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consistency Distillation or MeanFlow for SAM-Audio-Turbo"
    )
    parser.add_argument(
        "--objective",
        choices=("consistency_distillation", "meanflow"),
        default="consistency_distillation",
        help="Turbo training objective",
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
        "--meanflow-nonzero-ratio",
        type=float,
        default=0.25,
        help="Fraction of MeanFlow samples using a non-zero interval",
    )
    parser.add_argument(
        "--meanflow-time-distribution",
        choices=("logit_normal", "uniform"),
        default="logit_normal",
    )
    parser.add_argument("--meanflow-logit-mean", type=float, default=0.4)
    parser.add_argument("--meanflow-logit-std", type=float, default=1.0)
    parser.add_argument("--meanflow-adaptive-p", type=float, default=1.0)
    parser.add_argument("--meanflow-adaptive-eps", type=float, default=0.01)
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
    print(
        f"Mode: {'LoRA' if args.use_lora else 'full-weight'} "
        f"{args.objective}"
    )

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
    elif args.objective == "consistency_distillation":
        teacher = SAMAudio.from_pretrained(args.model_id)
        teacher = teacher.to(model_dtype)
        strip_vision_branch(teacher)

        student = SAMAudio.from_pretrained(args.model_id)
        student = student.to(model_dtype)
        strip_vision_branch(student)
    else:
        # MeanFlow is simulation-free and needs neither teacher nor EMA model.
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
        objective=args.objective,
        num_steps=args.num_steps,
        epochs=args.epochs,
        lr=args.lr,
        ema_decay=args.ema_decay,
        device=device,
        save_dir=args.save_dir,
        meanflow_nonzero_ratio=args.meanflow_nonzero_ratio,
        meanflow_time_distribution=args.meanflow_time_distribution,
        meanflow_logit_mean=args.meanflow_logit_mean,
        meanflow_logit_std=args.meanflow_logit_std,
        meanflow_adaptive_p=args.meanflow_adaptive_p,
        meanflow_adaptive_eps=args.meanflow_adaptive_eps,
    )

    print(f"{args.objective} training complete.")
    print(f"Checkpoints saved to {args.save_dir}/")
