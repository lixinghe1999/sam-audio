#!/usr/bin/env python3
"""Few-step SAM-Audio training with CD, MeanFlow, or AlphaFlow.

Consistency distillation uses teacher ODE segments and an EMA target.
MeanFlow is simulation-free: it learns interval-average velocity using the
MeanFlow identity and a no-grad Jacobian-vector product (JVP).
AlphaFlow uses a finite self-distillation step controlled by alpha, avoiding
the JVP for alpha > 0.

Usage:
    # Full-weight distillation (4-step student)
    python turbo_train.py --num-steps 4 --epochs 20

    # LoRA-based distillation (smaller GPU footprint)
    python turbo_train.py --num-steps 4 --use-lora --lora-rank 8

    # MeanFlow (alpha=0: JVP objective; no teacher or EMA)
    python turbo_train.py --objective meanflow --use-lora --epochs 20

    # Enable finite-step AlphaFlow through the same MeanFlow objective
    python turbo_train.py --objective meanflow --alpha 0.5 --use-lora

    # Four-GPU DDP (automatically detected from torchrun environment variables)
    torchrun --standalone --nproc_per_node=4 turbo_train.py \
        --objective meanflow --use-lora
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.dataset import build_dataloader
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
from sam_audio.turbo.meanflow import (
    alphaflow_loss,
    scheduled_alpha,
    sample_meanflow_times,
)


# ── training loop ────────────────────────────────────────────────────────


def _distributed_average(
    total: float,
    count: int,
    device: torch.device,
) -> float:
    """Average a scalar sum across the initialized process group."""
    values = torch.tensor([total, count], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return (values[0] / values[1].clamp_min(1)).item()


def _checkpoint_state_dict(
    student: nn.Module,
    ema: EMAHelper | None,
) -> dict[str, torch.Tensor]:
    """Build directly loadable weights, substituting EMA parameters for CD."""
    state_dict = student.state_dict()
    if ema is None:
        return state_dict
    if ema.names is None:
        raise RuntimeError("named EMA parameters are required for checkpointing")

    missing = [name for name in ema.names if name not in state_dict]
    if missing:
        raise RuntimeError(
            "EMA parameters are missing from the student state dict: "
            + ", ".join(missing[:3])
        )
    for name, shadow in zip(ema.names, ema.shadows):
        state_dict[name] = shadow.detach()
    return state_dict


def turbo_train(
    student: nn.Module,
    processor: SAMAudioProcessor,
    dataloader: DataLoader,
    *,
    teacher: nn.Module | None = None,
    objective: str = "consistency_distillation",
    num_steps: int = 4,
    epochs: int = 20,
    lr: float | None = None,
    ema_decay: float = 0.999,
    device: str = "cuda",
    save_dir: str = "turbo-checkpoint",
    log_every: int = 5,
    val_loader: DataLoader | None = None,
    val_every: int = 1,
    meanflow_nonzero_ratio: float = 0.5,
    meanflow_time_distribution: str = "logit_normal",
    meanflow_logit_mean: float = -0.4,
    meanflow_logit_std: float = 1.0,
    meanflow_adaptive_p: float = 1.0,
    meanflow_adaptive_eps: float = 0.001,
    alpha: float | None = None,
    alpha_start: float | None = None,
    alpha_end: float | None = None,
    alpha_schedule: str = "sigmoid",
    alpha_warmup_ratio: float = 0.0,
    alpha_transition_ratio: float = 1.0,
    alpha_sigmoid_gamma: float = 25.0,
    gradient_accumulation_steps: int | None = None,
    gradient_clip_norm: float | None = None,
):
    """Train SAM-Audio with consistency distillation, MeanFlow, or AlphaFlow.

    Args:
        student: Trainable student SAM-Audio (may be PeftModel).
        processor: SAMAudioProcessor for batch encoding.
        dataloader: DataLoader yielding ``(audios, descriptions)`` tuples.
        teacher: Frozen teacher SAM-Audio. When ``None`` (LoRA mode), the
            student's adapters are disabled for teacher calls, so the frozen
            pretrained base supplies the CD teacher without a second model copy.
            MeanFlow and AlphaFlow do not use a teacher.
        objective: ``"consistency_distillation"`` or ``"meanflow"``.
        num_steps: Number of CD intervals (1 / 2 / 4).
        epochs: Training epochs.
        lr: Learning rate. Defaults to ``1e-5`` for CD and ``1e-4`` for
            MeanFlow/AlphaFlow.
        ema_decay: EMA decay rate for target student.
        device: Training device.
        save_dir: Checkpoint directory.
        log_every: Log metrics every N steps.
        val_loader: Optional DataLoader for validation after each epoch.
        val_every: Validate every N epochs (default 1).
        meanflow_nonzero_ratio: Probability that a MeanFlow batch uses
            ``s != t``. Remaining batches train instantaneous velocity at
            ``s=t``.
        meanflow_time_distribution: ``"logit_normal"`` or ``"uniform"``.
        meanflow_logit_mean: Interval logit-normal mean. MeanFlow-TSE uses
            ``-0.4`` with the shared noise-at-zero time convention.
        meanflow_logit_std: Logit-normal standard deviation.
        meanflow_adaptive_p: Exponent for adaptive MeanFlow loss weighting.
        meanflow_adaptive_eps: Stabilizer for adaptive loss weighting.
        alpha: Fixed MeanFlow/AlphaFlow consistency-step ratio. ``0`` uses the
            original JVP MeanFlow objective; positive values enable finite-step
            AlphaFlow. When neither a fixed alpha nor curriculum endpoints are
            supplied, MeanFlow-TSE's ``1.0 -> 0.005`` curriculum is used.
        alpha_start: Optional curriculum starting alpha. Must be provided with
            ``alpha_end``; when set, it supersedes the fixed ``alpha``.
        alpha_end: Optional curriculum final alpha.
        alpha_schedule: Curriculum transition shape, ``"sigmoid"`` or
            ``"linear"``.
        alpha_warmup_ratio: Fraction of training held at ``alpha_start``.
        alpha_transition_ratio: Fraction annealing from start to end.
        alpha_sigmoid_gamma: Steepness of the sigmoid transition.
        gradient_accumulation_steps: Microbatches per optimizer update.
            Defaults to 1 for CD and 2 for MeanFlow/AlphaFlow.
        gradient_clip_norm: Maximum gradient norm. Defaults to disabled for CD
            and 0.5 for MeanFlow/AlphaFlow.
    """
    # ── freeze teacher ──────────────────────────────────────────────
    device = torch.device(device)
    amp_enabled = device.type == "cuda"
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    is_main_process = rank == 0
    valid_objectives = {"consistency_distillation", "meanflow"}
    if objective not in valid_objectives:
        raise ValueError(
            f"objective must be one of {sorted(valid_objectives)}, got {objective!r}"
        )
    use_cd = objective == "consistency_distillation"
    if lr is None:
        lr = 1e-5 if use_cd else 1e-4
    if lr <= 0:
        raise ValueError("lr must be positive")
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 1 if use_cd else 2
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if gradient_clip_norm is None:
        gradient_clip_norm = 0.0 if use_cd else 0.5
    if gradient_clip_norm < 0:
        raise ValueError("gradient_clip_norm must be non-negative")
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
        if alpha is None and alpha_start is None and alpha_end is None:
            alpha_start, alpha_end = 1.0, 0.005
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("MeanFlow alpha must satisfy 0 <= alpha <= 1")
        curriculum_enabled = alpha_start is not None or alpha_end is not None
        if curriculum_enabled:
            if alpha_start is None or alpha_end is None:
                raise ValueError("alpha_start and alpha_end must be set together")
            if alpha is not None:
                raise ValueError("fixed alpha and alpha curriculum cannot be combined")
            # Validate all schedule settings before allocating model state.
            scheduled_alpha(
                0,
                2,
                start=alpha_start,
                end=alpha_end,
                schedule=alpha_schedule,
                warmup_ratio=alpha_warmup_ratio,
                transition_ratio=alpha_transition_ratio,
                sigmoid_gamma=alpha_sigmoid_gamma,
            )
        else:
            curriculum_enabled = False
            if alpha is None:
                raise ValueError("a fixed alpha or alpha curriculum is required")
    else:
        curriculum_enabled = False

    if use_cd and teacher is not None:
        teacher.eval()
        teacher.to(device)
        for p in teacher.parameters():
            p.requires_grad = False

    # ── setup student ───────────────────────────────────────────────
    # Full-weight mixed-precision training keeps FP32 master parameters and
    # relies on autocast for BF16 compute. Updating BF16 parameters directly
    # can round away small optimizer steps. LoRA retains its low-precision,
    # frozen base while PEFT keeps trainable adapters in a suitable dtype.
    if hasattr(student, "peft_config"):
        student.to(device)
    else:
        student.to(device=device, dtype=torch.float32)
    student.train()
    freeze_encoders(student)

    # These modules only build forward_args and are never traversed by the
    # student's vector-field forward. Freeze them explicitly; eval() alone
    # does not clear requires_grad. With a separate teacher, its encoders build
    # the conditioning, so the duplicate student copies can stay on CPU.
    raw_student = student
    student_base = getattr(raw_student, "base_model", raw_student)
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

    # Keep the online MeanFlow/AlphaFlow model in training mode, matching
    # MeanFlow-TSE.  In particular, some PEFT/LoRA versions merge adapters in
    # eval mode, which can disconnect the prediction from trainable adapter
    # parameters after a stopped target forward.

    # ── EMA student (shadow weights, no optimizer) ──────────────────
    trainable_named = [
        (name, param)
        for name, param in raw_student.named_parameters()
        if param.requires_grad
    ]
    if not trainable_named:
        raise ValueError("student has no trainable parameters")
    ema = EMAHelper(trainable_named, decay=ema_decay) if use_cd else None

    # ── optimizer (single, no discriminator) ────────────────────────
    optimizer_kwargs = {"lr": lr}
    if use_cd:
        # Preserve the existing consistency-distillation optimizer recipe.
        optimizer_kwargs["betas"] = (0.5, 0.9)
    optimizer = torch.optim.AdamW(
        [param for _, param in trainable_named], **optimizer_kwargs
    )

    if distributed:
        ddp_kwargs = {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            ddp_kwargs.update(
                device_ids=[device.index],
                output_device=device.index,
            )
        student = DistributedDataParallel(raw_student, **ddp_kwargs)

    # ── stats ───────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in raw_student.parameters() if p.requires_grad)
    if is_main_process:
        print(f"Student trainable params: {trainable:,}")
        print(f"Training objective:       {objective}")
        print(f"Learning rate:            {lr}")
        if distributed:
            print(f"DDP world size:           {dist.get_world_size()}")
        if use_cd:
            print(f"CD intervals:             {num_steps}")
            print(f"EMA decay:                {ema_decay}")
        else:
            print(f"Flow interval ratio:      {meanflow_nonzero_ratio}")
            print(f"Gradient accumulation:    {gradient_accumulation_steps}")
            print(f"Gradient clip norm:       {gradient_clip_norm}")
            if curriculum_enabled:
                print(f"Alpha curriculum:         {alpha_start} -> {alpha_end}")
                print(f"Alpha schedule:           {alpha_schedule}")
            else:
                flow_method = "MeanFlow" if alpha == 0 else "AlphaFlow"
                print(f"Flow method:              {flow_method}")
                print(f"Flow alpha:               {alpha}")
        os.makedirs(save_dir, exist_ok=True)
    if distributed:
        dist.barrier()

    # ── epoch loop ──────────────────────────────────────────────────
    steps_per_epoch = len(dataloader)
    total_train_steps = epochs * steps_per_epoch
    current_alpha = alpha if alpha is not None else alpha_start
    for epoch in range(epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        epoch_start = time.time()
        losses = []
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            disable=not is_main_process,
        )
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            if curriculum_enabled:
                global_step = epoch * steps_per_epoch + step
                current_alpha = scheduled_alpha(
                    global_step,
                    total_train_steps,
                    start=alpha_start,
                    end=alpha_end,
                    schedule=alpha_schedule,
                    warmup_ratio=alpha_warmup_ratio,
                    transition_ratio=alpha_transition_ratio,
                    sigmoid_gamma=alpha_sigmoid_gamma,
                )
            # --- unpack / encode ---
            if isinstance(batch, tuple):
                audios, descriptions = batch
                batch = processor(audios=audios, descriptions=descriptions)
            batch = batch.to(device)

            conditioning_model = teacher if use_cd else raw_student
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

            should_step = (
                (step + 1) % gradient_accumulation_steps == 0
                or step + 1 == steps_per_epoch
            )
            sync_context = (
                student.no_sync()
                if distributed and not should_step
                else nullcontext()
            )
            with sync_context:
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
                            target_student=(
                                raw_student if distributed else None
                            ),
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
                        loss, _, _ = alphaflow_loss(
                            student=student,
                            target_student=(
                                raw_student if distributed else None
                            ),
                            clean=audio_features,
                            noise=z_1,
                            forward_args=forward_args,
                            t=t,
                            s=s,
                            alpha=current_alpha,
                            adaptive_p=meanflow_adaptive_p,
                            adaptive_eps=meanflow_adaptive_eps,
                        )

                (loss / gradient_accumulation_steps).backward()
            if should_step:
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for _, param in trainable_named],
                        gradient_clip_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # --- update EMA once per optimizer update ---
                if ema is not None:
                    ema.update()

            # --- logging ---
            losses.append(loss.item())
            if is_main_process:
                if use_cd:
                    pbar.set_postfix(loss=f"{losses[-1]:.6f}")
                else:
                    pbar.set_postfix(
                        loss=f"{losses[-1]:.6f}",
                        alpha=f"{current_alpha:.4f}",
                    )

        # --- epoch summary ---
        elapsed = time.time() - epoch_start
        if not losses:
            raise ValueError("dataloader produced no batches")
        avg_loss = _distributed_average(sum(losses), len(losses), device)
        if is_main_process:
            print(
                f"Epoch {epoch + 1:3d} | "
                f"{objective}_avg: {avg_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        # --- checkpoint ---
        ckpt_path = f"{save_dir}/epoch-{epoch + 1}"
        state = {
            "objective": objective,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "num_steps": num_steps,
            "saved_model_weights": "ema" if ema is not None else "online",
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
                "alpha": current_alpha,
                "alpha_curriculum": {
                    "enabled": curriculum_enabled,
                    "start": alpha_start,
                    "end": alpha_end,
                    "schedule": alpha_schedule,
                    "warmup_ratio": alpha_warmup_ratio,
                    "transition_ratio": alpha_transition_ratio,
                    "sigmoid_gamma": alpha_sigmoid_gamma,
                },
            }
        if is_main_process:
            checkpoint_weights = _checkpoint_state_dict(raw_student, ema)
            if hasattr(raw_student, "save_pretrained"):
                raw_student.save_pretrained(
                    ckpt_path, state_dict=checkpoint_weights
                )
            else:
                os.makedirs(ckpt_path, exist_ok=True)
                torch.save(checkpoint_weights, f"{ckpt_path}/model.pt")
            torch.save(state, f"{ckpt_path}/training_state.pt")
            print(f"  -> saved checkpoint to {ckpt_path}")
        if distributed:
            dist.barrier()

        # --- validation ---
        if val_loader is not None and (epoch + 1) % val_every == 0:
            student.eval()
            val_losses = []
            val_pbar = tqdm(
                val_loader,
                desc=f"Val {epoch+1}/{epochs}",
                disable=not is_main_process,
            )
            for batch in val_pbar:
                if isinstance(batch, tuple):
                    audios, descriptions = batch
                    batch = processor(audios=audios, descriptions=descriptions)
                batch = batch.to(device)

                conditioning_model = teacher if use_cd else raw_student
                base_t = getattr(conditioning_model, "base_model", conditioning_model)

                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    forward_args = base_t._get_forward_args(batch)
                del batch

                audio_features = forward_args["audio_features"]
                z_1 = torch.randn_like(audio_features)

                with torch.no_grad(), torch.autocast(
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
                            target_student=(
                                raw_student if distributed else None
                            ),
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
                        loss, _, _ = alphaflow_loss(
                            student=student,
                            target_student=(
                                raw_student if distributed else None
                            ),
                            clean=audio_features,
                            noise=z_1,
                            forward_args=forward_args,
                            t=t,
                            s=s,
                            alpha=current_alpha,
                            adaptive_p=meanflow_adaptive_p,
                            adaptive_eps=meanflow_adaptive_eps,
                        )

                val_losses.append(loss.item())
                if is_main_process:
                    val_pbar.set_postfix(val_loss=f"{loss.item():.6f}")

            avg_val = _distributed_average(
                sum(val_losses), len(val_losses), device
            )
            if is_main_process:
                print(f"  val_loss: {avg_val:.6f}")

            # restore training mode
            student.train()
            freeze_encoders(raw_student)

    return raw_student


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
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: CD 1e-5, MeanFlow 1e-4)",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.999, help="EMA decay for target student"
    )
    parser.add_argument(
        "--meanflow-nonzero-ratio",
        type=float,
        default=0.5,
        help="Probability of a MeanFlow batch using a non-zero interval",
    )
    parser.add_argument(
        "--meanflow-time-distribution",
        choices=("logit_normal", "uniform"),
        default="logit_normal",
    )
    parser.add_argument("--meanflow-logit-mean", type=float, default=-0.4)
    parser.add_argument("--meanflow-logit-std", type=float, default=1.0)
    parser.add_argument("--meanflow-adaptive-p", type=float, default=1.0)
    parser.add_argument("--meanflow-adaptive-eps", type=float, default=0.001)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "fixed alpha; omitted uses the MeanFlow-TSE 1.0 -> 0.005 "
            "curriculum"
        ),
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        default=None,
        help="Starting curriculum alpha; requires --alpha-end",
    )
    parser.add_argument(
        "--alpha-end",
        type=float,
        default=None,
        help="Final curriculum alpha; requires --alpha-start",
    )
    parser.add_argument(
        "--alpha-schedule",
        choices=("sigmoid", "linear"),
        default="sigmoid",
    )
    parser.add_argument("--alpha-warmup-ratio", type=float, default=0.0)
    parser.add_argument("--alpha-transition-ratio", type=float, default=1.0)
    parser.add_argument("--alpha-sigmoid-gamma", type=float, default=25.0)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="default: CD 1, MeanFlow/AlphaFlow 2",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=None,
        help="default: disabled for CD, 0.5 for MeanFlow/AlphaFlow",
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
    parser.add_argument(
        "--val",
        action="store_true",
        default=False,
        help="Enable validation on Clotho validation split after each epoch",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Validate every N epochs (default: 1)",
    )
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
        backend = "nccl" if use_cuda else "gloo"
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = torch.device(
            args.device if torch.cuda.is_available() else "cpu"
        )
    is_main_process = not distributed or dist.get_rank() == 0
    compute_dtype = (
        torch.bfloat16 if device.type != "cpu" else torch.float32
    )
    if is_main_process:
        print(f"Device: {device}")
        print(
            f"Mode: {'LoRA' if args.use_lora else 'full-weight'} "
            f"{args.objective}"
        )

    # ── Load model(s) ─────────────────────────────────────────────────
    teacher = None
    if args.use_lora:
        student = SAMAudio.from_pretrained(args.model_id)
        student = student.to(compute_dtype)
        strip_vision_branch(student)
        student = apply_lora(student, rank=args.lora_rank, alpha=16)
        freeze_encoders(student)
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student.parameters())
        if is_main_process:
            print(
                f"LoRA trainable: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.1f}%)"
            )
    elif args.objective == "consistency_distillation":
        teacher = SAMAudio.from_pretrained(args.model_id)
        teacher = teacher.to(compute_dtype)
        strip_vision_branch(teacher)

        # Keep full-weight student parameters in FP32. turbo_train uses BF16
        # autocast for forward/backward compute without sacrificing updates.
        student = SAMAudio.from_pretrained(args.model_id)
        strip_vision_branch(student)
    else:
        # MeanFlow/AlphaFlow need neither an external teacher nor an EMA model.
        student = SAMAudio.from_pretrained(args.model_id)
        strip_vision_branch(student)

    # ── Processor ─────────────────────────────────────────────────────
    processor = SAMAudioProcessor.from_pretrained(args.model_id)

    # ── Dataset ───────────────────────────────────────────────────────
    train_loader = build_dataloader(
        split="train", data_root=args.data_dir,
        batch_size=args.batch_size, num_workers=4,
        distributed=distributed,
    )
    if is_main_process:
        print(f"Train dataset: {len(train_loader.dataset)} items")

    val_loader = None
    if args.val:
        val_loader = build_dataloader(
            split="val", data_root=args.data_dir,
            batch_size=args.batch_size, num_workers=4,
            distributed=distributed,
        )
        if is_main_process:
            print(f"Val dataset:   {len(val_loader.dataset)} items")

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
        val_loader=val_loader,
        val_every=args.val_every,
        meanflow_nonzero_ratio=args.meanflow_nonzero_ratio,
        meanflow_time_distribution=args.meanflow_time_distribution,
        meanflow_logit_mean=args.meanflow_logit_mean,
        meanflow_logit_std=args.meanflow_logit_std,
        meanflow_adaptive_p=args.meanflow_adaptive_p,
        meanflow_adaptive_eps=args.meanflow_adaptive_eps,
        alpha=args.alpha,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_schedule=args.alpha_schedule,
        alpha_warmup_ratio=args.alpha_warmup_ratio,
        alpha_transition_ratio=args.alpha_transition_ratio,
        alpha_sigmoid_gamma=args.alpha_sigmoid_gamma,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_norm=args.gradient_clip_norm,
    )

    if is_main_process:
        print(f"{args.objective} training complete.")
        print(f"Checkpoints saved to {args.save_dir}/")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
