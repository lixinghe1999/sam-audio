"""Consistency Distillation (CD) for SAM-Audio-Turbo.

Replaces DMD (teacher + student + discriminator) with CD
(teacher + student + EMA) for lower memory and more stable training.

Core idea:
    A student predicts the clean endpoint from any noisy state ``x_t``.
    Teacher integrates from ``t`` to ``s`` along the ODE trajectory.
    Student from ``x_t`` and EMA from ``x_s`` must predict the same ``x_0``.

References:
    - Consistency Models (Song et al., 2023)  https://arxiv.org/abs/2303.01469
    - Latent Consistency Models (Luo et al., 2023) https://arxiv.org/abs/2310.04378
    - Multistep Consistency Models (Heek et al., 2024) https://arxiv.org/abs/2403.06807
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


# ── Time discretisation ───────────────────────────────────────────────


def cd_time_steps(num_steps: int = 4) -> list[float]:
    """Linearly spaced time grid from noise (t=0) to clean (t=1).

    This is the convention used by :class:`SAMAudio`: inference integrates
    its vector field from ``0`` to ``1``.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    return [i / num_steps for i in range(num_steps + 1)]


def cd_intervals(num_steps: int = 4) -> list[tuple[float, float]]:
    """Non-overlapping ``(noisier, cleaner)`` training intervals."""
    steps = cd_time_steps(num_steps)
    return [(steps[i], steps[i + 1]) for i in range(num_steps)]


# ── Flow-matching utilities ──────────────────────────────────────────


def velocity_to_clean(
    x_t: torch.Tensor,
    t: torch.Tensor,
    v_pred: torch.Tensor,
) -> torch.Tensor:
    """Convert predicted velocity to clean estimate via flow matching.

    SAM-Audio flow:  ``x_t = (1-t) * noise + t * clean``
    Vector field:    ``v = clean - noise``
    Therefore:       ``clean = x_t + (1-t) * v``.
    """
    scale = (1 - t).reshape((-1,) + (1,) * (x_t.ndim - 1))
    return x_t + scale * v_pred


def flow_interpolate(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    """Interpolate from ``x_0`` at time 0 to ``x_1`` at time 1."""
    if isinstance(t, float):
        return t * x_1 + (1 - t) * x_0
    return t[:, None, None] * x_1 + (1 - t[:, None, None]) * x_0


def pseudo_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    """Pseudo-Huber loss, more robust to outliers than MSE."""
    diff = pred - target
    return (delta**2) * ((1 + (diff / delta) ** 2).sqrt() - 1).mean()


# ── Teacher ODE segment (integrate from t to s) ─────────────────────


def _teacher_ode_segment(
    vector_field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z_t: torch.Tensor,
    t_start: float,
    t_end: float,
) -> torch.Tensor:
    """Integrate from ``t_start`` to ``t_end`` using teacher ODE.

    Supports backward integration (t_start > t_end).
    Uses adaptive number of midpoint steps proportional to interval length.
    """
    dt = abs(t_start - t_end)
    # Match original teacher's 16 step / full [0,1] schedule proportionally
    n_steps = max(1, round(16 * dt))
    step_size = dt / n_steps

    device = z_t.device
    states = odeint(
        vector_field_fn,
        z_t,
        torch.tensor([t_start, t_end], device=device),
        method="midpoint",
        options={"step_size": step_size},
    )
    return states[-1]


# ── EMA helper ──────────────────────────────────────────────────────


class EMAHelper:
    """Exponential Moving Average of student parameters.

    Usage::

        ema = EMAHelper(student.parameters(), decay=0.999)
        for step in steps:
            ...
            loss.backward()
            optimizer.step()
            ema.update()

        # Inference with EMA weights
        ema_model = ema.ema_model(student)
        ema_model.eval()
    """

    def __init__(
        self,
        params,
        decay: float = 0.999,
    ):
        if isinstance(params, nn.Parameter):
            params = [params]
        params = list(params)
        if params and isinstance(params[0], tuple):
            self.names = [name for name, _ in params]
            self._params = [param for _, param in params]
        else:
            self.names = None
            self._params = params
        self.decay = decay
        self.shadows: list[torch.Tensor] = []
        for p in self._params:
            self.shadows.append(p.data.clone().detach())
        self._update_count = 0
        # Cached frozen model — set externally via set_frozen_model()
        # to avoid per-step copy.deepcopy (which breaks on weight_norm).
        self._frozen_model: nn.Module | None = None

    def update(self, params: list[nn.Parameter] | None = None) -> None:
        """Update shadow weights:  shadow = decay * shadow + (1 - decay) * param."""
        self._update_count += 1
        # Use true decay (warmup from 0 in first few steps)
        d = min(self.decay, (1 + self._update_count) / (10 + self._update_count))

        current = self._params if params is None else list(params)
        if len(current) != len(self.shadows):
            raise ValueError("EMA parameter count changed during training")
        with torch.no_grad():
            for shadow, param in zip(self.shadows, current):
                shadow.lerp_(param.detach(), 1 - d)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "names": self.names,
            "shadows": self.shadows,
            "_update_count": self._update_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = state["decay"]
        self.names = state.get("names", self.names)
        if len(state["shadows"]) != len(self._params):
            raise ValueError("EMA checkpoint parameter count does not match model")
        self.shadows = [
            shadow.to(device=param.device, dtype=param.dtype)
            for shadow, param in zip(state["shadows"], self._params)
        ]
        self._update_count = state["_update_count"]

    def copy_to(self, model: nn.Module) -> None:
        """Copy shadow weights into model (for eval / checkpoint)."""
        if self.names is None:
            params = list(model.parameters())
        else:
            named = dict(model.named_parameters())
            params = [named[name] for name in self.names]
        for shadow, param in zip(self.shadows, params):
            param.data.copy_(shadow.data)

    def forward(self, model: nn.Module, **kwargs) -> torch.Tensor:
        """Run an EMA-weighted forward without cloning or mutating ``model``.

        Only the tracked parameters are substituted. Frozen base weights and
        buffers are shared, which is especially important for LoRA training.
        """
        if self.names is None:
            raise RuntimeError(
                "EMAHelper.forward requires (name, parameter) pairs at construction"
            )
        from torch.func import functional_call

        replacements = dict(zip(self.names, self.shadows))
        training = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            return functional_call(model, replacements, (), kwargs, strict=False)
        finally:
            # Restore each flag directly so nested frozen encoders remain frozen.
            for module, was_training in training:
                module.training = was_training

    def set_frozen_model(self, model: nn.Module) -> None:
        """Set a static copy of the student model for EMA weight loading.

        Call once before training to avoid per-step ``copy.deepcopy``
        (which fails on models with ``weight_norm`` hooks).
        The caller is responsible for constructing the clone using the same
        architecture (no deepcopy needed at this call site).
        """
        self._frozen_model = model

    def ema_model(self, model: nn.Module) -> nn.Module:
        """Return a copy of ``model`` with EMA weights loaded (inference only).

        When a frozen model has been registered via ``set_frozen_model()``,
        this simply copies the EMA shadows into it — no deepcopy per call.

        On first call without a pre-registered frozen model, attempts
        ``copy.deepcopy`` and falls back to state-dict reconstruction
        (works around weight_norm deepcopy limitations in timm layers).
        The constructed clone is cached for subsequent calls.
        """
        if self._frozen_model is not None:
            self.copy_to(self._frozen_model)
            self._frozen_model.eval()
            for p in self._frozen_model.parameters():
                p.requires_grad_(False)
            return self._frozen_model

        # --- first call: create EMA model copy ---
        try:
            ema = copy.deepcopy(model)
        except RuntimeError:
            # weight_norm non-leaf tensor → use state-dict reconstruction
            state = copy.deepcopy(model.state_dict())
            if hasattr(model, "base_model") and hasattr(model, "peft_config"):
                # PeftModel: reconstruct base model + re-apply LoRA
                from peft import get_peft_model

                base = model.base_model.__class__(
                    model.base_model.config
                )
                base.load_state_dict(
                    copy.deepcopy(model.base_model.state_dict()),
                    strict=False,
                )
                ema = get_peft_model(base, model.peft_config)
                ema.load_state_dict(state, strict=False)
            elif hasattr(model, "config"):
                ema = model.__class__(model.config)
                ema.load_state_dict(state)
            else:
                raise RuntimeError(
                    f"Cannot clone {type(model).__name__}: "
                    "no config or PeftModel attributes. "
                    "Call set_frozen_model() before training."
                )

        ema.eval()
        self.copy_to(ema)
        for p in ema.parameters():
            p.requires_grad_(False)
        self._frozen_model = ema  # cache
        return ema


# ── Consistency loss ────────────────────────────────────────────────


def _make_vf_fn(
    model: nn.Module,
    forward_args: dict[str, Any],
    *,
    disable_adapters: bool = False,
    force_eval: bool = False,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Wrap model forward as ``(t, z_t) → v`` for ODE integration."""
    def vf_fn(t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(z_t.size(0))
        adapter_context = (
            model.disable_adapter()
            if disable_adapters and hasattr(model, "disable_adapter")
            else nullcontext()
        )
        training = None
        if force_eval:
            training = [(module, module.training) for module in model.modules()]
            model.eval()
        try:
            with adapter_context:
                return model(noisy_audio=z_t, time=t, **forward_args)
        finally:
            if training is not None:
                for module, was_training in training:
                    module.training = was_training

    return vf_fn


def consistency_loss(
    teacher_vf: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    student: nn.Module,
    ema_model: nn.Module | EMAHelper,
    z_0: torch.Tensor,
    z_1: torch.Tensor,
    forward_args: dict[str, Any],
    t: float,
    s: float,
    loss_fn: str = "huber",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute consistency distillation loss for one (t, s) interval.

    Args:
        teacher_vf: Frozen teacher vector field ``(t, z_t) → v``.
        student: Trainable student model.
        ema_model: EMA-weighted student (frozen).
        z_0: Clean data ``[B, T, C]`` (audio_features).
        z_1: Noise ``[B, T, C]`` (randn_like(z_0)).
        forward_args: Conditioning forwarded to model.
        t: Noisier time point (e.g. 0.0).
        s: Cleaner time point (e.g. 0.25). Must satisfy ``t < s``.
        loss_fn: ``"mse"``, ``"l1"``, or ``"huber"`` (default).

    Returns:
        ``(loss, x0_student, x0_target)`` where ``loss`` has gradient
        attached to student parameters.
    """
    B = z_0.size(0)
    device = z_0.device

    # 1. Construct x_t and x_s
    t_s = torch.full((B,), t, device=device)
    s_s = torch.full((B,), s, device=device)
    if not 0 <= t < s <= 1:
        raise ValueError(f"expected 0 <= t < s <= 1, got t={t}, s={s}")
    x_t = flow_interpolate(z_1, z_0, t)  # noise at 0, clean at 1

    # 2. Teacher integrates from t → s (no grad)
    with torch.no_grad():
        x_s = _teacher_ode_segment(teacher_vf, x_t, t_start=t, t_end=s)

    # 3. EMA predicts clean from x_s (no grad). Do this before the student
    # forward so an implementation that swaps weights cannot invalidate its graph.
    with torch.no_grad():
        kwargs = dict(noisy_audio=x_s, time=s_s, **forward_args)
        if isinstance(ema_model, EMAHelper):
            v_ema = ema_model.forward(student, **kwargs)
        else:
            v_ema = ema_model(**kwargs)
        x0_target = velocity_to_clean(x_s, s_s, v_ema)

    # 4. Student predicts clean from x_t (with grad)
    v_student = student(noisy_audio=x_t, time=t_s, **forward_args)
    x0_student = velocity_to_clean(x_t, t_s, v_student)

    # 5. Consistency distance
    if loss_fn == "mse":
        loss = F.mse_loss(x0_student, x0_target.detach())
    elif loss_fn == "l1":
        loss = F.l1_loss(x0_student, x0_target.detach())
    else:
        loss = pseudo_huber_loss(x0_student, x0_target.detach())

    return loss, x0_student, x0_target


# ── Multistep consistency sampling (inference) ──────────────────────


@torch.inference_mode()
def multistep_sample(
    student: nn.Module,
    forward_args: dict[str, Any],
    num_steps: int = 4,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """4-step consistency sampling for inference.

    Procedure:
        1. Start from noise at t=0.
        2. Predict clean:  x0_hat = f_theta(x_{k}, k).
        3. Re-noise to next interval:  x_{k_next} = k_next * noise + (1 - k_next) * x0_hat.
        4. Repeat until t=0.

    Args:
        student:  Student model (in eval mode).
        forward_args:  Conditioning dict (must include ``audio_features``
            for shape inference).
        num_steps:  Number of denoising steps (default 4).
        noise:  Optional initial noise.  If ``None``, randn is used.

    Returns:
        ``x0`` -- predicted clean latent ``[B, T, C]``.
    """
    audio_features = forward_args["audio_features"]
    B, T, C = audio_features.shape
    device = audio_features.device

    if noise is None:
        noise = torch.randn_like(audio_features)

    steps = cd_time_steps(num_steps)  # [0.0, ..., 1.0]

    # Start from full noise
    z = noise  # x_{t=0}

    for i in range(num_steps):
        k = steps[i]
        k_next = steps[i + 1]
        t_k = torch.full((B,), k, device=device)

        v = student(
            noisy_audio=z,
            time=t_k,
            **forward_args,
        )
        x0_hat = velocity_to_clean(z, t_k, v)  # predict clean

        if k_next < 1:
            # Re-noise to next timestep (flow interpolation)
            noise_next = torch.randn_like(z)
            z = flow_interpolate(noise_next, x0_hat, k_next)
        else:
            z = x0_hat

    return z  # x_0 (clean estimate)
