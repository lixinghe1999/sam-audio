"""MeanFlow training and sampling for SAM-Audio.

SAM-Audio uses the forward convention ``noise @ t=0 -> data @ t=1``.
For an interval ending at ``s >= t``, the model predicts average velocity
``u(z_t, t, h)`` with ``h = s - t``.  In this convention the MeanFlow identity
is ``u = v + h * D_t u`` while holding the interval endpoint fixed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def sample_meanflow_times(
    batch_size: int,
    device: torch.device,
    *,
    nonzero_ratio: float = 0.25,
    distribution: str = "logit_normal",
    logit_mean: float = 0.4,
    logit_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample current/end times ``t <= s`` for MeanFlow training.

    ``1 - nonzero_ratio`` of samples use ``s=t`` and therefore retain the
    ordinary instantaneous flow-matching objective.  The default logit mean
    is the sign-flipped equivalent of the paper's ``-0.4`` because SAM-Audio's
    flow direction is reversed.
    """
    if not 0 <= nonzero_ratio <= 1:
        raise ValueError("nonzero_ratio must be between 0 and 1")
    if distribution not in {"uniform", "logit_normal"}:
        raise ValueError(
            "distribution must be either 'uniform' or 'logit_normal'"
        )

    def draw() -> torch.Tensor:
        if distribution == "uniform":
            return torch.rand(batch_size, device=device)
        return torch.sigmoid(
            torch.randn(batch_size, device=device) * logit_std + logit_mean
        )

    first, second = draw(), draw()
    t, s = torch.minimum(first, second), torch.maximum(first, second)
    use_interval = torch.rand(batch_size, device=device) < nonzero_ratio
    s = torch.where(use_interval, s, t)
    return t, s


def meanflow_loss(
    student: nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    forward_args: dict[str, Any],
    t: torch.Tensor,
    s: torch.Tensor,
    *,
    adaptive_p: float = 1.0,
    adaptive_eps: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the MeanFlow identity loss using a memory-conscious JVP.

    The prediction carrying parameter gradients is evaluated normally.  The
    Jacobian-vector product used to build the stopped target is evaluated
    separately under ``no_grad``, as recommended by the official PyTorch
    implementation when compilation is unavailable.
    """
    if t.shape != s.shape or t.ndim != 1 or t.size(0) != clean.size(0):
        raise ValueError("t and s must be [batch] tensors")
    if torch.any(t > s) or torch.any(t < 0) or torch.any(s > 1):
        raise ValueError("MeanFlow times must satisfy 0 <= t <= s <= 1")
    if adaptive_p < 0:
        raise ValueError("adaptive_p must be non-negative")
    if adaptive_eps <= 0:
        raise ValueError("adaptive_eps must be positive")

    h = s - t
    velocity = clean - noise
    t_view = t.reshape((-1,) + (1,) * (clean.ndim - 1))
    z_t = (1 - t_view) * noise + t_view * clean

    def u_fn(
        z_value: torch.Tensor,
        t_value: torch.Tensor,
        h_value: torch.Tensor,
    ) -> torch.Tensor:
        return student(
            noisy_audio=z_value,
            time=t_value,
            flow_interval=h_value,
            **forward_args,
        )

    # Keep the parameter-gradient path independent from the JVP target path.
    u_pred = u_fn(z_t, t, h)
    with torch.no_grad():
        from torch.func import jvp

        # s is fixed along the derivative, hence d(s-t)/dt = -1.
        _, du_dt = jvp(
            u_fn,
            (z_t, t, h),
            (velocity, torch.ones_like(t), -torch.ones_like(h)),
        )
        h_view = h.reshape((-1,) + (1,) * (clean.ndim - 1))
        target = velocity + h_view * du_dt

    per_sample = (u_pred.float() - target.float()).square().flatten(1).sum(1)
    if adaptive_p:
        weight = (per_sample.detach() + adaptive_eps).pow(adaptive_p)
        per_sample = per_sample / weight
    return per_sample.mean(), u_pred, target


@torch.inference_mode()
def meanflow_sample(
    student: nn.Module,
    forward_args: dict[str, Any],
    *,
    num_steps: int = 1,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate with one or more MeanFlow average-velocity steps."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    reference = forward_args["audio_features"]
    if noise is None:
        noise = torch.randn_like(reference)
    z = noise
    batch_size = z.size(0)
    for index in range(num_steps):
        t_value = index / num_steps
        s_value = (index + 1) / num_steps
        t = torch.full((batch_size,), t_value, device=z.device)
        h = torch.full((batch_size,), s_value - t_value, device=z.device)
        u = student(
            noisy_audio=z,
            time=t,
            flow_interval=h,
            **forward_args,
        )
        z = z + h.reshape((-1,) + (1,) * (z.ndim - 1)) * u
    return z
