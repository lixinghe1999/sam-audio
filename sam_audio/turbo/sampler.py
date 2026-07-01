"""Few-step ODE samplers for flow-matching models.

Replaces ``torchdiffeq.odeint`` (16 midpoint steps) with 1--4 explicit Euler
steps for ~4--16x inference speed-up.
"""

from typing import Any, Callable, Dict, Optional

import torch
from torchdiffeq import odeint


def euler_sample(
    vector_field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z_0: torch.Tensor,
    num_steps: int = 4,
) -> torch.Tensor:
    """K-step Euler integration of dz/dt = v(z, t) from t=0 to t=1.

    Args:
        vector_field_fn: ``(z_t, t) -> v`` where ``t`` is a scalar broadcast
            to batch dim.
        z_0: Initial noise ``[B, T, C]``.
        num_steps: Number of Euler steps (1 / 2 / 4).

    Returns:
        ``z_1`` -- generated latents ``[B, T, C]``.
    """
    z_t = z_0
    dt = 1.0 / num_steps
    device = z_0.device

    for step in range(num_steps):
        t = torch.full((z_0.size(0),), step * dt, device=device)
        v = vector_field_fn(t, z_t)
        z_t = z_t + v * dt

    return z_t


def teacher_ode_sample(
    vector_field_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z_0: torch.Tensor,
    ode_opt: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Teacher ODE integration (16 midpoint steps), matching original inference.

    Args:
        vector_field_fn: ``(z_t, t) -> v``.
        z_0: Initial noise ``[B, T, C]``.
        ode_opt: Keyword args for ``odeint``.  Defaults to 16 midpoint steps.

    Returns:
        ``z_1`` -- teacher-generated latents ``[B, T, C]``.
    """
    if ode_opt is None:
        ode_opt = {"method": "midpoint", "options": {"step_size": 2 / 32}}

    device = z_0.device
    states = odeint(
        vector_field_fn,
        z_0,
        torch.tensor([0.0, 1.0], device=device),
        **ode_opt,
    )
    return states[-1]