"""SAM-Audio-Turbo: few-step accelerated audio separation via consistency distillation.

Modules:
    sampler       -- Euler / ODE samplers for few-step generation
    consistency   -- Consistency Distillation loss, EMA helper, and multistep sampling
"""

from sam_audio.turbo.sampler import euler_sample, teacher_ode_sample
from sam_audio.turbo.consistency import (
    cd_time_steps,
    cd_intervals,
    velocity_to_clean,
    consistency_loss,
    EMAHelper,
    multistep_sample,
)

__all__ = [
    "euler_sample",
    "teacher_ode_sample",
    "cd_time_steps",
    "cd_intervals",
    "velocity_to_clean",
    "consistency_loss",
    "EMAHelper",
    "multistep_sample",
]