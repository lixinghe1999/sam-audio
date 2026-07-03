"""Few-step SAM-Audio via consistency distillation or MeanFlow.

Modules:
    sampler       -- Euler / ODE samplers for few-step generation
    consistency   -- Consistency Distillation loss, EMA helper, and multistep sampling
    meanflow      -- MeanFlow identity loss and average-velocity sampling
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
from sam_audio.turbo.meanflow import (
    meanflow_loss,
    meanflow_sample,
    sample_meanflow_times,
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
    "meanflow_loss",
    "meanflow_sample",
    "sample_meanflow_times",
]
