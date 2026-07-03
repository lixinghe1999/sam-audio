"""MeanFlow and AlphaFlow training and sampling for SAM-Audio.

SAM-Audio uses the forward convention ``noise @ t=0 -> data @ t=1``.
For an interval ending at ``s >= t``, the model predicts average velocity
``u(z_t, t, h)`` with ``h = s - t``.  In this convention the MeanFlow identity
is ``u = v + h * D_t u`` while holding the interval endpoint fixed.

AlphaFlow replaces the infinitesimal MeanFlow target with a finite
self-distillation step controlled by ``alpha``.  In this time convention,
the intermediate point is ``m = t + alpha * (s - t)``.  ``alpha=1`` is
trajectory flow matching, while the ``alpha -> 0`` gradient recovers
MeanFlow; the exact ``alpha=0`` case is evaluated with the MeanFlow JVP.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def scheduled_alpha(
    step: int,
    total_steps: int,
    *,
    start: float,
    end: float,
    schedule: str = "sigmoid",
    warmup_ratio: float = 0.1,
    transition_ratio: float = 0.7,
    sigmoid_gamma: float = 10.0,
) -> float:
    """Return a curriculum alpha for one global optimization step.

    The schedule holds ``start`` during warmup, anneals to ``end`` during the
    transition, and then holds ``end`` for the remaining steps.  The sigmoid
    curve is normalized so the transition reaches both endpoints exactly.
    """
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= step < total_steps:
        raise ValueError("step must satisfy 0 <= step < total_steps")
    if not 0 <= start <= 1 or not 0 <= end <= 1:
        raise ValueError("alpha schedule endpoints must be between 0 and 1")
    if start < end:
        raise ValueError("alpha curriculum must anneal from a larger value")
    if schedule not in {"linear", "sigmoid"}:
        raise ValueError("alpha schedule must be 'linear' or 'sigmoid'")
    if warmup_ratio < 0 or transition_ratio <= 0:
        raise ValueError("invalid alpha schedule ratios")
    if warmup_ratio + transition_ratio > 1:
        raise ValueError("alpha warmup and transition ratios must sum to <= 1")
    if sigmoid_gamma <= 0:
        raise ValueError("sigmoid_gamma must be positive")

    if total_steps == 1:
        return end
    progress = step / (total_steps - 1)
    if progress <= warmup_ratio:
        return start

    transition_progress = (progress - warmup_ratio) / transition_ratio
    if transition_progress >= 1:
        return end
    if schedule == "linear":
        weight = transition_progress
    else:
        def sigmoid(value: float) -> float:
            return 1 / (1 + math.exp(-value))

        # MeanFlow-TSE uses an unnormalised sigmoid and clamps both tails:
        # values above 1-alpha_min become exactly 1 and values below
        # alpha_min become exactly alpha_min.  Preserve that recipe for its
        # standard 1 -> alpha_min curriculum instead of entering the finite
        # AlphaFlow branch on the second optimization step.
        if start == 1.0 and end > 0:
            raw_alpha = 1 - sigmoid(
                sigmoid_gamma * (transition_progress - 0.5)
            )
            if raw_alpha > 1 - end:
                return 1.0
            if raw_alpha < end:
                return end
            return raw_alpha

        low = sigmoid(-0.5 * sigmoid_gamma)
        high = sigmoid(0.5 * sigmoid_gamma)
        value = sigmoid(sigmoid_gamma * (transition_progress - 0.5))
        weight = (value - low) / (high - low)
    return start + (end - start) * weight


def sample_meanflow_times(
    batch_size: int,
    device: torch.device,
    *,
    nonzero_ratio: float = 0.5,
    distribution: str = "logit_normal",
    logit_mean: float = -0.4,
    logit_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample current/end times ``t <= s`` for MeanFlow training.

    With probability ``1 - nonzero_ratio``, a whole batch uses ``s=t`` and
    therefore retains the ordinary instantaneous flow-matching objective.
    As in MeanFlow-TSE,
    interval samples use two logistic-normal draws with mean ``-0.4``, while
    instantaneous samples use a single standard logistic-normal draw.

    SAM-Audio and MeanFlow-TSE share the same time convention: the starting
    point is at ``t=0`` and clean data is at ``t=1``.  No sign flip is needed.
    """
    if not 0 <= nonzero_ratio <= 1:
        raise ValueError("nonzero_ratio must be between 0 and 1")
    if distribution not in {"uniform", "logit_normal"}:
        raise ValueError(
            "distribution must be either 'uniform' or 'logit_normal'"
        )

    def draw_interval() -> torch.Tensor:
        if distribution == "uniform":
            return torch.rand(batch_size, device=device)
        return torch.sigmoid(
            torch.randn(batch_size, device=device) * logit_std + logit_mean
        )

    def draw_instantaneous() -> torch.Tensor:
        if distribution == "uniform":
            return torch.rand(batch_size, device=device)
        return torch.sigmoid(torch.randn(batch_size, device=device))

    first, second = draw_interval(), draw_interval()
    interval_t = torch.minimum(first, second)
    interval_s = torch.maximum(first, second)
    instantaneous_t = draw_instantaneous()
    # MeanFlow-TSE selects rectified-flow versus AlphaFlow once per batch.
    use_interval = torch.rand((), device=device) < nonzero_ratio
    t = torch.where(use_interval, interval_t, instantaneous_t)
    s = torch.where(use_interval, interval_s, instantaneous_t)
    return t, s


def alphaflow_loss(
    student: nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    forward_args: dict[str, Any],
    t: torch.Tensor,
    s: torch.Tensor,
    *,
    alpha: float,
    adaptive_p: float = 1.0,
    adaptive_eps: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the AlphaFlow loss, including exact MeanFlow at ``alpha=0``.

    For ``alpha > 0``, the full-interval average velocity is trained against
    a weighted composition of the empirical velocity over the first part of
    the interval and a stopped model prediction over the remainder.  This is
    JVP-free.  At ``alpha=0``, the finite difference becomes degenerate, so
    the exact MeanFlow JVP target is used instead.
    """
    if t.shape != s.shape or t.ndim != 1 or t.size(0) != clean.size(0):
        raise ValueError("t and s must be [batch] tensors")
    if torch.any(t > s) or torch.any(t < 0) or torch.any(s > 1):
        raise ValueError("MeanFlow times must satisfy 0 <= t <= s <= 1")
    if adaptive_p < 0:
        raise ValueError("adaptive_p must be non-negative")
    if adaptive_eps <= 0:
        raise ValueError("adaptive_eps must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

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

    if alpha == 0:
        # Exact MeanFlow's JVP can be built before the online graph to reduce
        # peak memory.  Unlike finite AlphaFlow, it does not make a regular
        # stopped model call that can alter adapter/module runtime state.
        with torch.no_grad():
            from torch.func import jvp

            # s is fixed along the derivative, hence d(s-t)/dt = -1.
            primal, du_dt = jvp(
                u_fn,
                (z_t, t, h),
                (velocity, torch.ones_like(t), -torch.ones_like(h)),
            )
            h_view = h.reshape((-1,) + (1,) * (clean.ndim - 1))
            target = velocity + h_view * du_dt
            del primal, du_dt
        u_pred = u_fn(z_t, t, h)
    else:
        # Match MeanFlow-TSE's ordering: retain the online graph first, then
        # construct the stopped bootstrap target.  This is important for LoRA
        # wrappers whose eval/no-grad forward may update internal adapter
        # state; a target-first call can disconnect the following prediction
        # from all trainable parameters.
        u_pred = u_fn(z_t, t, h)
        with torch.no_grad():
            if alpha == 1:
                # The intermediate point is the endpoint, so AlphaFlow
                # reduces exactly to trajectory flow matching.
                target = velocity
            else:
                first_h = alpha * h
                first_h_view = first_h.reshape(
                    (-1,) + (1,) * (clean.ndim - 1)
                )
                intermediate_t = t + first_h
                intermediate_z = z_t + first_h_view * velocity
                remaining_h = s - intermediate_t
                remaining_u = u_fn(
                    intermediate_z,
                    intermediate_t,
                    remaining_h,
                )
                target = alpha * velocity + (1 - alpha) * remaining_u
                del intermediate_z, remaining_u

    # MeanFlow-TSE defines the adaptive statistic as a per-sample mean, not a
    # sum, so its stabilizer is independent of latent sequence length.
    per_sample = (u_pred.float() - target.float()).square().flatten(1).mean(1)
    if adaptive_p:
        # AlphaFlow scales the stopped adaptive weight by alpha**p.  This
        # deliberately reduces the update magnitude near the consistency end
        # of the curriculum; dividing the raw loss by alpha has the opposite
        # behavior and is not the MeanFlow-TSE objective.
        numerator = alpha**adaptive_p if alpha > 0 else 1.0
        weight = numerator / (
            per_sample.detach() + adaptive_eps
        ).pow(adaptive_p)
        per_sample = weight * per_sample
    return per_sample.mean(), u_pred, target


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
    """Compute the MeanFlow identity loss using a no-grad JVP target."""
    return alphaflow_loss(
        student=student,
        clean=clean,
        noise=noise,
        forward_args=forward_args,
        t=t,
        s=s,
        alpha=0.0,
        adaptive_p=adaptive_p,
        adaptive_eps=adaptive_eps,
    )


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


@torch.inference_mode()
def meanflow_separate(
    model: nn.Module,
    batch: Any,
    *,
    num_steps: int = 1,
    noise: torch.Tensor | None = None,
    reranking_candidates: int = 1,
    predict_spans: bool = False,
):
    """Run end-to-end SAM-Audio separation with a MeanFlow checkpoint.

    This is intentionally a separate turbo entry point: the original
    :meth:`SAMAudio.few_step_separate` continues to serve consistency-
    distillation checkpoints as an instantaneous vector field, while this
    function supplies MeanFlow's required interval length on every step.
    """
    if reranking_candidates < 1:
        raise ValueError("reranking_candidates must be positive")

    base = getattr(model, "base_model", model)
    forward_args = base._get_forward_args(
        batch, candidates=reranking_candidates
    )

    if (
        predict_spans
        and hasattr(base, "span_predictor")
        and batch.anchors is None
    ):
        batch = base.predict_spans(
            batch=batch,
            audio_features=base._unrepeat_from_reranking(
                forward_args["audio_features"], reranking_candidates
            ),
            audio_pad_mask=base._unrepeat_from_reranking(
                forward_args["audio_pad_mask"], reranking_candidates
            ),
        )
        forward_args.update(
            {
                "anchor_ids": base._repeat_for_reranking(
                    batch.anchor_ids, reranking_candidates
                ),
                "anchor_alignment": base._repeat_for_reranking(
                    batch.anchor_alignment, reranking_candidates
                ),
            }
        )

    audio_features = forward_args["audio_features"]
    batch_size, sequence_length, stacked_channels = audio_features.shape
    channels = stacked_channels // 2
    if noise is None:
        noise = torch.randn_like(audio_features)

    generated_features = meanflow_sample(
        model,
        forward_args,
        num_steps=num_steps,
        noise=noise,
    ).transpose(1, 2)
    wavs = base.audio_codec.decode(
        generated_features.reshape(
            2 * batch_size, channels, sequence_length
        )
    ).view(batch_size, 2, -1)

    output_batch_size = wavs.size(0) // reranking_candidates
    sizes = base.audio_codec.feature_idx_to_wav_idx(batch.sizes)
    target_wavs = base.unbatch(
        wavs[:, 0].view(output_batch_size, reranking_candidates, -1), sizes
    )
    residual_wavs = base.unbatch(
        wavs[:, 1].view(output_batch_size, reranking_candidates, -1), sizes
    )

    if (
        reranking_candidates > 1
        and batch.masked_video is not None
        and base.visual_ranker is not None
    ):
        scores = base.visual_ranker(
            extracted_audio=target_wavs,
            videos=batch.masked_video,
            sample_rate=base.audio_codec.sample_rate,
        )
        indices = scores.argmax(dim=1)
    elif reranking_candidates > 1 and base.text_ranker is not None:
        input_audio = [
            audio[:, :size].expand(reranking_candidates, -1)
            for audio, size in zip(batch.audios, sizes, strict=False)
        ]
        scores = base.text_ranker(
            extracted_audio=target_wavs,
            input_audio=input_audio,
            descriptions=batch.descriptions,
            sample_rate=base.audio_codec.sample_rate,
        )
        indices = scores.argmax(dim=1)
    else:
        indices = torch.zeros(
            output_batch_size, dtype=torch.long, device=noise.device
        )

    # Import lazily so the turbo module does not participate in the core model
    # module's import graph.
    from sam_audio.model.model import SeparationResult

    return SeparationResult(
        target=[
            wav[index]
            for wav, index in zip(target_wavs, indices, strict=False)
        ],
        residual=[
            wav[index]
            for wav, index in zip(residual_wavs, indices, strict=False)
        ],
        noise=noise,
    )
