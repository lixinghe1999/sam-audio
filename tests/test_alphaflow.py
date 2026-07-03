"""Focused tests for the generalized AlphaFlow/MeanFlow objective."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn as nn


_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "sam_audio" / "turbo" / "meanflow.py"
)
_SPEC = importlib.util.spec_from_file_location("_meanflow_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MEANFLOW = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MEANFLOW)


class ToyVelocityModel(nn.Module):
    """Small differentiable vector field with an analytic AlphaFlow target."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))

    def forward(
        self,
        *,
        noisy_audio: torch.Tensor,
        time: torch.Tensor,
        flow_interval: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor:
        view_shape = (-1,) + (1,) * (noisy_audio.ndim - 1)
        return (
            self.scale * noisy_audio
            + time.reshape(view_shape)
            + flow_interval.reshape(view_shape)
        )


class AlphaFlowLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ToyVelocityModel()
        self.clean = torch.tensor([[[2.0]], [[-1.0]]])
        self.noise = torch.tensor([[[0.5]], [[1.0]]])
        self.t = torch.tensor([0.2, 0.3])
        self.s = torch.tensor([0.8, 0.9])

    def _loss(self, alpha: float):
        return _MEANFLOW.alphaflow_loss(
            student=self.model,
            clean=self.clean,
            noise=self.noise,
            forward_args={},
            t=self.t,
            s=self.s,
            alpha=alpha,
            adaptive_p=0.0,
        )

    def test_alpha_one_is_trajectory_flow_matching(self) -> None:
        _, _, target = self._loss(alpha=1.0)
        torch.testing.assert_close(target, self.clean - self.noise)

    def test_finite_alpha_composes_at_intermediate_point(self) -> None:
        alpha = 0.5
        _, _, target = self._loss(alpha=alpha)

        velocity = self.clean - self.noise
        shape = (-1, 1, 1)
        h = self.s - self.t
        z_t = (1 - self.t.reshape(shape)) * self.noise + self.t.reshape(
            shape
        ) * self.clean
        first_h = alpha * h
        intermediate_t = self.t + first_h
        intermediate_z = z_t + first_h.reshape(shape) * velocity
        remaining_h = self.s - intermediate_t
        remaining_u = self.model(
            noisy_audio=intermediate_z,
            time=intermediate_t,
            flow_interval=remaining_h,
        )
        expected = alpha * velocity + (1 - alpha) * remaining_u.detach()
        torch.testing.assert_close(target, expected)

    def test_alpha_zero_matches_meanflow_wrapper(self) -> None:
        direct = self._loss(alpha=0.0)
        wrapped = _MEANFLOW.meanflow_loss(
            student=self.model,
            clean=self.clean,
            noise=self.noise,
            forward_args={},
            t=self.t,
            s=self.s,
            adaptive_p=0.0,
        )
        for actual, expected in zip(direct, wrapped):
            torch.testing.assert_close(actual, expected)

    def test_loss_backpropagates_only_through_online_prediction(self) -> None:
        loss, _, target = self._loss(alpha=0.5)
        self.assertFalse(target.requires_grad)
        loss.backward()
        self.assertIsNotNone(self.model.scale.grad)
        self.assertTrue(torch.isfinite(self.model.scale.grad))

    def test_separate_target_model_is_used_for_ddp_safe_bootstrap(self) -> None:
        target_model = ToyVelocityModel()
        loss, _, _ = _MEANFLOW.alphaflow_loss(
            student=self.model,
            target_student=target_model,
            clean=self.clean,
            noise=self.noise,
            forward_args={},
            t=self.t,
            s=self.s,
            alpha=0.5,
            adaptive_p=0.0,
        )
        loss.backward()
        self.assertIsNotNone(self.model.scale.grad)
        self.assertIsNone(target_model.scale.grad)

    def test_adaptive_alpha_weight_matches_meanflow_tse(self) -> None:
        alpha = 0.5
        eps = 1e-3
        loss, prediction, target = _MEANFLOW.alphaflow_loss(
            student=self.model,
            clean=self.clean,
            noise=self.noise,
            forward_args={},
            t=self.t,
            s=self.s,
            alpha=alpha,
            adaptive_p=1.0,
            adaptive_eps=eps,
        )
        delta_sq = (prediction.float() - target.float()).square().flatten(1).mean(1)
        expected = (
            alpha / (delta_sq.detach() + eps) * delta_sq
        ).mean()
        torch.testing.assert_close(loss, expected)


class MeanFlowTimeSamplingTest(unittest.TestCase):
    def test_instantaneous_samples_use_standard_logistic_normal(self) -> None:
        torch.manual_seed(0)
        t, s = _MEANFLOW.sample_meanflow_times(
            20_000,
            torch.device("cpu"),
            nonzero_ratio=0.0,
            distribution="logit_normal",
            # This parameter belongs only to interval sampling.
            logit_mean=8.0,
        )
        torch.testing.assert_close(t, s)
        self.assertAlmostEqual(t.mean().item(), 0.5, delta=0.02)

    def test_interval_samples_are_ordered(self) -> None:
        torch.manual_seed(1)
        t, s = _MEANFLOW.sample_meanflow_times(
            1_000,
            torch.device("cpu"),
            nonzero_ratio=1.0,
            distribution="logit_normal",
            logit_mean=-0.4,
        )
        self.assertTrue(torch.all(t <= s))
        self.assertTrue(torch.all(s > t))


class MeanFlowSamplerTest(unittest.TestCase):
    def test_sampler_supplies_each_step_interval(self) -> None:
        calls = []

        class RecordingModel(nn.Module):
            def forward(
                self,
                *,
                noisy_audio,
                time,
                flow_interval,
                **_kwargs,
            ):
                calls.append((time.clone(), flow_interval.clone()))
                return torch.ones_like(noisy_audio)

        noise = torch.zeros(2, 3, 4)
        result = _MEANFLOW.meanflow_sample(
            RecordingModel(),
            {"audio_features": noise},
            num_steps=4,
            noise=noise,
        )
        self.assertEqual(len(calls), 4)
        for index, (time, interval) in enumerate(calls):
            torch.testing.assert_close(
                time, torch.full_like(time, index / 4)
            )
            torch.testing.assert_close(
                interval, torch.full_like(interval, 0.25)
            )
        torch.testing.assert_close(result, torch.ones_like(noise))


class AlphaScheduleTest(unittest.TestCase):
    def test_linear_schedule_has_warmup_transition_and_hold(self) -> None:
        values = [
            _MEANFLOW.scheduled_alpha(
                step,
                11,
                start=1.0,
                end=0.1,
                schedule="linear",
                warmup_ratio=0.1,
                transition_ratio=0.7,
            )
            for step in range(11)
        ]
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[1], 1.0)
        self.assertAlmostEqual(values[8], 0.1)
        self.assertAlmostEqual(values[-1], 0.1)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_sigmoid_schedule_reaches_exact_endpoints(self) -> None:
        start = _MEANFLOW.scheduled_alpha(
            0, 11, start=1.0, end=0.01, schedule="sigmoid"
        )
        end = _MEANFLOW.scheduled_alpha(
            10, 11, start=1.0, end=0.01, schedule="sigmoid"
        )
        self.assertEqual(start, 1.0)
        self.assertEqual(end, 0.01)

    def test_meanflow_tse_sigmoid_clamps_near_one(self) -> None:
        value = _MEANFLOW.scheduled_alpha(
            1,
            1_000,
            start=1.0,
            end=0.005,
            schedule="sigmoid",
            warmup_ratio=0.0,
            transition_ratio=1.0,
            sigmoid_gamma=25.0,
        )
        self.assertEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
