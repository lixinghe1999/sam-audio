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


if __name__ == "__main__":
    unittest.main()
