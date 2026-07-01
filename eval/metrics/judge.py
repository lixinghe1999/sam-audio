# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from typing import Optional

import torch

from sam_audio import SAMAudioJudgeModel, SAMAudioJudgeProcessor


class Judge(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str = "facebook/sam-audio-judge",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # 延迟 GPU 加载：init 时留在 CPU，forward 时按需搬到 GPU
        # 避免与主模型 SAMAudio 同时抢占 GPU 内存导致 OOM
        self.model = SAMAudioJudgeModel.from_pretrained(checkpoint)
        self.processor = SAMAudioJudgeProcessor.from_pretrained(checkpoint)
        self._on_gpu = False

    def _ensure_gpu(self):
        if not self._on_gpu and self.device.type == "cuda":
            self.model = self.model.to(self.device)
            self._on_gpu = True

    def forward(
        self,
        input_wavs: list[torch.Tensor],
        target_wavs: list[torch.Tensor],
        descriptions: list[str],
        target_wavs_sample_rate: int = 48_000,
        **kwargs,
    ) -> torch.Tensor:
        with torch.inference_mode():
            processed = self.processor(
                text=descriptions,
                input_audio=[x.cpu() for x in input_wavs],
                separated_audio=[x.cpu() for x in target_wavs],
                sampling_rate=target_wavs_sample_rate,
            ).to(self.device)
            self._ensure_gpu()
            result = self.model(**processed)
            return {
                "JudgeOverall": result.overall.squeeze(-1).cpu().tolist(),
                "JudgeFaithfulness": result.faithfulness.squeeze(-1).cpu().tolist(),
                "JudgeRecall": result.recall.squeeze(-1).cpu().tolist(),
                "JudgePrecision": result.precision.squeeze(-1).cpu().tolist(),
            }
