"""Audio-text datasets for SAM-Audio training and evaluation.

Provides:
- ``ClothoDS``: Clotho dataset (development / validation / evaluation splits).
- ``FSD50KDS``: FSD50K dataset (dev / eval splits).
- ``build_dataloader``: Factory for train / val / eval DataLoaders.
- ``_collate``: Global collation function.
"""

from __future__ import annotations

import json
import os.path as osp

import pandas as pd
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Dataset

SR = 48_000


class ClothoDS(Dataset):
    """Clotho audio captioning dataset.

    Args:
        split: One of ``"development"``, ``"validation"``, ``"evaluation"``.
        data_root: Root dataset directory containing the CSV and ``clotho/``
            sub-directory.
        sr: Target sample rate for audio resampling.
    """

    SPLIT_MAP = {
        "development": {
            "csv": "clotho_captions_development.csv",
            "audio_dir": osp.join("clotho", "development", "audio"),
        },
        "validation": {
            "csv": "clotho_captions_validation.csv",
            "audio_dir": osp.join("clotho", "validation", "audio"),
        },
        "evaluation": {
            "csv": "clotho_captions_evaluation.csv",
            "audio_dir": osp.join("clotho", "evaluation", "audio"),
        },
    }

    def __init__(
        self,
        split: str = "development",
        data_root: str = "/home/lixing/audiolens/dataset",
        sr: int = SR,
    ):
        if split not in self.SPLIT_MAP:
            raise ValueError(
                f"Clotho split must be one of {list(self.SPLIT_MAP.keys())}, "
                f"got {split!r}"
            )
        self.sr = sr
        info = self.SPLIT_MAP[split]
        csv_path = osp.join(data_root, info["csv"])
        self.audio_dir = osp.join(data_root, info["audio_dir"])
        df = pd.read_csv(csv_path)
        self.items: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            fname = row["file_name"]
            for k in range(1, 6):
                self.items.append((fname, row[f"caption_{k}"]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        fname, caption = self.items[idx]
        wav, sr = torchaudio.load(osp.join(self.audio_dir, fname))
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav.mean(0, keepdim=True), caption


class FSD50KDS(Dataset):
    """FSD50K auto-captioned dataset.

    Args:
        split: One of ``"dev"`` or ``"eval"``.
        data_root: Root dataset directory containing the JSON and ``FSD50K/``
            sub-directory.
        sr: Target sample rate for audio resampling.
    """

    SPLIT_MAP = {
        "dev": {
            "json": "fsd50k_dev_auto_caption.json",
            "audio_dir": osp.join("FSD50K", "FSD50K.dev_audio"),
        },
        "eval": {
            "json": "fsd50k_eval_auto_caption.json",
            "audio_dir": osp.join("FSD50K", "FSD50K.eval_audio"),
        },
    }

    def __init__(
        self,
        split: str = "dev",
        data_root: str = "/home/lixing/audiolens/dataset",
        sr: int = SR,
    ):
        if split not in self.SPLIT_MAP:
            raise ValueError(
                f"FSD50K split must be one of {list(self.SPLIT_MAP.keys())}, "
                f"got {split!r}"
            )
        self.sr = sr
        info = self.SPLIT_MAP[split]
        json_path = osp.join(data_root, info["json"])
        self.audio_dir = osp.join(data_root, info["audio_dir"])
        with open(json_path) as f:
            entries = json.load(f)["data"]
        self.items: list[tuple[str, str]] = [
            (e["wav"], e["caption"]) for e in entries
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple:
        fname, caption = self.items[idx]
        wav, sr_ = torchaudio.load(osp.join(self.audio_dir, fname))
        if sr_ != self.sr:
            wav = torchaudio.functional.resample(wav, sr_, self.sr)
        return wav.mean(0, keepdim=True), caption


# ── collation ────────────────────────────────────────────────────────────


def _collate(items: list[tuple]) -> tuple[list, list]:
    """Collate function — unpack (audio, description) tuples into parallel lists."""
    audios, descriptions = zip(*items)
    return list(audios), list(descriptions)


# ── DataLoader factory ───────────────────────────────────────────────────


def build_dataloader(
    split: str,
    data_root: str = "/home/lixing/audiolens/dataset",
    batch_size: int = 1,
    num_workers: int = 4,
    sr: int = SR,
) -> DataLoader:
    """Build a DataLoader for the given split.

    Args:
        split: ``"train"`` (Clotho dev + FSD50K dev), ``"val"`` (Clotho val),
            or ``"eval"`` (Clotho eval + FSD50K eval).
        data_root: Root dataset directory.
        batch_size: Per-batch sample count.
        num_workers: DataLoader worker count.
        sr: Target sample rate.

    Returns:
        DataLoader yielding ``(list[waveform], list[description])`` tuples
        consumed by ``SAMAudioProcessor``.
    """
    if split == "train":
        dataset: Dataset = ConcatDataset(
            [ClothoDS("development", data_root, sr), FSD50KDS("dev", data_root, sr)]
        )
        shuffle = True
    elif split == "val":
        dataset = ClothoDS("validation", data_root, sr)
        shuffle = False
    elif split == "eval":
        dataset = ConcatDataset(
            [ClothoDS("evaluation", data_root, sr), FSD50KDS("eval", data_root, sr)]
        )
        shuffle = False
    else:
        raise ValueError(f"split must be 'train', 'val', or 'eval', got {split!r}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        num_workers=num_workers,
    )