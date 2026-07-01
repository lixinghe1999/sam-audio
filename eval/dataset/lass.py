# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import json
import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset


class LASS(Dataset):
    """LASS (Language-Audio Sound Separation) — DCASE Challenge 2024 Task 9.

    Text-queried sound separation: given a mixture audio and a natural-language
    query describing the target sound, separate the target from the mixture.

    Splits
    ------
    train              1000 sources × random noise + random SNR + random caption
                       per access.  Effectively unlimited — each epoch sees
                       different mixtures.
    validation         3000 fixed synthetic mixtures (1000 sources × 3 noise/SNR).
                       On-the-fly mixing.  Benchmark for development-time check.
    evaluation_real    200 pre-computed real-world recordings.
    evaluation_synth   3000 pre-computed synthetic mixtures.
    """

    def __init__(
        self,
        collate_fn,
        cache_path=None,  # accepted for make_dataset compatibility, unused
        split: str = "validation",
        sample_rate: int = 48_000,
        data_path: str = os.path.expanduser("~/audiolens/dataset/lass"),
    ):
        super().__init__()
        self.collate_fn = collate_fn
        self.split = split
        self.sample_rate = sample_rate
        self.data_path = data_path

        self.manifest = self._load_manifest()

    @property
    def visual(self):
        return False

    # ── manifest loading ────────────────────────────────────────────────

    def _load_manifest(self):
        if self.split == "train":
            return self._load_train_manifest()
        elif self.split == "validation":
            return self._load_validation_manifest()
        elif self.split == "evaluation_real":
            return self._load_eval_manifest("evaluation_real")
        elif self.split == "evaluation_synthetic":
            return self._load_eval_manifest("evaluation_synthetic")
        else:
            raise ValueError(f"Unknown LASS split: {self.split}")

    def _load_train_manifest(self):
        """Build manifest: each source once; noise/SNR picked randomly on-the-fly.

        Returns a list of dicts with only ``source``, ``captions`` and
        ``audio_dir`` — no fixed noise/SNR.  This lets training see a fresh
        random mixture every time ``__getitem__`` is called.
        """
        meta_path = os.path.join(self.data_path, "validation", "metadata.json")
        with open(meta_path) as f:
            sources = json.load(f)

        audio_dir = os.path.join(self.data_path, "validation", "audio")
        manifest = []
        for src in sources:
            manifest.append(
                {
                    "source": src["Index"],
                    "captions": src.get("Captions", []),
                    "audio_dir": audio_dir,
                }
            )
        return manifest

    def _load_validation_manifest(self):
        meta_path = os.path.join(self.data_path, "validation", "metadata.json")
        with open(meta_path) as f:
            sources = json.load(f)

        audio_dir = os.path.join(self.data_path, "validation", "audio")
        manifest = []
        for src in sources:
            for mix in src["synthetic_mixtures"]:
                manifest.append(
                    {
                        "source": src["Index"],
                        "noise": mix["noise"],
                        "snr": mix["snr"],
                        "description": mix["caption"],
                        "audio_dir": audio_dir,
                    }
                )
        return manifest

    def _load_eval_manifest(self, split_dir: str):
        import pandas as pd

        queries_path = os.path.join(self.data_path, split_dir, "queries.csv")
        df = pd.read_csv(queries_path)
        audio_dir = os.path.join(self.data_path, split_dir, "audio")
        manifest = []
        for _, row in df.iterrows():
            manifest.append(
                {
                    "description": row["query"],
                    "audio_path": os.path.join(audio_dir, row["file_name"]),
                }
            )
        return manifest

    # ── dataset interface ───────────────────────────────────────────────

    def __len__(self):
        return len(self.manifest)

    def collate(self, items):
        audios, descriptions = zip(*items, strict=False)
        return self.collate_fn(
            audios=list(audios),
            descriptions=list(descriptions),
        )

    def __getitem__(self, idx):
        item = self.manifest[idx]

        if self.split == "train":
            # Random noise + SNR + caption every call = effectively unlimited
            noise_src = random.choice(self.manifest)
            while noise_src["source"] == item["source"]:
                noise_src = random.choice(self.manifest)
            mix_item = {
                "source": item["source"],
                "noise": noise_src["source"],
                "snr": random.randint(-15, 15),
                "description": random.choice(item["captions"]),
                "audio_dir": item["audio_dir"],
            }
            mixture = self._mix_source_and_noise(mix_item)
            return mixture, mix_item["description"]

        elif self.split == "validation":
            mixture = self._mix_source_and_noise(item)
        else:
            mixture = self._load_wav(item["audio_path"])

        return mixture, item["description"]

    # ── audio helpers ───────────────────────────────────────────────────

    def _load_wav(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.mean(0, keepdim=True)

    def _mix_source_and_noise(self, item: dict) -> torch.Tensor:
        """Create mixture on-the-fly: source + noise scaled to target SNR."""
        source_path = os.path.join(item["audio_dir"], f"{item['source']}.wav")
        noise_path = os.path.join(item["audio_dir"], f"{item['noise']}.wav")

        source, sr = torchaudio.load(source_path)
        noise, _ = torchaudio.load(noise_path)

        # Trim to common length
        min_len = min(source.shape[-1], noise.shape[-1])
        source = source[..., :min_len]
        noise = noise[..., :min_len]

        # Scale noise to target SNR
        s_pow = (source**2).mean()
        n_pow = (noise**2).mean()
        target_n_pow = s_pow / (10 ** (item["snr"] / 10))
        scale = (target_n_pow / (n_pow + 1e-10)).sqrt()
        noise = noise * scale

        mixture = source + noise

        if sr != self.sample_rate:
            mixture = torchaudio.functional.resample(mixture, sr, self.sample_rate)

        return mixture.mean(0, keepdim=True)