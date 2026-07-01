#!/usr/bin/env python3
"""LoRA fine-tuning for SAM-Audio on Clotho + FSD50K + LASS.

Training:   Clotho (development captions) + FSD50K (dev auto-captions)
            + LASS validation sources (random on-the-fly mixtures)
Validation: LASS validation (3000 fixed synthetic mixtures)
Eval:       LASS evaluation splits via eval/main.py (eval_lass_val.sh)

Usage:
    python lora_train.py && bash eval_lass_val.sh
"""

import json
import os.path as osp

import pandas as pd
import torch
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.train import apply_lora, freeze_encoders, strip_vision_branch, train

# ── paths ────────────────────────────────────────────────────────────────
DATA_ROOT = osp.expanduser("~/audiolens/dataset")
SR = 48_000  # model sample rate


# ── Clotho ───────────────────────────────────────────────────────────────
class Clotho(Dataset):
    """Clotho v2 development split: (audio_tensor, caption) pairs.

    Each source wav has 5 reference captions → 5× items.
    """

    def __init__(self, data_root: str = DATA_ROOT, sr: int = SR):
        super().__init__()
        self.sr = sr
        csv_path = osp.join(data_root, "clotho_captions_development.csv")
        self.audio_dir = osp.join(data_root, "clotho", "development", "audio")
        df = pd.read_csv(csv_path)
        self.items = []
        for _, row in df.iterrows():
            fname = row["file_name"]
            for k in range(1, 6):  # caption_1 … caption_5
                self.items.append((fname, row[f"caption_{k}"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, caption = self.items[idx]
        wav, sr = torchaudio.load(osp.join(self.audio_dir, fname))
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav.mean(0, keepdim=True), caption


# ── FSD50K ───────────────────────────────────────────────────────────────
class FSD50KDev(Dataset):
    """FSD50K development split: (audio_tensor, caption) pairs.

    Uses auto-generated captions from fsd50k_dev_auto_caption.json.
    """

    def __init__(self, data_root: str = DATA_ROOT, sr: int = SR):
        super().__init__()
        self.sr = sr
        json_path = osp.join(data_root, "fsd50k_dev_auto_caption.json")
        self.audio_dir = osp.join(data_root, "FSD50K", "FSD50K.dev_audio")
        with open(json_path) as f:
            entries = json.load(f)["data"]
        self.items = [(e["wav"], e["caption"]) for e in entries]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, caption = self.items[idx]
        wav, sr = torchaudio.load(osp.join(self.audio_dir, fname))
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav.mean(0, keepdim=True), caption


# ── collate ──────────────────────────────────────────────────────────────
def _collate(items):
    """Unpack (audio, description) tuples into parallel lists."""
    audios, descriptions = zip(*items)
    return list(audios), list(descriptions)


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. Model & processor
    model = SAMAudio.from_pretrained("facebook/sam-audio-small")

    # ── memory optimizations ──────────────────────────────────────────
    # 5.8B params at FP32 = 23 GiB (fills 24 GiB GPU).  Cast to bfloat16
    # to halve weight memory (~11.6 GiB), leaving room for optimizer
    # states, gradients, and activations during backward.
    model = model.to(torch.bfloat16)
    # Vision / span branches are never called in text-only training.
    strip_vision_branch(model)

    processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-small")

    # 2. LoRA
    model = apply_lora(model, rank=8, alpha=16)
    freeze_encoders(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable (LoRA): {trainable:,}  /  Total: {total:,}  "
          f"({100 * trainable / total:.1f}%)")

    # 3. Training dataset: Clotho + FSD50K + LASS (random mixtures)
    from eval.dataset.lass import LASS

    clotho = Clotho()
    fsd50k = FSD50KDev()
    lass_train = LASS(
        collate_fn=processor,
        split="train",
        data_path=osp.join(DATA_ROOT, "lass"),
    )

    train_data = ConcatDataset([clotho, fsd50k, lass_train])
    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        collate_fn=_collate,
        num_workers=4,
    )

    print(f"Train dataset: {len(train_data)} items "
          f"(Clotho {len(clotho)} + FSD50K {len(fsd50k)} + LASS {len(lass_train)})")

    # 4. Validation dataset: LASS fixed synthetic mixtures
    val_data = LASS(
        collate_fn=processor,
        split="validation",
        data_path=osp.join(DATA_ROOT, "lass"),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=2,
        shuffle=False,
        collate_fn=_collate,
        num_workers=4,
    )
    print(f"Val dataset: {len(val_data)} items")

    # 5. Train
    model = train(
        model,
        processor,
        train_loader,
        val_loader=val_loader,
        epochs=10,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="lora-checkpoint",
    )

    # 6. Merge & save
    print("Merging LoRA weights into base model…")
    merged = model.merge_and_unload()
    merged.save_pretrained("lass-lora-merged")
    processor.save_pretrained("lass-lora-merged")
    print("  → saved merged model to lass-lora-merged/")