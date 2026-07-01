"""
Fine-tuning utilities for SAM-Audio using LoRA and flow matching.

Provides:
- apply_lora(): Wrap SAMAudio with LoRA adapters via peft
- freeze_encoders(): Set frozen encoders to eval mode
- flow_match_loss(): Compute conditional flow matching loss
- FlowMatchDataset: Simple (audio_path, text) dataset
- train(): Standard training loop with checkpointing
- strip_vision_branch(): Remove unused vision components to free GPU memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset


def strip_vision_branch(model: nn.Module) -> None:
    """Remove vision encoder / ranker / span predictor to free GPU memory.

    Only needed for text-only training -- these sub-modules are never called.
    Must be called BEFORE apply_lora() (operates on raw SAMAudio).

    Monkey-patches ``_get_video_features`` so it returns zero features when
    ``video=None`` instead of reading ``self.vision_encoder.dim``.
    """
    if not hasattr(model, "vision_encoder"):
        return

    vision_dim = model.vision_encoder.dim

    del model.vision_encoder
    if hasattr(model, "visual_ranker"):
        del model.visual_ranker
    if hasattr(model, "span_predictor"):
        del model.span_predictor
        if hasattr(model, "span_predictor_transform"):
            del model.span_predictor_transform

    _orig_get_video = model._get_video_features

    def _patched_get_video(video, audio_features):
        B, T, _ = audio_features.shape
        if video is None:
            return audio_features.new_zeros(B, vision_dim, T)
        return _orig_get_video(video, audio_features)

    model._get_video_features = _patched_get_video


def apply_lora(model, rank: int = 8, alpha: float = 16, dropout: float = 0.0):
    """Apply LoRA adapters to SAM-Audio's DiT attention layers.

    Targets ``wq``, ``wk``, ``wv``, ``wo`` in both self-attention and
    cross-attention blocks — the only modules with those names in the model.
    ``get_peft_model`` handles freezing the base parameters automatically.

    Args:
        model: SAMAudio instance (not yet wrapped).
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout probability.

    Returns:
        PeftModel-wrapped SAMAudio.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["wq", "wk", "wv", "wo"],
        lora_dropout=dropout,
        bias="none",
    )
    return get_peft_model(model, config)


def freeze_encoders(model):
    """Set frozen encoders to eval mode for deterministic inference.

    Call after ``model.train()`` so the transformer stays in training mode
    while the codec, T5, and vision encoder run without dropout.
    """
    base = model.base_model
    base.audio_codec.eval()
    base.text_encoder.eval()
    if hasattr(base, "vision_encoder"):
        base.vision_encoder.eval()


def flow_match_loss(model, batch, device: str = "cuda"):
    """Compute conditional flow-matching loss for one batch.

    Flow matching samples :math:`t \\sim U(0,1)` and noise
    :math:`x_0 \\sim N(0,I)`, constructs
    :math:`x_t = t \\cdot x_1 + (1-t) \\cdot x_0`, and trains the model
    to predict the vector field :math:`v = x_1 - x_0`.

    Args:
        model: PeftModel-wrapped SAMAudio.
        batch: :class:`Batch` from :class:`SAMAudioProcessor`.
        device: Target device string.

    Returns:
        Scalar MSE loss.
    """
    batch = batch.to(device)
    base = model.base_model

    # ---- encode clean audio and text with frozen encoders ----
    with torch.no_grad():
        audio_features = base._get_audio_features(batch.audios)
        text_features, text_mask = base.text_encoder(batch.descriptions)

    B = audio_features.size(0)
    dev = audio_features.device

    # ---- flow matching: sample time & noise, build x_t ----
    t = torch.rand(B, device=dev)
    noise = torch.randn_like(audio_features)
    xt = t[:, None, None] * audio_features + (1 - t[:, None, None]) * noise

    # ---- predict vector field ----
    v_pred = base(
        noisy_audio=xt,
        audio_features=audio_features,
        text_features=text_features,
        time=t,
        text_mask=text_mask,
        audio_pad_mask=batch.audio_pad_mask,
    )

    # ---- target: clean - noise ----
    v_true = audio_features - noise

    return F.mse_loss(v_pred, v_true)


class FlowMatchDataset(Dataset):
    """Thin dataset wrapper for (audio_path, text_description) pairs.

    Args:
        items: List of ``(audio_path: str, description: str)`` tuples.
    """

    def __init__(self, items: list[tuple[str, str]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(items):
    """Collate function — unpack tuples into parallel lists for the processor."""
    audios, descriptions = zip(*items)
    return list(audios), list(descriptions)


def train(
    model,
    processor,
    dataloader,
    *,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "lora-checkpoint",
    log_every: int = 10,
    val_loader=None,
):
    """Run a minimal flow-matching training loop.

    Saves a LoRA adapter checkpoint after every epoch via
    :meth:`peft.PeftModel.save_pretrained`.

    Args:
        model: PeftModel-wrapped SAMAudio.
        processor: :class:`SAMAudioProcessor` instance.
        dataloader: DataLoader yielding ``(audios, descriptions)`` tuples.
        epochs: Number of training epochs.
        lr: Learning rate for AdamW.
        device: Training device.
        save_dir: Directory for LoRA checkpoints.
        log_every: Log loss every *N* steps.
        val_loader: Optional DataLoader for validation after each epoch.

    Returns:
        The trained model.
    """
    model.to(device)
    model.train()
    freeze_encoders(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0

        for i, batch in enumerate(dataloader):
            # 兼容两种输入模式：
            #   1. tuple (audios, descriptions) → raw data，需 processor 处理
            #   2. 预处理好的 Batch 对象 → 跳过 processor
            if isinstance(batch, tuple):
                audios, descriptions = batch
                batch = processor(audios=audios, descriptions=descriptions)

            with autocast(dtype=torch.bfloat16, enabled=(device != "cpu")):
                loss = flow_match_loss(model, batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_every == 0:
                print(
                    f"Epoch {epoch + 1:3d} | "
                    f"Step {i + 1:4d} | "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1:3d} | Average Loss: {avg_loss:.6f}")

        # validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, tuple):
                        audios, descriptions = batch
                        batch = processor(audios=audios, descriptions=descriptions)
                    with autocast(dtype=torch.bfloat16, enabled=(device != "cpu")):
                        loss = flow_match_loss(model, batch, device)
                    val_loss += loss.item()
            val_loss /= max(len(val_loader), 1)
            print(f"  Validation Loss: {val_loss:.6f}")
            model.train()
            freeze_encoders(model)

        ckpt = f"{save_dir}/epoch-{epoch + 1}"
        model.save_pretrained(ckpt)
        print(f"  -> saved checkpoint to {ckpt}")

    return model
