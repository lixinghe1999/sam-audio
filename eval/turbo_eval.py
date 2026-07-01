#!/usr/bin/env python3
"""Turbo vs baseline evaluation: NFE, wall-clock time, quality metrics.

Compares SAM-Audio-Turbo (few-step Euler) against the original SAM-Audio
(16-step midpoint ODE) on LASS validation splits.

Usage:
    # Evaluate pre-distilled turbo checkpoint
    python eval/turbo_eval.py --turbo-path turbo-checkpoint/epoch-20 \\
        --num-steps 4 --samples 100

    # Evaluate baseline only (no turbo checkpoint available yet)
    python eval/turbo_eval.py --baseline-only --samples 50
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch

from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.turbo.sampler import euler_sample


def evaluate(
    model: SAMAudio,
    processor: SAMAudioProcessor,
    dataset,
    *,
    num_steps: int | None = None,
    max_samples: int = 100,
    device: str = "cuda",
    label: str = "model",
) -> dict:
    """Run inference on a subset of the dataset and collect timing / NFE stats.

    Args:
        model: SAM-Audio instance.
        processor: SAMAudioProcessor.
        dataset: PyTorch Dataset yielding batches.
        num_steps: If not None, use ``few_step_separate`` with this many
            Euler steps.  If None, use the original ``separate`` (ODE).
        max_samples: Max number of samples to evaluate.
        device: Inference device.
        label: Label for result keys.

    Returns:
        Dict with keys: nfe_total, time_total, time_per_sample, num_samples,
        and optionally per-sample metrics.
    """
    model.eval()
    model.to(device)

    stats = defaultdict(list)
    nfe_per_sample = []
    times = []
    count = 0

    for batch in dataset:
        if count >= max_samples:
            break

        if isinstance(batch, tuple):
            audios, descriptions = batch
            batch = processor(audios=audios, descriptions=descriptions)
        batch = batch.to(device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if num_steps is not None:
            # --- Turbo inference ---
            result = model.few_step_separate(batch, num_steps=num_steps)
            nfe = num_steps
        else:
            # --- Baseline ODE inference ---
            result = model.separate(batch)
            nfe = 32  # midpoint with step_size=2/32 → 32 NFE

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        nfe_per_sample.append(nfe)
        times.append(elapsed)
        count += 1

        if count % 10 == 0:
            print(f"  [{label}] {count}/{max_samples} samples, "
                  f"avg time: {sum(times) / len(times):.3f}s")

    return {
        f"{label}_nfe_total": sum(nfe_per_sample),
        f"{label}_nfe_per_sample": sum(nfe_per_sample) / len(nfe_per_sample),
        f"{label}_time_total": sum(times),
        f"{label}_time_per_sample": sum(times) / len(times),
        f"{label}_num_samples": count,
    }


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turbo vs baseline NFE / speed benchmark"
    )
    parser.add_argument("--turbo-path", type=str, default=None,
                        help="Path to distilled turbo checkpoint")
    parser.add_argument("--model-id", type=str,
                        default="facebook/sam-audio-small")
    parser.add_argument("--num-steps", type=int, default=4,
                        help="Euler steps for turbo inference")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--baseline-only", action="store_true",
                        help="Evaluate baseline only (no turbo model)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/turbo_eval.json")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load models ──────────────────────────────────────────────────
    baseline = SAMAudio.from_pretrained(args.model_id)
    baseline = baseline.to(torch.bfloat16)
    from sam_audio.train import strip_vision_branch
    strip_vision_branch(baseline)

    processor = SAMAudioProcessor.from_pretrained(args.model_id)

    turbo = None
    if not args.baseline_only and args.turbo_path:
        print(f"Loading turbo checkpoint from {args.turbo_path}")
        turbo = SAMAudio.from_pretrained(args.turbo_path)
        turbo = turbo.to(torch.bfloat16)
        if hasattr(turbo, "vision_encoder"):
            strip_vision_branch(turbo)

    # ── Dataset ──────────────────────────────────────────────────────
    print(f"Loading dataset (max {args.samples} samples)...")
    import os.path as osp
    from eval.dataset.lass import LASS

    def _collate(items):
        audios, descriptions = zip(*items)
        return list(audios), list(descriptions)

    lass_val = LASS(
        collate_fn=processor,
        split="validation",
        data_path=osp.expanduser("~/audiolens/dataset/lass"),
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        lass_val,
        batch_size=1,
        shuffle=False,
        collate_fn=_collate,
        num_workers=4,
    )

    # ── Benchmark ────────────────────────────────────────────────────
    results = {}

    print("\n=== Baseline (16-step midpoint ODE) ===")
    results.update(
        evaluate(
            baseline, processor, loader,
            num_steps=None, max_samples=args.samples,
            device=device, label="baseline",
        )
    )

    if turbo is not None:
        # Reset dataset iterator
        loader = DataLoader(
            lass_val, batch_size=1, shuffle=False,
            collate_fn=_collate, num_workers=4,
        )

        print(f"\n=== Turbo ({args.num_steps}-step Euler) ===")
        results.update(
            evaluate(
                turbo, processor, loader,
                num_steps=args.num_steps, max_samples=args.samples,
                device=device, label="turbo",
            )
        )

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    b_nfe = results.get("baseline_nfe_per_sample", "N/A")
    b_time = results.get("baseline_time_per_sample", "N/A")
    print(f"Baseline (ODE 16-step):  {b_nfe:.1f} NFE/sample, "
          f"{b_time:.3f}s/sample")

    if turbo is not None:
        t_nfe = results.get("turbo_nfe_per_sample", 0)
        t_time = results.get("turbo_time_per_sample", 0)
        speedup_nfe = b_nfe / t_nfe if t_nfe > 0 else 0
        speedup_time = b_time / t_time if t_time > 0 else 0
        print(f"Turbo    ({args.num_steps}-step Euler): "
              f"{t_nfe:.1f} NFE/sample, {t_time:.3f}s/sample")
        print(f"  NFE speedup:   {speedup_nfe:.1f}x")
        print(f"  Time speedup:  {speedup_time:.1f}x")

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")