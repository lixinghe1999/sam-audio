# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import argparse
import json
import os

import pandas as pd
import torch
import torchaudio
import torch.distributed as dist
from dataset import SETTINGS, make_dataset
from metrics import CLAP, Aesthetic, ImageBind, Judge
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sam_audio import SAMAudio, SAMAudioProcessor


def gather_and_average_results(results, world_size):
    if world_size == 1:
        return json.loads(results.mean().to_json())

    # 1. Gather all dictionaries to all ranks
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(
        all_results, {"sum": results.sum().to_json(), "count": len(results)}
    )

    summed = {}
    counts = 0

    for res in all_results:
        for k, v in json.loads(res["sum"]).items():
            if k not in summed:
                summed[k] = 0.0
            summed[k] += v
        counts += res["count"]

    # 3. Compute average for keys that appeared at least once
    averaged = {k: summed[k] / counts for k in summed}

    return averaged


def _strip_vision_branch(model: torch.nn.Module) -> None:
    """移除 text-only eval 不需要的 vision / span 分支，释放 GPU 内存。

    SAMAudio 的 vision_encoder (PerceptionEncoder) 和 visual_ranker
    在纯文本分离任务中完全不被调用，但占大量 GPU 内存。
    _get_video_features 在 video=None 时仅需 vision_encoder.dim，
    通过 monkey-patch 用缓存值替代。
    """
    if not hasattr(model, "vision_encoder"):
        return

    # 缓存 vision dim，后续 monkey-patch 使用
    vision_dim = model.vision_encoder.dim

    # 删除视觉分支
    del model.vision_encoder
    if hasattr(model, "visual_ranker"):
        del model.visual_ranker
    if hasattr(model, "span_predictor"):
        del model.span_predictor
        if hasattr(model, "span_predictor_transform"):
            del model.span_predictor_transform

    # monkey-patch _get_video_features：不再访问 self.vision_encoder
    _orig_get_video = model._get_video_features

    def _patched_get_video(video, audio_features):
        B, T, _ = audio_features.shape
        if video is None:
            return audio_features.new_zeros(B, vision_dim, T)
        else:
            return _orig_get_video(video, audio_features)

    model._get_video_features = _patched_get_video


def main(
    settings: list[str],
    cache_path: str,
    batch_size: int,
    checkpoint_path: str,
    num_workers: int = 4,
    reranking_candidates: int = 1,
    metrics: list[str] | None = None,
    no_strip: bool = False,
    num_data: int | None = None,
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

    if metrics is None:
        metrics = ["clap"]

    model = SAMAudio.from_pretrained(checkpoint_path).eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(checkpoint_path)

    if not no_strip:
        _strip_vision_branch(model)

    # 只实例化用户选择的 metric
    judge_metric = Judge(device=device) if "judge" in metrics else None
    aes_metric = Aesthetic(device=device) if "aes" in metrics else None
    clap_metric = CLAP(device=device) if "clap" in metrics else None
    _imagebind = None  # 懒加载

    for setting in settings:
        if rank == 0:
            print(f"Evaluating: {setting}")

        dset = make_dataset(setting, cache_path=cache_path, collate_fn=processor)
        sampler = DistributedSampler(dset) if world_size > 1 else None
        dl = DataLoader(
            dset, batch_size=batch_size, shuffle=False,
            collate_fn=dset.collate, num_workers=num_workers, sampler=sampler,
        )

        all_metrics = []
        if judge_metric is not None:
            all_metrics.append(judge_metric)
        if aes_metric is not None:
            all_metrics.append(aes_metric)
        if clap_metric is not None:
            all_metrics.append(clap_metric)
        if dset.visual:
            if _imagebind is None:
                _imagebind = ImageBind(device=device)
            all_metrics.append(_imagebind)

        dfs = []
        running_sum: dict[str, float] = {}
        running_n: int = 0
        saved_count: int = 0
        with torch.inference_mode():
            pbar = tqdm(dl, disable=rank > 1, desc=setting)
            for batch in pbar:
                batch = batch.to(device)
                result = model.separate(
                    batch, reranking_candidates=reranking_candidates
                )
                input_wavs = model.unbatch(batch.audios.squeeze(1), batch.wav_sizes)
                mets = {}
                for metric in all_metrics:
                    mets.update(
                        metric(
                            target_wavs=result.target,
                            target_wavs_sample_rate=model.sample_rate,
                            descriptions=batch.descriptions,
                            input_wavs=input_wavs,
                            videos=batch.masked_video,
                        )
                    )
                dfs.append(pd.DataFrame.from_dict(mets))

                # 累加运行统计，在 tqdm 中显示当前值和平均值
                running_n += 1
                for k, v_list in mets.items():
                    running_sum[k] = running_sum.get(k, 0.0) + sum(v_list) / len(v_list)
                postfix = {
                    k: f"{running_sum[k] / running_n:.3f}"
                    for k in running_sum
                }
                pbar.set_postfix(postfix)

                # --num-data: 保存输入/输出到 ./output/<setting>/ 并提前退出
                if num_data is not None and rank == 0:
                    save_dir = os.path.join("output", setting)
                    os.makedirs(save_dir, exist_ok=True)
                    batch_meta = []
                    for i in range(len(result.target)):
                        if saved_count >= num_data:
                            break
                        in_wav = input_wavs[i]
                        in_path = f"input_{saved_count:04d}.wav"
                        torchaudio.save(
                            os.path.join(save_dir, in_path),
                            (in_wav.unsqueeze(0) if in_wav.ndim == 1 else in_wav).cpu(),
                            model.sample_rate,
                        )
                        out_wav = result.target[i]
                        out_path = f"output_{saved_count:04d}.wav"
                        torchaudio.save(
                            os.path.join(save_dir, out_path),
                            (out_wav.unsqueeze(0) if out_wav.ndim == 1 else out_wav).cpu(),
                            model.sample_rate,
                        )
                        batch_meta.append({
                            "index": saved_count,
                            "input": in_path,
                            "output": out_path,
                            "description": batch.descriptions[i],
                        })
                        saved_count += 1

                    meta_path = os.path.join(save_dir, "metadata.json")
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            existing = json.load(f)
                    else:
                        existing = []
                    existing.extend(batch_meta)
                    with open(meta_path, "w") as f:
                        json.dump(existing, f, indent=2)

                    if saved_count >= num_data:
                        print(f"Saved {saved_count} samples to {save_dir}, exiting early.")
                        break

        df = pd.concat(dfs)
        averaged_results = gather_and_average_results(df, world_size)
        if rank == 0:
            results_dict = {k: f"{v:.3f}" for k, v in averaged_results.items()}
            print(json.dumps(results_dict, indent=4))
            os.makedirs("results", exist_ok=True)
            outfile = f"results/{setting}.json"
            with open(outfile, "w") as fout:
                print(json.dumps(results_dict), file=fout)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        "-s",
        choices=SETTINGS.keys(),
        help=f"Which setting to evaluate.  Choices: {SETTINGS.keys()}",
        default=["instr-pro"],
        nargs="+",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=os.path.expanduser("~/.cache/sam_audio"),
        help="Where to cache downloaded datasets",
    )
    parser.add_argument(
        "--checkpoint-path", "-p", type=str, default="facebook/sam-audio-small"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-workers", "-w", type=int, default=4, help="Number of workers"
    )
    parser.add_argument("--candidates", "-c", type=int, default=1)
    parser.add_argument(
        "--no-strip",
        action="store_true",
        help="Keep all model branches (vision, span) — needed for visual eval",
    )
    parser.add_argument(
        "--num-data",
        type=int,
        default=None,
        help="Limit to N samples, save input/output to ./output/<setting>/, and exit",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["clap", "aes", "judge"],
        default=["clap"],
        help="Metrics to compute (default: clap only, lightest)",
    )
    opt = parser.parse_args()
    main(
        settings=opt.setting,
        cache_path=opt.cache_path,
        batch_size=opt.batch_size,
        checkpoint_path=opt.checkpoint_path,
        num_workers=opt.num_workers,
        reranking_candidates=opt.candidates,
        metrics=opt.metrics,
        no_strip=opt.no_strip,
        num_data=opt.num_data,
    )
