"""Held-out PSNR/SSIM evaluation of lk | farneback | average | pretrained | finetuned.

  Triplets   : official tri_testlist.txt, im1, im3 -> im2 at t = 0.5
  Septuplets : official sep_testlist.txt, im1, im5 -> im2, im3, im4 at t = 0.25/0.5/0.75
  Synthetic  : --synthetic N rendered scenes (pipeline check only, not a benchmark)

Metrics are computed on the cropped (un-padded) uint8 RGB output scaled to [0, 1]:
PSNR = skimage peak_signal_noise_ratio(data_range=1); SSIM = skimage structural_similarity
(channel_axis=-1, data_range=1). Per-sample PSNR is capped at 100 dB (identical images).
Results are merged into OUT/summary.json and OUT/per_sample.csv keyed by dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import _common  # noqa: F401
import numpy as np
import torch

from superslomo.checkpoint import read_checkpoint
from superslomo.data import (SyntheticScene, VimeoSeptuplet, VimeoTriplet, check_vimeo_root, split_ids, subsample)
from superslomo.inference import Interpolator, pick_device
from superslomo.metrics import psnr, ssim
from superslomo.preprocess import load_image_rgb, save_image_rgb
from superslomo.viz import comparison_grid

LABELS = {"lk": "LK", "farneback": "Farneback", "average": "Average", "pretrained": "Super SloMo",
          "finetuned": "SSM finetuned"}
PSNR_CAP = 100.0

Sample = tuple[str, np.ndarray, np.ndarray, list[tuple[float, np.ndarray]]]


def triplet_samples(root: Path, ids: list[str]) -> Iterator[Sample]:
    ds = VimeoTriplet(root, ids)
    for i, sid in enumerate(ids):
        p = ds.frame_paths(i)
        yield sid, load_image_rgb(p[0]), load_image_rgb(p[2]), [(0.5, load_image_rgb(p[1]))]


def septuplet_samples(root: Path, ids: list[str], times: list[float]) -> Iterator[Sample]:
    ds = VimeoSeptuplet(root, ids)
    for i, sid in enumerate(ids):
        p = ds.frame_paths(i)
        gts = [(t, load_image_rgb(p[1 + VimeoSeptuplet.TIMES.index(t)])) for t in times]
        yield sid, load_image_rgb(p[0]), load_image_rgb(p[4]), gts


def synthetic_samples(n: int, times: list[float], seed: int) -> Iterator[Sample]:
    for k in range(n):
        sc = SyntheticScene(128, 224, seed=seed + 50_000 + k, motion=(0.15, 0.3, 0.6)[k % 3])
        yield f"synthetic/{k:04d}", sc.render(0.0), sc.render(1.0), [(t, sc.render(t)) for t in times]


def build_methods(args, device) -> dict[str, Interpolator]:
    methods = {}
    for m in args.methods:
        if m == "pretrained":
            if not Path(args.checkpoint).is_file():
                print(f"WARNING: skipping 'pretrained': checkpoint not found at {args.checkpoint}")
                continue
            methods[m] = Interpolator.from_checkpoint(m, args.checkpoint, device, args.trust_checkpoint,
                                                      fp16=args.fp16)
        elif m == "finetuned":
            if not args.finetuned or not Path(args.finetuned).is_file():
                print(f"WARNING: skipping 'finetuned': --finetuned {args.finetuned!r} not found")
                continue
            methods[m] = Interpolator.from_checkpoint(m, args.finetuned, device, args.trust_checkpoint,
                                                      fp16=args.fp16)
        else:
            methods[m] = Interpolator(m)
    if not methods:
        sys.exit("ERROR: no usable methods")
    return methods


def assert_no_training_leak(args, eval_ids: list[str]) -> None:
    """Fail if any evaluated sequence was used to train or validate the fine-tuned model."""
    if "finetuned" not in args.methods or not args.finetuned or not Path(args.finetuned).is_file():
        return
    ck = read_checkpoint(args.finetuned, "cpu", args.trust_checkpoint)
    used = set(ck.get("train_ids", [])) | set(ck.get("val_ids", []))
    if not used:
        print("WARNING: fine-tuned checkpoint records no train_ids; leakage cannot be checked")
    leak = used & set(eval_ids)
    # Septuplet and triplet ids share the 'xxxxx/yyyy' format but index different clips; only compare like with like.
    assert args.dataset != "triplet" or not leak, f"{len(leak)} evaluation samples were used in fine-tuning: {sorted(leak)[:5]}"


def summarize(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vimeo-triplet")
    ap.add_argument("--vimeo-septuplet")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--split", default="test", choices=("test", "train"))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["lk", "farneback", "pretrained"], choices=list(LABELS))
    ap.add_argument("--times", type=float, nargs="+", default=[0.25, 0.5, 0.75], help="septuplet/synthetic only")
    ap.add_argument("--checkpoint", default="checkpoints/SuperSloMo.ckpt")
    ap.add_argument("--finetuned")
    ap.add_argument("--trust-checkpoint", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-comparisons", type=int, default=5)
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    chosen = [k for k, v in (("triplet", args.vimeo_triplet), ("septuplet", args.vimeo_septuplet),
                             ("synthetic", args.synthetic)) if v]
    if len(chosen) != 1:
        sys.exit("ERROR: give exactly one of --vimeo-triplet, --vimeo-septuplet, --synthetic N")
    args.dataset = chosen[0]
    device = pick_device(args.device)

    if args.dataset == "synthetic":
        times, ids = sorted(args.times), [f"synthetic/{k:04d}" for k in range(args.synthetic)]
        samples = lambda: synthetic_samples(args.synthetic, times, args.seed)  # noqa: E731
    else:
        root_arg = args.vimeo_triplet or args.vimeo_septuplet
        try:
            root = check_vimeo_root(root_arg, args.dataset)
        except FileNotFoundError as err:
            sys.exit(f"ERROR: {err}\nFallback without data: python scripts/evaluate.py --synthetic 30 ...")
        ids = subsample(split_ids(root, args.split, args.dataset), args.max_samples, args.seed)
        if args.split == "test":
            assert not set(split_ids(root, "train", args.dataset)) & set(ids), "test ids overlap the official train list"
        if args.dataset == "triplet":
            times = [0.5]
            samples = lambda: triplet_samples(root, ids)  # noqa: E731
        else:
            times = sorted(args.times)
            bad = [t for t in times if t not in VimeoSeptuplet.TIMES]
            if bad:
                sys.exit(f"ERROR: septuplet times must be among {VimeoSeptuplet.TIMES}, got {bad}")
            samples = lambda: septuplet_samples(root, ids, times)  # noqa: E731
    assert_no_training_leak(args, ids)

    methods = build_methods(args, device)
    rows: list[dict] = []
    runtime = {m: 0.0 for m in methods}
    n_frames = 0
    print(f"evaluating {list(methods)} on {len(ids)} {args.dataset} samples ({args.split}), times {times}, {device}")
    for i, (sid, a, b, gts) in enumerate(samples()):
        ts = [t for t, _ in gts]
        for m, interp in methods.items():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds = interp(a, b, ts)
            if device.type == "cuda":
                torch.cuda.synchronize()
            runtime[m] += time.perf_counter() - t0
            for (t, gt), p in zip(gts, preds):
                rows.append({"sample_id": sid, "dataset": args.dataset, "t": t, "method": m,
                             "psnr": min(psnr(p, gt), PSNR_CAP), "ssim": ssim(p, gt)})
        n_frames += len(ts)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ids)}")

    # ---- summary
    split_label = "synthetic" if args.dataset == "synthetic" else args.split
    summary_block = {"dataset": args.dataset, "split": split_label, "n_samples": len(ids), "times": times,
                     "max_samples": args.max_samples, "seed": args.seed, "device": str(device),
                     "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                     "psnr_ssim_note": "RGB in [0,1]; skimage PSNR(data_range=1), SSIM(channel_axis=-1, data_range=1)",
                     "methods": {}}
    print(f"\n{'method':<14}{'t':<7}{'PSNR':>14}{'SSIM':>16}{'N':>6}")
    for m in methods:
        block = {"runtime_per_frame_s": runtime[m] / max(n_frames, 1),
                 "checkpoint": args.checkpoint if m == "pretrained" else args.finetuned if m == "finetuned" else None}
        for t in times + ["all"]:
            sel = [r for r in rows if r["method"] == m and (t == "all" or r["t"] == t)]
            pm, ps = summarize([r["psnr"] for r in sel])
            sm, ss = summarize([r["ssim"] for r in sel])
            block[str(t)] = {"psnr_mean": pm, "psnr_std": ps, "ssim_mean": sm, "ssim_std": ss, "n": len(sel)}
            if t == "all" and len(times) == 1:
                continue
            print(f"{m:<14}{str(t):<7}{pm:>8.3f} +-{ps:5.2f}{sm:>9.4f} +-{ss:6.4f}{len(sel):>6}")
        summary_block["methods"][m] = block
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary[args.dataset] = summary_block
    summary_path.write_text(json.dumps(summary, indent=2))

    csv_path = out / "per_sample.csv"
    fields = ["sample_id", "dataset", "t", "method", "psnr", "ssim"]
    kept = []
    if csv_path.is_file():
        with csv_path.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["dataset"] != args.dataset]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"\nwrote {summary_path} and {csv_path}")

    # ---- best / worst Super SloMo samples, all methods side by side
    ref = next((m for m in ("pretrained", "finetuned") if m in methods), next(iter(methods)))
    per_id: dict[str, list[float]] = {}
    for r in rows:
        if r["method"] == ref:
            per_id.setdefault(r["sample_id"], []).append(r["psnr"])
    ranked = sorted(per_id, key=lambda s: np.mean(per_id[s]))
    k = min(args.num_comparisons, len(ranked) // 2)
    picks = {sid: f"worst{j + 1}" for j, sid in enumerate(ranked[:k])}
    picks.update({sid: f"best{j + 1}" for j, sid in enumerate(ranked[::-1][:k])})
    for sid, a, b, gts in samples():
        if sid not in picks:
            continue
        t, gt = gts[len(gts) // 2]
        preds = {m: interp(a, b, [t])[0] for m, interp in methods.items()}
        cols = [("I0", a)] + [(f"{LABELS[m]} {psnr(p, gt):.1f}dB", p) for m, p in preds.items()] + [("GT", gt), ("I1", b)]
        focus = np.abs(preds[ref].astype(np.float32) - gt.astype(np.float32)).mean(-1)
        name = f"comparison_{args.dataset}_{picks[sid]}_{sid.replace('/', '-')}.png"
        save_image_rgb(out / name, comparison_grid(cols, gt, focus))
    print(f"wrote {2 * k} comparison images ({ref} best/worst) to {out}")


if __name__ == "__main__":
    main()
