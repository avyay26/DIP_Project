"""Qualitative side-by-side comparison: I0 | LK | Farnebäck | Super SloMo | GT | I1.

Scenes come from --inputs DIR, whose entries may be:
  * a folder with frame0.png, frame1.png and optional gt.png,
  * a Vimeo-style folder with im1.png, im2.png, im3.png (im2 is GT),
  * a video file: the triplet (k-1, k, k+1) with the most motion is used and frame k is GT
    (i.e. the method interpolates a 2x temporally-subsampled clip).
--synthetic N adds N synthetic motion/occlusion scenes (no data needed).
The zoom box marks where Super SloMo's visibility map is most decisive (occlusion), or
the largest |I1 - I0| region when no Super SloMo model is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from superslomo.data import SyntheticScene
from superslomo.inference import Interpolator, pick_device
from superslomo.metrics import psnr, ssim
from superslomo.preprocess import load_image_rgb, save_image_rgb
from superslomo.video import IMAGE_EXTS, FrameReader
from superslomo.viz import comparison_grid

LABELS = {"lk": "LK", "farneback": "Farneback", "average": "Average", "superslomo": "Super SloMo",
          "pretrained": "Super SloMo", "finetuned": "SSM finetuned"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def most_motion_triplet(path: Path, max_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window: list[np.ndarray] = []
    best, best_score = None, -1.0
    for i, frame in enumerate(FrameReader(path)):
        if i >= max_frames:
            break
        window = (window + [frame])[-3:]
        if len(window) == 3:
            score = float(np.abs(window[0].astype(np.int16) - window[2]).mean())
            if score > best_score:
                best, best_score = list(window), score
    if best is None:
        raise ValueError(f"{path} has fewer than 3 frames")
    return best[0], best[2], best[1]


def load_scenes(inputs: str | None, synthetic: int, max_frames: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]]:
    scenes = []
    if inputs:
        root = Path(inputs)
        if not root.exists():
            raise FileNotFoundError(f"--inputs {root} does not exist; put clips/pairs in data/clips/ or use --synthetic N")
        entries = [root] if root.is_file() else sorted(root.iterdir())
        for e in entries:
            if e.is_dir() and (e / "frame0.png").is_file():
                gt = load_image_rgb(e / "gt.png") if (e / "gt.png").is_file() else None
                scenes.append((e.name, load_image_rgb(e / "frame0.png"), load_image_rgb(e / "frame1.png"), gt))
            elif e.is_dir() and (e / "im1.png").is_file():
                scenes.append((e.name, load_image_rgb(e / "im1.png"), load_image_rgb(e / "im3.png"),
                               load_image_rgb(e / "im2.png")))
            elif e.is_file() and e.suffix.lower() in VIDEO_EXTS:
                a, b, gt = most_motion_triplet(e, max_frames)
                scenes.append((e.stem, a, b, gt))
            elif e.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in e.iterdir()):
                a, b, gt = most_motion_triplet(e, max_frames)
                scenes.append((e.name, a, b, gt))
    for k in range(synthetic):
        sc = SyntheticScene(256, 448, seed=100 + k, motion=(0.15, 0.3, 0.6)[k % 3])
        scenes.append((f"synthetic{k}", sc.render(0.0), sc.render(1.0), sc.render(0.5)))
    if not scenes:
        raise SystemExit("ERROR: no scenes found. Provide --inputs data/clips (pairs, Vimeo folders or videos) "
                         "or --synthetic N.")
    return scenes


def build_methods(args) -> dict[str, Interpolator]:
    device = pick_device(args.device)
    methods: dict[str, Interpolator] = {}
    for m in args.methods:
        if m in ("superslomo", "pretrained"):
            if not Path(args.checkpoint).is_file():
                print(f"WARNING: skipping {m}: checkpoint not found at {args.checkpoint}")
                continue
            methods[m] = Interpolator.from_checkpoint(m, args.checkpoint, device, args.trust_checkpoint)
        elif m == "finetuned":
            if not args.finetuned or not Path(args.finetuned).is_file():
                print(f"WARNING: skipping finetuned: --finetuned {args.finetuned!r} not found")
                continue
            methods[m] = Interpolator.from_checkpoint(m, args.finetuned, device, args.trust_checkpoint)
        else:
            methods[m] = Interpolator(m)
    return methods


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", help="directory of scenes (or a single video)")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["lk", "farneback", "superslomo"],
                    choices=list(LABELS))
    ap.add_argument("--checkpoint", default="checkpoints/SuperSloMo.ckpt")
    ap.add_argument("--finetuned")
    ap.add_argument("--trust-checkpoint", action="store_true")
    ap.add_argument("--max-frames", type=int, default=300, help="frames scanned per video for the triplet")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    methods = build_methods(args)
    out = Path(args.out)
    for name, a, b, gt in load_scenes(args.inputs, args.synthetic, args.max_frames):
        preds = {m: interp(a, b, [0.5])[0] for m, interp in methods.items()}
        focus = None
        for key in ("superslomo", "pretrained", "finetuned"):
            if key in methods:
                focus = np.abs(methods[key].visibility(a, b, 0.5) - 0.5)
                break
        if focus is None:
            focus = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1)
        cols = [("I0", a)] + [(LABELS[m], p) for m, p in preds.items()]
        if gt is not None:
            cols.append(("GT", gt))
        cols.append(("I1", b))
        save_image_rgb(out / f"comparison_{name}.png", comparison_grid(cols, gt, focus))
        scores = "  ".join(f"{LABELS[m]} {psnr(p, gt):.2f}dB/{ssim(p, gt):.3f}" for m, p in preds.items()) \
            if gt is not None else "(no GT)"
        print(f"{name}: {scores}  -> {out / f'comparison_{name}.png'}")


if __name__ == "__main__":
    main()
