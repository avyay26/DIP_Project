"""Interpolate an image pair or a video with lk | farneback | average | superslomo.

  Image pair : --frame0 a.png --frame1 b.png --times 0.25 0.5 0.75 --out DIR
  Video      : --input clip.mp4 (or a directory of frames) --factor 4 --out out.mp4 (or DIR for PNGs)
Videos are streamed: at most --batch-size + 1 input frames are held in memory.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import _common  # noqa: F401

from superslomo.inference import Interpolator, factor_times, interpolate_stream, pick_device
from superslomo.preprocess import load_image_rgb, save_image_rgb
from superslomo.video import FrameReader, FrameWriter

METHODS = ("lk", "farneback", "average", "superslomo")


def build_interpolator(args) -> Interpolator:
    device = pick_device(args.device)
    if args.method != "superslomo":
        return Interpolator(args.method)
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        sys.exit(f"ERROR: --method superslomo needs a checkpoint; not found: {args.checkpoint!r}. Place "
                 "SuperSloMo.ckpt at checkpoints/SuperSloMo.ckpt (or a fine-tuned runs/.../best.pt) and pass "
                 "--checkpoint PATH. For a no-download demo run scripts/smoke_test.py.")
    return Interpolator.from_checkpoint("superslomo", args.checkpoint, device, args.trust_checkpoint,
                                        args.warp_mode, args.fp16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--checkpoint", default="checkpoints/SuperSloMo.ckpt")
    ap.add_argument("--trust-checkpoint", action="store_true")
    ap.add_argument("--warp-mode", choices=("exact", "legacy"), default="exact")
    ap.add_argument("--frame0")
    ap.add_argument("--frame1")
    ap.add_argument("--times", type=float, nargs="+", default=[0.5])
    ap.add_argument("--input", help="video file or directory of frames")
    ap.add_argument("--factor", type=int, default=4, help="slow-motion factor N: times k/N, k=1..N-1")
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: input fps)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None, help="only use the first N input frames")
    ap.add_argument("--fp16", action="store_true", help="autocast to float16 on CUDA")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if bool(args.input) == bool(args.frame0 or args.frame1):
        sys.exit("ERROR: give either --input VIDEO/DIR or both --frame0 and --frame1")
    interp = build_interpolator(args)
    out = Path(args.out)
    start = time.time()

    if args.frame0:
        if not args.frame1:
            sys.exit("ERROR: --frame1 is required with --frame0")
        f0, f1 = load_image_rgb(args.frame0), load_image_rgb(args.frame1)
        ts = sorted(args.times)
        for t, frame in zip(ts, interp(f0, f1, ts)):
            save_image_rgb(out / f"{args.method}_t{t:.3f}.png", frame)
        print(f"wrote {len(ts)} frames to {out} in {time.time() - start:.2f}s")
        return

    reader = FrameReader(args.input)
    fps = args.fps or reader.fps
    frames = iter(reader)
    if args.max_frames:
        frames = (f for i, f in zip(range(args.max_frames), frames))
    with FrameWriter(out, fps) as writer:
        n = interpolate_stream(frames, interp, factor_times(args.factor), writer.write, args.batch_size)
    print(f"{args.method}: wrote {n} frames ({args.factor}x) to {out} at {fps:.2f} fps in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
