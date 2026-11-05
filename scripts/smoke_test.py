"""Synthetic end-to-end demo (no external data, no network).

Trains the tiny Super SloMo for a few steps on synthetic moving/occluding scenes, then
interpolates t = 0.25/0.5/0.75 on a held-out synthetic clip, writes PNGs + smoke.mp4 and
prints PSNR against the exactly rendered ground truth (with classical baselines for context).
"""
from __future__ import annotations

import argparse
import time

import _common  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from superslomo.baselines import interpolate_classical
from superslomo.data import SyntheticMotion, SyntheticScene
from superslomo.inference import Interpolator, pick_device
from superslomo.metrics import psnr
from superslomo.model import SuperSloMo, superslomo_loss
from superslomo.preprocess import save_image_rgb
from superslomo.video import FrameWriter
from superslomo.viz import comparison_grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/smoke")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    start = time.time()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- a few optimisation steps of the tiny model on synthetic triplets
    model = SuperSloMo(width="tiny").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(SyntheticMotion(num_samples=args.steps * args.batch_size, height=64, width=96,
                                        seed=args.seed + 1), batch_size=args.batch_size, shuffle=False)
    model.train()
    first = last = None
    for step, batch in enumerate(loader):
        loss = superslomo_loss(model, batch["I0"].to(device), batch["I1"].to(device),
                               batch["It"].to(device), batch["t"].to(device))
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        first = loss["recon"].item() if first is None else first
        last = loss["recon"].item()
        if step % 10 == 0:
            print(f"step {step:3d}  loss {loss['total'].item():.4f}  recon L1 {last:.4f}")
    print(f"train recon L1: first {first:.4f} -> last {last:.4f} ({args.steps} steps, tiny model)")

    # --- held-out synthetic clip: 4 key frames, factor 4 -> 13 frames
    scene = SyntheticScene(96, 128, seed=args.seed + 999, motion=3.0)
    keys = [scene.render(k / 3) for k in range(4)]
    ts = [0.25, 0.5, 0.75]
    methods = {"tiny_superslomo": Interpolator("superslomo", model, device),
               "average": None, "farneback": None, "lk": None}
    scores: dict[str, dict[float, list[float]]] = {m: {t: [] for t in ts} for m in methods}
    with FrameWriter(out / "smoke.mp4", fps=8) as writer:
        for k in range(3):
            a, b = keys[k], keys[k + 1]
            gts = [scene.render((k + t) / 3) for t in ts]
            preds = {m: (fn(a, b, ts) if fn else interpolate_classical(a, b, ts, m)) for m, fn in methods.items()}
            for m in methods:
                for t, p, g in zip(ts, preds[m], gts):
                    scores[m][t].append(psnr(p, g))
            writer.write(a)
            for j, t in enumerate(ts):
                save_image_rgb(out / f"pair{k}_t{t:.2f}_pred.png", preds["tiny_superslomo"][j])
                save_image_rgb(out / f"pair{k}_t{t:.2f}_gt.png", gts[j])
                writer.write(preds["tiny_superslomo"][j])
            if k == 1:
                cols = [("I0", a), ("LK", preds["lk"][1]), ("Farneback", preds["farneback"][1]),
                        ("tiny SSM", preds["tiny_superslomo"][1]), ("GT", gts[1]), ("I1", b)]
                focus = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1)
                save_image_rgb(out / "comparison_smoke.png", comparison_grid(cols, gts[1], focus))
        writer.write(keys[-1])
    print(f"wrote {writer.count} frames to {out / 'smoke.mp4'} (expected 4*(4-1)+1 = 13)")

    print("\nPSNR vs exact synthetic GT (N = 3 pairs per t, synthetic, NOT a benchmark):")
    print(f"{'method':<16}" + "".join(f"t={t:<7}" for t in ts) + "mean")
    for m in methods:
        vals = [float(np.mean(scores[m][t])) for t in ts]
        print(f"{m:<16}" + "".join(f"{v:<9.2f}" for v in vals) + f"{np.mean(vals):.2f}")
    print(f"\nsmoke test finished in {time.time() - start:.1f}s on {device}")


if __name__ == "__main__":
    main()
