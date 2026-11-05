"""Verify that the full-width model loads the legacy SuperSloMo.ckpt strictly and interpolates sensibly.

Prints parameter counts, the key/shape match summary and the checkpoint metadata; writes the
t = 0.5 frame and a [frame0 | pred | frame1] strip. Optionally compares both warp modes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch

from superslomo.checkpoint import (AT_KEY, FC_KEY, compare_state_dict, format_mismatch_table, load_state_into,
                                   metadata_of, read_checkpoint, strip_module_prefix)
from superslomo.data import SyntheticScene
from superslomo.inference import Interpolator, pick_device
from superslomo.metrics import psnr
from superslomo.model import SuperSloMo, count_parameters
from superslomo.preprocess import load_image_rgb, save_image_rgb
from superslomo.viz import label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/SuperSloMo.ckpt")
    ap.add_argument("--frame0")
    ap.add_argument("--frame1")
    ap.add_argument("--gt", help="optional ground-truth middle frame for a PSNR check")
    ap.add_argument("--out", default="outputs/pretrained/check")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--trust-checkpoint", action="store_true")
    ap.add_argument("--compare-warp-modes", action="store_true",
                    help="also run warp_mode='legacy' (reference backWarp formula) and report the difference")
    args = ap.parse_args()
    out = Path(args.out)
    device = pick_device(args.device)

    model = SuperSloMo(width="full")
    print(f"flow U-Net (6->4)       parameters: {count_parameters(model.flow_net):,}")
    print(f"refinement U-Net (20->5) parameters: {count_parameters(model.refine_net):,}")

    if not Path(args.checkpoint).is_file():
        print(f"\nERROR: checkpoint not found at {args.checkpoint}.\n"
              "Download SuperSloMo.ckpt manually (link in the avinashpaliwal/Super-SloMo README) and place it at "
              "checkpoints/SuperSloMo.ckpt, or pass --checkpoint PATH. Nothing was verified.\n"
              "Fallback: `python scripts/smoke_test.py` runs the synthetic demo without any checkpoint.")
        return 2

    ckpt = read_checkpoint(args.checkpoint, "cpu", args.trust_checkpoint)
    print(f"\ncheckpoint keys: {sorted(ckpt.keys())}")
    for key, net, name in ((FC_KEY, model.flow_net, "flow U-Net"), (AT_KEY, model.refine_net, "refinement U-Net")):
        if key in ckpt:
            print(format_mismatch_table(f"{name} <- {key}", compare_state_dict(net, strip_module_prefix(ckpt[key]))))
    print("metadata:")
    for k, v in metadata_of(ckpt).items():
        print(f"  {k}: {v}")
    load_state_into(ckpt, model, source=args.checkpoint)
    print("strict load: OK (state_dictFC -> flow_net, state_dictAT -> refine_net)")

    gt = None
    if args.frame0 and args.frame1:
        f0, f1 = load_image_rgb(args.frame0), load_image_rgb(args.frame1)
        gt = load_image_rgb(args.gt) if args.gt else None
    else:
        print("\nWARNING: --frame0/--frame1 not given; using a synthetic pair (with exact GT). "
              "Use a real pair for the visual check.")
        scene = SyntheticScene(256, 448, seed=7)
        f0, f1, gt = scene.render(0.0), scene.render(1.0), scene.render(0.5)

    interp = Interpolator("superslomo", model, device)
    pred = interp(f0, f1, [0.5])[0]
    save_image_rgb(out / "pred_t0.50.png", pred)
    strip = [label(f0, "frame0"), label(pred, "Super SloMo t=0.5"), label(f1, "frame1")]
    if gt is not None:
        strip.append(label(gt, "GT"))
    save_image_rgb(out / "side_by_side.png", np.hstack(strip))
    print(f"\nwrote {out / 'pred_t0.50.png'} and {out / 'side_by_side.png'}")
    avg = ((f0.astype(np.float32) + f1) / 2).round().astype(np.uint8)
    if gt is not None:
        print(f"PSNR vs GT (N=1): Super SloMo {psnr(pred, gt):.2f} dB | frame average {psnr(avg, gt):.2f} dB")

    if args.compare_warp_modes:
        model.warp_mode = "legacy"
        pred_legacy = interp(f0, f1, [0.5])[0]
        model.warp_mode = "exact"
        save_image_rgb(out / "pred_t0.50_legacywarp.png", pred_legacy)
        diff = np.abs(pred.astype(np.float32) - pred_legacy).mean()
        print(f"warp mode exact vs legacy: mean |diff| = {diff:.3f} (0-255 scale)")
        if gt is not None:
            print(f"PSNR vs GT (N=1): exact {psnr(pred, gt):.2f} dB | legacy {psnr(pred_legacy, gt):.2f} dB")
    return 0


if __name__ == "__main__":
    with torch.inference_mode():
        sys.exit(main())
