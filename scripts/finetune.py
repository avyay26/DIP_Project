"""Small controlled fine-tune of the pretrained Super SloMo on Vimeo-90K triplets.

Train/val subsets are drawn (seeded, disjoint) only from tri_trainlist.txt; the test list is
never used for training. With --freeze-flow only the refinement U-Net is trained, so the
warping/smoothness terms are constant and the effective loss is L1 reconstruction.
No perceptual (VGG) loss: it would need a weight download.
--synthetic runs the same loop on synthetic data with the tiny model (pipeline check only).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from superslomo.checkpoint import load_legacy_checkpoint, read_checkpoint, load_state_into, save_legacy_checkpoint
from superslomo.data import (SyntheticMotion, VimeoTriplet, check_vimeo_root, sample_train_val, split_ids)
from superslomo.metrics import psnr, ssim
from superslomo.model import SuperSloMo, superslomo_loss
from superslomo.preprocess import tensor_to_float01


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=_common.ROOT).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git repository)"


def skip(reason: str) -> None:
    print(f"FINE-TUNING SKIPPED: {reason}")
    sys.exit(3)


@torch.no_grad()
def validate(model: SuperSloMo, loader: DataLoader, device) -> dict[str, float]:
    model.eval()
    ps, ss = [], []
    for b in loader:
        pred = model(b["I0"].to(device), b["I1"].to(device), b["t"].to(device))["frame"]
        for p, g in zip(tensor_to_float01(pred), tensor_to_float01(b["It"])):
            ps.append(min(psnr(p, g), 100.0))
            ss.append(ssim(p, g))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(ps)}


def set_train_mode(model: SuperSloMo, freeze_flow: bool) -> None:
    model.train()
    if freeze_flow:
        model.flow_net.eval()


def probe_batch_size(model, sample, batch_size, device, amp) -> int:
    """Halve the batch size until one forward/backward step fits in GPU memory."""
    while batch_size >= 1:
        try:
            x = {k: sample[k].unsqueeze(0).repeat(batch_size, *[1] * sample[k].dim()).to(device)
                 for k in ("I0", "I1", "It", "t")}
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = superslomo_loss(model, x["I0"], x["I1"], x["It"], x["t"])["total"]
            loss.backward()
            model.zero_grad(set_to_none=True)
            return batch_size
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"OOM at batch size {batch_size}; halving")
            batch_size //= 2
    raise RuntimeError("even batch size 1 does not fit in GPU memory; use --crop H W")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/SuperSloMo.ckpt")
    ap.add_argument("--trust-checkpoint", action="store_true")
    ap.add_argument("--vimeo-triplet", default="data/vimeo_triplet")
    ap.add_argument("--num-train", type=int, default=700)
    ap.add_argument("--num-val", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--cosine", action="store_true", help="cosine LR decay over all steps")
    ap.add_argument("--freeze-flow", action="store_true")
    ap.add_argument("--crop", type=int, nargs=2, metavar=("H", "W"), help="random crop (multiples of 32)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-cpu", action="store_true", help="run on CPU (very slow; for debugging)")
    ap.add_argument("--synthetic", action="store_true", help="pipeline check: tiny model + synthetic data")
    ap.add_argument("--max-steps", type=int, default=None, help="stop each epoch after N steps (debug)")
    ap.add_argument("--resume", help="resume from runs/.../last.pt")
    ap.add_argument("--out", default="runs/finetune")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)

    # ---- preconditions
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        if not args.allow_cpu:
            skip("CUDA GPU not available (pass --device cpu --allow-cpu to force a slow CPU run)")
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cpu" and not (args.allow_cpu or args.synthetic):
        skip("--device cpu needs --allow-cpu")
    amp = args.amp and device.type == "cuda"

    if args.synthetic:
        model = SuperSloMo(width="tiny")
        train_ids = [f"synthetic/{i:05d}" for i in range(args.num_train)]
        val_ids = [f"synthetic/{i:05d}" for i in range(args.num_train, args.num_train + args.num_val)]
        train_ds = SyntheticMotion(args.num_train, 64, 96, seed=args.seed)
        val_ds = SyntheticMotion(args.num_val, 64, 96, seed=args.seed + 10_000, times=[0.5])
    else:
        try:
            root = check_vimeo_root(args.vimeo_triplet, "triplet")
        except FileNotFoundError as err:
            skip(str(err))
        if not Path(args.checkpoint).is_file() and not args.resume:
            skip(f"pretrained checkpoint not found at {args.checkpoint} (Phase 2 must pass first)")
        model = SuperSloMo(width="full")
        if not args.resume:
            load_legacy_checkpoint(args.checkpoint, model, "cpu", args.trust_checkpoint)
        test_ids = split_ids(root, "test")
        train_ids, val_ids = sample_train_val(split_ids(root, "train"), args.num_train, args.num_val,
                                              args.seed, test_ids)
        train_ds = VimeoTriplet(root, train_ids, augment=True, crop=tuple(args.crop) if args.crop else None)
        val_ds = VimeoTriplet(root, val_ids)
    model.to(device)

    if args.freeze_flow:
        for p in model.flow_net.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, history, best_psnr = 0, [], -math.inf
    if args.resume:
        ck = read_checkpoint(args.resume, "cpu", args.trust_checkpoint)
        load_state_into(ck, model, source=args.resume)
        if list(ck.get("train_ids", [])) != train_ids or list(ck.get("val_ids", [])) != val_ids:
            raise ValueError("resume checkpoint was trained on a different subset; use the same --seed/--num-*")
        opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck and amp:
            scaler.load_state_dict(ck["scaler"])
        start_epoch, history = int(ck["epoch"]), list(ck.get("history", []))
        best_psnr = max([h["val_psnr"] for h in history] or [-math.inf])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    if device.type == "cuda" and not args.synthetic:
        set_train_mode(model, args.freeze_flow)
        bs = probe_batch_size(model, train_ds[0], args.batch_size, device, amp)
        if bs != args.batch_size:
            print(f"using batch size {bs} (requested {args.batch_size})")
            args.batch_size = bs

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen, drop_last=True,
                              num_workers=0 if args.synthetic else args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False,
                            num_workers=0 if args.synthetic else args.workers)
    steps_per_epoch = min(len(train_loader), args.max_steps or len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / max(total_steps, 1))))
        if args.cosine else (lambda s: 1.0))
    for _ in range(start_epoch * steps_per_epoch):
        sched.step()

    config = {"args": vars(args), "git_commit": git_commit(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "python": platform.python_version(), "os": platform.platform(),
              "num_train": len(train_ids), "num_val": len(val_ids), "trainable_params": sum(p.numel() for p in params),
              "loss": "L1 recon + 0.5 warp + 0.005 smooth (warp/smooth constant when flow net is frozen); no perceptual"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    if not history:
        base = validate(model, val_loader, device)
        print(f"epoch 0 (before fine-tuning): val PSNR {base['psnr']:.3f}  SSIM {base['ssim']:.4f}  (N={base['n']})")
        history.append({"epoch": 0, "train_loss": None, "val_psnr": base["psnr"], "val_ssim": base["ssim"],
                        "val_n": base["n"], "time_s": 0.0})
        best_psnr = base["psnr"]

    for epoch in range(start_epoch + 1, args.epochs + 1):
        set_train_mode(model, args.freeze_flow)
        t0, losses, recons = time.time(), [], []
        for step, b in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = superslomo_loss(model, b["I0"].to(device, non_blocking=True), b["I1"].to(device),
                                       b["It"].to(device), b["t"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss["total"].item())
            recons.append(loss["recon"].item())
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {losses[-1]:.4f} recon {recons[-1]:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}")
        val = validate(model, val_loader, device)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_recon_l1": float(np.mean(recons)),
                 "val_psnr": val["psnr"], "val_ssim": val["ssim"], "val_n": val["n"], "time_s": time.time() - t0}
        history.append(entry)
        print(f"epoch {epoch}: train loss {entry['train_loss']:.4f}  val PSNR {val['psnr']:.3f}  "
              f"SSIM {val['ssim']:.4f}  ({entry['time_s']:.0f}s)")
        extra = dict(epoch=epoch, history=history, train_ids=train_ids, val_ids=val_ids,
                     optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                     Detail=f"fine-tuned from {args.checkpoint}; freeze_flow={args.freeze_flow}")
        save_legacy_checkpoint(out / "last.pt", model, **extra)
        if val["psnr"] > best_psnr:
            best_psnr = val["psnr"]
            save_legacy_checkpoint(out / "best.pt", model, **extra)
            print(f"  new best -> {out / 'best.pt'}")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    (out / "history.json").write_text(json.dumps(history, indent=2))
    if not (out / "best.pt").exists():
        print("note: no epoch beat the pre-fine-tuning validation PSNR; best.pt = last.pt")
        (out / "best.pt").write_bytes((out / "last.pt").read_bytes())
    print(f"done. history: {out / 'history.json'}")


if __name__ == "__main__":
    main()
