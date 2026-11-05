"""Super SloMo (Jiang et al., CVPR 2018) network, warping and fusion.

Layer names and widths mirror the reference implementation (avinashpaliwal/Super-SloMo)
so that its `SuperSloMo.ckpt` (keys `state_dictFC`, `state_dictAT`) loads with strict=True.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import pad_to_multiple, crop_to

FULL_WIDTHS: tuple[int, ...] = (32, 64, 128, 256, 512, 512)
TINY_WIDTHS: tuple[int, ...] = (8, 16, 32, 32, 64, 64)
WIDTHS = {"full": FULL_WIDTHS, "tiny": TINY_WIDTHS}
PAD_MULTIPLE = 32  # five 2x down-samplings
WARP_MODES = ("exact", "legacy")


def _lrelu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=0.1)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel, stride=1, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, 2)
        return _lrelu(self.conv2(_lrelu(self.conv1(x))))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * out_ch, out_ch, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = _lrelu(self.conv1(x))
        return _lrelu(self.conv2(torch.cat((x, skip), dim=1)))


class UNet(nn.Module):
    """Reference U-Net. Note: the reference applies leaky ReLU to the output layer too;
    this is kept because the pretrained weights were trained with it."""

    def __init__(self, in_ch: int, out_ch: int, widths: Sequence[int] = FULL_WIDTHS):
        super().__init__()
        if len(widths) != 6:
            raise ValueError(f"widths must have 6 entries, got {len(widths)}: {widths}")
        w0, w1, w2, w3, w4, w5 = widths
        self.in_ch, self.out_ch = in_ch, out_ch
        self.conv1 = nn.Conv2d(in_ch, w0, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(w0, w0, 7, stride=1, padding=3)
        self.down1 = Down(w0, w1, 5)
        self.down2 = Down(w1, w2, 3)
        self.down3 = Down(w2, w3, 3)
        self.down4 = Down(w3, w4, 3)
        self.down5 = Down(w4, w5, 3)
        self.up1 = Up(w5, w4)
        self.up2 = Up(w4, w3)
        self.up3 = Up(w3, w2)
        self.up4 = Up(w2, w1)
        self.up5 = Up(w1, w0)
        self.conv3 = nn.Conv2d(w0, out_ch, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_ch:
            raise ValueError(f"UNet expects {self.in_ch} input channels, got {x.shape[1]}")
        h, w = x.shape[-2:]
        if h % PAD_MULTIPLE or w % PAD_MULTIPLE:
            raise ValueError(f"UNet input H, W must be multiples of {PAD_MULTIPLE}, got {h}x{w}; "
                             "use pad_to_multiple() first")
        x = _lrelu(self.conv1(x))
        s1 = _lrelu(self.conv2(x))
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        return _lrelu(self.conv3(x))


def backward_warp(img: torch.Tensor, flow: torch.Tensor, mode: str = "exact") -> torch.Tensor:
    """g(I, F): out(x) = I(x + F(x)), bilinear, zeros outside the image.

    flow[:, 0] is the x (horizontal) displacement, flow[:, 1] the y displacement, in pixels.
    mode="exact":  normalise with 2x/(W-1) - 1 and align_corners=True (zero flow == identity).
    mode="legacy": the reference backWarp formula 2(x/W - 0.5) with the pre-1.3 grid_sample
                   default align_corners=True (a sub-pixel shrink towards the top-left).
    """
    if mode not in WARP_MODES:
        raise ValueError(f"warp mode must be one of {WARP_MODES}, got {mode!r}")
    if flow.shape[1] != 2 or flow.shape[-2:] != img.shape[-2:]:
        raise ValueError(f"flow must be (B,2,H,W) matching img {tuple(img.shape)}, got {tuple(flow.shape)}")
    _, _, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    x = xs.unsqueeze(0) + flow[:, 0]
    y = ys.unsqueeze(0) + flow[:, 1]
    if mode == "exact":
        gx = 2.0 * x / max(w - 1, 1) - 1.0
        gy = 2.0 * y / max(h - 1, 1) - 1.0
    else:
        gx = 2.0 * (x / w - 0.5)
        gy = 2.0 * (y / h - 0.5)
    grid = torch.stack((gx, gy), dim=3)
    return F.grid_sample(img, grid.to(img.dtype), mode="bilinear", padding_mode="zeros", align_corners=True)


def _as_time(t: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Return t as a (B,1,1,1) tensor (scalar t is broadcast)."""
    t = torch.as_tensor(t, dtype=ref.dtype, device=ref.device)
    if t.dim() == 0:
        return t.view(1, 1, 1, 1)
    return t.view(-1, 1, 1, 1)


def intermediate_flows(f01: torch.Tensor, f10: torch.Tensor, t: float | torch.Tensor):
    """Paper eq. (4): F_t0 = -t(1-t) F_01 + t^2 F_10 ;  F_t1 = (1-t)^2 F_01 - t(1-t) F_10."""
    t = _as_time(t, f01)
    f_t0 = -t * (1 - t) * f01 + t * t * f10
    f_t1 = (1 - t) * (1 - t) * f01 - t * (1 - t) * f10
    return f_t0, f_t1


class SuperSloMo(nn.Module):
    """Flow U-Net (6->4) + arbitrary-time refinement U-Net (20->5).

    All inputs are normalised RGB tensors (see preprocess.normalize), shape (B,3,H,W).
    """

    def __init__(self, width: str = "full", warp_mode: str = "exact", eps: float = 1e-8):
        super().__init__()
        if width not in WIDTHS:
            raise ValueError(f"width must be one of {list(WIDTHS)}, got {width!r}")
        if warp_mode not in WARP_MODES:
            raise ValueError(f"warp_mode must be one of {WARP_MODES}, got {warp_mode!r}")
        self.width = width
        self.warp_mode = warp_mode
        self.eps = eps
        self.flow_net = UNet(6, 4, WIDTHS[width])    # checkpoint key: state_dictFC
        self.refine_net = UNet(20, 5, WIDTHS[width])  # checkpoint key: state_dictAT

    def warp(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return backward_warp(img, flow, self.warp_mode)

    def compute_flows(self, i0: torch.Tensor, i1: torch.Tensor):
        out = self.flow_net(torch.cat((i0, i1), dim=1))
        return out[:, 0:2], out[:, 2:4]

    def synthesize(self, i0, i1, f01, f10, t) -> dict[str, torch.Tensor]:
        """Refine flows for time t and fuse. Refinement-net input channel order (20 ch),
        identical to the reference: I0, I1, F_01, F_10, F_t1, F_t0, g(I1, F_t1), g(I0, F_t0).
        Output: dF_t0 = out[:, 0:2], dF_t1 = out[:, 2:4], visibility logit = out[:, 4:5]."""
        tt = _as_time(t, i0)
        f_t0, f_t1 = intermediate_flows(f01, f10, tt)
        g_i0_ft0 = self.warp(i0, f_t0)
        g_i1_ft1 = self.warp(i1, f_t1)
        out = self.refine_net(torch.cat((i0, i1, f01, f10, f_t1, f_t0, g_i1_ft1, g_i0_ft0), dim=1))
        f_t0_ref = f_t0 + out[:, 0:2]
        f_t1_ref = f_t1 + out[:, 2:4]
        v_t0 = torch.sigmoid(out[:, 4:5])
        v_t1 = 1 - v_t0
        g0 = self.warp(i0, f_t0_ref)
        g1 = self.warp(i1, f_t1_ref)
        w0, w1 = (1 - tt) * v_t0, tt * v_t1
        frame = (w0 * g0 + w1 * g1) / (w0 + w1 + self.eps)
        return {"frame": frame, "F_t0": f_t0, "F_t1": f_t1, "F_t0_refined": f_t0_ref,
                "F_t1_refined": f_t1_ref, "V_t0": v_t0, "V_t1": v_t1,
                "g_I0_F_t0": g_i0_ft0, "g_I1_F_t1": g_i1_ft1}

    def forward(self, i0: torch.Tensor, i1: torch.Tensor, t: float | torch.Tensor) -> dict[str, torch.Tensor]:
        """Training-style forward on inputs whose H, W are multiples of 32."""
        f01, f10 = self.compute_flows(i0, i1)
        out = self.synthesize(i0, i1, f01, f10, t)
        out["F_01"], out["F_10"] = f01, f10
        return out

    def interpolate(self, i0: torch.Tensor, i1: torch.Tensor, ts: Iterable[float],
                    return_aux: bool = False) -> list:
        """Direct multi-frame interpolation of arbitrary-size normalised inputs.

        The flow U-Net runs once; the refinement U-Net runs once per t. Frames are returned
        in the order of `ts` (callers pass ascending times). Predictions are never fed back."""
        ts = [float(t) for t in ts]
        for t in ts:
            if not 0.0 <= t <= 1.0:
                raise ValueError(f"interpolation times must lie in [0, 1], got {t}")
        if i0.shape != i1.shape:
            raise ValueError(f"I0 and I1 shapes differ: {tuple(i0.shape)} vs {tuple(i1.shape)}")
        p0, size = pad_to_multiple(i0, PAD_MULTIPLE)
        p1, _ = pad_to_multiple(i1, PAD_MULTIPLE)
        f01, f10 = self.compute_flows(p0, p1)
        results = []
        for t in ts:
            out = self.synthesize(p0, p1, f01, f10, t)
            if return_aux:
                results.append({k: crop_to(v, size) for k, v in out.items()})
            else:
                results.append(crop_to(out["frame"], size))
        return results


def superslomo_loss(model: SuperSloMo, i0, i1, it, t, w_recon: float = 1.0, w_warp: float = 0.5,
                    w_smooth: float = 0.005) -> dict[str, torch.Tensor]:
    """L1 reconstruction + warping + flow smoothness (paper sec. 3.3, without the perceptual term).

    Default ratio 1 : 0.5 : 0.005 follows the reference weights 204 : 102 : 1. When the flow net is
    frozen, the warping and smoothness terms carry no gradient to the refinement net."""
    out = model(i0, i1, t)
    recon = F.l1_loss(out["frame"], it)
    warp = (F.l1_loss(out["g_I0_F_t0"], it) + F.l1_loss(out["g_I1_F_t1"], it)
            + F.l1_loss(model.warp(i0, out["F_10"]), i1) + F.l1_loss(model.warp(i1, out["F_01"]), i0))
    smooth = sum(
        (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean() + (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
        for f in (out["F_01"], out["F_10"])
    )
    total = w_recon * recon + w_warp * warp + w_smooth * smooth
    return {"total": total, "recon": recon, "warp": warp, "smooth": smooth, "frame": out["frame"]}


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
