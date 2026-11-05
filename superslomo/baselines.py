"""Classical interpolation baselines: Lucas-Kanade (sparse -> dense) and Farnebäck.

Both estimate F_01 and F_10, approximate F_t0 / F_t1 with the same quadratic scheme as
Super SloMo (paper eq. 4), backward-warp with cv2.remap and blend with weights (1-t, t).
No occlusion reasoning is done, which is what Super SloMo's visibility maps add.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

CLASSICAL_METHODS = ("lk", "farneback", "average")


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def farneback_flow(gray0: np.ndarray, gray1: np.ndarray) -> np.ndarray:
    """Dense flow F_01 (H,W,2): gray0(x) ~ gray1(x + F_01(x))."""
    return cv2.calcOpticalFlowFarneback(gray0, gray1, None, pyr_scale=0.5, levels=5, winsize=15,
                                        iterations=5, poly_n=7, poly_sigma=1.5, flags=0)


def _idw_densify(points: np.ndarray, vectors: np.ndarray, h: int, w: int, power: float = 2.0) -> np.ndarray:
    """Inverse-distance-weighted interpolation of sparse vectors onto a coarse grid
    (about 4096 cells), then bilinear resize to full resolution."""
    if len(points) == 0:
        return np.zeros((h, w, 2), np.float32)
    stride = max(4, int(np.ceil(np.sqrt(h * w / 4096))))
    gh, gw = int(np.ceil(h / stride)), int(np.ceil(w / stride))
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    cells = np.stack([(gx.ravel() + 0.5) * stride - 0.5, (gy.ravel() + 0.5) * stride - 0.5], axis=1)
    d2 = ((cells[:, None, :] - points[None, :, :]) ** 2).sum(-1) + 1e-6
    wts = 1.0 / d2 ** (power / 2)
    grid = (wts @ vectors) / wts.sum(1, keepdims=True)
    coarse = grid.reshape(gh, gw, 2).astype(np.float32)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)


def lucas_kanade_flow(gray0: np.ndarray, gray1: np.ndarray, max_corners: int = 1500) -> np.ndarray:
    """Sparse pyramidal LK on Shi-Tomasi corners with a forward-backward consistency check,
    densified by inverse-distance weighting. Returns F_01 (H,W,2)."""
    h, w = gray0.shape
    pts0 = cv2.goodFeaturesToTrack(gray0, maxCorners=max_corners, qualityLevel=0.01, minDistance=5, blockSize=7)
    if pts0 is None:
        return np.zeros((h, w, 2), np.float32)
    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts1, st, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts0, None, **lk)
    back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, pts1, None, **lk)
    fb_err = np.linalg.norm((back - pts0).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < 1.0)
    p0 = pts0.reshape(-1, 2)[ok]
    vec = (pts1.reshape(-1, 2) - pts0.reshape(-1, 2))[ok]
    return _idw_densify(p0.astype(np.float32), vec.astype(np.float32), h, w)


def warp_cv(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward warp: out(x) = img(x + flow(x)), bilinear, border replicate."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def bidirectional_flows(frame0: np.ndarray, frame1: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    g0, g1 = _gray(frame0), _gray(frame1)
    if method == "farneback":
        return farneback_flow(g0, g1), farneback_flow(g1, g0)
    if method == "lk":
        return lucas_kanade_flow(g0, g1), lucas_kanade_flow(g1, g0)
    raise ValueError(f"unknown flow method {method!r}; expected 'lk' or 'farneback'")


def interpolate_classical(frame0: np.ndarray, frame1: np.ndarray, ts: Sequence[float], method: str) -> list[np.ndarray]:
    """uint8 RGB frames in -> list of uint8 RGB frames, one per t (flows computed once)."""
    if method not in CLASSICAL_METHODS:
        raise ValueError(f"method must be one of {CLASSICAL_METHODS}, got {method!r}")
    if frame0.shape != frame1.shape:
        raise ValueError(f"frame shapes differ: {frame0.shape} vs {frame1.shape}")
    a, b = frame0.astype(np.float32), frame1.astype(np.float32)
    if method == "average":
        return [np.clip((1 - t) * a + t * b, 0, 255).round().astype(np.uint8) for t in ts]
    f01, f10 = bidirectional_flows(frame0, frame1, method)
    out = []
    for t in ts:
        f_t0 = -t * (1 - t) * f01 + t * t * f10
        f_t1 = (1 - t) * (1 - t) * f01 - t * (1 - t) * f10
        blend = (1 - t) * warp_cv(a, f_t0) + t * warp_cv(b, f_t1)
        out.append(np.clip(blend, 0, 255).round().astype(np.uint8))
    return out
