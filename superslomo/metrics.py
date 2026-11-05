"""PSNR / SSIM on RGB images in [0, 1] (float) or uint8.

PSNR: skimage.metrics.peak_signal_noise_ratio(data_range=1.0); identical images return +inf.
SSIM: skimage.metrics.structural_similarity(channel_axis=-1, data_range=1.0) with default
      7x7 uniform window, i.e. SSIM computed per RGB channel and averaged.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def _check(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred, gt)
    p, g = _to01(pred), _to01(gt)
    if np.array_equal(p, g):
        return float("inf")
    return float(peak_signal_noise_ratio(g, p, data_range=1.0))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    _check(pred, gt)
    return float(structural_similarity(_to01(gt), _to01(pred), channel_axis=-1, data_range=1.0))
