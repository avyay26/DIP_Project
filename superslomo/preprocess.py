"""Image <-> tensor conversion, normalisation and pad/crop helpers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Reference normalisation (mean of the Adobe-240fps training set), std = 1.
MEAN = (0.429, 0.431, 0.397)
STD = (1.0, 1.0, 1.0)


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Read an image as uint8 RGB (H, W, 3). OpenCV loads BGR, so convert explicitly."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write image: {path}")


def _stats(x: torch.Tensor):
    mean = torch.tensor(MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return mean, std


def normalize(x: torch.Tensor) -> torch.Tensor:
    """x: float RGB in [0, 1], shape (B,3,H,W)."""
    mean, std = _stats(x)
    return (x - mean) / std


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize, clamped to [0, 1]."""
    mean, std = _stats(x)
    return (x * std + mean).clamp(0.0, 1.0)


def rgb_to_tensor(rgb: np.ndarray | list[np.ndarray], device: str | torch.device = "cpu") -> torch.Tensor:
    """uint8 RGB (H,W,3) or list of them -> normalised float tensor (B,3,H,W)."""
    arrs = rgb if isinstance(rgb, list) else [rgb]
    for a in arrs:
        if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB (H,W,3), got dtype={a.dtype} shape={a.shape}")
    x = torch.from_numpy(np.stack(arrs)).to(device).permute(0, 3, 1, 2).float() / 255.0
    return normalize(x)


def tensor_to_rgb(x: torch.Tensor) -> list[np.ndarray]:
    """Normalised tensor (B,3,H,W) -> list of uint8 RGB arrays."""
    x = denormalize(x.detach().float())
    arr = (x * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return [a for a in arr]


def tensor_to_float01(x: torch.Tensor) -> list[np.ndarray]:
    """Normalised tensor (B,3,H,W) -> list of float32 RGB arrays in [0, 1] (for metrics)."""
    x = denormalize(x.detach().float())
    return [a for a in x.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)]


def pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, tuple[int, int]]:
    """Replicate-pad bottom/right so H and W are multiples of `multiple`; returns (padded, (H, W))."""
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x, (h, w)
    return F.pad(x, (0, pw, 0, ph), mode="replicate"), (h, w)


def crop_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    h, w = size
    return x[..., :h, :w]
