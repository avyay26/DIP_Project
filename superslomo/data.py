"""Vimeo-90K triplet/septuplet datasets, split handling and a synthetic motion/occlusion scene."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import load_image_rgb, rgb_to_tensor

TRIPLET_LISTS = {"train": "tri_trainlist.txt", "test": "tri_testlist.txt"}
SEPTUPLET_LISTS = {"train": "sep_trainlist.txt", "test": "sep_testlist.txt"}


# ----------------------------------------------------------------------------- splits
def read_id_list(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Vimeo split list not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def check_vimeo_root(root: str | Path, kind: str = "triplet") -> Path:
    """Verify a Vimeo-90K root has `sequences/` and the split lists; raise with a clear message."""
    root = Path(root)
    lists = TRIPLET_LISTS if kind == "triplet" else SEPTUPLET_LISTS
    missing = [str(root / p) for p in ("sequences", *lists.values()) if not (root / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Vimeo-90K {kind} data incomplete; missing: {missing}. Download vimeo_{kind} manually and place it "
            f"at data/vimeo_{kind}/ (or pass --vimeo-{kind} PATH).")
    return root


def split_ids(root: str | Path, split: str, kind: str = "triplet") -> list[str]:
    lists = TRIPLET_LISTS if kind == "triplet" else SEPTUPLET_LISTS
    if split not in lists:
        raise ValueError(f"split must be one of {list(lists)}, got {split!r}")
    return read_id_list(Path(root) / lists[split])


def sample_train_val(train_ids: Sequence[str], num_train: int, num_val: int, seed: int,
                     test_ids: Sequence[str] = ()) -> tuple[list[str], list[str]]:
    """Disjoint random train/val subsets drawn only from the training list."""
    test_set = set(test_ids)
    pool = sorted(set(train_ids) - test_set)
    if num_train + num_val > len(pool):
        raise ValueError(f"requested {num_train}+{num_val} samples but train list has only {len(pool)}")
    rng = random.Random(seed)
    chosen = rng.sample(pool, num_train + num_val)
    train, val = chosen[:num_train], chosen[num_train:]
    assert not set(train) & set(val), "train/val overlap"
    assert not (set(train) | set(val)) & test_set, "train/val subset contains test sequences"
    return train, val


def subsample(ids: Sequence[str], max_samples: int | None, seed: int) -> list[str]:
    ids = list(ids)
    if max_samples is None or max_samples >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, max_samples))


# ----------------------------------------------------------------------------- Vimeo datasets
def _augment(frames: list[np.ndarray], crop: tuple[int, int] | None, rng: random.Random) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]
    if crop is not None:
        ch, cw = crop
        if ch > h or cw > w:
            raise ValueError(f"crop {crop} larger than frame {h}x{w}")
        y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
        frames = [f[y:y + ch, x:x + cw] for f in frames]
    if rng.random() < 0.5:
        frames = [f[:, ::-1] for f in frames]
    if rng.random() < 0.5:
        frames = [f[::-1] for f in frames]
    return [np.ascontiguousarray(f) for f in frames]


class VimeoTriplet(Dataset):
    """Returns I0=im1, It=im2 (ground truth at t=0.5), I1=im3 as normalised (3,H,W) tensors."""

    def __init__(self, root: str | Path, ids: Sequence[str], augment: bool = False,
                 crop: tuple[int, int] | None = None):
        self.root = Path(root)
        self.ids = list(ids)
        self.augment = augment
        self.crop = crop

    def __len__(self) -> int:
        return len(self.ids)

    def frame_paths(self, idx: int) -> list[Path]:
        seq = self.root / "sequences" / self.ids[idx]
        return [seq / f"im{k}.png" for k in (1, 2, 3)]

    def __getitem__(self, idx: int) -> dict:
        frames = [load_image_rgb(p) for p in self.frame_paths(idx)]
        t = 0.5
        if self.augment:
            # torch's RNG is seeded per DataLoader worker and epoch, so augmentations differ across workers
            rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            frames = _augment(frames, self.crop, rng)
            if rng.random() < 0.5:  # temporal reverse: t -> 1 - t (still 0.5)
                frames = frames[::-1]
                t = 1.0 - t
        i0, it, i1 = (rgb_to_tensor(f)[0] for f in frames)
        return {"I0": i0, "It": it, "I1": i1, "t": torch.tensor(t), "id": self.ids[idx]}


class FolderTriplet(VimeoTriplet):
    """Collected data: root/<id>/{im1,im2,im3}.png; target is always t=0.5."""

    def frame_paths(self, idx: int) -> list[Path]:
        return [self.root / self.ids[idx] / f"im{k}.png" for k in (1, 2, 3)]

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        shapes = {tuple(sample[k].shape) for k in ("I0", "It", "I1")}
        if len(shapes) != 1:
            raise ValueError(f"Mismatched triplet dimensions: {self.ids[idx]}")
        # Training forward requires multiples of 32; validation uses unmodified images.
        if self.augment:
            from .preprocess import pad_to_multiple
            for k in ("I0", "It", "I1"):
                sample[k] = pad_to_multiple(sample[k].unsqueeze(0))[0][0]
        return sample


def discover_triplets(root: str | Path) -> list[str]:
    root = Path(root)
    ids = sorted(p.parent.relative_to(root).as_posix() for p in root.rglob("im1.png"))
    if not ids:
        raise ValueError(f"No triplets found under {root}")
    for name in ids:
        for k in (1, 2, 3):
            if not (root / name / f"im{k}.png").is_file():
                raise FileNotFoundError(root / name / f"im{k}.png")
    return ids


class VimeoSeptuplet(Dataset):
    """Evaluation view: I0=im1, I1=im5, targets im2, im3, im4 at t = 0.25, 0.5, 0.75."""

    TIMES = (0.25, 0.5, 0.75)

    def __init__(self, root: str | Path, ids: Sequence[str]):
        self.root = Path(root)
        self.ids = list(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def frame_paths(self, idx: int) -> list[Path]:
        seq = self.root / "sequences" / self.ids[idx]
        return [seq / f"im{k}.png" for k in (1, 2, 3, 4, 5)]

    def __getitem__(self, idx: int) -> dict:
        frames = [load_image_rgb(p) for p in self.frame_paths(idx)]
        tensors = [rgb_to_tensor(f)[0] for f in frames]
        return {"I0": tensors[0], "I1": tensors[4], "targets": tensors[1:4],
                "times": list(self.TIMES), "id": self.ids[idx]}


# ----------------------------------------------------------------------------- synthetic scene
def _texture(h: int, w: int, rng: np.random.Generator, blur: float) -> np.ndarray:
    noise = rng.random((h, w, 3)).astype(np.float32)
    tex = cv2.GaussianBlur(noise, (0, 0), blur)
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-6)
    stripes = 0.5 + 0.5 * np.sin(np.arange(w, dtype=np.float32) / rng.uniform(3, 7))[None, :, None]
    return np.clip(0.75 * tex + 0.25 * stripes, 0, 1)


def _shift(img: np.ndarray, dx: float, dy: float, size: tuple[int, int]) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (size[1], size[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class SyntheticScene:
    """Textured square translating over a slowly panning background, with an occluding bar
    crossing in the opposite direction. Frames at any t in [0, 1] are rendered exactly."""

    def __init__(self, height: int = 96, width: int = 128, seed: int = 0, motion: float = 1.0):
        rng = np.random.default_rng(seed)
        self.h, self.w = height, width
        pad = 32
        self.bg = _texture(height + 2 * pad, width + 2 * pad, rng, blur=2.0)
        self.pad = pad
        s = max(8, height // 3)
        cell = max(2, s // 6)
        yy, xx = np.mgrid[0:s, 0:s]
        checker = (((yy // cell) + (xx // cell)) % 2).astype(np.float32)[..., None]
        c0, c1 = rng.uniform(0.0, 0.3, 3), rng.uniform(0.7, 1.0, 3)
        self.square = (checker * c1 + (1 - checker) * c0).astype(np.float32)
        self.sq_mask = np.ones((s, s), np.float32)
        bw = max(4, width // 10)
        self.bar = np.ones((height, bw, 3), np.float32) * rng.uniform(0.1, 0.9, 3).astype(np.float32)
        self.bar_mask = np.ones((height, bw), np.float32)
        self.sq_start = np.array([rng.uniform(0.05, 0.2) * width, rng.uniform(0.2, 0.5) * height])
        self.sq_vel = np.array([rng.uniform(0.25, 0.4) * width, rng.uniform(-0.1, 0.1) * height]) * motion
        self.bar_start = rng.uniform(0.55, 0.75) * width
        self.bar_vel = -rng.uniform(0.2, 0.35) * width * motion
        self.bg_vel = np.array([rng.uniform(-4, 4), rng.uniform(-2, 2)]) * motion

    def render(self, t: float) -> np.ndarray:
        size = (self.h, self.w)
        bx, by = self.bg_vel * t
        frame = _shift(self.bg, bx - self.pad, by - self.pad, size)
        sx, sy = self.sq_start + self.sq_vel * t
        m = _shift(self.sq_mask, sx, sy, size)[..., None]
        frame = frame * (1 - m) + _shift(self.square, sx, sy, size) * m
        bxp = self.bar_start + self.bar_vel * t
        m = _shift(self.bar_mask, bxp, 0, size)[..., None]
        frame = frame * (1 - m) + _shift(self.bar, bxp, 0, size) * m
        return (np.clip(frame, 0, 1) * 255).round().astype(np.uint8)


class SyntheticMotion(Dataset):
    """Synthetic triplets (I0, It, I1) with random t, for tests and the smoke demo (no downloads)."""

    def __init__(self, num_samples: int = 64, height: int = 64, width: int = 96, seed: int = 0,
                 times: Sequence[float] | None = None):
        self.n, self.h, self.w, self.seed = num_samples, height, width, seed
        self.times = list(times) if times else None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed * 100003 + idx)
        scene = SyntheticScene(self.h, self.w, seed=self.seed * 100003 + idx)
        t = rng.choice(self.times) if self.times else rng.uniform(0.1, 0.9)
        i0, it, i1 = (rgb_to_tensor(scene.render(x))[0] for x in (0.0, t, 1.0))
        return {"I0": i0, "It": it, "I1": i1, "t": torch.tensor(t, dtype=torch.float32), "id": f"synthetic/{idx:05d}"}
