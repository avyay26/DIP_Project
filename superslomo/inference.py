"""Uniform interface over all interpolation methods: uint8 RGB frames in, uint8 RGB frames out."""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .baselines import CLASSICAL_METHODS, interpolate_classical
from .checkpoint import build_model_from_checkpoint
from .model import SuperSloMo
from .preprocess import rgb_to_tensor, tensor_to_rgb


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(name)


class Interpolator:
    """method in {'lk', 'farneback', 'average'} or a Super SloMo model."""

    def __init__(self, method: str, model: SuperSloMo | None = None, device: torch.device | str = "cpu",
                 fp16: bool = False):
        self.method = method
        self.model = model
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == "cuda"
        if model is None and method not in CLASSICAL_METHODS:
            raise ValueError(f"method {method!r} needs a model; classical methods are {CLASSICAL_METHODS}")
        if model is not None:
            self.model = model.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, name: str, checkpoint: str | Path, device, trust_checkpoint: bool = False,
                        warp_mode: str = "exact", fp16: bool = False) -> "Interpolator":
        model, _ = build_model_from_checkpoint(checkpoint, device, trust_checkpoint, warp_mode)
        return cls(name, model, device, fp16)

    def _autocast(self):
        if self.fp16:
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def __call__(self, frame0: np.ndarray, frame1: np.ndarray, ts: Sequence[float]) -> list[np.ndarray]:
        return self.batch([frame0], [frame1], ts)[0]

    def batch(self, frames0: list[np.ndarray], frames1: list[np.ndarray], ts: Sequence[float]) -> list[list[np.ndarray]]:
        """Interpolate several pairs; returns out[pair][k] for time ts[k]."""
        if self.model is None:
            return [interpolate_classical(a, b, ts, self.method) for a, b in zip(frames0, frames1)]
        with torch.inference_mode(), self._autocast():
            i0 = rgb_to_tensor(list(frames0), self.device)
            i1 = rgb_to_tensor(list(frames1), self.device)
            preds = self.model.interpolate(i0, i1, ts)
        per_t = [tensor_to_rgb(p) for p in preds]  # [t][pair]
        return [[per_t[k][j] for k in range(len(ts))] for j in range(len(frames0))]


    def visibility(self, frame0: np.ndarray, frame1: np.ndarray, t: float) -> np.ndarray | None:
        """V_t0 map (H, W) in [0, 1] for Super SloMo, None for classical methods."""
        if self.model is None:
            return None
        with torch.inference_mode(), self._autocast():
            aux = self.model.interpolate(rgb_to_tensor(frame0, self.device), rgb_to_tensor(frame1, self.device),
                                         [t], return_aux=True)[0]
        return aux["V_t0"][0, 0].float().cpu().numpy()


def factor_times(factor: int) -> list[float]:
    if factor < 2:
        raise ValueError(f"--factor must be >= 2, got {factor}")
    return [k / factor for k in range(1, factor)]


def interpolate_stream(frames: Iterable[np.ndarray], interpolator: Interpolator, ts: Sequence[float],
                       write: Callable[[np.ndarray], None], batch_size: int = 1) -> int:
    """Stream frames -> write(original, intermediates..., original, ...). Holds at most
    batch_size + 1 input frames in memory. Every intermediate frame comes from the two
    *input* frames around it (never from a previous prediction). Returns frames written."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    ts = sorted(float(t) for t in ts)
    written = 0
    buffer: list[np.ndarray] = []

    def flush(buf: list[np.ndarray]) -> int:
        outs = interpolator.batch(buf[:-1], buf[1:], ts)
        n = 0
        for a, mids in zip(buf[:-1], outs):
            write(a)
            for m in mids:
                write(m)
            n += 1 + len(mids)
        return n

    for frame in frames:
        if buffer and frame.shape != buffer[0].shape:
            raise ValueError(f"frame size changed mid-stream: {frame.shape} vs {buffer[0].shape}")
        buffer.append(frame)
        if len(buffer) == batch_size + 1:
            written += flush(buffer)
            buffer = buffer[-1:]
    if not buffer:
        raise ValueError("input contains no frames")
    if len(buffer) > 1:
        written += flush(buffer)
    write(buffer[-1])
    return written + 1
