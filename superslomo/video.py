"""Streaming frame I/O with OpenCV: frames are read and written one at a time (RGB uint8)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .preprocess import load_image_rgb, save_image_rgb

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class FrameReader:
    """Iterate RGB frames from a video file or a directory of images (sorted by name)."""

    def __init__(self, source: str | Path, default_fps: float = 30.0):
        self.source = Path(source)
        if not self.source.exists():
            raise FileNotFoundError(f"Input not found: {self.source}")
        self.is_dir = self.source.is_dir()
        if self.is_dir:
            self.paths = sorted(p for p in self.source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self.paths:
                raise FileNotFoundError(f"No images ({sorted(IMAGE_EXTS)}) in directory {self.source}")
            self.fps = default_fps
            self.frame_count = len(self.paths)
        else:
            cap = cv2.VideoCapture(str(self.source))
            if not cap.isOpened():
                raise ValueError(f"OpenCV cannot open video {self.source}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 0 else default_fps
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        if self.is_dir:
            for p in self.paths:
                yield load_image_rgb(p)
            return
        cap = cv2.VideoCapture(str(self.source))
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()


class FrameWriter:
    """Write RGB frames to an MP4 (mp4v) or, if `path` has no video suffix, to numbered PNGs."""

    def __init__(self, path: str | Path, fps: float):
        self.path = Path(path)
        self.fps = fps
        self.count = 0
        self.writer: cv2.VideoWriter | None = None
        self.to_video = self.path.suffix.lower() in {".mp4", ".avi", ".mov"}
        if self.to_video:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.path.mkdir(parents=True, exist_ok=True)

    def write(self, rgb: np.ndarray) -> None:
        if self.to_video:
            if self.writer is None:
                h, w = rgb.shape[:2]
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                if not self.writer.isOpened():
                    raise OSError(f"cv2.VideoWriter could not open {self.path}")
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            save_image_rgb(self.path / f"frame_{self.count:06d}.png", rgb)
        self.count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
