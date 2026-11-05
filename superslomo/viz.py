"""Labelled comparison grids (shared by compare.py and evaluate.py)."""
from __future__ import annotations

import cv2
import numpy as np


def label(img: np.ndarray, text: str, height: int = 22) -> np.ndarray:
    bar = np.full((height, img.shape[1], 3), 255, np.uint8)
    scale = 0.45 if img.shape[1] >= 160 else 0.35
    cv2.putText(bar, text, (4, height - 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def error_map(pred: np.ndarray, gt: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """Mean absolute RGB error, amplified and colour-mapped (returns RGB uint8)."""
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(-1)
    err = np.clip(err * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(err, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def find_focus_box(score: np.ndarray, size: int) -> tuple[int, int, int, int]:
    """Top-left (y, x) and size of the square window with the largest mean `score`."""
    h, w = score.shape
    size = max(8, min(size, h, w))
    box = cv2.blur(score.astype(np.float32), (size, size))
    half = size // 2
    inner = box[half:h - size + half + 1, half:w - size + half + 1]
    y, x = np.unravel_index(int(np.argmax(inner)), inner.shape)
    return int(y), int(x), size, size


def crop_zoom(img: np.ndarray, box: tuple[int, int, int, int], out_w: int) -> np.ndarray:
    y, x, h, w = box
    crop = img[y:y + h, x:x + w]
    return cv2.resize(crop, (out_w, int(round(out_w * h / w))), interpolation=cv2.INTER_NEAREST)


def draw_box(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    y, x, h, w = box
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 1)
    return out


def comparison_grid(columns: list[tuple[str, np.ndarray]], gt: np.ndarray | None = None,
                    focus: np.ndarray | None = None, zoom_frac: float = 0.35) -> np.ndarray:
    """Row 1: labelled images (box marks the zoom region); row 2: zoomed crops;
    row 3 (if gt): |pred - gt| error maps for every predicted column."""
    h, w = columns[0][1].shape[:2]
    if focus is None:
        focus = np.zeros((h, w), np.float32)
    box = find_focus_box(focus, int(min(h, w) * zoom_frac))
    row1 = [label(draw_box(img, box), name) for name, img in columns]
    row2 = [label(crop_zoom(img, box, w), f"zoom {name}") for name, img in columns]
    rows = [np.hstack(row1), np.hstack(row2)]
    if gt is not None:
        blank = np.full_like(columns[0][1], 255)
        maps = []
        for name, img in columns:
            is_pred = name not in ("I0", "I1", "GT")
            maps.append(label(error_map(img, gt) if is_pred else blank, f"|{name}-GT|" if is_pred else ""))
        rows.append(np.hstack(maps))
    widths = {r.shape[1] for r in rows}
    if len(widths) != 1:
        raise ValueError(f"inconsistent row widths {widths}")
    return np.vstack(rows)
