import cv2
import numpy as np
import pytest

from superslomo.baselines import interpolate_classical
from superslomo.metrics import psnr


def _textured(h=96, w=128, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 2.0)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def _shift(img, dx, dy=0.0):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("method", ["lk", "farneback", "average"])
def test_shape_dtype(method):
    a, b = _textured(), _textured(seed=1)
    out = interpolate_classical(a, b, [0.25, 0.5, 0.75], method)
    assert len(out) == 3
    assert all(o.shape == a.shape and o.dtype == np.uint8 for o in out)


@pytest.mark.parametrize("method", ["lk", "farneback"])
def test_beats_frame_averaging_on_translation(method):
    base = _textured(seed=2)
    a, gt, b = _shift(base, 0), _shift(base, 3, 1.5), _shift(base, 6, 3)
    m = 12  # ignore borders where content enters/leaves
    pred = interpolate_classical(a, b, [0.5], method)[0]
    avg = interpolate_classical(a, b, [0.5], "average")[0]
    p_pred, p_avg = psnr(pred[m:-m, m:-m], gt[m:-m, m:-m]), psnr(avg[m:-m, m:-m], gt[m:-m, m:-m])
    assert p_pred > p_avg + 3, (p_pred, p_avg)


def test_unknown_method():
    with pytest.raises(ValueError):
        interpolate_classical(_textured(), _textured(), [0.5], "raft")
