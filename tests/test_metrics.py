import math

import numpy as np
import pytest

from superslomo.metrics import psnr, ssim


def test_identical_images():
    img = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    assert psnr(img, img) == math.inf
    assert ssim(img, img) == pytest.approx(1.0)


def test_known_noise_psnr():
    gt = np.full((32, 32, 3), 0.5, np.float32)
    pred = gt + 0.1 * np.where(np.random.default_rng(1).random(gt.shape) < 0.5, -1, 1)  # MSE = 0.01
    assert psnr(pred, gt) == pytest.approx(20.0, abs=1e-4)


def test_uint8_matches_float():
    rng = np.random.default_rng(2)
    a, b = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8), rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert psnr(a, b) == pytest.approx(psnr(a / 255.0, b / 255.0))
    assert ssim(a, b) < 0.5


def test_shape_mismatch():
    with pytest.raises(ValueError):
        psnr(np.zeros((8, 8, 3)), np.zeros((8, 9, 3)))
