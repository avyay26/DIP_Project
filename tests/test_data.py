from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from superslomo.data import (SyntheticMotion, VimeoSeptuplet, VimeoTriplet, check_vimeo_root, sample_train_val,
                             split_ids)
from superslomo.preprocess import MEAN


def _make_vimeo(root: Path, kind: str, n_frames: int, train: list[str], test: list[str]) -> Path:
    for sid in train + test:
        seq = root / "sequences" / sid
        seq.mkdir(parents=True)
        for k in range(1, n_frames + 1):
            img = np.zeros((32, 48, 3), np.uint8)
            img[..., 2] = 10 * k  # red channel encodes the frame index (written as BGR)
            cv2.imwrite(str(seq / f"im{k}.png"), img)
    prefix = "tri" if kind == "triplet" else "sep"
    (root / f"{prefix}_trainlist.txt").write_text("\n".join(train) + "\n")
    (root / f"{prefix}_testlist.txt").write_text("\n".join(test) + "\n")
    return root


def _red(x: torch.Tensor) -> float:
    return round((x[0].mean().item() + MEAN[0]) * 255 / 10)


def test_triplet_frame_order(tmp_path):
    root = _make_vimeo(tmp_path, "triplet", 3, ["00001/0001"], ["00002/0001"])
    ds = VimeoTriplet(root, split_ids(root, "train"))
    s = ds[0]
    assert s["I0"].shape == (3, 32, 48)
    assert (_red(s["I0"]), _red(s["It"]), _red(s["I1"])) == (1, 2, 3)  # RGB order, im1/im2/im3
    assert float(s["t"]) == 0.5


def test_triplet_augment_crop(tmp_path):
    root = _make_vimeo(tmp_path, "triplet", 3, ["00001/0001"], [])
    s = VimeoTriplet(root, ["00001/0001"], augment=True, crop=(32, 32))[0]
    assert s["I0"].shape == (3, 32, 32)
    assert {_red(s["I0"]), _red(s["I1"])} == {1, 3} and _red(s["It"]) == 2


def test_septuplet_frame_order(tmp_path):
    root = _make_vimeo(tmp_path, "septuplet", 7, [], ["00003/0002"])
    s = VimeoSeptuplet(root, split_ids(root, "test", "septuplet"))[0]
    assert _red(s["I0"]) == 1 and _red(s["I1"]) == 5
    assert [_red(t) for t in s["targets"]] == [2, 3, 4]
    assert s["times"] == [0.25, 0.5, 0.75]


def test_split_isolation():
    train = [f"{i:05d}/0001" for i in range(50)]
    test = [f"{i:05d}/0001" for i in range(40, 60)]  # overlap 40..49 must be excluded
    tr, va = sample_train_val(train, 20, 5, seed=0, test_ids=test)
    assert len(tr) == 20 and len(va) == 5
    assert not set(tr) & set(va)
    assert not (set(tr) | set(va)) & set(test)
    assert sample_train_val(train, 20, 5, seed=0, test_ids=test) == (tr, va)  # seeded


def test_split_too_large():
    with pytest.raises(ValueError):
        sample_train_val(["a", "b"], 2, 1, seed=0)


def test_missing_vimeo_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="vimeo-triplet"):
        check_vimeo_root(tmp_path / "vimeo_triplet")


def test_synthetic_dataset_shapes():
    s = SyntheticMotion(2, 64, 96, seed=1)[1]
    assert s["I0"].shape == s["It"].shape == s["I1"].shape == (3, 64, 96)
    assert 0 < float(s["t"]) < 1
