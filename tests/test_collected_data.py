import numpy as np
import pytest
from superslomo.data import FolderTriplet, discover_triplets
from superslomo.preprocess import save_image_rgb


def test_folder_midpoint_and_padding(tmp_path):
    for k in (1, 2, 3):
        save_image_rgb(tmp_path / 'sample' / f'im{k}.png', np.full((35, 49, 3), k * 30, np.uint8))
    ids = discover_triplets(tmp_path)
    assert ids == ['sample']
    val = FolderTriplet(tmp_path, ids)[0]
    assert val['t'].item() == .5
    assert val['It'].shape == (3, 35, 49)
    train = FolderTriplet(tmp_path, ids, augment=True)[0]
    assert train['It'].shape == (3, 64, 64)
    (tmp_path / 'sample' / 'im2.png').unlink()
    with pytest.raises(FileNotFoundError):
        discover_triplets(tmp_path)
