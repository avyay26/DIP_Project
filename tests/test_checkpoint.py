import os
from pathlib import Path

import pytest
import torch

from superslomo.checkpoint import (CheckpointError, load_legacy_checkpoint, save_legacy_checkpoint,
                                   build_model_from_checkpoint)
from superslomo.model import SuperSloMo


@pytest.fixture(scope="module")
def full_ckpt(tmp_path_factory) -> Path:
    """A fake legacy checkpoint from a freshly initialised full-width model (reference layout)."""
    torch.manual_seed(0)
    model = SuperSloMo(width="full")
    path = tmp_path_factory.mktemp("ckpt") / "fake_legacy.ckpt"
    torch.save({"Detail": "fake", "epoch": 3, "learningRate": 1e-4, "loss": [0.1, 0.2],
                "state_dictFC": model.flow_net.state_dict(), "state_dictAT": model.refine_net.state_dict()}, path)
    return path


def test_legacy_roundtrip_strict(full_ckpt):
    model = SuperSloMo(width="full")
    ckpt = load_legacy_checkpoint(full_ckpt, model)
    for k, v in model.flow_net.state_dict().items():
        assert torch.equal(v, ckpt["state_dictFC"][k])
    for k, v in model.refine_net.state_dict().items():
        assert torch.equal(v, ckpt["state_dictAT"][k])


def test_missing_key_raises(tmp_path):
    m = SuperSloMo(width="tiny")
    path = tmp_path / "bad.ckpt"
    torch.save({"state_dictFC": m.flow_net.state_dict()}, path)
    with pytest.raises(KeyError, match="state_dictAT"):
        load_legacy_checkpoint(path, SuperSloMo(width="tiny"))


def test_module_prefix_is_stripped(tmp_path):
    m = SuperSloMo(width="tiny")
    path = tmp_path / "dp.ckpt"
    torch.save({"state_dictFC": {f"module.{k}": v for k, v in m.flow_net.state_dict().items()},
                "state_dictAT": {f"module.{k}": v for k, v in m.refine_net.state_dict().items()}}, path)
    target = SuperSloMo(width="tiny")
    load_legacy_checkpoint(path, target)
    assert torch.equal(target.flow_net.conv1.weight, m.flow_net.conv1.weight)


def test_tiny_model_with_full_checkpoint_fails_loudly(full_ckpt, capsys):
    with pytest.raises(CheckpointError, match="width"):
        load_legacy_checkpoint(full_ckpt, SuperSloMo(width="tiny"))
    assert "SHAPE" in capsys.readouterr().out


def test_missing_file_error_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoints/SuperSloMo.ckpt"):
        load_legacy_checkpoint(tmp_path / "nope.ckpt", SuperSloMo(width="tiny"))


class Custom:  # arbitrary object -> not loadable with weights_only=True
    pass


def test_untrusted_pickle_needs_flag(tmp_path):
    m = SuperSloMo(width="tiny")
    path = tmp_path / "pickled.ckpt"
    torch.save({"state_dictFC": m.flow_net.state_dict(), "state_dictAT": m.refine_net.state_dict(),
                "obj": Custom()}, path)
    with pytest.raises(CheckpointError, match="--trust-checkpoint"):
        load_legacy_checkpoint(path, SuperSloMo(width="tiny"))


def test_finetune_format_roundtrip(tmp_path):
    m = SuperSloMo(width="tiny")
    path = tmp_path / "ft.pt"
    save_legacy_checkpoint(path, m, epoch=1, train_ids=["00001/0001"])
    loaded, ckpt = build_model_from_checkpoint(path)
    assert loaded.width == "tiny" and ckpt["train_ids"] == ["00001/0001"]
    assert torch.equal(loaded.refine_net.conv3.weight, m.refine_net.conv3.weight)


REAL = os.environ.get("SUPERSLOMO_CKPT")


@pytest.mark.skipif(not (REAL and Path(REAL).is_file()),
                    reason="set SUPERSLOMO_CKPT=/path/to/SuperSloMo.ckpt to test the real pretrained checkpoint")
def test_real_checkpoint_loads_and_runs():
    model = SuperSloMo(width="full")
    load_legacy_checkpoint(REAL, model)
    model.eval()
    x = torch.rand(1, 3, 64, 96) - 0.43
    with torch.inference_mode():
        out = model.interpolate(x, x, [0.5])[0]
    assert torch.isfinite(out).all()
    assert (out - x).abs().mean() < 0.02  # identical inputs -> (nearly) identical output
