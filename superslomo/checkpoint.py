"""Legacy checkpoint loading/saving (avinashpaliwal/Super-SloMo format: state_dictFC, state_dictAT)."""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .model import SuperSloMo

FC_KEY = "state_dictFC"
AT_KEY = "state_dictAT"


class CheckpointError(RuntimeError):
    pass


def read_checkpoint(path: str | Path, device: str | torch.device = "cpu", trust_checkpoint: bool = False) -> dict:
    """torch.load with weights_only=True; falls back to full unpickling only if trusted."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Download SuperSloMo.ckpt from the avinashpaliwal/Super-SloMo "
            "README manually and place it at checkpoints/SuperSloMo.ckpt (or pass --checkpoint PATH).")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError, TypeError) as err:
        if not trust_checkpoint:
            raise CheckpointError(
                f"Could not load {path} with weights_only=True ({type(err).__name__}: {err}). If you trust this "
                "file, re-run with --trust-checkpoint to allow full unpickling.") from err
        warnings.warn(f"Loading {path} with weights_only=False (arbitrary code execution possible).")
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise CheckpointError(f"Checkpoint {path} is a {type(ckpt).__name__}, expected a dict")
    return ckpt


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove the `module.` prefix added by nn.DataParallel."""
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return dict(state)


def compare_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, list]:
    own = module.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    shape_mismatch = [(k, tuple(own[k].shape), tuple(state[k].shape))
                      for k in sorted(set(own) & set(state)) if own[k].shape != state[k].shape]
    return {"missing": missing, "unexpected": unexpected, "shape_mismatch": shape_mismatch,
            "matched": len(set(own) & set(state)) - len(shape_mismatch), "expected": len(own)}


def format_mismatch_table(name: str, report: dict[str, list]) -> str:
    lines = [f"[{name}] matched {report['matched']}/{report['expected']} tensors"]
    for k in report["missing"]:
        lines.append(f"  MISSING     {k}")
    for k in report["unexpected"]:
        lines.append(f"  UNEXPECTED  {k}")
    for k, own, got in report["shape_mismatch"]:
        lines.append(f"  SHAPE       {k:<30} model {own} vs checkpoint {got}")
    return "\n".join(lines)


def metadata_of(ckpt: dict) -> dict[str, Any]:
    """Non-weight entries of a checkpoint, summarised for printing/logging."""
    meta = {}
    for k, v in ckpt.items():
        if k in (FC_KEY, AT_KEY) or k in ("optimizer", "scaler", "scheduler"):
            continue
        if isinstance(v, (list, tuple)) and len(v) > 10:
            meta[k] = f"{type(v).__name__} of len {len(v)} (last: {v[-1]!r})"
        else:
            meta[k] = v
    return meta


def load_legacy_checkpoint(path: str | Path, model: SuperSloMo, device: str | torch.device = "cpu",
                           trust_checkpoint: bool = False, verbose: bool = False) -> dict:
    """Strictly load state_dictFC -> model.flow_net and state_dictAT -> model.refine_net.

    Returns the full checkpoint dict (weights included) so callers can read metadata."""
    ckpt = read_checkpoint(path, device, trust_checkpoint)
    load_state_into(ckpt, model, source=str(path), verbose=verbose)
    return ckpt


def load_state_into(ckpt: dict, model: SuperSloMo, source: str = "checkpoint", verbose: bool = False) -> None:
    """Strict load of an already-read legacy checkpoint dict into `model`."""
    path = source
    missing_keys = [k for k in (FC_KEY, AT_KEY) if k not in ckpt]
    if missing_keys:
        raise KeyError(f"Checkpoint {path} lacks {missing_keys}; keys found: {sorted(ckpt.keys())}")
    problems = []
    for key, net, name in ((FC_KEY, model.flow_net, "flow U-Net"), (AT_KEY, model.refine_net, "refinement U-Net")):
        state = strip_module_prefix(ckpt[key])
        report = compare_state_dict(net, state)
        if verbose or report["missing"] or report["unexpected"] or report["shape_mismatch"]:
            print(format_mismatch_table(f"{name} <- {key}", report))
        if report["missing"] or report["unexpected"] or report["shape_mismatch"]:
            problems.append(name)
            continue
        net.load_state_dict(state, strict=True)
    if problems:
        raise CheckpointError(
            f"Checkpoint {path} does not match the {model.width}-width model for {problems}. "
            "Only width='full' is compatible with the pretrained SuperSloMo.ckpt.")


def save_legacy_checkpoint(path: str | Path, model: SuperSloMo, **extra: Any) -> None:
    """Save in the legacy layout so every script loads it with load_legacy_checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {FC_KEY: model.flow_net.state_dict(), AT_KEY: model.refine_net.state_dict(), "width": model.width}
    payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def build_model_from_checkpoint(path: str | Path, device: str | torch.device = "cpu",
                                trust_checkpoint: bool = False, warp_mode: str = "exact") -> tuple[SuperSloMo, dict]:
    """Build a model of the width recorded in the checkpoint (legacy files default to full)."""
    ckpt = read_checkpoint(path, "cpu", trust_checkpoint)
    width = ckpt.get("width", "full") if isinstance(ckpt.get("width", "full"), str) else "full"
    model = SuperSloMo(width=width, warp_mode=warp_mode)
    load_state_into(ckpt, model, source=str(path))
    return model.to(device).eval(), ckpt
