import numpy as np
import pytest
import torch

from superslomo.inference import Interpolator, factor_times, interpolate_stream
from superslomo.preprocess import pad_to_multiple

from conftest import zero_heads


def _count_calls(module):
    calls = []
    module.register_forward_hook(lambda m, inp, out: calls.append(inp[0].detach().clone()))
    return calls


def test_flow_net_once_refine_net_per_t(tiny_model):
    flow_calls, refine_calls = _count_calls(tiny_model.flow_net), _count_calls(tiny_model.refine_net)
    i0, i1 = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        tiny_model.interpolate(i0, i1, [0.25, 0.5, 0.75])
    assert len(flow_calls) == 1
    assert len(refine_calls) == 3


def test_no_recursion_refine_inputs_are_original_frames(tiny_model):
    refine_calls = _count_calls(tiny_model.refine_net)
    i0, i1 = torch.randn(1, 3, 40, 50), torch.randn(1, 3, 40, 50)
    with torch.no_grad():
        tiny_model.interpolate(i0, i1, [0.25, 0.5, 0.75])
    p0, _ = pad_to_multiple(i0)
    p1, _ = pad_to_multiple(i1)
    for x in refine_calls:
        assert torch.equal(x[:, 0:3], p0) and torch.equal(x[:, 3:6], p1)


def test_outputs_follow_requested_times(tiny_model):
    i0, i1 = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
    ts = [0.25, 0.5, 0.75]
    with torch.no_grad():
        together = tiny_model.interpolate(i0, i1, ts)
        single = [tiny_model.interpolate(i0, i1, [t])[0] for t in ts]
    for a, b in zip(together, single):
        assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(together[0], together[2])


def test_odd_sizes_roundtrip(tiny_model):
    i0, i1 = torch.randn(2, 3, 97, 131), torch.randn(2, 3, 97, 131)
    with torch.no_grad():
        out = tiny_model.interpolate(i0, i1, [0.5])
    assert out[0].shape == (2, 3, 97, 131)


def test_endpoints_and_identical_frames(tiny_model):
    model = zero_heads(tiny_model)
    i0, i1 = torch.rand(1, 3, 45, 70), torch.rand(1, 3, 45, 70)
    with torch.no_grad():
        o0, o1 = model.interpolate(i0, i1, [0.0, 1.0])
        same = model.interpolate(i0, i0, [0.25, 0.5, 0.75])
    assert torch.allclose(o0, i0, atol=1e-5)
    assert torch.allclose(o1, i1, atol=1e-5)
    for s in same:
        assert torch.allclose(s, i0, atol=1e-5)


def test_times_out_of_range(tiny_model):
    with pytest.raises(ValueError):
        tiny_model.interpolate(torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32), [1.5])


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("n", [2, 5])
def test_stream_frame_count_and_order(n, batch_size):
    frames = [np.full((8, 8, 3), 40 * k, np.uint8) for k in range(n)]
    out: list[np.ndarray] = []
    written = interpolate_stream(iter(frames), Interpolator("average"), factor_times(4), out.append, batch_size)
    assert written == len(out) == 4 * (n - 1) + 1
    values = [int(f[0, 0, 0]) for f in out]
    assert values == sorted(values)  # temporal order (brightness ramps linearly)
    assert values[::4] == [40 * k for k in range(n)]  # originals in place


def test_superslomo_interpolator_batch_matches_single(tiny_model):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (33, 47, 3), dtype=np.uint8) for _ in range(3)]
    interp = Interpolator("superslomo", tiny_model)
    batched = interp.batch(frames[:-1], frames[1:], [0.5])
    for j in range(2):
        single = interp(frames[j], frames[j + 1], [0.5])
        assert np.abs(batched[j][0].astype(int) - single[0].astype(int)).max() <= 1
