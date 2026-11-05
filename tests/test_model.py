import pytest
import torch
import torch.nn as nn

from superslomo.data import SyntheticMotion
from superslomo.model import (FULL_WIDTHS, SuperSloMo, UNet, backward_warp, count_parameters,
                              intermediate_flows, superslomo_loss)


def test_unet_channel_contract(tiny_model):
    x6 = torch.randn(2, 6, 64, 96)
    assert tiny_model.flow_net(x6).shape == (2, 4, 64, 96)
    x20 = torch.randn(2, 20, 64, 96)
    assert tiny_model.refine_net(x20).shape == (2, 5, 64, 96)


def test_unet_rejects_unpadded_and_wrong_channels(tiny_model):
    with pytest.raises(ValueError, match="multiples of 32"):
        tiny_model.flow_net(torch.randn(1, 6, 50, 64))
    with pytest.raises(ValueError, match="input channels"):
        tiny_model.flow_net(torch.randn(1, 5, 64, 64))


def test_full_width_matches_reference_size():
    net = UNet(6, 4, FULL_WIDTHS)

    def conv(i, o, k):
        return i * o * k * k + o
    # Layer list of the reference model.UNet(6, 4) (avinashpaliwal/Super-SloMo).
    expected = (conv(6, 32, 7) + conv(32, 32, 7)
                + conv(32, 64, 5) + conv(64, 64, 5) + conv(64, 128, 3) + conv(128, 128, 3)
                + conv(128, 256, 3) + conv(256, 256, 3) + conv(256, 512, 3) + conv(512, 512, 3)
                + 2 * conv(512, 512, 3)
                + conv(512, 512, 3) + conv(1024, 512, 3) + conv(512, 256, 3) + conv(512, 256, 3)
                + conv(256, 128, 3) + conv(256, 128, 3) + conv(128, 64, 3) + conv(128, 64, 3)
                + conv(64, 32, 3) + conv(64, 32, 3) + conv(32, 4, 3))
    assert count_parameters(net) == expected
    assert count_parameters(UNet(20, 5, FULL_WIDTHS)) == count_parameters(net) + 14 * 32 * 49 + 32 * 9 + 1


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_intermediate_flow_equations(t):
    f01, f10 = torch.randn(2, 2, 8, 8), torch.randn(2, 2, 8, 8)
    f_t0, f_t1 = intermediate_flows(f01, f10, t)
    assert torch.allclose(f_t0, -t * (1 - t) * f01 + t ** 2 * f10)
    assert torch.allclose(f_t1, (1 - t) ** 2 * f01 - t * (1 - t) * f10)


def test_intermediate_flow_per_sample_t():
    f01, f10 = torch.randn(3, 2, 4, 4), torch.randn(3, 2, 4, 4)
    ts = torch.tensor([0.1, 0.5, 0.9])
    f_t0, _ = intermediate_flows(f01, f10, ts)
    for i, t in enumerate(ts.tolist()):
        assert torch.allclose(f_t0[i], -t * (1 - t) * f01[i] + t * t * f10[i], atol=1e-6)


def test_warp_zero_flow_is_identity():
    img = torch.rand(2, 3, 17, 23)
    out = backward_warp(img, torch.zeros(2, 2, 17, 23))
    assert torch.allclose(out, img, atol=1e-5)


def test_warp_integer_shift():
    img = torch.rand(1, 3, 20, 30)
    flow = torch.zeros(1, 2, 20, 30)
    flow[:, 0], flow[:, 1] = 3.0, -2.0  # out(y, x) = img(y - 2, x + 3)
    out = backward_warp(img, flow)
    assert torch.allclose(out[..., 2:, :-3], img[..., :-2, 3:], atol=1e-5)
    assert out[..., :2, :].abs().max() < 1e-5  # sampled outside -> zeros padding


def test_warp_legacy_mode_differs_by_subpixel_scale():
    img = torch.arange(32, dtype=torch.float32).view(1, 1, 1, 32).expand(1, 1, 4, 32).contiguous()
    out = backward_warp(img, torch.zeros(1, 2, 4, 32), mode="legacy")
    # legacy samples x * (W-1)/W: off by < 1 px, exactly 0 at the left edge
    assert torch.allclose(out[0, 0, 0], torch.arange(32.0) * 31 / 32, atol=1e-4)


class _ConstNet(nn.Module):
    def __init__(self, out_ch: int, logit: float):
        super().__init__()
        self.out_ch, self.logit = out_ch, logit

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.out_ch, *x.shape[-2:])
        out[:, -1] = self.logit
        return out


def test_fusion_visibility_and_eps(tiny_model):
    i0, i1 = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    f = torch.zeros(1, 2, 32, 32)
    tiny_model.refine_net = _ConstNet(5, logit=0.3)
    out = tiny_model.synthesize(i0, i1, f, f, 0.4)
    assert torch.allclose(out["V_t1"], 1 - out["V_t0"])
    v0 = torch.sigmoid(torch.tensor(0.3))
    w0, w1 = 0.6 * v0, 0.4 * (1 - v0)
    assert torch.allclose(out["frame"], (w0 * i0 + w1 * i1) / (w0 + w1), atol=1e-5)
    # V_t0 -> 0 at t = 0 makes both weights 0: eps keeps the result finite
    tiny_model.refine_net = _ConstNet(5, logit=-1e4)
    out = tiny_model.synthesize(i0, i1, f, f, 0.0)
    assert torch.isfinite(out["frame"]).all()


def test_frozen_flow_net_gets_no_grads(tiny_model):
    model = tiny_model.train()
    for p in model.flow_net.parameters():
        p.requires_grad_(False)
    b = next(iter(torch.utils.data.DataLoader(SyntheticMotion(2, 64, 64), batch_size=2)))
    superslomo_loss(model, b["I0"], b["I1"], b["It"], b["t"])["total"].backward()
    assert all(p.grad is None for p in model.flow_net.parameters())
    grads = [p.grad for p in model.refine_net.parameters()]
    assert all(g is not None for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    model = SuperSloMo(width="tiny").train()
    b = next(iter(torch.utils.data.DataLoader(SyntheticMotion(4, 64, 64, seed=3), batch_size=4)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = superslomo_loss(model, b["I0"], b["I1"], b["It"], b["t"])["total"]
    opt.zero_grad()
    before.backward()
    opt.step()
    after = superslomo_loss(model, b["I0"], b["I1"], b["It"], b["t"])["total"]
    assert after.item() < before.item()


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        SuperSloMo(width="huge")
    with pytest.raises(ValueError):
        SuperSloMo(warp_mode="nearest")
