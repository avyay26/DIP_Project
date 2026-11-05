import pytest
import torch

from superslomo.model import SuperSloMo


@pytest.fixture
def tiny_model() -> SuperSloMo:
    torch.manual_seed(0)
    return SuperSloMo(width="tiny").eval()


def zero_heads(model: SuperSloMo) -> SuperSloMo:
    """Zero the output conv of both U-Nets: flows = 0, flow residuals = 0, V_t0 = 0.5."""
    for net in (model.flow_net, model.refine_net):
        torch.nn.init.zeros_(net.conv3.weight)
        torch.nn.init.zeros_(net.conv3.bias)
    return model
