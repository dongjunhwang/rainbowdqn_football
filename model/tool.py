import torch
import numpy as np

from torch import nn

def init_lecun_normal(tensor, scale=1.0):
    """Initializes the tensor with LeCunNormal."""
    fan_in = torch.nn.init._calculate_correct_fan(tensor, "fan_in")
    std = scale * np.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)

@torch.no_grad()
def init_chainer_default(layer):
    """Initializes the layer with the chainer default.
    weights with LeCunNormal(scale=1.0) and zeros as biases
    """
    assert isinstance(layer, nn.Module)

    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        init_lecun_normal(layer.weight)
        if layer.bias is not None:
            # layer may be initialized with bias=False
            nn.init.zeros_(layer.bias)
    return layer

def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias