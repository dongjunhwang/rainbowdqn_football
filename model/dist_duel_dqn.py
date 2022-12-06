import torch

from torch import nn
from torch.nn import functional as F
from abc import ABCMeta, abstractmethod
from .tool import init_chainer_default
from .tool import constant_bias_initializer
from .action_value import DistributionalDiscreteActionValue

__all__ = ['DistributionalDuelingDQN']

class StateQFunction(object, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError()

class DistributionalDuelingDQN(nn.Module, StateQFunction):
    def __init__(
        self,
        n_actions,
        n_atoms,
        v_min,
        v_max,
        n_input_channels=4,
        activation=torch.relu,
        bias=0.1,
    ):
        assert n_atoms >= 2
        assert v_min < v_max

        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms

        super().__init__()
        self.z_values = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        self.main_stream = nn.Linear(3136, 1024)
        self.a_stream = nn.Linear(512, n_actions * n_atoms)
        self.v_stream = nn.Linear(512, n_atoms)

        self.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for layer in self.conv_layers:
            h = self.activation(layer(h))

        # Advantage
        batch_size = x.shape[0]

        h = self.activation(self.main_stream(h.view(batch_size, -1)))
        h_a, h_v = torch.chunk(h, 2, dim=1)
        ya = self.a_stream(h_a).reshape((batch_size, self.n_actions, self.n_atoms))

        mean = ya.sum(dim=1, keepdim=True) / self.n_actions

        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h_v).reshape((batch_size, 1, self.n_atoms))
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)

        self.z_values = self.z_values.to(x.device)
        return DistributionalDiscreteActionValue(q, self.z_values)