import torch

from abc import ABCMeta, abstractmethod, abstractproperty
from torch.distributions.utils import lazy_property
from torch.nn import functional as F

class ActionValue(object, metaclass=ABCMeta):
    """Struct that holds state-fixed Q-functions and its subproducts.

    Every operation it supports is done in a batch manner.
    """

    @abstractproperty
    def greedy_actions(self):
        """Get argmax_a Q(s,a)."""
        raise NotImplementedError()

    @abstractproperty
    def max(self):
        """Evaluate max Q(s,a)."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate_actions(self, actions):
        """Evaluate Q(s,a) with a = given actions."""
        raise NotImplementedError()

    @abstractproperty
    def params(self):
        """Learnable parameters of this action value.

        Returns:
            tuple of torch.Tensor
        """
        raise NotImplementedError()

    def __getitem__(self, i) -> "ActionValue":
        """ActionValue is expected to be indexable."""
        raise NotImplementedError()


class DistributionalDiscreteActionValue(ActionValue):
    """distributional Q-function output for discrete action space.

    Args:
        q_dist: Probabilities of atoms. Its shape must be
            (batchsize, n_actions, n_atoms).
        z_values (ndarray): Values represented by atoms.
            Its shape must be (n_atoms,).
    """

    def __init__(self, q_dist, z_values, q_values_formatter=lambda x: x):
        assert isinstance(q_dist, torch.Tensor)
        assert isinstance(z_values, torch.Tensor)
        assert q_dist.ndim == 3
        assert z_values.ndim == 1
        assert q_dist.shape[2] == z_values.shape[0]
        self.z_values = z_values
        q_scaled = q_dist * self.z_values[None, None, ...]
        self.q_values = q_scaled.sum(dim=2)
        self.q_dist = q_dist
        self.n_actions = q_dist.shape[1]
        self.q_values_formatter = q_values_formatter

    @lazy_property
    def greedy_actions(self):
        return self.q_values.argmax(dim=1).detach()

    @lazy_property
    def max(self):
        return torch.gather(self.q_values, 1, self.greedy_actions[:, None])[:, 0]

    @lazy_property
    def max_as_distribution(self):
        """Return the return distributions of the greedy actions.

        Returns:
            torch.Tensor: Return distributions. Its shape will be
                (batch_size, n_atoms).
        """
        return self.q_dist[
            torch.arange(self.q_values.shape[0]), self.greedy_actions.detach()
        ]

    def evaluate_actions(self, actions):
        return torch.gather(self.q_values, 1, actions[:, None])[:, 0]

    def evaluate_actions_as_distribution(self, actions):
        """Return the return distributions of given actions.

        Args:
            actions (torch.Tensor): Array of action indices.
                Its shape must be (batch_size,).

        Returns:
            torch.Tensor: Return distributions. Its shape will be
                (batch_size, n_atoms).
        """
        return self.q_dist[torch.arange(self.q_values.shape[0]), actions]

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return self.evaluate_actions(actions) - self.evaluate_actions(argmax_actions)

    def compute_expectation(self, beta):
        return (F.softmax(beta * self.q_values) * self.q_values).sum(dim=1)

    def __repr__(self):
        return "DistributionalDiscreteActionValue greedy_actions:{} q_values:{}".format(  # NOQA
            self.greedy_actions.detach(),
            self.q_values_formatter(self.q_values.detach()),
        )

    @property
    def params(self):
        return (self.q_dist,)

    def __getitem__(self, i):
        return DistributionalDiscreteActionValue(
            self.q_dist[i],
            self.z_values,
            q_values_formatter=self.q_values_formatter,
        )


