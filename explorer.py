from abc import ABCMeta, abstractmethod

class Explorer(object, metaclass=ABCMeta):
    """Abstract explorer."""

    @abstractmethod
    def select_action(self, t, greedy_action_func, action_value=None):
        """Select an action.

        Args:
          t: current time step
          greedy_action_func: function with no argument that returns an action
          action_value (ActionValue): ActionValue object
        """
        raise NotImplementedError()


class Greedy(Explorer):
    """No exploration"""

    def select_action(self, t, greedy_action_func, action_value=None):
        return greedy_action_func()

    def __repr__(self):
        return "Greedy()"