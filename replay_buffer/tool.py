import numpy as np
import itertools
import collections
import pickle

from typing import Optional
from abc import ABCMeta, abstractmethod

def sample_n_k(n, k):
    """Sample k distinct elements uniformly from range(n)"""

    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")
    if k == 0:
        return np.empty((0,), dtype=np.int64)
    elif 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        result = np.random.choice(n, 2 * k)
        selected = set()
        selected_add = selected.add
        j = k
        for i in range(k):
            x = result[i]
            while x in selected:
                x = result[i] = result[j]
                j += 1
                if j == 2 * k:
                    # This is slow, but it rarely happens.
                    result[k:] = np.random.choice(n, k)
                    j = k
            selected_add(x)
        return result[:k]


class RandomAccessQueue(object):
    """FIFO queue with fast indexing

    Operations getitem, setitem, append, popleft, and len
    are amortized O(1)-time, if this data structure is used ephemerally.
    """

    def __init__(self, *args, **kwargs):
        self.maxlen = kwargs.pop("maxlen", None)
        assert self.maxlen is None or self.maxlen >= 0
        self._queue_front = []
        self._queue_back = list(*args, **kwargs)
        self._apply_maxlen()

    def _apply_maxlen(self):
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popleft()

    def __iter__(self):
        return itertools.chain(reversed(self._queue_front), iter(self._queue_back))

    def __repr__(self):
        return "RandomAccessQueue({})".format(str(list(iter(self))))

    def __len__(self):
        return len(self._queue_front) + len(self._queue_back)

    def __getitem__(self, i):
        if i >= 0:
            nf = len(self._queue_front)
            if i < nf:
                return self._queue_front[~i]
            else:
                i -= nf
                if i < len(self._queue_back):
                    return self._queue_back[i]
                else:
                    raise IndexError("RandomAccessQueue index out of range")

        else:
            nb = len(self._queue_back)
            if i >= -nb:
                return self._queue_back[i]
            else:
                i += nb
                if i >= -len(self._queue_front):
                    return self._queue_front[~i]
                else:
                    raise IndexError("RandomAccessQueue index out of range")

    def __setitem__(self, i, x):
        if i >= 0:
            nf = len(self._queue_front)
            if i < nf:
                self._queue_front[~i] = x
            else:
                i -= nf
                if i < len(self._queue_back):
                    self._queue_back[i] = x
                else:
                    raise IndexError("RandomAccessQueue index out of range")

        else:
            nb = len(self._queue_back)
            if i >= -nb:
                self._queue_back[i] = x
            else:
                i += nb
                if i >= -len(self._queue_front):
                    self._queue_front[~i] = x
                else:
                    raise IndexError("RandomAccessQueue index out of range")

    def append(self, x):
        self._queue_back.append(x)
        if self.maxlen is not None and len(self) > self.maxlen:
            self.popleft()

    def extend(self, xs):
        self._queue_back.extend(xs)
        self._apply_maxlen()

    def popleft(self):
        if not self._queue_front:
            if not self._queue_back:
                raise IndexError("pop from empty RandomAccessQueue")

            self._queue_front = self._queue_back
            self._queue_back = []
            self._queue_front.reverse()

        return self._queue_front.pop()

    def sample(self, k):
        return [self[i] for i in sample_n_k(len(self), k)]


class AbstractReplayBuffer(object, metaclass=ABCMeta):
    """Defines a common interface of replay buffer.

    You can append transitions to the replay buffer and later sample from it.
    Replay buffers are typically used in experience replay.
    """

    @abstractmethod
    def append(
        self,
        state,
        action,
        reward,
        next_state=None,
        next_action=None,
        is_state_terminal=False,
        env_id=0,
        **kwargs
    ):
        """Append a transition to this replay buffer.

        Args:
            state: s_t
            action: a_t
            reward: r_t
            next_state: s_{t+1} (can be None if terminal)
            next_action: a_{t+1} (can be None for off-policy algorithms)
            is_state_terminal (bool)
            env_id (object): Object that is unique to each env. It indicates
                which env a given transition came from in multi-env training.
            **kwargs: Any other information to store.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n):
        """Sample n unique transitions from this replay buffer.

        Args:
            n (int): Number of transitions to sample.
        Returns:
            Sequence of n sampled transitions.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Return the number of transitions in the buffer.

        Returns:
            Number of transitions in the buffer.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filename):
        """Save the content of the buffer to a file.

        Args:
            filename (str): Path to a file.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, filename):
        """Load the content of the buffer from a file.

        Args:
            filename (str): Path to a file.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def capacity(self) -> Optional[int]:
        """Returns the capacity of the buffer in number of transitions.

        If unbounded, returns None instead.
        """
        raise NotImplementedError

    @abstractmethod
    def stop_current_episode(self, env_id=0):
        """Notify the buffer that the current episode is interrupted.

        You may want to interrupt the current episode and start a new one
        before observing a terminal state. This is typical in continuing envs.
        In such cases, you need to call this method before appending a new
        transition so that the buffer will treat it as an initial transition of
        a new episode.

        This method should not be called after an episode whose termination is
        already notified by appending a transition with is_state_terminal=True.

        Args:
            env_id (object): Object that is unique to each env. It indicates
                which env's current episode is interrupted in multi-env
                training.
        """
        raise NotImplementedError



class ReplayBuffer(AbstractReplayBuffer):
    """Experience Replay Buffer

    As described in
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf.

    Args:
        capacity (int): capacity in terms of number of transitions
        num_steps (int): Number of timesteps per stored transition
            (for N-step updates)
    """

    # Implements AbstractReplayBuffer.capacity
    capacity: Optional[int] = None

    def __init__(self, capacity: Optional[int] = None, num_steps: int = 1):
        self.capacity = capacity
        assert num_steps > 0
        self.num_steps = num_steps
        self.memory = RandomAccessQueue(maxlen=capacity)
        self.last_n_transitions: collections.defaultdict = collections.defaultdict(
            lambda: collections.deque([], maxlen=num_steps)
        )

    def append(
        self,
        state,
        action,
        reward,
        next_state=None,
        next_action=None,
        is_state_terminal=False,
        env_id=0,
        **kwargs
    ):
        last_n_transitions = self.last_n_transitions[env_id]
        experience = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            **kwargs
        )
        last_n_transitions.append(experience)
        if is_state_terminal:
            while last_n_transitions:
                self.memory.append(list(last_n_transitions))
                del last_n_transitions[0]
            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                self.memory.append(list(last_n_transitions))

    def stop_current_episode(self, env_id=0):
        last_n_transitions = self.last_n_transitions[env_id]
        # if n-step transition hist is not full, add transition;
        # if n-step hist is indeed full, transition has already been added;
        if 0 < len(last_n_transitions) < self.num_steps:
            self.memory.append(list(last_n_transitions))
        # avoid duplicate entry
        if 0 < len(last_n_transitions) <= self.num_steps:
            del last_n_transitions[0]
        while last_n_transitions:
            self.memory.append(list(last_n_transitions))
            del last_n_transitions[0]
        assert len(last_n_transitions) == 0

    def sample(self, num_experiences):
        assert len(self.memory) >= num_experiences
        return self.memory.sample(num_experiences)

    def __len__(self):
        return len(self.memory)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.memory = pickle.load(f)
        if isinstance(self.memory, collections.deque):
            # Load v0.2
            self.memory = RandomAccessQueue(self.memory, maxlen=self.memory.maxlen)