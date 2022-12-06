import gym
import numpy as np
import cv2

import gfootball.env as football_env

from gym import spaces
from .random_action import RandomizeAction

__all__ = ['make_env']


class TransEnv(gym.ObservationWrapper):
    def __init__(self, env, channel_order="hwc"):
        gym.ObservationWrapper.__init__(self, env)
        self.height = 84
        self.width = 84
        self.ch = env.observation_space.shape[2]
        shape = {
            "hwc": (self.height, self.width, self.ch),
            "chw": (self.ch, self.height, self.width),
        }
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape[channel_order], dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame.reshape(self.observation_space.low.shape)


def make_env(test_seed, train_seed, test=False):
    # Use different random seeds for train and test envs
    env_seed = test_seed if test else train_seed

    # env = gym.make('GFootball-11_vs_11_kaggle-SMM-v0')
    env = football_env.create_environment(
        env_name='11_vs_11_easy_stochastic',  # easy mode
        stacked=False,
        representation='extracted',  # SMM
        rewards='scoring,checkpoints',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
        dump_frequency=1,
        logdir='./',
        extra_players=None,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0
    )
    env = TransEnv(env, channel_order="chw")

    env.seed(int(env_seed))
    if test:
        env = RandomizeAction(env, random_fraction=0.0)
    return env