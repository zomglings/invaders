"""
invaders environment
"""
from typing import Optional

import gym
import numpy as np

DIM_ACTION_SPACE = 6
DIM_OBSERVATION_SPACE = (210, 160, 3)
INPUT_DIMENSIONS = (
    DIM_OBSERVATION_SPACE[0]*DIM_OBSERVATION_SPACE[1]*DIM_OBSERVATION_SPACE[2]
)

def get_environment(seed: Optional[int] = None):
    env = gym.make('SpaceInvaders-v0')
    assert repr(env.action_space) == f'Discrete({DIM_ACTION_SPACE})'
    assert env.observation_space.shape == DIM_OBSERVATION_SPACE
    env.seed(seed=42)
    return env

def random_action_probabilities(batch_size: int = 1):
    unnormalized = np.random.rand(batch_size, DIM_ACTION_SPACE)
    row_sums = unnormalized.sum(axis=1)
    normalized = (unnormalized.T/row_sums).T
    return normalized
