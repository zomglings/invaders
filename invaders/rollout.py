"""
invaders rollouts
"""
import torch
from torch.distributions import Categorical

from .env import INPUT_DIMENSIONS

def rollout(model, env, render=True):
    raw_state = env.reset()
    done = False
    rewards = []
    log_probabilities = []
    while not done:
        # All rollouts are assumed to take place on CPUs. If we want to be able to make use of
        # GPUs to generate rollouts, we will need to add code here to pin state to GPUs if a
        # (new) GPU argument is set to True when rollout is called.
        state = torch.from_numpy(raw_state).float().reshape((-1, INPUT_DIMENSIONS)).cuda()
        action_probabilities = model(state)
        policy = Categorical(action_probabilities)
        action = policy.sample()
        log_probabilities.append(policy.log_prob(action))
        move = action.item()
        raw_state, reward, done, _ = env.step(move)
        rewards.append(reward)
        if render:
            env.render()

    return rewards, log_probabilities
