"""
invaders training script
"""
import argparse
from typing import Optional

import torch

from .env import get_environment
from .models import SingleFrameSimpleModel
from .rollout import rollout

def training_loop(model, optimizer, steps: int, discount_rate: float):
    for j in range(steps):
        env = get_environment()
        rewards, log_probabilities = rollout(model, env)
        retrospective_rewards = []
        current_reward = 0
        for step_reward in rewards[::-1]:
            current_reward = step_reward + discount_rate*current_reward
            retrospective_rewards.insert(0, current_reward)
        print(f'Final reward at step {j}: {sum(rewards)}')

        rollout_loss = []
        for log_probability, reward in zip(log_probabilities, retrospective_rewards):
            rollout_loss.append(-log_probability*reward)

        optimizer.zero_grad()
        rollout_loss = torch.cat(rollout_loss).sum().cuda()
        rollout_loss.backward()
        optimizer.step()

def main():
    model = SingleFrameSimpleModel(256).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    training_loop(model, optimizer, 3, 0.9)

if __name__ == '__main__':
    main()
