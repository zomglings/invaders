"""
invaders models
"""

import torch
import torch.nn as nn

from .env import DIM_ACTION_SPACE, INPUT_DIMENSIONS

def SingleFrameSimpleModel(latent_dimensions: int = 256):
    return nn.Sequential(
        nn.Linear(INPUT_DIMENSIONS, latent_dimensions*4),
        nn.ReLU(),
        nn.Linear(latent_dimensions*4, latent_dimensions*2),
        nn.ReLU(),
        nn.Linear(latent_dimensions*2, latent_dimensions),
        nn.ReLU(),
        nn.Linear(latent_dimensions, 10*DIM_ACTION_SPACE),
        nn.ReLU(),
        nn.Linear(10*DIM_ACTION_SPACE, DIM_ACTION_SPACE),
        nn.ReLU(),
        nn.Softmax(),
    )
