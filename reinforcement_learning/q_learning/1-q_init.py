#!/usr/bin/env python3
"""Module to initialize the Q-table."""
import numpy as np


def q_init(env):
    """Initialize the Q-table with zeros.

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        the Q-table as a numpy.ndarray of zeros
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
