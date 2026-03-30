#!/usr/bin/env python3
"""Module to implement epsilon-greedy action selection."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Use epsilon-greedy to determine the next action.

    Args:
        Q: a numpy.ndarray containing the q-table
        state: the current state
        epsilon: the epsilon to use

    Returns:
        the next action index
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, Q.shape[1])
    return action
