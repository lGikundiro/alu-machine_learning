#!/usr/bin/env python3
"""
Module that performs the forward algorithm a hidden markov model
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm using a hidden markov model

    Parameters:
    Observation: numpy.ndarray of shape (T,) containing observation indices
    Emission: numpy.ndarray of shape (N, M) containing emission probabilities
    Transition: numpy.ndarray of shape (N, N) containing transition probs
    Initial: numpy.ndarray of shape (N, 1) containing initial state probs

    Returns:
    P: likelihood of the observations given the model
    F: numpy.ndarray of shape (N, T) containing forward path probabilities
    or None, None on failure
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    F = np.zeros((N, T))

    # Initialize first column
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Iterate through observations
    for t in range(1, T):
        F[:, t] = np.matmul(F[:, t - 1], Transition) * Emission[
            :, Observation[t]
        ]

    P = np.sum(F[:, T - 1])

    return P, F
