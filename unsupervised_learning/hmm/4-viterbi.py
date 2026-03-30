#!/usr/bin/env python3
"""
Module that calculates the most likely sequence of hidden states using Viterbi
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states using Viterbi

    Parameters:
    Observation: numpy.ndarray of shape (T,) containing observation indices
    Emission: numpy.ndarray of shape (N, M) containing emission probabilities
    Transition: numpy.ndarray of shape (N, N) containing transition probs
    Initial: numpy.ndarray of shape (N, 1) containing initial state probs

    Returns:
    path: list of length T containing most likely sequence of hidden states
    P: probability of obtaining the path sequence
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

    # Viterbi probability matrix and backpointer matrix
    V = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Initialize first column
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Fill in the rest
    for t in range(1, T):
        # For each state, compute max over previous states
        trans_prob = V[:, t - 1:t] * Transition  # shape (N, N)
        backpointer[:, t] = np.argmax(trans_prob, axis=0)
        V[:, t] = np.max(trans_prob, axis=0) * Emission[:, Observation[t]]

    # Backtrack to find most likely path
    path = [0] * T
    path[T - 1] = int(np.argmax(V[:, T - 1]))
    P = np.max(V[:, T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]

    return path, P
