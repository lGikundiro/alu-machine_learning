#!/usr/bin/env python3
"""
Module that performs the Baum-Welch algorithm using a hidden markov model
"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm using a hidden markov model

    Parameters:
    Observations: numpy.ndarray of shape (T,) containing observation indices
    Transition: numpy.ndarray of shape (M, M) containing transition probs
    Emission: numpy.ndarray of shape (M, N) containing emission probs
    Initial: numpy.ndarray of shape (M, 1) containing initial state probs
    iterations: number of times expectation-maximization should be performed

    Returns:
    Transition, Emission (converged), or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or \
            len(Observations.shape) != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or \
            len(Transition.shape) != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    M = Transition.shape[0]
    if Transition.shape[1] != M:
        return None, None
    if Emission.shape[0] != M:
        return None, None
    if Initial.shape[0] != M or Initial.shape[1] != 1:
        return None, None

    T = Observations.shape[0]
    N = Emission.shape[1]

    for _ in range(iterations):
        # Forward pass
        F = np.zeros((M, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            F[:, t] = np.matmul(F[:, t - 1], Transition) * Emission[
                :, Observations[t]
            ]

        # Backward pass
        B = np.zeros((M, T))
        B[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            B[:, t] = np.matmul(
                Transition, B[:, t + 1] * Emission[:, Observations[t + 1]]
            )

        # Compute likelihood
        P = np.sum(F[:, T - 1])

        # Compute gamma: state occupancy probabilities (M, T)
        gamma = (F * B) / P

        # Compute xi: transition occupancy probabilities (M, M, T-1)
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            num = (F[:, t:t + 1] * Transition *
                   Emission[:, Observations[t + 1]] * B[:, t + 1])
            xi[:, :, t] = num / P

        # Update Transition
        Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1,
                                                  keepdims=True)

        # Update Emission
        new_Emission = np.zeros((M, N))
        for k in range(N):
            mask = Observations == k
            new_Emission[:, k] = np.sum(gamma[:, mask], axis=1)
        Emission = new_Emission / np.sum(gamma, axis=1, keepdims=True)

    return Transition, Emission
