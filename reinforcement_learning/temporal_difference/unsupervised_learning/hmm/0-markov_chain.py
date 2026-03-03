#!/usr/bin/env python3
"""
Module that determines probability of a markov chain state after t iterations
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular state
    after a specified number of iterations

    Parameters:
    P: square 2D numpy.ndarray of shape (n, n) representing transition matrix
    s: numpy.ndarray of shape (1, n) representing probability of starting state
    t: number of iterations the markov chain has been through

    Returns:
    numpy.ndarray of shape (1, n) representing the probability of being in
    a specific state after t iterations, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t < 1:
        return None

    result = s
    for _ in range(t):
        result = np.matmul(result, P)

    return result
