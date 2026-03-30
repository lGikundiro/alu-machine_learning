#!/usr/bin/env python3
"""
Module that determines steady state probabilities of a regular markov chain
"""

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain

    Parameters:
    P: square 2D numpy.ndarray of shape (n, n) representing transition matrix

    Returns:
    numpy.ndarray of shape (1, n) containing steady state probabilities,
    or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n = P.shape[0]
    if n != P.shape[1]:
        return None
    if not np.isclose(P.sum(axis=1), np.ones(n)).all():
        return None

    # Check if chain is regular: some power of P has all positive entries
    power = np.copy(P)
    for _ in range(n * n):
        if np.all(power > 0):
            break
        power = np.matmul(power, P)
    else:
        return None

    # Iterate to find steady state
    s = np.ones((1, n)) / n
    for _ in range(10000):
        s_next = np.matmul(s, P)
        if np.allclose(s, s_next):
            return s_next
        s = s_next

    return None
