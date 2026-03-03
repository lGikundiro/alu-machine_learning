#!/usr/bin/env python3
"""
Module that determines if a markov chain is absorbing
"""

import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing

    Parameters:
    P: square 2D numpy.ndarray of shape (n, n) representing transition matrix

    Returns:
    True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n = P.shape[0]
    if n != P.shape[1]:
        return False
    if not np.isclose(P.sum(axis=1), np.ones(n)).all():
        return False

    # Find absorbing states (diagonal entry == 1)
    absorbing_states = np.where(np.isclose(np.diag(P), 1))[0]

    if len(absorbing_states) == 0:
        return False

    # If all states are absorbing
    if len(absorbing_states) == n:
        return True

    # Check if every non-absorbing state can reach an absorbing state
    # Use iterative power to see if all rows eventually have positive
    # weight on absorbing states
    reachable = set(absorbing_states)
    prev_size = 0

    while len(reachable) != prev_size:
        prev_size = len(reachable)
        for i in range(n):
            if i in reachable:
                continue
            for j in reachable:
                if P[i, j] > 0:
                    reachable.add(i)
                    break

    return len(reachable) == n
