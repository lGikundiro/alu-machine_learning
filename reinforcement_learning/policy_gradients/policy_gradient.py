#!/usr/bin/env python3
"""
Module implementing policy gradient functions using softmax policy
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight matrix using softmax

    Parameters:
    matrix: state matrix (1, s)
    weight: weight matrix (s, a)

    Returns:
    softmax probabilities over actions
    """
    z = np.dot(matrix, weight)
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient

    Parameters:
    state: matrix representing the current observation (1, s)
    weight: matrix of random weights (s, a)

    Returns:
    action: the chosen action
    gradient: the policy gradient
    """
    # Get action probabilities
    probs = policy(state, weight)

    # Sample action from the probability distribution
    action = np.random.choice(probs.shape[1], p=probs[0])

    # Compute gradient: d log pi(a|s) / d weight
    # = state.T * (one_hot(a) - probs)
    d_softmax = -probs.copy()
    d_softmax[0, action] += 1

    gradient = np.dot(state.T, d_softmax)

    return action, gradient
