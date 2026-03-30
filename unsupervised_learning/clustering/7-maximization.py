#!/usr/bin/env python3
"""
Module that calculates maximization step in EM algorithm using GMM
"""

import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm using GMM

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the data set
    g: numpy.ndarray of shape (k, n) containing posterior probabilities

    Returns:
    pi: numpy.ndarray of shape (k,) containing updated priors each cluster
    m: numpy.ndarray of shape (k, d) containing updated centroid means
    S: numpy.ndarray of shape (k, d, d) containing updated covariance matrices
    or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    
    n, d = X.shape
    k, n_g = g.shape
    
    if n != n_g:
        return None, None, None
    
    # Calculate sum of posterior probabilities each cluster
    g_sum = np.sum(g, axis=1)  # shape (k,)
    
    # Update priors
    pi = g_sum / n
    
    # Update means: m[k] = sum(g[k, i] * X[i]) / sum(g[k, i])
    m = (g @ X) / g_sum[:, np.newaxis]  # shape (k, d)
    
    # Update covariance matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]  # shape (n, d)
        weighted_diff = g[i, :, np.newaxis] * diff  # shape (n, d)
        S[i] = (weighted_diff.T @ diff) / g_sum[i]
    
    return pi, m, S
