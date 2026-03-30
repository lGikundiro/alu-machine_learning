#!/usr/bin/env python3
"""
Module that calculates expectation step in EM algorithm using GMM
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm using GMM

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the data set
    pi: numpy.ndarray of shape (k,) containing priors each cluster
    m: numpy.ndarray of shape (k, d) containing centroid means each cluster
    S: numpy.ndarray of shape (k, d, d) containing covariance matrices

    Returns:
    g: numpy.ndarray of shape (k, n) containing posterior probabilities
    l: total log likelihood
    or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    
    n, d = X.shape
    k = pi.shape[0]
    
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None
    
    # Calculate weighted pdf each cluster
    likelihood = np.zeros((k, n))
    
    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        likelihood[i] = pi[i] * P
    
    # Calculate total likelihood each point (sum over clusters)
    total_likelihood = np.sum(likelihood, axis=0)
    
    # Calculate log likelihood
    l = np.sum(np.log(total_likelihood))
    
    # Calculate posterior probabilities (normalize)
    g = likelihood / total_likelihood
    
    return g, l
