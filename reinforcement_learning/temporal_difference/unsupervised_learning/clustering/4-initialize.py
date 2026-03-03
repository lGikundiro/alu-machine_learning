#!/usr/bin/env python3
"""
Module that initializes variables using Gaussian Mixture Model
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables using Gaussian Mixture Model

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the data set
    k: positive integer containing the number of clusters

    Returns:
    pi: numpy.ndarray of shape (k,) containing priors each cluster
    m: numpy.ndarray of shape (k, d) containing centroid means each cluster
    S: numpy.ndarray of shape (k, d, d) containing covariance matrices
    or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    
    n, d = X.shape
    
    if k > n:
        return None, None, None
    
    # Initialize priors evenly
    pi = np.full(k, 1 / k)
    
    # Initialize means using K-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    
    # Initialize covariance matrices as identity matrices
    S = np.tile(np.eye(d), (k, 1, 1))
    
    return pi, m, S
