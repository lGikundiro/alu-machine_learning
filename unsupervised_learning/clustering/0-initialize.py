#!/usr/bin/env python3
"""
Module that initializes cluster centroids using K-means
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids using K-means

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the dataset
       n is the number of data points
       d is the number of dimensions in each data point
    k: positive integer containing the number of clusters

    Returns:
    numpy.ndarray of shape (k, d) containing the initialized centroids
    of each cluster, or None on failure
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(k) is not int or k <= 0:
        return None

    n, d = X.shape

    if n == 0 or k > n:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    return centroids
