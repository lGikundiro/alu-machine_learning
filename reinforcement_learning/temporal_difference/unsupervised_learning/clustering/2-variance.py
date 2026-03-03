#!/usr/bin/env python3
"""
Module that calculates total intra-cluster variance
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the data set
    C: numpy.ndarray of shape (k, d) containing centroid means

    Returns:
    var: total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    
    n, d = X.shape
    k, d_c = C.shape
    
    if d != d_c:
        return None
    
    # Calculate distances from each point to all centroids
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    
    # Get minimum distance to nearest centroid
    min_distances = np.min(distances, axis=1)
    
    # Calculate total variance as sum of squared distances
    var = np.sum(min_distances ** 2)
    
    return var
