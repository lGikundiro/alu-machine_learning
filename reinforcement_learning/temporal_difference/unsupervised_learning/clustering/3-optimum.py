#!/usr/bin/env python3
"""
Module that tests optimum number of clusters by variance
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests optimum number of clusters by variance

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the data set
    kmin: positive integer containing minimum number of clusters (inclusive)
    kmax: positive integer containing maximum number of clusters (inclusive)
    iterations: positive integer containing maximum iterations K-means

    Returns:
    results: list containing outputs of K-means each cluster size
    d_vars: list containing difference in variance from smallest cluster size
    or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    
    n, d = X.shape
    
    if kmax is None:
        kmax = n
    
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if kmax > n:
        return None, None
    
    results = []
    variances = []
    
    # Loop through different cluster sizes
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))
        var = variance(X, C)
        variances.append(var)
    
    # Calculate differences from first variance
    first_var = variances[0]
    d_vars = [first_var - var for var in variances]
    
    return results, d_vars
