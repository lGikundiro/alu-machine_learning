#!/usr/bin/env python3
"""
Module that finds best number of clusters using GMM with BIC
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters using GMM with BIC

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the data set
    kmin: positive integer containing minimum number of clusters (inclusive)
    kmax: positive integer containing maximum number of clusters (inclusive)
    iterations: positive integer containing maximum iterations EM algorithm
    tol: non-negative float containing tolerance EM algorithm
    verbose: boolean determining if EM should print info

    Returns:
    best_k: best value k based on BIC
    best_result: tuple containing pi, m, S best number of clusters
    l: numpy.ndarray of shape (kmax - kmin + 1) containing log likelihood
    b: numpy.ndarray of shape (kmax - kmin + 1) containing BIC value
    or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    
    n, d = X.shape
    
    if kmax is None:
        kmax = n
    
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    
    num_tests = kmax - kmin + 1
    l = np.zeros(num_tests)
    b = np.zeros(num_tests)
    results = []
    
    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        
        if pi is None:
            return None, None, None, None
        
        results.append((pi, m, S))
        idx = k - kmin
        l[idx] = log_l
        
        # Calculate number of parameters
        # p = (k-1) + k*d + k*d*(d+1)/2
        p = (k - 1) + k * d + k * d * (d + 1) / 2
        
        # Calculate BIC: p * ln(n) - 2 * l
        b[idx] = p * np.log(n) - 2 * log_l
    
    # Find best k (minimum BIC)
    best_idx = np.argmin(b)
    best_k = best_idx + kmin
    best_result = results[best_idx]
    
    return best_k, best_result, l, b
