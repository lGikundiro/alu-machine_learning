#!/usr/bin/env python3
"""
Module that performs EM algorithm using GMM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization using GMM

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the data set
    k: positive integer containing the number of clusters
    iterations: positive integer containing maximum iterations
    tol: non-negative float containing tolerance of log likelihood
    verbose: boolean determining if info should be printed

    Returns:
    pi: numpy.ndarray of shape (k,) containing priors each cluster
    m: numpy.ndarray of shape (k, d) containing centroid means
    S: numpy.ndarray of shape (k, d, d) containing covariance matrices
    g: numpy.ndarray of shape (k, n) containing probabilities each point
    l: log likelihood of the model
    or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None
    
    # Initialize parameters
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    
    prev_l = 0
    
    for i in range(iterations):
        # E-step
        g, l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        
        # Print if verbose
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {l:.5f}")
        
        # Check convergence
        if i > 0 and abs(l - prev_l) <= tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {l:.5f}")
            break
        
        prev_l = l
        
        # M-step
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None
    else:
        # Loop completed without break - run final E-step and print
        g, l = expectation(X, pi, m, S)
        if verbose:
            print(f"Log Likelihood after {iterations} iterations: {l:.5f}")
    
    return pi, m, S, g, l
