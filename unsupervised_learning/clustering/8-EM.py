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
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    n, d = X.shape
    if n == 0 or k > n:
        return None, None, None, None, None

    # Initialize parameters
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    g, log_l = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(log_l))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        g, l_new = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        printed = False
        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {:.5f}"
                  .format(i, l_new))
            printed = True

        if abs(l_new - log_l) <= tol:
            log_l = l_new
            if verbose and not printed:
                print("Log Likelihood after {} iterations: {:.5f}"
                      .format(i, log_l))
            break

        log_l = l_new

    return pi, m, S, g, log_l
