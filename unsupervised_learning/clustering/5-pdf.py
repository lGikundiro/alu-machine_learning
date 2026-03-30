#!/usr/bin/env python3
"""
Module that calculates PDF of a Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    Parameters:
    X: numpy.ndarray of shape (n, d) containing data points
    m: numpy.ndarray of shape (d,) containing mean of the distribution
    S: numpy.ndarray of shape (d, d) containing covariance of distribution

    Returns:
    P: numpy.ndarray of shape (n,) containing PDF values each data point
    or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    
    n, d = X.shape
    
    if m.shape[0] != d:
        return None
    if S.shape[0] != d or S.shape[1] != d:
        return None
    
    # Calculate determinant
    det = np.linalg.det(S)
    if det == 0:
        return None
    
    # Calculate inverse of covariance matrix
    S_inv = np.linalg.inv(S)
    
    # Calculate normalization factor
    norm = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    
    # Calculate difference from mean
    diff = X - m
    
    # Calculate Mahalanobis distance: (X-m) @ S^(-1) @ (X-m)^T
    mahalanobis = np.sum(diff @ S_inv * diff, axis=1)
    
    # Calculate PDF
    P = norm * np.exp(-0.5 * mahalanobis)
    
    # Ensure minimum value of 1e-300
    P = np.maximum(P, 1e-300)
    
    return P
