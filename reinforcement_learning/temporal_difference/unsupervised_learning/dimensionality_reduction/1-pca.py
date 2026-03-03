#!/usr/bin/env python3
"""
Module that performs PCA on a dataset to a specified number of dimensions
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset reducing to ndim dimensions

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the dataset
    ndim: new dimensionality of the transformed X

    Returns:
    T: numpy.ndarray of shape (n, ndim) containing the transformed X
    """
    # Mean-center the data
    X_m = X - np.mean(X, axis=0)

    # Compute SVD
    U, S, Vt = np.linalg.svd(X_m)

    # Take the first ndim principal components
    W = Vt[:ndim].T

    # Project data onto new dimensions
    T = np.matmul(X_m, W)

    return T
