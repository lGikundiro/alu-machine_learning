#!/usr/bin/env python3
"""
Module that performs PCA on a dataset maintaining a fraction of variance
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the dataset
       all dimensions have a mean of 0 across all data points
    var: fraction of the variance the PCA transformation should maintain

    Returns:
    W: numpy.ndarray of shape (d, nd) containing the weights matrix
       nd is the new dimensionality of the transformed X
    """
    # Compute SVD of X
    U, S, Vt = np.linalg.svd(X)

    # Calculate cumulative explained variance ratio
    variance_explained = np.cumsum(S ** 2) / np.sum(S ** 2)

    # Find number of dimensions needed to maintain var fraction of variance
    nd = np.argmax(variance_explained >= var) + 1

    # W is the first nd rows of Vt, transposed
    W = Vt[:nd].T

    return W
