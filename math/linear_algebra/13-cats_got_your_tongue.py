#!/usr/bin/env python3
"""Function to concatenate two numpy.ndarrays along a specific axis"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis

    Args:
        mat1: First numpy.ndarray
        mat2: Second numpy.ndarray
        axis: Axis along which to concatenate (default=0)

    Returns:
        A new numpy.ndarray containing the concatenated matrices
    """
    return np.concatenate((mat1, mat2), axis=axis)
