#!/usr/bin/env python3
"""Function to perform element-wise operations on numpy.ndarrays"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication, and
    division

    Args:
        mat1: First numpy.ndarray
        mat2: Second numpy.ndarray or scalar

    Returns:
        A tuple containing the element-wise sum, difference, product, and
        quotient
    """
    addition = mat1 + mat2
    subtraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2

    return (addition, subtraction, multiplication, division)
