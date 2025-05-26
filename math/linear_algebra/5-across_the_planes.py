#!/usr/bin/env python3
"""Function to add two 2D matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise

    Args:
        mat1: First 2D matrix (list of lists of ints/floats)
        mat2: Second 2D matrix (list of lists of ints/floats)

    Returns:
        A new matrix containing the sum of the matrices element-wise,
        or None if mat1 and mat2 are not the same shape
    """
    # Check if matrices have the same shape
    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        # Check if rows have the same length
        if len(mat1[i]) != len(mat2[i]):
            return None

        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
