#!/usr/bin/env python3
"""Function to perform matrix multiplication"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication

    Args:
        mat1: First 2D matrix (list of lists of ints/floats)
        mat2: Second 2D matrix (list of lists of ints/floats)

    Returns:
        A new matrix resulting from the multiplication of mat1 by mat2,
        or None if the matrices cannot be multiplied
    """
    # Check if matrices can be multiplied: columns of mat1 = rows of mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Get dimensions of the resulting matrix
    rows = len(mat1)
    cols = len(mat2[0])

    # Initialize result matrix with zeros
    result = [[0 for _ in range(cols)] for _ in range(rows)]

    # Perform matrix multiplication
    for i in range(rows):
        for j in range(cols):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
