#!/usr/bin/env python3
"""Function to transpose a 2D matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix.

    Args:
        matrix: A 2D matrix to transpose

    Returns:
        A new matrix that is the transpose of the input matrix
    """
    # Get dimensions of the input matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Create a new matrix with swapped dimensions
    transpose = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(matrix[i][j])
        transpose.append(new_row)

    return transpose
