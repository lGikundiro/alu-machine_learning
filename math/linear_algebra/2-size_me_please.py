#!/usr/bin/env python3
"""Function to calculate the shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix

    Args:
        matrix: the matrix to calculate the shape of

    Returns:
        A list of integers representing the dimensions of the matrix
    """
    shape = []
    current = matrix

    while isinstance(current, list):
        shape.append(len(current))
        if len(current) == 0:
            break
        current = current[0]

    return shape
