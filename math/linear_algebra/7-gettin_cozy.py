#!/usr/bin/env python3
"""Function to concatenate two matrices along a specified axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specified axis

    Args:
        mat1: First 2D matrix (list of lists of ints/floats)
        mat2: Second 2D matrix (list of lists of ints/floats)
        axis: Axis along which to concatenate (0 for rows, 1 for columns)

    Returns:
        A new matrix containing the concatenated matrices,
        or None if they cannot be concatenated
    """
    # Create deep copies to avoid modifying the original matrices
    result = [row[:] for row in mat1]

    # Concatenate along rows (axis=0)
    if axis == 0:
        # Check if column dimensions match
        if len(mat1[0]) != len(mat2[0]):
            return None

        # Concatenate by adding new rows
        for row in mat2:
            result.append(row[:])

    # Concatenate along columns (axis=1)
    elif axis == 1:
        # Check if row dimensions match
        if len(mat1) != len(mat2):
            return None

        # Concatenate by extending each row
        for i in range(len(result)):
            result[i].extend(mat2[i][:])

    else:
        return None

    return result
