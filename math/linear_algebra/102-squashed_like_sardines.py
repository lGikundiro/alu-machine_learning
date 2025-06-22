#!/usr/bin/env python3

def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1: First matrix.
        mat2: Second matrix.
        axis: Axis along which the matrices should be concatenated.

    Returns:
        A new concatenated matrix if valid, otherwise None.
    """
    if axis < 0:
        return None

    # Check if dimensions are compatible for concatenation
    try:
        if len(mat1) == 0 or len(mat2) == 0:
            return None

        # Recursive concatenation if depth > 1
        if isinstance(mat1[0], list) and isinstance(mat2[0], list):
            if len(mat1[0]) != len(mat2[0]):
                return None

            return [cat_matrices(row1, row2, axis - 1) for row1, row2 in zip(mat1, mat2)]

        # Concatenate at the base level
        return mat1 + mat2 if axis == 0 else None

    except IndexError:
        return None

# Example usage:
if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    result = cat_matrices(mat1, mat2, axis=0)
    print("Concatenated along axis 0:", result)

    result = cat_matrices(mat1, mat2, axis=1)
    print("Concatenated along axis 1:", result)

