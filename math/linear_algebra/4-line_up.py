#!/usr/bin/env python3
"""Function to add two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise

    Args:
        arr1: First array (list of ints/floats)
        arr2: Second array (list of ints/floats)

    Returns:
        A new list containing the sum of the arrays element-wise,
        or None if arr1 and arr2 are not the same shape
    """
    # Check if arrays have the same shape
    if len(arr1) != len(arr2):
        return None

    # Add arrays element-wise
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
