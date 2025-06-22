#!/usr/bin/env python3
"""Transpose a list-based matrix"""


def matrix_transpose(matrix):
    """Perform matrix transpose"""
    nrows = len(matrix)
    ncols = len(matrix[0])
    transpose = [[0 for _ in range(nrows)] for _ in range(ncols)]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            transpose[j][i] = matrix[i][j]
    return transpose
