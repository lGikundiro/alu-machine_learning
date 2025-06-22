#!/usr/bin/env python3
"""Demonstrate numpy slicing along specifix axes"""


def np_slice(matrix, axes={}):
    """Slice matrix along specified axes"""
    return matrix[
        tuple([slice(*axes.get(ax, (None, None)))
               for ax in range(max(axes)+1)])]
