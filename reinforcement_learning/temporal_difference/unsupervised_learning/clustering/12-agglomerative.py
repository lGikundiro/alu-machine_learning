#!/usr/bin/env python3
"""
Module that performs agglomerative clustering
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the dataset
    dist: maximum cophenetic distance all clusters

    Returns:
    clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    # Perform hierarchical/agglomerative clustering using Ward linkage
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, method='ward')
    
    # Create dendrogram
    scipy.cluster.hierarchy.dendrogram(linkage_matrix, color_threshold=dist)
    plt.show()
    
    # Cut the dendrogram at the specified distance
    clss = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, t=dist, criterion='distance'
    )
    
    return clss
