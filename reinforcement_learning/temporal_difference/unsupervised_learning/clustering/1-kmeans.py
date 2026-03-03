#!/usr/bin/env python3
"""
Module that performs K-means clustering on a dataset
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset

    Parameters:
    X: numpy.ndarray of shape (n, d) containing the dataset
       n is the number of data points
       d is the number of dimensions in each data point
    k: positive integer containing the number of clusters
    iterations: positive integer containing the maximum number of iterations

    Returns:
    C: numpy.ndarray of shape (k, d) containing centroid means
    clss: numpy.ndarray of shape (n,) containing cluster index of each point
    or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    
    n, d = X.shape
    
    if k > n:
        return None, None
    
    # Initialize centroids using multivariate uniform distribution
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))
    
    # Main K-means loop
    for _ in range(iterations):
        # Store previous centroids to check convergence
        C_prev = np.copy(C)
        
        # Assign each point to nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)
        
        # Update centroids
        for i in range(k):
            # Get points assigned to cluster i
            cluster_points = X[clss == i]
            
            if len(cluster_points) == 0:
                # Reinitialize empty cluster
                C[i] = np.random.uniform(low=min_vals, high=max_vals, size=(d,))
            else:
                # Update centroid as mean of assigned points
                C[i] = np.mean(cluster_points, axis=0)
        
        # Check convergence
        if np.allclose(C, C_prev):
            break
    
    # Final assignment
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    
    return C, clss
