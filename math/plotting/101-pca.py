#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Load data
lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

# Perform PCA
data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color mapping
scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
                     c=labels, cmap='plasma', s=50)

# Add labels and title
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
ax.set_title('PCA of Iris Dataset')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Iris Species')

# Show plot
plt.show()
