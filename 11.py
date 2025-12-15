import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Step 1: Generate synthetic 2D data with 3 clusters
X, y_true = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=[1.0, 1.5, 0.5], random_state=42)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50)
plt.title("Generated 2D Data with 3 Clusters")
plt.show()

# Step 2: Fit Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Step 3: Get estimated parameters
print("Estimated Means:\n", gmm.means_)
print("Estimated Covariances:\n", gmm.covariances_)
print("Estimated Weights:\n", gmm.weights_)

# Step 4: Compare with actual clusters (optional visual comparison)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100, label="Estimated Means")
plt.title("GMM Clustering with Estimated Means")
plt.legend()
plt.show()
