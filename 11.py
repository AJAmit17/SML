import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Step 1: Generate data with 3 clusters
X, y_true = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=[1.0, 1.5, 0.5], random_state=42)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title("Original Data")
plt.show()

# Step 2: Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Step 3: Show parameters
print("Means:\n", gmm.means_)
print("Covariances:\n", gmm.covariances_)
print("Weights:\n", gmm.weights_)

# Step 4: Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap='viridis')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100, label="Means")
plt.title("GMM Clustering")
plt.legend()
plt.show()
