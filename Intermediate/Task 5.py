# Task 5: Implementing K-Means Clustering

# Goals:
# Determine optimal clusters using the elbow method
# Apply K-Means clustering
# Visualize the clusters

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Visualize the points (unlabeled)
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], s=30)
plt.title("Synthetic Blobs (Unlabeled Dataset)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()