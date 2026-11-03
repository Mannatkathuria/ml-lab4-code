
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ic272_lab9_customers.csv")
X = df[["x_km", "y_km"]].to_numpy(float)

print(X.shape)


def random_init(X, K, seed):
    """K distinct data points, chosen at random, as starting centroids."""
    idx = np.random.default_rng(seed).choice(len(X), K, replace=False)
    return X[idx]



def assign_clusters(X, centroids):
    distances = np.sqrt(
        np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    )
    return np.argmin(distances, axis=1)



def update_centroids(X, labels, centroids):
    new_centroids = centroids.copy()

    for k in range(len(centroids)):
        points = X[labels == k]

        if len(points) > 0:
            new_centroids[k] = np.mean(points, axis=0)

    return new_centroids



def distortion(X, labels, centroids):
    return np.sum((X - centroids[labels]) ** 2)


# 1(d) Complete K-means algorithm
def kmeans(X, init_centroids, max_iter=100):
    centroids = init_centroids.copy()

    for iteration in range(1, max_iter + 1):
        labels = assign_clusters(X, centroids)

        new_centroids = update_centroids(X, labels, centroids)

        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            labels = assign_clusters(X, centroids)
            break

        centroids = new_centroids

    J = distortion(X, labels, centroids)

    return centroids, labels, J, iteration



init_centroids = random_init(X, 6, 0)

centroids, labels, J, iterations = kmeans(X, init_centroids)

print("Final Centroids:\n", centroids)
print("Distortion:", J)
print("Iterations:", iterations)


fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=14)
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    color="red",
    s=150,
    label="Centroids"
)

ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_title("K-means Clustering (K=6)")
ax.set_aspect("equal")
ax.legend()

plt.show()
