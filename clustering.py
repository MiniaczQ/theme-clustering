import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans
from typing import List, Optional


def normalized_pca(embeddings: NDArray, n_dims: int) -> NDArray:
    reducer = PCA(n_components=n_dims)
    reduced = reducer.fit_transform(embeddings)
    (min, max) = np.min(reduced, axis=0), np.max(reduced, axis=0)
    reduced /= min
    reduced -= max - min
    return reduced


def distance(a, b):
    delta = b - a
    return np.dot(delta, delta) ** 0.5


def cluster_kmeans(embeddings: NDArray, lines: List[str], n: Optional[int]):
    cluster_centroids = kmeans(embeddings, k_or_guess=n)[0]
    clusters = list()
    for _ in range(n):
        clusters.append(list())

    for line, emb in zip(lines, embeddings):
        min_dist = 1e10
        maybe_centroid = 0
        for idx, centroid in enumerate(cluster_centroids):
            dist = distance(emb, centroid)
            if dist < min_dist:
                min_dist = dist
                maybe_centroid = idx
        clusters[maybe_centroid].append(line)

    return clusters
