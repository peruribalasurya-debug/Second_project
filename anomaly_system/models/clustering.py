from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


@dataclass(frozen=True)
class KMeansConfig:
    n_clusters: int
    random_state: int


def fit_kmeans(X: np.ndarray, cfg: KMeansConfig) -> KMeans:
    km = KMeans(
        n_clusters=cfg.n_clusters,
        n_init="auto",
        random_state=cfg.random_state,
    )
    km.fit(X)
    return km


def kmeans_distance_score(model: KMeans, X: np.ndarray) -> np.ndarray:
    centers = model.cluster_centers_
    d = pairwise_distances(X, centers, metric="euclidean")
    s = np.min(d, axis=1)
    return s.astype(np.float64)

