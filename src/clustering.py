from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from .utils import save_joblib


def fit_kmeans(X, n_clusters: int = 20, random_state: int = 42, save_name: str = "kmeans") -> Tuple[KMeans, np.ndarray]:
    # For sparse X, scikit-learn KMeans expects dense; use mini-batch alternative
    # We'll convert to MiniBatchKMeans for scalability
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=4096)
    km.fit(X)
    labels = km.labels_
    save_joblib(km, save_name)
    return km, labels

