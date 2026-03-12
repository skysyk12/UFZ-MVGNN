"""Clustering algorithms for embeddings."""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def hdbscan_clustering(
    embeddings: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int = 1,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
) -> np.ndarray:
    """HDBSCAN: density-based clustering."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

    return labels


def dbscan_clustering(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 15,
    metric: str = "euclidean",
) -> np.ndarray:
    """DBSCAN: grid-based clustering."""
    from sklearn.cluster import DBSCAN

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

    return labels


def kmeans_clustering(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """KMeans: centroid-based clustering."""
    from sklearn.cluster import KMeans

    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = clusterer.fit_predict(embeddings)

    logger.info(f"KMeans: {n_clusters} clusters")
    return labels


def leiden_clustering(
    embeddings: np.ndarray,
    resolution: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Leiden: community detection via modularity."""
    import scanpy as sc
    import anndata

    adata = anndata.AnnData(X=embeddings)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(100, embeddings.shape[1]))
    
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=resolution, random_state=seed)

    labels = adata.obs["leiden"].astype(int).values
    n_clusters = len(set(labels))
    logger.info(f"Leiden: {n_clusters} clusters (resolution={resolution})")

    return labels


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    **kwargs,
) -> np.ndarray:
    """Unified clustering interface."""
    methods = {
        "hdbscan": hdbscan_clustering,
        "dbscan": dbscan_clustering,
        "kmeans": kmeans_clustering,
        "leiden": leiden_clustering,
    }

    if method not in methods:
        raise ValueError(f"Unknown clustering method: {method}")

    return methods[method](embeddings, **kwargs)
