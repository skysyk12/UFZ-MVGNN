"""Test analysis components."""

import pytest
import numpy as np
from ufz.analysis import cluster_embeddings, PCAReducer, UMAPReducer


def test_pca_reducer():
    """Test PCA dimensionality reduction."""
    embeddings = np.random.randn(100, 50)
    
    reducer = PCAReducer(n_components=2)
    reduced, variance = reducer.fit_transform(embeddings)
    
    assert reduced.shape == (100, 2)
    assert 0 < variance <= 1


def test_umap_reducer():
    """Test UMAP dimensionality reduction."""
    embeddings = np.random.randn(50, 20)  # Small for speed
    
    reducer = UMAPReducer(n_components=2, n_neighbors=5)
    reduced = reducer.fit_transform(embeddings)
    
    assert reduced.shape == (50, 2)


def test_clustering():
    """Test clustering methods."""
    embeddings = np.random.randn(100, 20)
    
    # DBSCAN
    labels = cluster_embeddings(embeddings, method="dbscan", eps=0.5)
    assert len(labels) == 100
    assert labels.min() >= -1  # -1 for noise
    
    # KMeans
    labels = cluster_embeddings(embeddings, method="kmeans", n_clusters=5)
    assert len(labels) == 100
    assert labels.min() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
