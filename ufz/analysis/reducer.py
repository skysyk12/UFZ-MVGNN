"""Dimensionality reduction: PCA, UMAP."""

import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


class PCAReducer:
    """PCA dimensionality reduction wrapper."""

    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        self.variance_ratio = None

    def fit_transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit PCA and transform embeddings."""
        from sklearn.decomposition import PCA

        self.reducer = PCA(n_components=self.n_components, random_state=self.random_state)
        reduced = self.reducer.fit_transform(embeddings)
        self.variance_ratio = sum(self.reducer.explained_variance_ratio_)

        logger.info(
            f"PCA: {embeddings.shape[1]} → {self.n_components} dims "
            f"(variance: {self.variance_ratio:.2%})"
        )

        return reduced, self.variance_ratio

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings."""
        if self.reducer is None:
            raise RuntimeError("PCA not fitted. Call fit_transform first.")
        return self.reducer.transform(embeddings)


class UMAPReducer:
    """UMAP dimensionality reduction wrapper."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.reducer = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit UMAP and transform embeddings."""
        import umap

        self.reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        reduced = self.reducer.fit_transform(embeddings)

        logger.info(
            f"UMAP: {embeddings.shape[1]} → {self.n_components} dims "
            f"(neighbors={self.n_neighbors}, min_dist={self.min_dist})"
        )

        return reduced

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings."""
        if self.reducer is None:
            raise RuntimeError("UMAP not fitted. Call fit_transform first.")
        return self.reducer.transform(embeddings)
