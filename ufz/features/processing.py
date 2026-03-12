"""
Feature Post-Processing

Provides standardization (Z-score) and dimensionality reduction (PCA / UMAP)
for computed feature matrices. Pure computation, no I/O.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Standardization and dimensionality reduction for feature matrices."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def standard_scale_subset(
        self,
        data_df: pd.DataFrame,
        feature_subset: list[str],
        by_row: bool = False,
    ) -> pd.DataFrame:
        """
        Z-score standardize a subset of columns (or rows) in a DataFrame.

        Args:
            data_df: DataFrame with all features.
            feature_subset: Column names to standardize.
            by_row: If True, standardize across columns per row instead.

        Returns:
            DataFrame copy with the specified columns standardized.
        """
        if not feature_subset:
            return data_df.copy()

        subset_data = data_df[feature_subset].fillna(0).astype(float)
        scaler = StandardScaler()

        if by_row:
            scaled_array = scaler.fit_transform(subset_data.T).T
        else:
            scaled_array = scaler.fit_transform(subset_data)

        output_df = data_df.copy()
        output_df[feature_subset] = pd.DataFrame(
            scaled_array, columns=feature_subset, index=data_df.index
        )
        return output_df

    def pca_reducer(
        self,
        data_array: np.ndarray,
        n_components: int | float = 0.95,
    ) -> tuple[np.ndarray, list[str]]:
        """Apply PCA dimensionality reduction."""
        logger.info("Running PCA reduction...")
        reducer = PCA(n_components=n_components, random_state=self.random_state)
        reduced = reducer.fit_transform(data_array)
        final_dim = reducer.n_components_
        cols = [f'PCA_{i}' for i in range(final_dim)]
        logger.info("PCA: %dD -> %dD", data_array.shape[1], final_dim)
        return reduced, cols

    def umap_reducer(
        self,
        data_array: np.ndarray,
        n_components: int = 4,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
    ) -> tuple[np.ndarray, list[str]]:
        """Apply UMAP dimensionality reduction."""
        import umap

        logger.info("Running UMAP reduction...")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_state,
        )
        reduced = reducer.fit_transform(data_array)
        cols = [f'UMAP_{i}' for i in range(n_components)]
        logger.info("UMAP: %dD -> %dD", data_array.shape[1], n_components)
        return np.asarray(reduced), cols
