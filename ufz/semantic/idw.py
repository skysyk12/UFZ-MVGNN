"""Gaussian IDW: POI spatial diffusion to all buildings."""

import logging
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def apply_gaussian_idw(
    building_centroids: np.ndarray,
    poi_coords: np.ndarray,
    poi_categories: np.ndarray,
    num_classes: int = 17,
    R: float = 100.0,
    c: float = 30.0,
) -> np.ndarray:
    """Apply Gaussian IDW to diffuse POI influence to all buildings.

    For each building: find POIs within radius R, weight by Gaussian kernel
    exp(-d^2 / 2c^2), aggregate by category, normalize to probability vector.

    Args:
        building_centroids: [N, 2] building centroid coordinates (projected CRS)
        poi_coords: [M, 2] POI coordinates (same CRS)
        poi_categories: [M] integer category index for each POI
        num_classes: Number of POI categories
        R: Search radius in meters
        c: Gaussian bandwidth in meters

    Returns:
        x_sem_idw: [N, num_classes] probability matrix
    """
    # Implementation withheld — will be released upon paper publication.
    raise NotImplementedError


def compute_idw_from_dataframes(
    gdf,
    poi_csv_path: str,
    target_crs: str = "EPSG:32650",
    lon_col: str = "update_wgs84_lon",
    lat_col: str = "update_wgs84_lat",
    type_col: str = "poi类型",
    num_classes: int = 17,
    R: float = 100.0,
    c: float = 30.0,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """End-to-end IDW computation from GeoDataFrame + POI CSV.

    Handles CRS projection, POI loading, category encoding, caching,
    and delegates to apply_gaussian_idw for the core computation.

    Returns:
        Tuple of (probability_matrix [N, num_classes], category_names)
    """
    # Implementation withheld — will be released upon paper publication.
    raise NotImplementedError
