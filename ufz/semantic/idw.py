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
    
    For each building: find POIs within radius R, weight by Gaussian kernel,
    aggregate by category, normalize to probability vector.
    
    Returns:
        x_sem_idw: [N, num_classes] probability matrix
    """
    N = len(building_centroids)
    M = len(poi_coords)
    logger.info(f"Gaussian IDW: {N:,} buildings × {M:,} POIs, R={R}m, c={c}")

    result = np.zeros((N, num_classes), dtype=np.float64)

    if M == 0:
        logger.warning("No POIs provided")
        return result.astype(np.float32)

    poi_tree = cKDTree(poi_coords)

    try:
        neighbors_list = poi_tree.query_ball_point(
            building_centroids, r=R, workers=-1
        )
    except TypeError:
        neighbors_list = poi_tree.query_ball_point(building_centroids, r=R)

    inv_2c2 = 1.0 / (2.0 * c * c)
    n_with_poi = 0

    for i, poi_indices in enumerate(neighbors_list):
        if len(poi_indices) == 0:
            continue

        poi_idx = np.array(poi_indices, dtype=np.intp)
        diff = building_centroids[i] - poi_coords[poi_idx]
        dist_sq = (diff * diff).sum(axis=1)
        weights = np.exp(-dist_sq * inv_2c2)
        cats = poi_categories[poi_idx]
        np.add.at(result[i], cats, weights)
        n_with_poi += 1

    row_sums = result.sum(axis=1, keepdims=True)
    nonzero_mask = row_sums.ravel() > 0
    result[nonzero_mask] = result[nonzero_mask] / row_sums[nonzero_mask]

    coverage = n_with_poi / N * 100
    logger.info(f"  IDW coverage: {n_with_poi:,}/{N:,} ({coverage:.1f}%) buildings have POIs within {R}m")

    return result.astype(np.float32)


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
    """End-to-end IDW from GeoDataFrame + POI CSV."""
    
    import pandas as pd
    import geopandas as gpd
    import json

    if cache_path:
        cat_cache = str(Path(cache_path).with_suffix(".json"))
        if Path(cache_path).exists() and Path(cat_cache).exists():
            try:
                x_sem_idw = np.load(cache_path)
                with open(cat_cache, "r", encoding="utf-8") as f:
                    categories = json.load(f)
                if x_sem_idw.shape[0] == len(gdf):
                    logger.info(f"Loaded cached IDW from {cache_path}")
                    return x_sem_idw, categories
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

    logger.info(f"Loading POIs from: {poi_csv_path}")
    try:
        poi_df = pd.read_csv(poi_csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        poi_df = pd.read_csv(poi_csv_path, encoding="gbk")

    poi_df = poi_df.dropna(subset=[lon_col, lat_col, type_col])
    logger.info(f"  Valid POIs: {len(poi_df)}")

    categories = sorted(poi_df[type_col].unique().tolist())
    cat2idx = {c: i for i, c in enumerate(categories)}
    poi_cat_idx = poi_df[type_col].map(cat2idx).values.astype(np.intp)

    poi_gdf = gpd.GeoDataFrame(
        poi_df,
        geometry=gpd.points_from_xy(poi_df[lon_col], poi_df[lat_col]),
        crs="EPSG:4326",
    )

    if gdf.crs is None:
        bounds = gdf.total_bounds
        if -180 <= bounds[0] <= 180:
            gdf = gdf.set_crs("EPSG:4326")

    buildings_proj = gdf.to_crs(target_crs)
    poi_proj = poi_gdf.to_crs(target_crs)

    bldg_centroids = np.column_stack([
        buildings_proj.geometry.centroid.x.values,
        buildings_proj.geometry.centroid.y.values,
    ])
    poi_xy = np.column_stack([
        poi_proj.geometry.x.values,
        poi_proj.geometry.y.values,
    ])

    actual_classes = len(categories)
    x_sem_idw = apply_gaussian_idw(
        bldg_centroids, poi_xy, poi_cat_idx,
        num_classes=actual_classes, R=R, c=c,
    )

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, x_sem_idw)
        cat_cache = str(Path(cache_path).with_suffix(".json"))
        with open(cat_cache, "w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved IDW to {cache_path}")

    return x_sem_idw, categories
