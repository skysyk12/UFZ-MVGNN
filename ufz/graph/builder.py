"""Delaunay graph construction with quantile-based edge pruning."""

import logging
from typing import Optional

import numpy as np
import torch
import geopandas as gpd
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _build_delaunay_edge_pairs(
    points: np.ndarray,
    max_edge_length_m: Optional[float] = None,
) -> np.ndarray:
    """
    Build unique undirected edge pairs from Delaunay triangulation.

    Args:
        points: (N, 2) array of metric coordinates.
        max_edge_length_m: Drop edges longer than this (metres).

    Returns:
        (E, 2) int array of node-index pairs.
    """
    tri = Delaunay(points)
    simplices = tri.simplices
    if simplices.size == 0:
        return np.empty((0, 2), dtype=int)

    edges = np.concatenate(
        [simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]],
        axis=0,
    ).astype(int, copy=False)

    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]

    if max_edge_length_m is not None and edges.size > 0:
        dx = points[edges[:, 0], 0] - points[edges[:, 1], 0]
        dy = points[edges[:, 0], 1] - points[edges[:, 1], 1]
        keep = (dx * dx + dy * dy) <= float(max_edge_length_m) ** 2
        edges = edges[keep]

    if edges.size == 0:
        return np.empty((0, 2), dtype=int)

    return np.unique(edges, axis=0)


def convert_to_utm(
    gdf: gpd.GeoDataFrame,
    utm_epsg: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    Project *gdf* to a UTM CRS (auto-detected from centroid if *utm_epsg* is None).
    """
    if gdf.crs is None:
        minx, miny, maxx, maxy = gdf.total_bounds
        if -180 <= minx <= 180 and -90 <= miny <= 90:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            raise ValueError("Cannot determine CRS from data")

    if utm_epsg is None:
        gdf_wgs = gdf.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = gdf_wgs.total_bounds
        lon = float((minx + maxx) / 2)
        lat = float((miny + maxy) / 2)
        zone = max(1, min(60, int(np.floor((lon + 180) / 6) + 1)))
        utm_epsg = (32600 + zone) if lat >= 0 else (32700 + zone)

    return gdf.to_crs(epsg=int(utm_epsg))


def compute_edge_length_threshold(
    gdf: gpd.GeoDataFrame,
    percentile: float = 0.99,
) -> float:
    """Return the *percentile*-quantile of all Delaunay edge lengths (metres)."""
    gdf_utm = convert_to_utm(gdf)
    centroids = gdf_utm.geometry.centroid
    points = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()]).astype(
        float, copy=False
    )
    edge_pairs = _build_delaunay_edge_pairs(points)
    if edge_pairs.size == 0:
        return 200.0

    dx = points[edge_pairs[:, 0], 0] - points[edge_pairs[:, 1], 0]
    dy = points[edge_pairs[:, 0], 1] - points[edge_pairs[:, 1], 1]
    lengths = np.sqrt(dx * dx + dy * dy)

    q = float(percentile) if percentile <= 1.0 else percentile / 100.0
    return float(np.quantile(lengths, q))


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class GraphBuilder:
    """
    Delaunay triangulation + configurable edge-length pruning.

    If *max_edge_length_m* is ``None``, the threshold is auto-detected at the
    given *auto_threshold_percentile* of all raw Delaunay edge lengths.
    """

    def __init__(
        self,
        max_edge_length_m: Optional[float] = None,
        auto_threshold_percentile: float = 0.99,
    ):
        self.max_edge_length_m = max_edge_length_m
        self.auto_threshold_percentile = auto_threshold_percentile
        self._computed_threshold: Optional[float] = None

    def build(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """Return (E, 2) undirected edge pairs."""
        gdf_utm = convert_to_utm(gdf)
        centroids = gdf_utm.geometry.centroid
        points = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()]).astype(
            float, copy=False
        )

        if self.max_edge_length_m is None:
            self._computed_threshold = compute_edge_length_threshold(
                gdf, self.auto_threshold_percentile
            )
            max_len = self._computed_threshold
        else:
            max_len = self.max_edge_length_m

        logger.debug(f"Building Delaunay graph with max edge length: {max_len:.1f}m")
        edge_pairs = _build_delaunay_edge_pairs(points, max_edge_length_m=max_len)
        logger.debug(f"Graph built: {len(points)} nodes, {len(edge_pairs)} edges")
        return edge_pairs

    def build_edge_index(self, gdf: gpd.GeoDataFrame) -> torch.Tensor:
        """Return PyG ``edge_index`` tensor of shape ``(2, 2E)`` (bi-directional)."""
        edge_pairs = self.build(gdf)
        if edge_pairs.size == 0:
            return torch.empty((2, 0), dtype=torch.long)
        ei = torch.as_tensor(edge_pairs.T, dtype=torch.long)
        return torch.cat([ei, ei.flip(0)], dim=1)

    @property
    def computed_threshold(self) -> Optional[float]:
        return self._computed_threshold


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def create_edge_index(
    shp_file_path: str,
    max_edge_length_m: Optional[float] = None,
) -> torch.Tensor:
    """One-liner: shapefile path -> PyG edge_index."""
    gdf = gpd.read_file(shp_file_path)
    return GraphBuilder(max_edge_length_m=max_edge_length_m).build_edge_index(gdf)
