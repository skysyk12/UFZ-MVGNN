"""
Graphlet Feature Computation (Parallelized)

Computes 4-node graphlet orbit counts (15D) using the ORCA algorithm.
Supports multi-core processing by splitting the graph into buffered subgraphs.
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import torch
from joblib import Parallel, cpu_count, delayed
from torch_geometric.utils import k_hop_subgraph, to_undirected

logger = logging.getLogger(__name__)

GRAPHLET_ORBIT_NAMES = [f'orbit_{i}' for i in range(15)]


class GraphletCalculator:
    """Compute graphlet features using the ORCA binary with parallel support."""

    def __init__(
        self,
        orca_path: str = "./orca",
        max_edge_length_m: float = 200.0,
        graphlet_size: int = 4,
    ):
        if not orca_path or not orca_path.strip():
            raise ValueError("orca_path is empty")
        self.orca_path = os.path.abspath(orca_path.strip())
        self.max_edge_length_m = max_edge_length_m
        self.graphlet_size = graphlet_size

        if not os.path.exists(self.orca_path):
            logger.warning("ORCA binary not found at %s", self.orca_path)
        elif not os.access(self.orca_path, os.X_OK):
            logger.warning("ORCA binary is not executable: %s", self.orca_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def write_orca_input(edges: np.ndarray, num_nodes: int, path: str):
        """Write edge list to file in ORCA format."""
        with open(path, 'w') as f:
            f.write(f"{num_nodes} {len(edges)}\n")
            for u, v in edges:
                f.write(f"{u} {v}\n")

    def _process_subgraph(
        self,
        target_nodes: torch.Tensor,
        full_edge_index: torch.Tensor,
        num_hops: int,
        total_num_nodes: int,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Extract subgraph -> run ORCA -> return features for target nodes."""
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            target_nodes,
            num_hops,
            full_edge_index,
            relabel_nodes=True,
            num_nodes=total_num_nodes,
        )

        num_sub_nodes = len(subset)
        sub_edges_np = sub_edge_index.T.cpu().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            in_file = os.path.join(tmpdir, "in.txt")
            out_file = os.path.join(tmpdir, "out.txt")

            # Keep one direction (u < v) for ORCA
            mask = sub_edges_np[:, 0] < sub_edges_np[:, 1]
            orca_edges = sub_edges_np[mask]

            self.write_orca_input(orca_edges, num_sub_nodes, in_file)
            cmd = [self.orca_path, "node", str(self.graphlet_size), in_file, out_file]

            try:
                res = subprocess.run(cmd, capture_output=True, text=True)
            except (FileNotFoundError, PermissionError, OSError) as e:
                raise RuntimeError(
                    f"Failed to execute ORCA at {self.orca_path!r}: {e}"
                ) from e

            if res.returncode != 0:
                return target_nodes, np.zeros((len(target_nodes), 15))

            try:
                with open(out_file, 'r') as f:
                    all_counts = np.array([
                        [int(x) for x in line.strip().split()] for line in f
                    ])
            except Exception:
                return target_nodes, np.zeros((len(target_nodes), 15))

            target_counts = all_counts[mapping.cpu().numpy()]
            return target_nodes, target_counts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        gdf: gpd.GeoDataFrame,
        edge_index: Optional[torch.Tensor] = None,
        normalize: str = "log",
        cache_path: Optional[str] = None,
    ) -> torch.Tensor:
        normalize_key = (normalize or "log").strip().lower()

        # --- Cache check ---
        if cache_path and os.path.exists(cache_path):
            logger.info("[Cache Hit] Loading graphlets from %s", cache_path)
            cached_obj = torch.load(cache_path, map_location="cpu", weights_only=False)
            if isinstance(cached_obj, torch.Tensor):
                if normalize_key in ("log", "log1p"):
                    return cached_obj
            elif isinstance(cached_obj, dict) and "features" in cached_obj:
                if str(cached_obj.get("normalize", "")).strip().lower() == normalize_key:
                    features = cached_obj.get("features")
                    if isinstance(features, torch.Tensor):
                        return features
            logger.info("Cache normalize mismatch. Recomputing graphlets...")

        logger.info("Computing %d-node graphlet features (Parallel)...", self.graphlet_size)

        # --- Prepare centroids & edges ---
        working_gdf = gdf.to_crs(epsg=3857) if gdf.crs and gdf.crs.is_geographic else gdf
        centroids = np.array([[p.x, p.y] for p in working_gdf.geometry.centroid])
        num_nodes = len(gdf)

        if edge_index is None:
            from scipy.spatial import cKDTree
            logger.info("Building KDTree for edges...")
            tree = cKDTree(centroids)
            pairs = tree.query_pairs(r=self.max_edge_length_m, output_type='ndarray')
            edge_index = torch.from_numpy(pairs.T).long()
            edge_index = to_undirected(edge_index)
        else:
            if edge_index.device.type != 'cpu':
                edge_index = edge_index.cpu()
            # Validate edge indices
            mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            if not mask.all():
                invalid_count = (~mask).sum().item()
                logger.warning(
                    "Found %d edges pointing to invalid nodes (>= %d). Removing them.",
                    invalid_count, num_nodes,
                )
                edge_index = edge_index[:, mask]

        edge_index = to_undirected(edge_index.cpu())

        # --- Parallel processing ---
        total_cores = cpu_count()
        n_jobs = max(1, total_cores // 2)
        chunk_size = int(np.ceil(num_nodes / (n_jobs * 4)))
        chunks = torch.split(torch.arange(num_nodes), chunk_size)

        logger.info("Parallel execution: %d workers, %d chunks", n_jobs, len(chunks))

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._process_subgraph)(chunk, edge_index, num_hops=3, total_num_nodes=num_nodes)
            for chunk in chunks
        )

        # --- Aggregate ---
        final_features = np.zeros((num_nodes, 15), dtype=np.float32)
        for batch_nodes, batch_counts in results:
            final_features[batch_nodes.cpu().numpy()] = batch_counts

        # --- Normalize ---
        norm_tokens = [t for t in normalize_key.replace("+", "_").replace("-", "_").split("_") if t]

        if "log" in norm_tokens or "log1p" in norm_tokens:
            final_features = np.log1p(final_features)
        if "zscore" in norm_tokens:
            mean = final_features.mean(axis=0, keepdims=True)
            std = final_features.std(axis=0, keepdims=True)
            final_features = (final_features - mean) / (std + 1e-8)

        features_tensor = torch.tensor(final_features, dtype=torch.float32)

        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save({"features": features_tensor, "normalize": normalize_key}, cache_path)
            logger.info("Saved graphlet features to %s", cache_path)

        return features_tensor


def compute_graphlet_features(
    gdf: gpd.GeoDataFrame,
    edge_index: Optional[torch.Tensor] = None,
    orca_path: str = "./orca",
    max_edge_length_m: float = 200.0,
    normalize: str = "log",
    cache_path: Optional[str] = None,
) -> torch.Tensor:
    """Convenience function wrapping GraphletCalculator."""
    calculator = GraphletCalculator(orca_path, max_edge_length_m)
    return calculator.compute(gdf, edge_index, normalize, cache_path)
