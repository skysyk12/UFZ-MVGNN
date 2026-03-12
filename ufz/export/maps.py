"""Export cluster results as visualization maps (GeoJSON, etc)."""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


def export_geojson(
    positions: np.ndarray,
    labels: np.ndarray,
    output_path: str = "clusters.geojson",
    properties: Optional[Dict] = None,
    crs: str = "EPSG:4326",
):
    """Export clusters as GeoJSON FeatureCollection.
    
    Args:
        positions: [N, 2] coordinates (lon, lat)
        labels: [N] cluster labels (-1 for noise)
        output_path: Output GeoJSON file
        properties: Optional node properties dict
        crs: Coordinate reference system
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    features = []
    for i in range(len(positions)):
        cluster = int(labels[i])
        props = {
            "cluster": cluster,
            "noise": cluster == -1,
        }
        
        if properties and i in properties:
            props.update(properties[i])
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(positions[i, 0]), float(positions[i, 1])],
            },
            "properties": props,
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features,
    }
    
    with open(output_path, "w") as f:
        json.dump(geojson, f)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(
        f"Exported {len(features)} points to {output_path} "
        f"({n_clusters} clusters, {n_noise} noise)"
    )


def export_cluster_summary(
    labels: np.ndarray,
    embeddings: np.ndarray,
    output_path: str = "cluster_summary.json",
):
    """Export cluster statistics.
    
    Args:
        labels: [N] cluster labels
        embeddings: [N, D] embeddings (for statistics)
        output_path: Output JSON file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    unique_labels = sorted(np.unique(labels))
    summary = {}
    
    for label in unique_labels:
        mask = labels == label
        cluster_emb = embeddings[mask]
        
        summary[int(label)] = {
            "size": int(mask.sum()),
            "mean_embedding": cluster_emb.mean(axis=0).tolist()[:10],  # First 10 dims
            "std_embedding": cluster_emb.std(axis=0).tolist()[:10],
        }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Exported cluster summary to {output_path}")


def export_to_shapefile(
    positions: np.ndarray,
    labels: np.ndarray,
    gdf,
    output_path: str = "clusters.shp",
):
    """Export clusters with original geometries as Shapefile.
    
    Args:
        positions: [N, 2] centroids (for reference)
        labels: [N] cluster labels
        gdf: Original GeoDataFrame with geometries
        output_path: Output Shapefile path
    """
    import geopandas as gpd
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    output_gdf = gdf.copy()
    output_gdf["cluster"] = labels.astype(int)
    output_gdf["noise"] = labels == -1
    
    output_gdf.to_file(output_path)
    
    logger.info(f"Exported clusters to {output_path}")


if __name__ == "__main__":
    # Example usage
    positions = np.random.rand(100, 2) * 10
    labels = np.random.randint(0, 5, 100)
    export_geojson(positions, labels, "test_clusters.geojson")
