"""Cluster result visualization on map."""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def visualize_clusters(
    positions: np.ndarray,
    labels: np.ndarray,
    values: Optional[np.ndarray] = None,
    title: str = "Cluster Visualization",
    max_nodes: int = 10000,
):
    """Visualize clusters on geographic/spatial map.
    
    Args:
        positions: [N, 2] node coordinates
        labels: [N] cluster labels
        values: [N] optional values for coloring
        max_nodes: Subsample if N > max_nodes
        
    Returns:
        Plotly Figure
    """
    import plotly.graph_objects as go

    if len(positions) > max_nodes:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(positions), size=max_nodes, replace=False)
        positions = positions[idx]
        labels = labels[idx]
        if values is not None:
            values = values[idx]
        logger.info(f"Subsampled: {len(positions)} nodes (max={max_nodes})")

    fig = go.Figure()

    unique_labels = sorted(np.unique(labels))
    for label in unique_labels:
        mask = labels == label
        
        if values is not None:
            color_vals = values[mask]
        else:
            color_vals = None

        fig.add_trace(
            go.Scattergl(
                x=positions[mask, 0],
                y=positions[mask, 1],
                mode="markers",
                name=f"Cluster {label}",
                marker=dict(
                    size=4,
                    color=color_vals,
                    colorscale="Viridis" if values is not None else None,
                    showscale=False,
                    opacity=0.7,
                ),
                hovertemplate=f"Cluster {label}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        hovermode="closest",
        height=700,
        scattergl=dict(renderer="webgl"),
    )

    return fig


def export_cluster_geojson(
    positions: np.ndarray,
    labels: np.ndarray,
    properties: Optional[dict] = None,
    output_path: str = "clusters.geojson",
):
    """Export clusters as GeoJSON.
    
    Args:
        positions: [N, 2] coordinates (lon, lat)
        labels: [N] cluster labels
        properties: Optional feature properties dict
        output_path: Output GeoJSON file path
    """
    import json

    features = []
    for i in range(len(positions)):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(positions[i, 0]), float(positions[i, 1])],
            },
            "properties": {
                "cluster": int(labels[i]),
                **(properties[i] if properties and i in properties else {}),
            },
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    with open(output_path, "w") as f:
        json.dump(geojson, f)

    logger.info(f"Exported {len(features)} cluster points to {output_path}")
