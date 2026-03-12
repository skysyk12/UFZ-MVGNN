"""Graph structure visualization using Plotly."""

import logging
import numpy as np
import torch
from typing import Optional, List

logger = logging.getLogger(__name__)


def visualize_graph(
    node_positions: np.ndarray,
    edge_index: Optional[torch.Tensor] = None,
    node_colors: Optional[np.ndarray] = None,
    node_labels: Optional[List[str]] = None,
    max_edges: int = 5000,
    node_size: int = 4,
    edge_width: float = 0.5,
    title: str = "Graph Visualization",
    colorscale: str = "Viridis",
):
    """Visualize graph with node positions.
    
    Args:
        node_positions: Node coordinates [N, 2]
        edge_index: Edge indices [2, E] or [E, 2]
        node_colors: Node color values [N]
        node_labels: Hover labels
        max_edges: Max edges to display (subsample if larger)
        
    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go

    edge_x, edge_y = [], []
    if edge_index is not None:
        edge_arr = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index
        if edge_arr.shape[0] == 2:
            edge_pairs = edge_arr.T
        else:
            edge_pairs = edge_arr

        if len(edge_pairs) > max_edges:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(edge_pairs), size=max_edges, replace=False)
            edge_pairs = edge_pairs[idx]
            logger.info(f"Subsampled edges: {len(edge_pairs)}/{len(edge_pairs)} (max={max_edges})")

        for src, dst in edge_pairs:
            edge_x.extend([node_positions[src, 0], node_positions[dst, 0], None])
            edge_y.extend([node_positions[src, 1], node_positions[dst, 1], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=edge_width, color="rgba(125,125,125,0.2)"),
        hoverinfo="none",
        showlegend=False,
    )

    node_trace = go.Scatter(
        x=node_positions[:, 0],
        y=node_positions[:, 1],
        mode="markers",
        marker=dict(
            size=node_size,
            color=node_colors if node_colors is not None else "blue",
            colorscale=colorscale if node_colors is not None else None,
            showscale=node_colors is not None,
            line=dict(width=0),
        ),
        text=node_labels,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
    )

    return fig
