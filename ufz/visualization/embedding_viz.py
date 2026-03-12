"""Embedding space visualization using UMAP/t-SNE."""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "umap",
    n_components: int = 2,
    title: str = "Embedding Visualization",
    **kwargs,
):
    """Visualize high-dim embeddings in 2D/3D.
    
    Args:
        embeddings: [N, D] embedding matrix
        labels: [N] optional cluster/class labels
        method: 'umap' or 'tsne'
        n_components: 2 or 3
        
    Returns:
        Plotly Figure
    """
    import plotly.graph_objects as go

    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
        reduced = reducer.fit_transform(embeddings)
        logger.info(f"UMAP: {embeddings.shape[1]} → {n_components} dims")
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42, **kwargs)
        reduced = reducer.fit_transform(embeddings)
        logger.info(f"t-SNE: {embeddings.shape[1]} → {n_components} dims")
    else:
        raise ValueError(f"Unknown method: {method}")

    if n_components == 2:
        fig = go.Figure()
        if labels is not None:
            for label in sorted(np.unique(labels)):
                mask = labels == label
                fig.add_trace(
                    go.Scatter(
                        x=reduced[mask, 0],
                        y=reduced[mask, 1],
                        mode="markers",
                        name=f"Cluster {label}",
                        marker=dict(size=5),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode="markers",
                    marker=dict(size=5, color=np.arange(len(embeddings)), colorscale="Viridis"),
                )
            )
        fig.update_layout(title=title, hovermode="closest", height=700)

    elif n_components == 3:
        fig = go.Figure()
        if labels is not None:
            for label in sorted(np.unique(labels)):
                mask = labels == label
                fig.add_trace(
                    go.Scatter3d(
                        x=reduced[mask, 0],
                        y=reduced[mask, 1],
                        z=reduced[mask, 2],
                        mode="markers",
                        name=f"Cluster {label}",
                        marker=dict(size=3),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode="markers",
                    marker=dict(size=3, color=np.arange(len(embeddings)), colorscale="Viridis"),
                )
            )
        fig.update_layout(title=title, hovermode="closest", height=700)

    else:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")

    return fig
