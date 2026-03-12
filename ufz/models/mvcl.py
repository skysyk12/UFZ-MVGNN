"""Multi-View Contrastive Learning (MVCL) Model."""

import torch
import torch.nn as nn
from typing import Optional

from .backbones import BackboneRegistry
from .losses import InfoNCELoss


class ProjectionHead(nn.Module):
    """Projection head: MLP for mapping representations to contrastive space."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # Implementation withheld — will be released upon paper publication.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MVCLModel(nn.Module):
    """Multi-View Contrastive Learning: dual encoders + projection heads.

    Architecture:
        Physical View (27D) ──→ GNN Encoder ──→ Projection Head ──┐
                                                                   ├─→ InfoNCE Loss
        Semantic View (17D) ──→ GNN Encoder ──→ Projection Head ──┘

    Features:
        - Pluggable GNN backbone via BackboneRegistry (GIN/GAT/GCN)
        - Semantic-aware negative sample reweighting
        - Averaged dual-view embeddings for downstream tasks

    Args:
        physical_dim: Input dimension for physical view
        semantic_dim: Input dimension for semantic view
        hidden_dim: GNN hidden layer dimension
        repr_dim: Representation dimension (encoder output)
        proj_dim: Projection dimension (contrastive space)
        backbone: GNN backbone type ('gin', 'gat', 'gcn')
        num_layers: Number of GNN layers
        dropout: Dropout rate
        temperature: InfoNCE temperature parameter
    """

    def __init__(
        self,
        physical_dim: int,
        semantic_dim: int,
        hidden_dim: int = 256,
        repr_dim: int = 128,
        proj_dim: int = 128,
        backbone: str = "gin",
        num_layers: int = 2,
        dropout: float = 0.5,
        temperature: float = 0.07,
        **backbone_kwargs,
    ):
        super().__init__()
        # Implementation withheld — will be released upon paper publication.
        raise NotImplementedError

    def forward(
        self, x_phys: torch.Tensor, x_sem: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple:
        """Forward: encode both views and project to contrastive space.

        Returns:
            (h_phys, h_sem, z_phys, z_sem): representations and projections
        """
        raise NotImplementedError

    def compute_loss(
        self,
        z_phys: torch.Tensor,
        z_sem: torch.Tensor,
        sem_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute contrastive loss with optional semantic-aware weighting."""
        raise NotImplementedError

    def get_embeddings(
        self, x_phys: torch.Tensor, x_sem: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get merged embeddings (averaged dual-view) for downstream tasks."""
        raise NotImplementedError
