"""RefineNet: Dual-stream gated fusion of physical + IDW semantic features."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class RefineNet(nn.Module):
    """Dual-stream gated fusion: z · H_phys + (1-z) · H_sem.

    Architecture:
        - Two independent GAT encoders for physical and semantic streams
        - Learned gating mechanism to balance stream contributions
        - MLP decoder for refined POI distribution prediction

    Input:  Physical features [N, phys_dim] + Semantic features [N, sem_dim] + edge_index
    Output: Refined POI distribution [N, num_classes]
    """

    def __init__(
        self,
        phys_dim: int,
        sem_dim: int = 17,
        hidden_dim: int = 128,
        num_classes: int = 17,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Implementation withheld — will be released upon paper publication.
        raise NotImplementedError

    def forward(
        self,
        x_phys: torch.Tensor,
        x_sem: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with gated dual-stream fusion."""
        raise NotImplementedError

    def forward_relu_embedding(
        self,
        x_phys: torch.Tensor,
        x_sem: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return intermediate fused embedding (before final classification)."""
        raise NotImplementedError

    def get_gate_values(
        self,
        x_phys: torch.Tensor,
        x_sem: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return gate values z ∈ [0,1] for analysis/visualization."""
        raise NotImplementedError
