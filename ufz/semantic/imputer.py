"""CrossViewImputer: GAT encoder + MLP decoder for POI distribution prediction."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class CrossViewImputer(nn.Module):
    """Physics → POI distribution via GAT + MLP.

    Architecture:
        - 2-layer GAT encoder with multi-head attention and BatchNorm
        - MLP decoder with softmax output

    Input:  Physical features [N, in_dim] + edge_index [2, E]
    Output: POI distribution  [N, num_classes]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 17,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        # Implementation withheld — will be released upon paper publication.
        raise NotImplementedError

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode physical features into latent space via GAT layers."""
        raise NotImplementedError

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode latent representation into POI category probabilities."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode → decode."""
        raise NotImplementedError
