"""Base encoder abstract class for GNN backbones."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for GNN encoders."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass: x [N, in_dim], edge_index [2, E] → [N, out_dim]."""
        pass

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
