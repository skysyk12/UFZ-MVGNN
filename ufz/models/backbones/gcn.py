"""GCN (Graph Convolutional Network) encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .registry import BackboneRegistry
from .base import BaseEncoder


@BackboneRegistry.register("gcn")
class GCNEncoder(BaseEncoder):
    """GCN: Simplified spectral convolution."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            elif i == num_layers - 1:
                self.convs.append(GCNConv(hidden_dim, out_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
