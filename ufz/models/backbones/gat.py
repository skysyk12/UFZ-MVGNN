"""GAT (Graph Attention Network) encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .registry import BackboneRegistry
from .base import BaseEncoder


@BackboneRegistry.register("gat")
class GATEncoder(BaseEncoder):
    """GAT: Uses multi-head attention for node aggregation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 4,
    ):
        super().__init__(in_dim, hidden_dim, out_dim, num_layers, dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                conv = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
                self.convs.append(conv)
                self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
            elif i == num_layers - 1:
                conv = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False)
                self.convs.append(conv)
            else:
                conv = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
                self.convs.append(conv)
                self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
