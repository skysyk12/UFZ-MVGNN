"""CrossViewImputer: GAT encoder + MLP decoder for POI distribution prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class CrossViewImputer(nn.Module):
    """Physics → POI distribution via GAT + MLP."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 17,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(
            in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(h)
        return F.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.decode(h)
