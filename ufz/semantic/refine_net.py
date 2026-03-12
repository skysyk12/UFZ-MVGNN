"""RefineNet: Dual-stream gated fusion of physical + IDW semantic features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class _GATEncoder(nn.Module):
    """Shared 2-layer GATConv encoder."""

    def __init__(self, in_dim: int, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        return h


class RefineNet(nn.Module):
    """Dual-stream fusion: z·H_phys + (1-z)·H_sem."""

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

        self.phys_encoder = _GATEncoder(phys_dim, hidden_dim, heads, dropout)
        self.sem_encoder = _GATEncoder(sem_dim, hidden_dim, heads, dropout)

        self.gate_linear = nn.Linear(hidden_dim * 2, 1)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x_phys: torch.Tensor,
        x_sem: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h_phys = self.phys_encoder(x_phys, edge_index)
        h_sem = self.sem_encoder(x_sem, edge_index)
        z = torch.sigmoid(self.gate_linear(torch.cat([h_phys, h_sem], dim=-1)))
        h_fused = z * h_phys + (1.0 - z) * h_sem
        logits = self.decoder(h_fused)
        return logits

    def forward_relu_embedding(
        self,
        x_phys: torch.Tensor,
        x_sem: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h_phys = self.phys_encoder(x_phys, edge_index)
        h_sem = self.sem_encoder(x_sem, edge_index)
        z = torch.sigmoid(self.gate_linear(torch.cat([h_phys, h_sem], dim=-1)))
        h_fused = z * h_phys + (1.0 - z) * h_sem
        h = self.decoder[0](h_fused)
        h = self.decoder[1](h)
        return h

    def get_gate_values(
        self,
        x_phys: torch.Tensor,
        x_sem: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            h_phys = self.phys_encoder(x_phys, edge_index)
            h_sem = self.sem_encoder(x_sem, edge_index)
            z = torch.sigmoid(self.gate_linear(torch.cat([h_phys, h_sem], dim=-1)))
        return z.squeeze(-1)
