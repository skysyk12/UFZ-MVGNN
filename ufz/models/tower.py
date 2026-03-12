"""Multi-Tower Input Encoder: separate MLPs for geo/topo/semantic features."""

import torch
import torch.nn as nn
from typing import Tuple


class FeatureTower(nn.Module):
    """Single feature tower: Linear → BN → ReLU → Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiTowerEncoder(nn.Module):
    """Multi-tower input encoder: geo + topo + semantic towers."""

    def __init__(
        self,
        geo_dim: int,
        topo_dim: int,
        sem_dim: int,
        tower_dim: int = 32,
        fusion_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Args:
            geo_dim: Geometric features (shape, size, orientation)
            topo_dim: Topological features (graphlet)
            sem_dim: Semantic features (POI/IDW)
            tower_dim: Output dimension of each tower
            fusion_dim: Final fused feature dimension
        """
        super().__init__()

        self.geo_tower = FeatureTower(geo_dim, tower_dim * 2, tower_dim, dropout)
        self.topo_tower = FeatureTower(topo_dim, tower_dim * 2, tower_dim, dropout)
        self.sem_tower = FeatureTower(sem_dim, tower_dim * 4, tower_dim, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(tower_dim * 3, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = fusion_dim

    def forward(
        self, x_geo: torch.Tensor, x_topo: torch.Tensor, x_sem: torch.Tensor
    ) -> torch.Tensor:
        """Fuse three feature towers."""
        h_geo = self.geo_tower(x_geo)
        h_topo = self.topo_tower(x_topo)
        h_sem = self.sem_tower(x_sem)

        h_concat = torch.cat([h_geo, h_topo, h_sem], dim=1)
        return self.fusion(h_concat)

    def forward_with_tower_outputs(
        self, x_geo: torch.Tensor, x_topo: torch.Tensor, x_sem: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return fused output + individual tower outputs."""
        h_geo = self.geo_tower(x_geo)
        h_topo = self.topo_tower(x_topo)
        h_sem = self.sem_tower(x_sem)

        h_concat = torch.cat([h_geo, h_topo, h_sem], dim=1)
        h_fused = self.fusion(h_concat)

        return h_fused, (h_geo, h_topo, h_sem)
