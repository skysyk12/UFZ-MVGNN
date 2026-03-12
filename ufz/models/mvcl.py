"""Multi-View Contrastive Learning (MVCL) Model."""

import torch
import torch.nn as nn
from typing import Optional

from .backbones import BackboneRegistry
from .losses import InfoNCELoss


class ProjectionHead(nn.Module):
    """Projection head: MLP for contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MVCLModel(nn.Module):
    """Multi-View Contrastive Learning: dual encoders + projection heads."""

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

        self.physical_dim = physical_dim
        self.semantic_dim = semantic_dim
        self.repr_dim = repr_dim

        BackboneClass = BackboneRegistry.get(backbone)

        self.physical_encoder = BackboneClass(
            physical_dim, hidden_dim, repr_dim, num_layers, dropout, **backbone_kwargs
        )
        self.semantic_encoder = BackboneClass(
            semantic_dim, hidden_dim, repr_dim, num_layers, dropout, **backbone_kwargs
        )

        self.physical_proj = ProjectionHead(repr_dim, repr_dim, proj_dim)
        self.semantic_proj = ProjectionHead(repr_dim, repr_dim, proj_dim)

        self.loss_fn = InfoNCELoss(temperature=temperature)

    def forward(
        self, x_phys: torch.Tensor, x_sem: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple:
        """Forward: encode both views and project."""
        h_phys = self.physical_encoder(x_phys, edge_index)
        h_sem = self.semantic_encoder(x_sem, edge_index)

        z_phys = self.physical_proj(h_phys)
        z_sem = self.semantic_proj(h_sem)

        return h_phys, h_sem, z_phys, z_sem

    def compute_loss(
        self,
        z_phys: torch.Tensor,
        z_sem: torch.Tensor,
        sem_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute contrastive loss."""
        return self.loss_fn(z_phys, z_sem, sem_probs=sem_probs)

    def get_embeddings(
        self, x_phys: torch.Tensor, x_sem: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get merged embeddings for downstream tasks."""
        h_phys, h_sem, _, _ = self.forward(x_phys, x_sem, edge_index)
        return (h_phys + h_sem) / 2.0
