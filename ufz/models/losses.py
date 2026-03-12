"""Contrastive loss functions: InfoNCE, NT-Xent, DGI."""

import torch
import torch.nn as nn
from typing import Optional


class InfoNCELoss(nn.Module):
    """InfoNCE loss with semantic-aware negative sample reweighting.

    When semantic probabilities are provided, negative pairs with similar
    POI distributions receive lower repulsion weights, preventing the model
    from pushing semantically similar buildings apart.

    Args:
        temperature: Softmax temperature for similarity scaling
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        sem_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss between two views.

        Args:
            z1, z2: Projected embeddings [N, D]
            sem_probs: Optional semantic distribution [N, C] for negative weighting
        """
        # Implementation withheld — will be released upon paper publication.
        raise NotImplementedError


class NTXentLoss(nn.Module):
    """NT-Xent: Normalized Temperature-scaled Cross Entropy Loss."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # Implementation withheld — will be released upon paper publication.
        raise NotImplementedError


class DGILoss(nn.Module):
    """DGI: Deep Graph Infomax loss."""

    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        # Implementation withheld — will be released upon paper publication.
        raise NotImplementedError


class ContrastiveLoss(nn.Module):
    """Generic contrastive loss selector.

    Supported types: 'infonce', 'ntxent', 'dgi'
    """

    LOSS_REGISTRY = {
        "infonce": InfoNCELoss,
        "ntxent": NTXentLoss,
        "dgi": DGILoss,
    }

    def __init__(self, loss_type: str = "infonce", **kwargs):
        super().__init__()
        if loss_type not in self.LOSS_REGISTRY:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss_fn = self.LOSS_REGISTRY[loss_type](**kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.loss_fn(*args, **kwargs)
