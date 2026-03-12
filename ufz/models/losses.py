"""Contrastive loss functions: InfoNCE, NT-Xent, DGI."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE: Contrastive learning loss with semantic-aware weighting."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def _weighted_loss_oneway(
        self, sim: torch.Tensor, repulsion_weights: torch.Tensor
    ) -> torch.Tensor:
        pos = torch.diag(sim)
        weights = repulsion_weights.clone()
        weights.fill_diagonal_(0.0)

        row_max = sim.max(dim=1).values
        sim_shifted = sim - row_max.unsqueeze(1)

        exp_sim = torch.exp(sim_shifted)
        pos_exp = torch.exp(pos - row_max)
        neg_exp_sum = torch.sum(weights * exp_sim, dim=1)
        denom = pos_exp + neg_exp_sum

        loss = -pos + row_max + torch.log(denom.clamp(min=1e-12))
        return loss.mean()

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        sem_probs: torch.Tensor = None,
        repulsion_min: float = 1e-4,
        repulsion_max: float = 1.0,
    ) -> torch.Tensor:
        """Compute InfoNCE loss between two views.
        
        Args:
            z1, z2: Embeddings [N, D]
            sem_probs: Optional semantic distribution [N, C] for weighting
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.mm(z1, z2.t()) / self.temperature

        if sem_probs is None:
            loss_1to2 = F.cross_entropy(sim, torch.arange(z1.size(0), device=z1.device))
            loss_2to1 = F.cross_entropy(sim.t(), torch.arange(z1.size(0), device=z1.device))
            return (loss_1to2 + loss_2to1) / 2

        sem_probs_norm = F.normalize(sem_probs, p=2, dim=1)
        sem_sim = torch.mm(sem_probs_norm, sem_probs_norm.t())
        repulsion_weights = torch.clamp(
            1.0 - sem_sim, min=float(repulsion_min), max=float(repulsion_max)
        )

        loss_1to2 = self._weighted_loss_oneway(sim, repulsion_weights)
        loss_2to1 = self._weighted_loss_oneway(sim.t(), repulsion_weights.t())
        return (loss_1to2 + loss_2to1) / 2


class NTXentLoss(nn.Module):
    """NT-Xent: Normalized Temperature-scaled Cross Entropy Loss."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)
        sim = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2,
        ) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0)

        mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
        negatives = sim[~mask].view(2 * batch_size, -1)

        logits = torch.cat([positive_samples.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, device=z1.device, dtype=torch.long)

        return F.cross_entropy(logits, labels)


class DGILoss(nn.Module):
    """DGI: Deep Graph Infomax loss."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        labels_pos = torch.ones_like(pos_scores)
        labels_neg = torch.zeros_like(neg_scores)
        loss_pos = self.loss_fn(pos_scores, labels_pos)
        loss_neg = self.loss_fn(neg_scores, labels_neg)
        return loss_pos + loss_neg


class ContrastiveLoss(nn.Module):
    """Generic contrastive loss selector."""

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
