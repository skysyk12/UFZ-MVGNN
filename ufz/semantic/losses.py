"""Weighted KL divergence loss for long-tail POI category handling."""

import torch
import torch.nn as nn


class WeightedKLDivLoss(nn.Module):
    """Class-weighted KL Divergence: penalizes errors on rare categories.

    Computes inverse-frequency weights so that rare POI categories
    contribute more to the loss, addressing long-tail distribution.

    Args:
        class_freq: [C] tensor of per-class sample counts
    """

    def __init__(self, class_freq: torch.Tensor):
        super().__init__()
        # Implementation withheld — will be released upon paper publication.
        raise NotImplementedError

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_is_logits: bool = False,
    ) -> torch.Tensor:
        """Compute weighted KL divergence.

        Args:
            pred: [N, C] predicted distribution or logits
            target: [N, C] target distribution
            pred_is_logits: If True, apply log_softmax to pred first
        """
        raise NotImplementedError
