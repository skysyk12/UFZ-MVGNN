"""Weighted KL divergence loss for long-tail POI category handling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedKLDivLoss(nn.Module):
    """Class-weighted KL Divergence: penalizes errors on rare categories."""

    def __init__(self, class_freq: torch.Tensor):
        super().__init__()
        freq = class_freq.float().clamp(min=1e-8)
        weights = 1.0 / torch.sqrt(freq)
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_is_logits: bool = False,
    ) -> torch.Tensor:
        if pred.size(0) == 0:
            return pred.sum() * 0.0

        if pred_is_logits:
            log_pred = F.log_softmax(pred, dim=-1)
        else:
            log_pred = torch.log(pred.clamp(min=1e-8))

        kl = F.kl_div(log_pred, target, reduction="none")
        kl = torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
        weighted_kl = kl * self.class_weights.unsqueeze(0)
        return weighted_kl.sum(dim=-1).mean()
