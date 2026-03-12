"""Test semantic enhancement components."""

import pytest
import torch
import numpy as np
from ufz.semantic import CrossViewImputer, RefineNet, WeightedKLDivLoss


def test_imputer():
    """Test CrossViewImputer."""
    model = CrossViewImputer(in_dim=36, hidden_dim=128, num_classes=17)
    x = torch.randn(50, 36)
    edge_index = torch.randint(0, 50, (2, 100))
    
    pred = model(x, edge_index)
    assert pred.shape == (50, 17)
    assert (pred.sum(dim=1) - 1).abs().max() < 1e-5  # Normalized


def test_refine_net():
    """Test RefineNet."""
    model = RefineNet(phys_dim=36, sem_dim=17, hidden_dim=128)
    x_phys = torch.randn(50, 36)
    x_sem = torch.randn(50, 17)
    edge_index = torch.randint(0, 50, (2, 100))
    
    logits = model(x_phys, x_sem, edge_index)
    assert logits.shape == (50, 17)
    
    gates = model.get_gate_values(x_phys, x_sem, edge_index)
    assert gates.shape == (50,)
    assert (gates >= 0).all() and (gates <= 1).all()


def test_weighted_kl_loss():
    """Test weighted KL divergence loss."""
    class_freq = torch.tensor([100, 50, 10, 5])
    loss_fn = WeightedKLDivLoss(class_freq)
    
    pred = torch.softmax(torch.randn(10, 4), dim=1)
    target = torch.softmax(torch.randn(10, 4), dim=1)
    
    loss = loss_fn(pred, target)
    assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
