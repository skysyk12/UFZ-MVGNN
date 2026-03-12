"""Test model components."""

import pytest
import torch
from ufz.models import MVCLModel, BackboneRegistry
from ufz.models.backbones import GINEncoder, GATEncoder


def test_backbone_registry():
    """Test backbone registry."""
    assert "gin" in BackboneRegistry.list_available()
    assert "gat" in BackboneRegistry.list_available()
    
    GINClass = BackboneRegistry.get("gin")
    assert GINClass == GINEncoder


def test_gin_encoder():
    """Test GIN encoder."""
    encoder = GINEncoder(in_dim=32, hidden_dim=64, out_dim=128, num_layers=2)
    x = torch.randn(100, 32)
    edge_index = torch.randint(0, 100, (2, 200))
    
    out = encoder(x, edge_index)
    assert out.shape == (100, 128)


def test_mvcl_model():
    """Test MVCL model."""
    model = MVCLModel(
        physical_dim=36,
        semantic_dim=17,
        backbone="gin",
        repr_dim=128,
    )
    
    x_phys = torch.randn(100, 36)
    x_sem = torch.randn(100, 17)
    edge_index = torch.randint(0, 100, (2, 200))
    
    h_phys, h_sem, z_phys, z_sem = model(x_phys, x_sem, edge_index)
    
    assert h_phys.shape == (100, 128)
    assert h_sem.shape == (100, 128)
    assert z_phys.shape == (100, 128)
    assert z_sem.shape == (100, 128)


def test_mvcl_loss():
    """Test MVCL loss computation."""
    model = MVCLModel(physical_dim=36, semantic_dim=17)
    z_phys = torch.randn(32, 128)
    z_sem = torch.randn(32, 128)
    
    loss = model.compute_loss(z_phys, z_sem)
    assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
