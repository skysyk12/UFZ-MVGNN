"""GNN backbone encoders with registry pattern."""

from .base import BaseEncoder
from .registry import BackboneRegistry
from .gin import GINEncoder
from .gat import GATEncoder
from .gcn import GCNEncoder

__all__ = ["BaseEncoder", "BackboneRegistry", "GINEncoder", "GATEncoder", "GCNEncoder"]
