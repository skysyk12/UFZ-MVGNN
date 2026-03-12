"""Backbone registry for dynamic encoder selection."""

import logging

logger = logging.getLogger(__name__)


class BackboneRegistry:
    """Registry pattern for backbone encoders."""

    _registry = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a backbone encoder."""
        def decorator(encoder_cls):
            cls._registry[name] = encoder_cls
            logger.info(f"Registered backbone: {name}")
            return encoder_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        """Get encoder class by name."""
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown backbone: {name}. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list_available(cls):
        """List all registered backbones."""
        return list(cls._registry.keys())
