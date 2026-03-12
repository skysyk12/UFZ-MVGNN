"""
Feature Registry Module

Implements a decorator-based registration pattern for feature calculation functions.

Usage:
    @FeatureRegistry.register('shape')
    def calculate_my_feature(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf['my_feature'] = ...
        return gdf
"""

import logging
from typing import Callable, Dict, List

import geopandas as gpd

logger = logging.getLogger(__name__)

FeatureFunc = Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame]


class FeatureRegistry:
    """
    Central registry for feature calculation functions.

    Features are organized into groups (e.g., 'shape', 'size', 'orientation').
    Each group can contain multiple calculation functions that are executed
    sequentially on the GeoDataFrame.
    """

    _registry: Dict[str, List[FeatureFunc]] = {}

    @classmethod
    def register(cls, group: str):
        """
        Decorator to register a feature calculation function.

        Args:
            group: The feature group name (e.g., 'shape', 'size', 'orientation')

        Example:
            @FeatureRegistry.register('shape')
            def calculate_area(gdf):
                gdf['area'] = gdf.geometry.area
                return gdf
        """
        def decorator(func: FeatureFunc) -> FeatureFunc:
            if group not in cls._registry:
                cls._registry[group] = []
            cls._registry[group].append(func)
            return func
        return decorator

    @classmethod
    def get_functions(cls, groups: List[str]) -> List[FeatureFunc]:
        """
        Get all registered functions for the specified groups.

        Args:
            groups: List of group names to retrieve functions for

        Returns:
            List of feature calculation functions in registration order
        """
        funcs = []
        for group in groups:
            if group in cls._registry:
                funcs.extend(cls._registry[group])
            else:
                logger.warning("Feature group '%s' not found in registry.", group)
        return funcs

    @classmethod
    def list_groups(cls) -> List[str]:
        """List all registered feature groups."""
        return list(cls._registry.keys())

    @classmethod
    def list_functions(cls, group: str) -> List[str]:
        """List all function names in a group."""
        if group not in cls._registry:
            return []
        return [f.__name__ for f in cls._registry[group]]

    @classmethod
    def clear(cls):
        """Clear all registrations (useful for testing)."""
        cls._registry.clear()
