"""
Feature Manager

Orchestrates feature calculation by retrieving registered functions
and applying them sequentially to GeoDataFrames.
"""

import logging
from typing import List

import geopandas as gpd

from .registry import FeatureRegistry

# Import feature modules to trigger @register decorators
from . import density, orientation, shape, size  # noqa: F401

logger = logging.getLogger(__name__)


class FeatureManager:
    """
    Manages feature calculation for GeoDataFrames.

    Retrieves registered feature functions based on the requested groups
    and applies them sequentially to the input data.
    """

    def __init__(self, groups: List[str]):
        """
        Args:
            groups: Feature group names to activate (e.g., ['shape', 'size']).
        """
        self.groups = groups

    def calculate_features(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate all registered features for the given GeoDataFrame.

        Args:
            gdf: Input GeoDataFrame with a geometry column.

        Returns:
            GeoDataFrame with additional feature columns appended.
        """
        funcs = FeatureRegistry.get_functions(self.groups)

        for func in funcs:
            logger.info("Calculating: %s", func.__name__)
            gdf = func(gdf)

        return gdf

    def get_feature_names(self) -> List[str]:
        """Get the names of all functions that will be executed."""
        return [f.__name__ for f in FeatureRegistry.get_functions(self.groups)]
