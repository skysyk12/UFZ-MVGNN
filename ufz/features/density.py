"""
Density Features

- covered_area_ratio: each polygon's area as a fraction of total area
"""

import logging

from .registry import FeatureRegistry

logger = logging.getLogger(__name__)


@FeatureRegistry.register('density')
def calculate_covered_area_ratio_from_gdf(gdf):
    """
    Compute the covered area ratio for each polygon.

    Defined as each polygon's area divided by the total area of all polygons
    in the GeoDataFrame.
    """
    total_area = gdf['geometry'].area.sum()

    if total_area > 0:
        gdf['covered_area_ratio'] = gdf['geometry'].area / total_area
    else:
        logger.warning("Total area is zero; cannot compute covered area ratio.")
        gdf['covered_area_ratio'] = 0.0

    return gdf
