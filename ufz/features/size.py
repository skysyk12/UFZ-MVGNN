"""
Size Features (2D + height)

- longest_chord_length: polygon diameter via convex hull
- mean_radius: average vertex-to-centroid distance
"""

import numpy as np
from scipy.spatial.distance import pdist
from shapely.geometry import MultiPolygon, Polygon

from .registry import FeatureRegistry


@FeatureRegistry.register('size')
def calculate_longest_chord(gdf):
    """
    Compute the longest chord (diameter) for each polygon.

    Uses the convex hull to reduce vertex count before pairwise distance
    calculation, making this safe even for high-vertex geometries.
    """
    longest_chords = []

    for _, row in gdf.iterrows():
        geometry = row['geometry']
        if geometry is None or geometry.is_empty:
            longest_chords.append(np.nan)
            continue

        hull = geometry.convex_hull
        vertices = []
        try:
            if hasattr(hull, 'exterior') and hull.exterior is not None:
                vertices = np.array(hull.exterior.coords)
            elif hasattr(hull, 'geoms'):
                all_coords = []
                for geom in hull.geoms:
                    if hasattr(geom, 'exterior') and geom.exterior:
                        all_coords.append(np.array(geom.exterior.coords))
                if all_coords:
                    vertices = np.vstack(all_coords)
            elif hasattr(hull, 'coords'):
                vertices = np.array(hull.coords)
        except Exception:
            longest_chords.append(np.nan)
            continue

        if len(vertices) > 1:
            try:
                distances = pdist(vertices, 'euclidean')
                longest_chords.append(np.max(distances) if len(distances) > 0 else 0)
            except Exception:
                longest_chords.append(np.nan)
        else:
            longest_chords.append(0)

    gdf['longest_chord_length'] = longest_chords
    return gdf


@FeatureRegistry.register('size')
def mean_radius(gdf):
    """
    Compute the mean distance from each vertex to the polygon centroid.

    Uses NumPy broadcasting for vectorized distance computation,
    ~10-100x faster than naive Python loops with Shapely Point objects.
    """
    avg_distances = []

    for _, row in gdf.iterrows():
        geometry = row['geometry']
        if geometry is None or geometry.is_empty:
            avg_distances.append(np.nan)
            continue

        cx, cy = geometry.centroid.x, geometry.centroid.y

        coords_list = []
        try:
            if isinstance(geometry, Polygon):
                coords_list.append(np.array(geometry.exterior.coords))
                for interior in geometry.interiors:
                    coords_list.append(np.array(interior.coords))
            elif isinstance(geometry, MultiPolygon):
                for part in geometry.geoms:
                    coords_list.append(np.array(part.exterior.coords))
                    for interior in part.interiors:
                        coords_list.append(np.array(interior.coords))
        except AttributeError:
            avg_distances.append(np.nan)
            continue

        if coords_list:
            all_points = np.vstack(coords_list)
            dists = np.linalg.norm(all_points - np.array([cx, cy]), axis=1)
            avg_distances.append(np.mean(dists))
        else:
            avg_distances.append(0)

    gdf['mean_radius'] = avg_distances
    return gdf
