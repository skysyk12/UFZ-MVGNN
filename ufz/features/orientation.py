"""
Orientation Features (4D)

- sbr_orientation_degrees: PCA-based main axis orientation (0-180)
- longest_chord_orientation_degrees: convex hull diameter direction
- bisector_orientation_degrees: perpendicular to longest chord
- weighted_edge_orientation_degrees: edge-length-weighted circular mean
"""

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from .registry import FeatureRegistry


@FeatureRegistry.register('orientation')
def calculate_sbr_orientation(gdf):
    """PCA-based orientation (0-180 degrees)."""
    orientations = []

    for geometry in gdf['geometry']:
        if geometry is None or geometry.is_empty:
            orientations.append(np.nan)
            continue

        coords = []
        try:
            if isinstance(geometry, Polygon):
                coords.extend(list(geometry.exterior.coords))
            elif isinstance(geometry, MultiPolygon):
                for part in geometry.geoms:
                    coords.extend(list(part.exterior.coords))
            else:
                orientations.append(np.nan)
                continue

            if len(coords) > 1:
                points = np.array(coords)
                centered = points - np.mean(points, axis=0)
                cov = np.cov(centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                principal = eigenvectors[:, np.argmax(eigenvalues)]
                deg = np.degrees(np.arctan2(principal[1], principal[0]))
                orientations.append(np.mod(deg, 180))
            else:
                orientations.append(np.nan)
        except Exception:
            orientations.append(np.nan)

    gdf['sbr_orientation_degrees'] = orientations
    return gdf


@FeatureRegistry.register('orientation')
def calculate_hull_orientations(gdf):
    """Convex hull longest chord direction and its perpendicular bisector."""
    chord_orientations = []
    bisector_orientations = []

    for geometry in gdf['geometry']:
        if geometry is None or geometry.is_empty:
            chord_orientations.append(np.nan)
            bisector_orientations.append(np.nan)
            continue

        try:
            convex_hull = geometry.convex_hull
            if isinstance(convex_hull, MultiPolygon):
                if not convex_hull.geoms:
                    raise ValueError("Empty MultiPolygon hull")
                convex_hull = max(convex_hull.geoms, key=lambda p: p.area)

            if not isinstance(convex_hull, Polygon):
                chord_orientations.append(np.nan)
                bisector_orientations.append(np.nan)
                continue

            vertices = np.array(convex_hull.exterior.coords)
        except Exception:
            chord_orientations.append(np.nan)
            bisector_orientations.append(np.nan)
            continue

        if len(vertices) <= 1:
            chord_orientations.append(np.nan)
            bisector_orientations.append(np.nan)
            continue

        # Brute-force farthest pair (convex hull has few vertices)
        max_dist_sq = 0
        v1_best = v2_best = None
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist_sq = (vertices[i][0] - vertices[j][0]) ** 2 + (vertices[i][1] - vertices[j][1]) ** 2
                if dist_sq > max_dist_sq:
                    max_dist_sq = dist_sq
                    v1_best, v2_best = vertices[i], vertices[j]

        if v1_best is not None:
            dx = v2_best[0] - v1_best[0]
            dy = v2_best[1] - v1_best[1]
            chord_deg = np.degrees(np.arctan2(dy, dx))
            bisector_deg = chord_deg + 90
            chord_orientations.append(np.mod(chord_deg, 180))
            bisector_orientations.append(np.mod(bisector_deg, 180))
        else:
            chord_orientations.append(np.nan)
            bisector_orientations.append(np.nan)

    gdf['longest_chord_orientation_degrees'] = chord_orientations
    gdf['bisector_orientation_degrees'] = bisector_orientations
    return gdf


@FeatureRegistry.register('orientation')
def calculate_weighted_edge_orientation(gdf):
    """
    Edge-length-weighted average orientation using circular mean.

    Uses the doubling trick (2*theta) to correctly handle the 0-180 cyclic
    nature of directions (e.g., 1 deg and 179 deg are nearly parallel).
    """
    orientations = []

    for geometry in gdf['geometry']:
        if geometry is None or geometry.is_empty:
            orientations.append(np.nan)
            continue

        vertices = []
        try:
            if isinstance(geometry, Polygon):
                vertices.extend(list(geometry.exterior.coords))
            elif isinstance(geometry, MultiPolygon):
                for part in geometry.geoms:
                    vertices.extend(list(part.exterior.coords))
        except Exception:
            orientations.append(np.nan)
            continue

        sum_sin = sum_cos = total_length = 0.0

        if len(vertices) > 1:
            for i in range(len(vertices) - 1):
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[i + 1])
                edge_len = np.linalg.norm(p2 - p1)
                if edge_len == 0:
                    continue
                theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                double_angle = 2 * theta
                sum_sin += edge_len * np.sin(double_angle)
                sum_cos += edge_len * np.cos(double_angle)
                total_length += edge_len

        if total_length > 0:
            mean_angle = np.arctan2(sum_sin, sum_cos) / 2.0
            orientations.append(np.mod(np.degrees(mean_angle), 180))
        else:
            orientations.append(np.nan)

    gdf['weighted_edge_orientation_degrees'] = orientations
    return gdf
