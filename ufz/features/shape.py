"""
Shape Features (13D)

Computes 13 morphological shape descriptors for building footprint polygons:
Area, Perimeter, Complexity, IPQ, Fractality, Max Circularity, Gibbs Compactness,
Elongation, Ellipticity, Concavity, DCM, Exchange Index, BCI, Moments of Inertia,
Eccentricity.
"""

import warnings
from math import cos, sin

import numpy as np
import shapely
from scipy.spatial import distance
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from .registry import FeatureRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_base_attributes(geometry):
    """Extract base geometric attributes, handling MultiPolygon by keeping the largest part."""
    if geometry is None or geometry.is_empty:
        return np.nan, np.nan, np.nan, np.nan, None

    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return np.nan, np.nan, np.nan, np.nan, None

    if isinstance(geometry, MultiPolygon):
        valid_polygons = [p for p in geometry.geoms if isinstance(p, Polygon) and p.area > 0]
        if not valid_polygons:
            return np.nan, np.nan, np.nan, np.nan, None
        geometry = max(valid_polygons, key=lambda p: p.area)

    if geometry.area <= 0:
        return np.nan, np.nan, np.nan, np.nan, None

    A = geometry.area
    P = geometry.length
    A_CH = geometry.convex_hull.area
    A_SCC = shapely.minimum_bounding_circle(geometry).area
    return A, P, A_CH, A_SCC, geometry


def _resolve_polygon(geometry):
    """Resolve geometry to a single Polygon (largest part if MultiPolygon)."""
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, MultiPolygon):
        valid = [p for p in geometry.geoms if isinstance(p, Polygon) and p.area > 0]
        return max(valid, key=lambda p: p.area) if valid else None
    if isinstance(geometry, Polygon):
        return geometry
    return None


# ---------------------------------------------------------------------------
# Registered feature functions
# ---------------------------------------------------------------------------

@FeatureRegistry.register('shape')
def calculate_area_perimeter(gdf):
    """Compute polygon area and perimeter."""
    gdf['Area'] = gdf.geometry.area
    gdf['Perimeter'] = gdf.geometry.length
    return gdf


@FeatureRegistry.register('shape')
def calculate_complexity_index(gdf):
    """Complexity index (A / P)."""
    results = []
    for _, row in gdf.iterrows():
        A, P, _, _, _ = _get_base_attributes(row['geometry'])
        results.append(A / P if P and P > 0 else np.nan)
    gdf['Complexity_index'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_ipq(gdf):
    """Isoperimetric quotient (4piA / P^2)."""
    results = []
    for _, row in gdf.iterrows():
        A, P, _, _, _ = _get_base_attributes(row['geometry'])
        results.append(4 * np.pi * A / (P ** 2) if P and P > 0 else np.nan)
    gdf['IPQ'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_fractality(gdf):
    """Fractal dimension (1 - log(A) / 2log(P))."""
    results = []
    for _, row in gdf.iterrows():
        A, P, _, _, _ = _get_base_attributes(row['geometry'])
        if A and A > 0 and P and P > 0:
            results.append(1 - np.log10(A) / (2 * np.log10(P)))
        else:
            results.append(np.nan)
    gdf['Fractality'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_max_circularity(gdf):
    """Max circularity (sqrt(A/pi) / r_max)."""
    results = []
    for _, row in gdf.iterrows():
        A, _, _, _, geometry = _get_base_attributes(row['geometry'])
        if A and A > 0 and geometry is not None:
            centroid = geometry.centroid
            vertices = np.array(list(geometry.exterior.coords))
            distances = distance.cdist([centroid.coords[0]], vertices, 'euclidean')
            r_max = np.max(distances)
            results.append(np.sqrt(A / np.pi) / r_max if r_max > 0 else np.nan)
        else:
            results.append(np.nan)
    gdf['Max_circularity'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_gibbs_compactness(gdf):
    """Gibbs compactness (4A / piL_max^2)."""
    results = []
    for _, row in gdf.iterrows():
        A, _, _, _, geometry = _get_base_attributes(row['geometry'])
        if A and A > 0 and isinstance(geometry, Polygon):
            convex_hull = geometry.convex_hull
            if not isinstance(convex_hull, Polygon):
                results.append(np.nan)
                continue
            vertices = np.array(list(convex_hull.exterior.coords))
            if len(vertices) > 1:
                L_max = np.max(distance.pdist(vertices, 'euclidean'))
                results.append(4 * A / (np.pi * L_max ** 2) if L_max > 0 else np.nan)
            else:
                results.append(np.nan)
        else:
            results.append(np.nan)
    gdf['Gibbs_compactness'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_elongation_index(gdf):
    """Elongation index (L_SBR / W_SBR) from minimum rotated rectangle."""
    results = []
    for _, row in gdf.iterrows():
        geometry = row['geometry']
        if geometry is None or geometry.is_empty:
            results.append(np.nan)
            continue
        sbr = geometry.minimum_rotated_rectangle
        if sbr is None or sbr.is_empty:
            results.append(np.nan)
            continue
        try:
            coords = list(sbr.exterior.coords)[:4]
            if len(coords) < 4:
                results.append(np.nan)
                continue
            side_lengths = [
                float(Point(coords[i]).distance(Point(coords[(i + 1) % 4])))
                for i in range(4)
            ]
            side_lengths = [s for s in side_lengths if s > 0]
            if not side_lengths:
                results.append(np.nan)
                continue
            results.append(max(side_lengths) / min(side_lengths) if min(side_lengths) > 0 else np.nan)
        except Exception:
            results.append(np.nan)
    gdf['Elongation_index'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_ellipticity(gdf):
    """Ellipticity (L_width / L_max)."""
    results = []
    for _, row in gdf.iterrows():
        geometry = _resolve_polygon(row['geometry'])
        if geometry is None:
            results.append(np.nan)
            continue

        convex_hull = geometry.convex_hull
        if not isinstance(convex_hull, Polygon):
            results.append(np.nan)
            continue
        vertices = np.array(list(convex_hull.exterior.coords))
        if len(vertices) < 2:
            results.append(np.nan)
            continue

        # Find longest chord (L_max)
        max_dist_sq = 0
        v1_furthest = v2_furthest = None
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist_sq = (vertices[i][0] - vertices[j][0]) ** 2 + (vertices[i][1] - vertices[j][1]) ** 2
                if dist_sq > max_dist_sq:
                    max_dist_sq = dist_sq
                    v1_furthest, v2_furthest = vertices[i], vertices[j]

        L_max = np.sqrt(max_dist_sq)
        if L_max == 0 or v1_furthest is None:
            results.append(np.nan)
            continue

        # Maximum perpendicular distance from chord (L_width)
        chord_vector = v2_furthest - v1_furthest
        L_width = 0
        for vertex in vertices:
            vec = vertex - v1_furthest
            cross_z = chord_vector[0] * vec[1] - chord_vector[1] * vec[0]
            d = abs(cross_z) / L_max
            if d > L_width:
                L_width = d

        results.append(L_width / L_max)
    gdf['Ellipticity'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_concavity(gdf):
    """Concavity (A / A_CH)."""
    results = []
    for _, row in gdf.iterrows():
        A, _, A_CH, _, _ = _get_base_attributes(row['geometry'])
        results.append(A / A_CH if A_CH and A_CH > 0 else np.nan)
    gdf['Concavity'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_dcm(gdf):
    """Digital compactness measure (A / A_SCC)."""
    results = []
    for _, row in gdf.iterrows():
        A, _, _, A_SCC, _ = _get_base_attributes(row['geometry'])
        results.append(A / A_SCC if A_SCC and A_SCC > 0 else np.nan)
    gdf['DCM'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_exchange_index(gdf):
    """Exchange index (1 - A intersection EAC / A_EAC)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning, module='shapely')
    results = []
    for _, row in gdf.iterrows():
        A, P, _, _, geometry = _get_base_attributes(row['geometry'])
        if geometry is None or geometry.is_empty:
            results.append(np.nan)
            continue
        try:
            r_EAC = np.sqrt(A / np.pi)
            eac = geometry.centroid.buffer(r_EAC)
            A_EAC = eac.area
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if geometry.is_valid and eac.is_valid and A_EAC > 0:
                A_intersection = geometry.intersection(eac).area
                results.append(1 - A_intersection / A_EAC)
            else:
                results.append(np.nan)
        except Exception:
            results.append(np.nan)
    gdf['Exchange_index'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_bci(gdf, angle_step=10):
    """Boyce-Clark shape index using radial method."""
    results = []
    angles_rad = np.radians(np.arange(0, 360, angle_step))

    for _, row in gdf.iterrows():
        geometry = _resolve_polygon(row['geometry'])
        if geometry is None or geometry.exterior is None:
            results.append(np.nan)
            continue

        centroid = geometry.centroid
        cx, cy = centroid.coords[0]

        minx, miny, maxx, maxy = geometry.bounds
        ray_length = max(maxx - minx, maxy - miny) * 1.5
        r_list = []

        for angle in angles_rad:
            end_point = Point(cx + cos(angle) * ray_length, cy + sin(angle) * ray_length)
            ray = LineString([centroid, end_point])
            intersection = ray.intersection(geometry.exterior)

            def flatten_geom(geom):
                if geom.geom_type == 'Point':
                    return [geom]
                elif geom.geom_type in ('MultiPoint', 'MultiLineString', 'GeometryCollection'):
                    pts = []
                    for g in geom.geoms:
                        pts.extend(flatten_geom(g))
                    return pts
                elif geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    if len(coords) >= 2:
                        return [Point(coords[0]), Point(coords[-1])]
                    elif len(coords) == 1:
                        return [Point(coords[0])]
                    return []
                return []

            min_r = float('inf')
            for p in flatten_geom(intersection):
                try:
                    r = centroid.distance(p)
                    if r < min_r:
                        min_r = r
                except Exception:
                    continue

            if min_r != float('inf') and min_r > 1e-9:
                r_list.append(min_r)

        if len(r_list) < 3:
            results.append(np.nan)
            continue

        n = len(r_list)
        sum_r = sum(r_list)
        if sum_r == 0:
            results.append(np.nan)
        else:
            average_ratio = 100 / n
            bci = sum(abs((r / sum_r) * 100 - average_ratio) for r in r_list)
            results.append(bci)

    gdf['BCI'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_moments_of_inertia(gdf):
    """Moments of inertia (mu_11 = sum(x_i * y_i))."""
    results = []
    for _, row in gdf.iterrows():
        geometry = _resolve_polygon(row['geometry'])
        if geometry is None or geometry.exterior is None:
            results.append(np.nan)
            continue
        vertices = np.array(list(geometry.exterior.coords))
        if len(vertices) > 1:
            centered = vertices - vertices.mean(axis=0)
            results.append(np.sum(centered[:, 0] * centered[:, 1]))
        else:
            results.append(np.nan)
    gdf['Moments_of_inertia'] = results
    return gdf


@FeatureRegistry.register('shape')
def calculate_eccentricity(gdf):
    """Eccentricity (r_max / r_min)."""
    results = []
    for _, row in gdf.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            results.append(np.nan)
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry is None or geometry.is_empty:
            results.append(np.nan)
            continue

        geometry = _resolve_polygon(geometry)
        if geometry is None or geometry.exterior is None:
            results.append(np.nan)
            continue

        centroid = geometry.centroid
        center = np.array(centroid.coords[0])
        vertices = np.array(geometry.exterior.coords)

        r_list = []
        # Centroid -> vertices
        if len(vertices) > 1:
            r_list.extend(distance.cdist([center], vertices, 'euclidean').flatten())

        # Centroid -> edge projections
        for i in range(len(vertices) - 1):
            p1, p2 = vertices[i], vertices[i + 1]
            edge_vec = p2 - p1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq == 0:
                r_list.append(np.linalg.norm(center - p1))
            else:
                t = np.clip(np.dot(edge_vec, center - p1) / edge_len_sq, 0.0, 1.0)
                proj = p1 + t * edge_vec
                r_list.append(np.linalg.norm(center - proj))

        if r_list:
            r_max, r_min = np.max(r_list), np.min(r_list)
            results.append(r_max / r_min if r_min > 0 else np.nan)
        else:
            results.append(np.nan)

    gdf['Eccentricity'] = results
    return gdf
