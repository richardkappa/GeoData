"""Distance from each postcode to the nearest feature (metres).

* points            exact, scipy cKDTree (C++), all cores
* lines / polygons  vertices densified every ``step`` m (GEOS ``segmentize``)
                    then a KD-tree: accurate to ~step/2 and far faster than
                    exact geometry distance for millions of postcodes
* polygons          0 when the postcode is inside (STRtree point-in-polygon)
"""
from __future__ import annotations

import numpy as np


def nearest_points(px, py, qx, qy, k: int = 1):
    from scipy.spatial import cKDTree
    if len(qx) == 0:
        return np.full(len(px), np.nan), np.full(len(px), -1)
    tree = cKDTree(np.column_stack([qx, qy]))
    d, i = tree.query(np.column_stack([px, py]), k=k, workers=-1)
    return d, i


def sample_lines(geoms, step: float = 25.0) -> np.ndarray:
    import shapely
    if len(geoms) == 0:
        return np.empty((0, 2))
    return shapely.get_coordinates(shapely.segmentize(np.asarray(geoms), step))


def nearest_lines(px, py, geoms, step: float = 25.0):
    pts = sample_lines(geoms, step)
    d, i = nearest_points(px, py, pts[:, 0], pts[:, 1])
    return d, pts, i


def inside_any(px, py, polys) -> np.ndarray:
    import shapely
    inside = np.zeros(len(px), dtype=bool)
    if len(polys) == 0:
        return inside
    tree = shapely.STRtree(shapely.points(px, py))
    _, p_idx = tree.query(np.asarray(polys), predicate="intersects")
    inside[np.unique(p_idx)] = True
    return inside


def nearest_polygons(px, py, polys, step: float = 25.0):
    import shapely
    if len(polys) == 0:
        return np.full(len(px), np.nan)
    d, _, _ = nearest_lines(px, py, shapely.boundary(np.asarray(polys)), step)
    d = np.asarray(d, dtype="f8")
    d[inside_any(px, py, polys)] = 0.0
    return d
