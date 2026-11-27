"""Neighbourhoods of areas (e.g. LSOAs) with a sparse adjacency matrix.

``A[i, j] = 1`` when areas i and j share a boundary. With ``M1 = A + I``,
the areas within k boundary-steps of i are the non-zeros of row i of
``M1 ** k``. A neighbourhood average of value v weighted by w (e.g.
population) is then just two sparse matrix-vector products::

    avg_k = (Mk @ (v * w)) / (Mk @ w)

which does all 35,000 LSOAs at once in compiled code, instead of one graph
search per area.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def contiguity(geoms, tolerance: float = 1.0):
    """Queen contiguity (shared edge or corner) as a CSR matrix.

    ``tolerance`` metres lets generalised boundaries that should touch but have
    hairline gaps still count as neighbours.
    """
    import shapely
    from scipy.sparse import coo_matrix
    geoms = np.asarray(geoms)
    tree = shapely.STRtree(geoms)
    i, j = tree.query(geoms, predicate="dwithin", distance=tolerance)
    keep = i != j
    n = len(geoms)
    A = coo_matrix((np.ones(keep.sum(), dtype=np.float32), (i[keep], j[keep])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    log.info("adjacency %s areas, %.1f neighbours on average", f"{n:,}", A.nnz / max(n, 1))
    return A


def road_bridges(geoms, roads, A=None, min_gap_m: float = 5.0):
    """Extra links between areas that do not touch but are joined by a road (bridges and tunnels
    over rivers and estuaries). A road link whose two ends fall in different, non-adjacent areas,
    and which runs outside every area for more than ``min_gap_m`` (i.e. over water), joins them.
    ``roads`` is an array of LineStrings in the same CRS. Returns (CSR matrix of the new links,
    array of road indices that made them)."""
    import shapely
    from scipy.sparse import coo_matrix
    geoms = np.asarray(geoms)
    roads = np.asarray(roads)
    n = len(geoms)
    tree = shapely.STRtree(geoms)

    def area_of(pts):
        out = np.full(len(pts), -1, dtype=np.int64)
        p_idx, a_idx = tree.query(pts, predicate="within")
        out[p_idx] = a_idx
        return out

    a = area_of(shapely.get_point(roads, 0))
    b = area_of(shapely.get_point(roads, -1))
    ok = (a >= 0) & (b >= 0) & (a != b)
    if A is not None and ok.any():   # drop pairs that already touch
        ok[ok] = np.asarray(A[a[ok], b[ok]]).ravel() == 0
    # a bridge must actually leave the areas (cross water or a gap), not just pass through a
    # narrow third area: keep links with more than `min_gap_m` of their length outside every area
    cand = np.flatnonzero(ok)
    if len(cand):
        r_idx, g_idx = tree.query(roads[cand], predicate="intersects")
        inside = np.bincount(r_idx, weights=shapely.length(shapely.intersection(roads[cand][r_idx], geoms[g_idx])),
                             minlength=len(cand))
        outside = shapely.length(roads[cand]) - inside
        ok[cand[outside <= min_gap_m]] = False
    ia, ib = a[ok], b[ok]
    B = coo_matrix((np.ones(2 * len(ia), dtype=np.float32), (np.r_[ia, ib], np.r_[ib, ia])), shape=(n, n)).tocsr()
    B.data[:] = 1.0
    log.info("bridges %s new links between areas that do not touch (%s road links)", f"{B.nnz // 2:,}",
             f"{int(ok.sum()):,}")
    return B, np.flatnonzero(ok)


def k_step(A, k: int):
    """Boolean reachability within k steps (including the area itself)."""
    from scipy.sparse import identity
    M1 = (A + identity(A.shape[0], format="csr", dtype=np.float32)).tocsr()
    M1.data[:] = 1.0
    M = M1
    for _ in range(k - 1):
        M = (M @ M1).tocsr()
        M.data[:] = 1.0
    return M


def neighbourhood_mean(M, values: np.ndarray, weights: np.ndarray | None = None):
    v = np.asarray(values, dtype="f8")
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype="f8")
    ok = np.isfinite(v) & np.isfinite(w)
    num = M @ np.where(ok, v * w, 0.0)
    den = M @ np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def neighbourhood_sum(M, values: np.ndarray):
    v = np.asarray(values, dtype="f8")
    return M @ np.where(np.isfinite(v), v, 0.0)
