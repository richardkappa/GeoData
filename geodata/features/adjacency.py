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
