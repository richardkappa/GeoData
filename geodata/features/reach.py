"""What can be reached over the road network within a time limit, for the whole country.

For each measure (configured under ``network.reach.measures`` in config/features.yaml), every postcode
gets the number of points - or the sum of a weight column - on roads reachable within N minutes'
drive or walk:

    collisions_within_30min_drive: {mode: drive, minutes: 30, points: collisions}
    casualties_within_30min_drive: {mode: drive, minutes: 30, points: collisions, weight: number_of_casualties}
    bus_stops_within_10min_walk:   {mode: walk,  minutes: 10, points: bus_stops}

``points`` names an entry under ``points:`` (file + filter). Each point is placed on the nearest road
junction within ``snap_m``. Drive times use typical free-flow speeds by road class (as the other
drive-time features); walking uses ``walk.speed_kmh`` on every road except motorways.

One search per postcode would be millions of searches, so the starting points are the national grid
squares (``cell_m``, 500 m) that contain postcodes, each starting from the junction nearest the middle
of its postcodes; every postcode takes its square's value. Searches are exact but local: nothing more
than ``minutes x top speed`` away can be reached, so the country is cut into tiles and each tile's
searches use only the roads within that distance of it. Tiles run in parallel worker processes.
Measures with the same mode and time limit share one set of searches (one weight column each).
"""
from __future__ import annotations

import logging
import os
import tempfile
import time

import numpy as np

log = logging.getLogger(__name__)

_G: dict = {}          # per worker process: the graph and weights (loaded once from the .npz)


def _init(path: str):
    z = np.load(path)
    _G.update({k: z[k] for k in z.files})


def _tile(job):
    """Searches from the source nodes of one tile over the roads inside `box`.
    Returns (source node ids, sums of each weight column over the nodes reached)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    box, src, limit, chunk = job
    xy, u, v, w, W = _G["xy"], _G["u"], _G["v"], _G["w"], _G["W"]
    inside = (xy[:, 0] >= box[0]) & (xy[:, 0] <= box[2]) & (xy[:, 1] >= box[1]) & (xy[:, 1] <= box[3])
    m = inside[u] & inside[v]
    nodes = np.flatnonzero(inside)
    lu, lv = np.searchsorted(nodes, u[m]), np.searchsorted(nodes, v[m])
    n = len(nodes)
    A = csr_matrix((np.r_[w[m], w[m]], (np.r_[lu, lv], np.r_[lv, lu])), shape=(n, n))
    ls = np.searchsorted(nodes, src)
    wt = W[nodes].astype("f8")
    out = np.zeros((len(src), W.shape[1]))
    for i in range(0, len(ls), chunk):
        d = dijkstra(A, directed=True, indices=ls[i:i + chunk], limit=limit)
        out[i:i + chunk] = (d <= limit).astype("f8") @ wt
    return src, out


def simple_edges(u, v, secs, mask=None):
    """One link per node pair (the quickest) and no loops: a sparse matrix would add parallel links'
    times together."""
    keep = (u != v) if mask is None else (u != v) & mask
    a, b, s = np.minimum(u, v)[keep], np.maximum(u, v)[keep], secs[keep]
    o = np.lexsort((s, b, a))
    a, b, s = a[o], b[o], s[o]
    first = np.r_[True, (a[1:] != a[:-1]) | (b[1:] != b[:-1])]
    return a[first], b[first], s[first]


def node_weights(tree, nodes, px, py, w, n, snap_m=200.0):
    """Points -> per-node totals: each point on its nearest node (of `nodes`) within snap_m."""
    d, i = tree.query(np.column_stack([px, py]), workers=-1)
    ok = d <= snap_m
    return np.bincount(nodes[i[ok]], weights=w[ok], minlength=n), int(ok.sum())


def reach_sums(node_xy, u, v, secs, W, pc_x, pc_y, *, minutes, top_kmh, nodes=None, cell_m=500.0,
               tile_m=20000.0, max_hop_m=3000.0, workers=None, chunk=32, label="reach"):
    """Per postcode: the sum of each column of W (n_nodes x k) over the nodes reachable within `minutes`,
    plus a per-square table. u/v/secs: undirected links (see simple_edges). `nodes`: the nodes the
    postcodes may start from (default: every node on a link). Postcodes whose square's start is more
    than max_hop_m away (e.g. Northern Ireland, no GB road data) get NaN."""
    from concurrent.futures import ProcessPoolExecutor
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    n = len(node_xy)
    nodes = np.unique(np.r_[u, v]) if nodes is None else nodes
    tree = cKDTree(node_xy[nodes])
    gx, gy = np.floor(pc_x / cell_m).astype(np.int64), np.floor(pc_y / cell_m).astype(np.int64)
    cells, inv = np.unique(gx * 10_000_000 + gy, return_inverse=True)
    cnt = np.bincount(inv)
    cx, cy = np.bincount(inv, weights=pc_x) / cnt, np.bincount(inv, weights=pc_y) / cnt
    hd, si = tree.query(np.column_stack([cx, cy]), workers=-1)
    src = nodes[si]
    ok = hd <= max_hop_m
    pad = minutes / 60 * top_kmh * 1000 + 1000          # beyond this nothing is reachable in time
    tk = np.floor(node_xy[src, 0] / tile_m).astype(np.int64) * 100_000 + np.floor(node_xy[src, 1] / tile_m).astype(np.int64)
    jobs = []
    for k in np.unique(tk[ok]):
        x0, y0 = (k // 100_000) * tile_m, (k % 100_000) * tile_m
        jobs.append(((x0 - pad, y0 - pad, x0 + tile_m + pad, y0 + tile_m + pad), np.unique(src[ok & (tk == k)]),
                     minutes * 60, chunk))
    jobs.sort(key=lambda j: -len(j[1]))                 # biggest first keeps the workers busy to the end
    res = np.full((n, W.shape[1]), np.nan)
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "graph.npz")
        np.savez(f, xy=node_xy.astype("f8"), u=u, v=v, w=secs.astype("f8"), W=W.astype("f4"))
        workers = int(workers or max(1, min(10, (os.cpu_count() or 2) - 2)))
        done = 0
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(f,)) as ex:
            for sn, out in ex.map(_tile, jobs):
                res[sn] = out
                done += 1
                if done % 100 == 0 or done == len(jobs):
                    log.info("%s: %d of %d tiles (%.0fs)", label, done, len(jobs), time.perf_counter() - t0)
    cell_vals = np.where(ok[:, None], res[src], np.nan)
    cells_tab = {"cell_x": (cells // 10_000_000) * cell_m + cell_m / 2, "cell_y": (cells % 10_000_000) * cell_m + cell_m / 2,
                 "start_x": node_xy[src, 0], "start_y": node_xy[src, 1], "postcodes": cnt}
    log.info("%s: %s squares in %.0fs", label, f"{len(cells):,}", time.perf_counter() - t0)
    return cell_vals[inv], cell_vals, cells_tab


def reach_at(node_xy, u, v, secs, W, src, *, minutes, top_kmh):
    """Exact sums for a few given nodes (single process; for checks and hotspot refinement)."""
    _G.update(xy=node_xy, u=u, v=v, w=secs, W=W)
    src = np.asarray(src)
    pad = minutes / 60 * top_kmh * 1000 + 1000
    xs, ys = node_xy[src, 0], node_xy[src, 1]
    box = (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)
    order = np.argsort(src)
    _, out = _tile((box, src[order], minutes * 60, 32))
    res = np.empty_like(out)
    res[order] = out
    return res
