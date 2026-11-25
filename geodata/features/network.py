"""Travel distances and times over OS Open Roads.

The road network becomes a sparse matrix (scipy CSR): one row/column per road
node, one stored value per road link (its length in metres, or its travel time
in seconds). Shortest paths use scipy's C Dijkstra.

Two tricks keep this fast on a laptop:

* **Multi-source search with a virtual source node.** To get, for every road
  node, the distance to the *nearest* station, we add one extra node joined to
  every station's nearest road node (edge = the station's walk onto the road)
  and run Dijkstra once from it. One run, all postcodes, any number of stations.
* **Limited searches for counts.** "Stations within 15 minutes' walk" runs a
  small search from each station, stopped at the time limit, and adds 1 to
  every node it reaches.

Postcodes and targets are attached to their nearest road node (KD-tree); the
straight-line hop onto the network is added to the network distance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# typical speeds (km/h) by road class for drive times; overridable in config
DEFAULT_SPEEDS = {"Motorway": 100, "A Road": 70, "B Road": 55, "Classified Unnumbered": 45,
                  "Unclassified": 35, "Not Classified": 25, "Unknown": 25}


@dataclass
class RoadGraph:
    node_xy: np.ndarray      # (n, 2)
    u: np.ndarray            # edge start node index
    v: np.ndarray            # edge end node index
    length: np.ndarray       # metres
    cls: np.ndarray          # road_classification per edge (object)
    _tree: object = None

    @property
    def n(self):
        return len(self.node_xy)

    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.node_xy)
        return self._tree

    def snap(self, x, y):
        d, i = self.tree().query(np.column_stack([x, y]), workers=-1)
        return d, i

    def matrix(self, weight: np.ndarray, mask: np.ndarray | None = None, extra_nodes: int = 0):
        from scipy.sparse import coo_matrix
        from .reach import simple_edges
        n = self.n + extra_nodes
        # one link per junction pair, the quickest: a sparse matrix would add parallel links' costs together
        u, v, w = simple_edges(self.u, self.v, weight, mask)
        # undirected: one entry each way
        return coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()


def load_roads(path, bbox=None) -> RoadGraph:
    """Road links -> graph. Node ids are factorised strings; node coordinates
    are the first/last vertex of each link (vectorised GEOS)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import shapely

    t = pq.read_table(path, columns=["start_node", "end_node", "length", "road_classification", "geometry"])
    geoms = shapely.from_wkb(t["geometry"].to_numpy(zero_copy_only=False))
    s_xy = shapely.get_coordinates(shapely.get_point(geoms, 0))
    e_xy = shapely.get_coordinates(shapely.get_point(geoms, -1))
    keep = np.ones(len(geoms), dtype=bool)
    if bbox is not None:
        keep = (s_xy[:, 0] >= bbox[0]) & (s_xy[:, 0] <= bbox[2]) & (s_xy[:, 1] >= bbox[1]) & (s_xy[:, 1] <= bbox[3])
    ids = pc.dictionary_encode(pa_concat(t["start_node"], t["end_node"])).combine_chunks()
    codes = ids.indices.to_numpy()
    n_e = t.num_rows
    u, v = codes[:n_e], codes[n_e:]
    node_xy = np.full((len(ids.dictionary), 2), np.nan)
    node_xy[u] = s_xy
    node_xy[v] = e_xy
    u, v = u[keep], v[keep]
    length = t["length"].to_numpy()[keep].astype("f8")
    cls = t["road_classification"].to_numpy(zero_copy_only=False)[keep]
    # re-index to the nodes actually used
    used = np.unique(np.concatenate([u, v]))
    remap = np.full(len(node_xy), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    g = RoadGraph(node_xy[used], remap[u], remap[v], length, cls)
    log.info("roads   %s nodes, %s links", f"{g.n:,}", f"{len(g.u):,}")
    return g


def pa_concat(a, b):
    import pyarrow as pa
    return pa.chunked_array(a.chunks + b.chunks)


def nearest_via_network(g: RoadGraph, weight: np.ndarray, mask, tx, ty, px, py, *,
                        hop_speed: float = 1.0, limit: float = np.inf):
    """Network cost from each postcode (px, py) to the nearest target (tx, ty).

    ``weight`` is per-edge cost (metres or seconds). ``hop_speed`` converts the
    straight-line hops onto the network into the same unit (1 for metres,
    metres-per-second for seconds).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    if len(tx) == 0:
        return np.full(len(px), np.nan)
    A = g.matrix(weight, mask, extra_nodes=1)
    src = g.n                                   # the virtual source node
    td, ti = g.snap(tx, ty)
    # virtual edges src -> target node (cost of walking from the road to the target)
    hop = coo_matrix((td / hop_speed + 1e-6, (np.full(len(ti), src), ti)), shape=A.shape).tocsr()
    A = (A + hop).tocsr()
    dist = dijkstra(A, directed=True, indices=src, limit=limit)
    pd, pi = g.snap(px, py)
    out = dist[pi] + pd / hop_speed
    out[~np.isfinite(out)] = np.nan
    return out


def count_within_network(g: RoadGraph, weight, mask, tx, ty, px, py, limit: float, *, hop_speed=1.0):
    """How many targets each postcode can reach within ``limit`` (same unit as weight)."""
    from scipy.sparse.csgraph import dijkstra
    counts = np.zeros(g.n, dtype=np.int32)
    if len(tx) == 0:
        return np.zeros(len(px))
    A = g.matrix(weight, mask)
    td, ti = g.snap(tx, ty)
    for node, hop in zip(ti, td / hop_speed):
        budget = limit - hop
        if budget <= 0:
            continue
        d = dijkstra(A, directed=True, indices=int(node), limit=budget)
        counts[np.isfinite(d)] += 1
    pd, pi = g.snap(px, py)
    return counts[pi].astype("f8")
