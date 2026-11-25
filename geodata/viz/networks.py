"""Road and area networks around each key location, for the map explorer (explorer.py).

Views (shown on the map explorer page, explorer.py; data loaded per location on demand):

* road centrality - how many shortest routes between places in the square use each
  road (sampled edge betweenness on drive times): the through-routes stand out;
* walking / driving time from the centre over the road network;
* shortest walking route from the centre to the nearest railway station;
* major road junction network - A roads and motorways simplified to a graph of
  junctions (nodes of degree 2 removed), as in "Network Analysis in Python";
* LSOA neighbour network - areas that touch, plus "bridges" (roads joining areas
  across a river), and the rings within 1, 2 and 3 boundaries of the centre LSOA
  used for spatial smoothing.

Shortest paths use scipy's C Dijkstra; betweenness is accumulated from the
shortest-path trees of a sample of sources.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .. import io
from ..features.network import DEFAULT_SPEEDS
from . import square

log = logging.getLogger(__name__)

CLASS_ORDER = ["Motorway", "A Road", "B Road", "Classified Unnumbered", "Unclassified", "Not Classified", "Unknown"]


def _graph(sub):
    """Road links -> (u, v, n_nodes, node_xy) with node ids factorised locally."""
    import pandas as pd
    import shapely
    ids, uniq = pd.factorize(pd.concat([sub["start_node"], sub["end_node"]], ignore_index=True))
    n_e = len(sub)
    u, v = ids[:n_e], ids[n_e:]
    xy = np.full((len(uniq), 2), np.nan)
    g = sub.geometry.values
    xy[u] = shapely.get_coordinates(shapely.get_point(g, 0))
    xy[v] = shapely.get_coordinates(shapely.get_point(g, -1))
    return u, v, len(uniq), xy, uniq


def _wide_measures(roads, box, cfg, speeds, walk_kmh, stations, coll, sub_ids):
    """Measures that need a much wider road network than the square (30 minutes' drive can be
    40+ km): walking / driving time from every node of the square to the nearest railway station,
    and road collisions (total, fatal) reachable within N minutes' drive of a grid of points."""
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    import pandas as pd
    wpad = float(cfg.get("wide_pad_m", 45000))
    wb = (box[0] - wpad, box[1] - wpad, box[2] + wpad, box[3] + wpad)
    ws = roads.cx[wb[0]:wb[2], wb[1]:wb[3]]
    ws = ws[ws["start_node"].notna() & ws["end_node"].notna()]
    u, v, n, xy, uniq = _graph(ws)
    cls = ws["road_classification"].astype(str).to_numpy()
    length = ws["length"].to_numpy("f8")
    drive_s = length / (np.array([speeds.get(c, 25) for c in cls], dtype="f8") / 3.6)
    walkable = cls != "Motorway"
    walk_s = length / (walk_kmh / 3.6)
    A_drive = _csr(u, v, drive_s, n)
    A_walk = _csr(u, v, walk_s, n, walkable)
    dnodes = np.unique(np.r_[u, v])
    wnodes = np.unique(np.r_[u[walkable], v[walkable]])
    out = {"walk_station": None, "drive_station": None, "reach": [], "reach_min": None}
    pos = pd.Index(uniq).get_indexer(pd.Index(sub_ids))    # square's nodes -> wide network nodes
    # time to the nearest station: one multi-source search per mode
    if stations is not None and len(stations):
        s = stations[stations["easting"].between(wb[0], wb[2]) & stations["northing"].between(wb[1], wb[3])]
        if len(s):
            sxy = s[["easting", "northing"]].to_numpy()
            for key, A, nodes in (("walk_station", A_walk, wnodes), ("drive_station", A_drive, dnodes)):
                _, si = cKDTree(xy[nodes]).query(sxy)
                d = dijkstra(A, directed=False, indices=np.unique(nodes[si]), min_only=True)
                out[key] = np.where(pos >= 0, d[np.clip(pos, 0, None)], np.inf)
    # collisions reachable within `reach_min` minutes' drive of grid points in the square
    minutes = float(cfg.get("collision_reach_min", 30))
    if coll is not None and len(coll):
        c = coll[coll["easting"].between(wb[0], wb[2]) & coll["northing"].between(wb[1], wb[3])]
        tree = cKDTree(xy[dnodes])
        dist, ci = tree.query(c[["easting", "northing"]].to_numpy())
        near = dist <= 200                                 # collisions on (or beside) a mapped road
        node = dnodes[ci[near]]
        cnt_t = np.bincount(node, minlength=n).astype("f8")
        cnt_f = np.bincount(node[c["collision_severity"].to_numpy()[near] == 1], minlength=n).astype("f8")
        step = float(cfg.get("collision_reach_grid_m", 400))
        gx, gy = np.meshgrid(np.arange(box[0] + step / 2, box[2], step), np.arange(box[1] + step / 2, box[3], step))
        gd, gi = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
        src = np.unique(dnodes[gi[gd <= 300]])               # grid points on or near a road
        limit = minutes * 60
        for i in range(0, len(src), 20):                     # chunks keep the distance matrix small
            ch = src[i:i + 20]
            d = dijkstra(A_drive, directed=False, indices=ch, limit=limit)
            reach = (d <= limit).astype("f8")
            for k, sn in enumerate(ch):
                out["reach"].append((xy[sn], int(reach[k] @ cnt_t), int(reach[k] @ cnt_f)))
        out["reach_min"] = minutes
    log.info("networks: wide network %s links; %d reach points", f"{len(ws):,}", len(out["reach"]))
    return out


def _csr(u, v, w, n, mask=None):
    from scipy.sparse import coo_matrix
    from ..features.reach import simple_edges
    a, b, s = simple_edges(np.asarray(u), np.asarray(v), np.asarray(w, dtype="f8"), mask)   # quickest of parallel links
    return coo_matrix((np.r_[s, s], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()


def _edge_lookup(u, v, w):
    """(node, node) -> index of the cheapest link between them."""
    best = {}
    for k, (a, b, c) in enumerate(zip(u, v, w)):
        key = (a, b) if a < b else (b, a)
        if key not in best or c < w[best[key]]:
            best[key] = k
    return best


def _betweenness(u, v, w, n, sources, look):
    """Edge use counted over the shortest-path trees from `sources` (sampled betweenness)."""
    from scipy.sparse.csgraph import dijkstra
    A = _csr(u, v, w, n)
    use = np.zeros(len(u))
    dist, pred = dijkstra(A, directed=False, indices=sources, return_predecessors=True)
    for d, p in zip(dist, pred):
        reach = np.flatnonzero(np.isfinite(d))
        order = reach[np.argsort(-d[reach])]
        cnt = np.ones(n)
        for node in order:          # farthest first: pass each node's subtree size to its parent
            par = p[node]
            if par >= 0:
                cnt[par] += cnt[node]
                k = look.get((par, node) if par < node else (node, par))
                if k is not None:
                    use[k] += cnt[node]
    return use


def _contract(u, v, n, xy, mask):
    """Major-road graph with degree-2 nodes removed -> (junction xy list, edges between them)."""
    adj = {}
    for a, b in zip(u[mask], v[mask]):
        if a != b:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    junction = {x for x, nb in adj.items() if len(nb) != 2}
    if not junction and adj:
        junction = {next(iter(adj))}
    edges, seen = set(), set()
    for j in junction:
        for nb in adj[j]:
            prev, cur = j, nb
            while cur not in junction:
                nxt = [x for x in adj[cur] if x != prev]
                if not nxt or (prev, cur) in seen:
                    break
                seen.add((prev, cur))
                prev, cur = cur, nxt[0]
            if cur in junction and cur != j:
                edges.add((min(j, cur), max(j, cur)))
    nodes = sorted(junction)
    pos = {x: i for i, x in enumerate(nodes)}
    return xy[nodes], [(pos[a], pos[b]) for a, b in edges if a in pos and b in pos]


def build_networks(data_dir: Path, out_dir: Path, locs: dict, cfg: dict) -> Path | None:
    """Per-location network data (network_data/<name>.js) for the map explorer (explorer.py)."""
    import geopandas as gpd
    import shapely
    from pyproj import Transformer
    from scipy.sparse.csgraph import dijkstra
    from ..features.adjacency import contiguity, road_bridges

    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    rf = data_dir / "extract" / cfg.get("roads", "os_open/openroads__road_link.parquet")
    if not rf.exists():
        log.warning("networks skipped: %s missing (enable OpenRoads)", rf.name)
        return None
    size = float(cfg.get("size_m", 5000))
    pad = float(cfg.get("route_pad_m", 1500))
    speeds = {**DEFAULT_SPEEDS, **(cfg.get("speeds_kmh") or {})}
    walk_kmh = float(cfg.get("walk_kmh", 4.8))
    n_src = int(cfg.get("betweenness_sources", 200))
    with io.Timer("networks: load roads, LSOAs, stations"):
        roads = gpd.read_parquet(rf, columns=["start_node", "end_node", "length", "road_classification",
                                              "form_of_way", "geometry"])
        roads.sindex
        lf = data_dir / "extract" / cfg.get("lsoa", "ons_boundaries/lsoa21_detail.parquet")
        lsoa = gpd.read_parquet(lf if lf.exists() else data_dir / "extract" / "ons_boundaries/lsoa21.parquet",
                                columns=["lsoa21cd", "lsoa21nm", "geometry"])
        lsoa.sindex
        nf = data_dir / "extract" / "naptan" / "naptan_stops.parquet"
        stations = None
        if nf.exists():
            import pyarrow.parquet as pq
            st = pq.read_table(nf, columns=["stoptype", "commonname", "easting", "northing"]).to_pandas()
            stations = st[(st["stoptype"] == "RLY") & st["easting"].notna()]
        cf = data_dir / "extract" / cfg.get("collisions", "road_safety/collision.parquet")
        coll = None
        if cf.exists():
            import pyarrow.parquet as pq
            coll = pq.read_table(cf, columns=["easting", "northing", "collision_severity"]).to_pandas().dropna()

    def ll(xy):
        lon, lat = to_ll.transform(xy[:, 0], xy[:, 1])
        return np.round(np.column_stack([lat, lon]), 5)

    nd = out_dir / "network_data"
    nd.mkdir(parents=True, exist_ok=True)
    meta = []
    for name, loc in locs.items():
        with io.Timer(f"networks: {name}"):
            box = square(loc, size)
            big = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
            sub = roads.cx[big[0]:big[2], big[1]:big[3]]
            sub = sub[sub["start_node"].notna() & sub["end_node"].notna()].reset_index(drop=True)
            if len(sub) < 10:
                log.warning("networks: %s has too few roads - skipped", name)
                continue
            u, v, n, xy, uniq = _graph(sub)
            length = sub["length"].to_numpy("f8")
            cls = sub["road_classification"].astype(str).to_numpy()
            kmh = np.array([speeds.get(c, 25) for c in cls], dtype="f8")
            drive_s = length / (kmh / 3.6)
            walkable = cls != "Motorway"
            walk_s = length / (walk_kmh / 3.6)
            # centre node: nearest walkable node to the location
            wnodes = np.unique(np.r_[u[walkable], v[walkable]])
            centre = wnodes[np.argmin(np.hypot(xy[wnodes, 0] - loc["x"], xy[wnodes, 1] - loc["y"]))]
            dw, pw = dijkstra(_csr(u, v, walk_s, n, walkable), directed=False, indices=centre, return_predecessors=True)
            dd = dijkstra(_csr(u, v, drive_s, n), directed=False, indices=centre)
            # sampled betweenness over drive times, sources inside the square
            inbox = np.flatnonzero((xy[:, 0] >= box[0]) & (xy[:, 0] <= box[2]) & (xy[:, 1] >= box[1]) & (xy[:, 1] <= box[3]))
            rng = np.random.default_rng(0)
            src = rng.choice(inbox, size=min(n_src, len(inbox)), replace=False) if len(inbox) else np.array([], int)
            look = _edge_lookup(u, v, drive_s)
            btw = _betweenness(u, v, drive_s, n, src, look) if len(src) else np.zeros(len(u))
            btw = btw / btw.max() if btw.max() > 0 else btw
            # walking route to the nearest station
            route = []
            station = None
            if stations is not None and len(stations):
                s = stations[(stations["easting"].between(big[0], big[2])) & (stations["northing"].between(big[1], big[3]))]
                if len(s):
                    from scipy.spatial import cKDTree
                    tw = cKDTree(xy[wnodes])
                    _, si = tw.query(s[["easting", "northing"]].to_numpy())
                    snodes = wnodes[si]
                    t = dw[snodes]
                    if np.isfinite(t).any():
                        best = int(np.nanargmin(t))
                        node = snodes[best]
                        station = {"name": str(s.iloc[best]["commonname"]), "min": round(float(t[best]) / 60, 1),
                                   "ll": ll(s.iloc[[best]][["easting", "northing"]].to_numpy())[0].tolist()}
                        wlook = _edge_lookup(u, v, np.where(walkable, walk_s, np.inf))
                        while node != centre and node >= 0:
                            par = pw[node]
                            if par < 0:
                                break
                            k = wlook.get((par, node) if par < node else (node, par))
                            if k is not None:
                                route.append(k)
                            node = par
            # wider network: time to the nearest station, collisions within N minutes' drive
            wide = _wide_measures(roads, box, cfg, speeds, walk_kmh, stations, coll, uniq)
            # collision hotspots: each collision on the nearest road link within 30 m
            ct = np.zeros(len(sub))
            cfat = np.zeros(len(sub))
            if coll is not None and len(coll):
                cc = coll[coll["easting"].between(big[0], big[2]) & coll["northing"].between(big[1], big[3])]
                if len(cc):
                    pin, lk = shapely.STRtree(sub.geometry.values).query_nearest(
                        shapely.points(cc["easting"].to_numpy(), cc["northing"].to_numpy()), max_distance=30)
                    first = np.unique(pin, return_index=True)[1]   # one road per collision (ties: first)
                    pin, lk = pin[first], lk[first]
                    np.add.at(ct, lk, 1)
                    np.add.at(cfat, lk[cc["collision_severity"].to_numpy()[pin] == 1], 1)
            # display: links touching the square (plus a margin), simplified and in lat/lon
            show = np.flatnonzero(shapely.intersects(sub.geometry.values,
                                                     shapely.box(box[0] - 300, box[1] - 300, box[2] + 300, box[3] + 300)))
            g = shapely.simplify(sub.geometry.values[show], 2.0)
            links = []
            on_route = set(route)
            # collisions within N minutes' drive: each road section takes the value of the nearest
            # reach point (computed on a grid along the roads), so the measure is drawn on the network
            if wide["reach"]:
                from scipy.spatial import cKDTree
                rxy = np.array([p for p, _, _ in wide["reach"]])
                rtree, rvals = cKDTree(rxy), [(t, f) for _, t, f in wide["reach"]]
            else:
                rtree = None

            def reach_of(coords):
                if rtree is None:
                    return None, None
                d, i = rtree.query(coords[len(coords) // 2])
                return rvals[i] if d <= 2 * float(cfg.get("collision_reach_grid_m", 250)) else (None, None)
            for k, geom in zip(show, g):
                c = shapely.get_coordinates(geom)
                if len(c) < 2:
                    continue
                tmid = lambda d: None if not (np.isfinite(d[u[k]]) or np.isfinite(d[v[k]])) else \
                    round(float(np.nanmin([d[u[k]], d[v[k]]])) / 60, 1)
                rt, rf = reach_of(c)
                links.append([ll(c).ravel().tolist(), CLASS_ORDER.index(cls[k]) if cls[k] in CLASS_ORDER else 6,
                              round(float(btw[k]), 3), tmid(dw) if walkable[k] else None, tmid(dd),
                              1 if k in on_route else 0, int(ct[k]), int(cfat[k]),
                              tmid(wide["walk_station"]) if wide["walk_station"] is not None and walkable[k] else None,
                              tmid(wide["drive_station"]) if wide["drive_station"] is not None else None, rt, rf])
            # major road junction network
            major = np.isin(cls, ["Motorway", "A Road"])
            jxy, jedges = _contract(u, v, n, xy, major)
            # LSOA network with bridges and rings
            ls = lsoa.cx[box[0]:box[2], box[1]:box[3]].reset_index(drop=True)
            lnet = None
            if len(ls):
                A = contiguity(ls.geometry.values, 1.0)
                B, _ = road_bridges(ls.geometry.values, sub.geometry.values, A)
                pts = shapely.get_coordinates(shapely.point_on_surface(ls.geometry.values))
                cpt = shapely.points(loc["x"], loc["y"])
                home = np.flatnonzero(shapely.contains(ls.geometry.values, cpt))
                ring = np.full(len(ls), -1)
                if len(home):
                    M = (A + B).tocsr()
                    frontier, ring[home[0]] = [home[0]], 0
                    for lvl in (1, 2, 3):
                        nxt = set()
                        for i in frontier:
                            nxt.update(j for j in M[i].indices if ring[j] < 0)
                        for j in nxt:
                            ring[j] = lvl
                        frontier = list(nxt)
                ea = [(int(i), int(j), 0) for i, j in zip(*A.nonzero()) if i < j]
                eb = [(int(i), int(j), 1) for i, j in zip(*B.nonzero()) if i < j]
                lnet = {"pts": ll(pts).tolist(), "names": ls["lsoa21nm"].astype(str).tolist(),
                        "ring": ring.tolist(), "edges": ea + eb}
            lon, lat = to_ll.transform([box[0], box[2], box[2], box[0]], [box[1], box[1], box[3], box[3]])
            data = {"box": [[round(a, 6), round(b, 6)] for a, b in zip(lat, lon)], "links": links,
                    "centre": ll(xy[[centre]])[0].tolist(), "station": station,
                    "junctions": {"pts": ll(jxy).tolist() if len(jxy) else [], "edges": jedges},
                    "lsoa": lnet, "reach_min": wide["reach_min"]}
            (nd / f"{name}.js").write_text(f"window.NET[{json.dumps(name)}]=" + json.dumps(data, separators=(",", ":"))
                                           + ";", encoding="utf-8")
            meta.append({"name": name, "label": name.replace("_", " ") + (f" - {loc['note']}" if loc.get("note") else "")})
    if not meta:
        return None
    # the page itself is the map explorer (explorer.py), which reads this
    out = nd / "meta.json"
    out.write_text(json.dumps({"locations": meta}), encoding="utf-8")
    mb = sum(p.stat().st_size for p in nd.glob("*.js")) / 1e6
    log.info("wrote   network_data (%d locations, %.1f MB)", len(meta), mb)
    return out

