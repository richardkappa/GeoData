"""Network explorer: road and area networks around each key location.

Views (one Leaflet page, data loaded per location on demand):

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
from . import PAGE_CSS, square, tile_js

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
    return u, v, len(uniq), xy


def _csr(u, v, w, n, mask=None):
    from scipy.sparse import coo_matrix
    m = np.ones(len(u), bool) if mask is None else mask
    return coo_matrix((np.r_[w[m], w[m]], (np.r_[u[m], v[m]], np.r_[v[m], u[m]])), shape=(n, n)).tocsr()


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
            u, v, n, xy = _graph(sub)
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
            # display: links touching the square (plus a margin), simplified and in lat/lon
            show = np.flatnonzero(shapely.intersects(sub.geometry.values,
                                                     shapely.box(box[0] - 300, box[1] - 300, box[2] + 300, box[3] + 300)))
            g = shapely.simplify(sub.geometry.values[show], 2.0)
            links = []
            on_route = set(route)
            for k, geom in zip(show, g):
                c = shapely.get_coordinates(geom)
                if len(c) < 2:
                    continue
                tmid = lambda d: None if not (np.isfinite(d[u[k]]) or np.isfinite(d[v[k]])) else \
                    round(float(np.nanmin([d[u[k]], d[v[k]]])) / 60, 1)
                links.append([ll(c).ravel().tolist(), CLASS_ORDER.index(cls[k]) if cls[k] in CLASS_ORDER else 6,
                              round(float(btw[k]), 3), tmid(dw) if walkable[k] else None, tmid(dd),
                              1 if k in on_route else 0])
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
                    "lsoa": lnet}
            (nd / f"{name}.js").write_text(f"window.NET[{json.dumps(name)}]=" + json.dumps(data, separators=(",", ":"))
                                           + ";", encoding="utf-8")
            meta.append({"name": name, "label": name.replace("_", " ") + (f" - {loc['note']}" if loc.get("note") else "")})
    if not meta:
        return None
    page = (TEMPLATE.replace("__CSS__", PAGE_CSS).replace("__TILES__", tile_js("light"))
            .replace("__KM__", f"{size / 1000:g} km").replace("__META__", json.dumps({"locations": meta})))
    out = out_dir / "network_explorer.html"
    out.write_text(page, encoding="utf-8")
    mb = sum(p.stat().st_size for p in nd.glob("*.js")) / 1e6
    log.info("wrote   %s (%d locations, %.1f MB of data files)", out.name, len(meta), mb)
    return out


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Network explorer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>__CSS__
html,body{height:100%} body{display:flex;flex-direction:column}
#bar{display:flex;gap:18px;align-items:center;padding:10px 26px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
#bar select{font-size:14px;padding:4px;max-width:520px} #map{flex:1}
.legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px #0003;font-size:12px;line-height:18px;max-width:280px}
.legend i{display:inline-block;width:22px;height:4px;margin-right:6px;vertical-align:middle}
.legend b.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}
</style></head><body>
<header><h1>Network explorer</h1><p>Road and area networks in __KM__ squares around the key locations: which roads carry
the through-routes, how far you can walk or drive from the centre, the major road junction graph, and the network of
neighbouring LSOAs with bridges across rivers. Roads: OS Open Roads; areas: ONS LSOA 2021. <a href="index.html">All visualisations</a></p></header>
<div id="bar"><label>Location <select id="loc"></select></label>
<label>Show <select id="view">
<option value="btw">Road centrality: the through-routes</option>
<option value="walk">Walking time from the centre</option>
<option value="drive">Driving time from the centre</option>
<option value="route">Shortest walk to the nearest railway station</option>
<option value="junc">Major road junction network (A roads and motorways, simplified)</option>
<option value="lsoa">LSOA neighbour network with bridges</option>
<option value="rings">LSOAs within 1, 2 and 3 boundaries of the centre</option>
</select></label>
<label><input type="checkbox" id="bg" checked> background map</label><span id="note"></span></div>
<div id="map"></div>
<script>
const M = __META__;
window.NET = {};
const map = L.map('map', {preferCanvas: true});
__TILES__
let tiles = null; map.eachLayer(l => { if (l instanceof L.TileLayer) tiles = l; });
const locSel = document.getElementById('loc'), view = document.getElementById('view'), note = document.getElementById('note');
M.locations.forEach(l => locSel.add(new Option(l.label, l.name)));
let layer = null, cur = null;
const HEAT = ['#ffffcc','#fed976','#fd8d3c','#e31a1c','#800026'];
const TIME = ['#1a9850','#91cf60','#d9ef8b','#fee08b','#fc8d59','#d73027'], TB = [5, 10, 15, 20, 30];
const RING = ['#111111', '#1b9e77', '#d95f02', '#7570b3'];
const legend = L.control({position: 'bottomright'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.id = 'legend'; return d; };
legend.addTo(map);
const pl = c => { const a = []; for (let k = 0; k < c.length; k += 2) a.push([c[k], c[k + 1]]); return a; };
function tband(t) { if (t === null) return null; let k = 0; while (k < TB.length && t > TB[k]) k++; return TIME[k]; }
function draw() {
  const D = window.NET[cur]; if (!D) return;
  if (layer) map.removeLayer(layer);
  layer = L.layerGroup();
  const v = view.value; let lg = '';
  const faint = () => D.links.forEach(l => L.polyline(pl(l[0]), {color: '#999', weight: 0.6, opacity: 0.5, interactive: false}).addTo(layer));
  if (v === 'btw') {
    D.links.slice().sort((a, b) => a[2] - b[2]).forEach(l => { const b = l[2], k = Math.min(4, Math.floor(Math.sqrt(b) * 5));
      L.polyline(pl(l[0]), {color: HEAT[k], weight: 0.8 + 6 * Math.sqrt(b), opacity: 0.9}).bindTooltip('share of sampled routes: ' + (b * 100).toFixed(0) + '% of the busiest road', {sticky: true}).addTo(layer); });
    lg = '<b>Road centrality</b><br>' + HEAT.map((c, i) => '<i style="background:' + c + ';height:' + (2 + i) + 'px"></i>' + ['quiet', '', 'busy', '', 'main through-route'][i]).join('<br>') +
      '<br><span style="color:#777">how many shortest drives between places in the square pass along each road (sampled)</span>';
  } else if (v === 'walk' || v === 'drive') {
    const idx = v === 'walk' ? 3 : 4;
    D.links.forEach(l => { const c = tband(l[idx]); if (c) L.polyline(pl(l[0]), {color: c, weight: 2.2, opacity: 0.9})
      .bindTooltip(l[idx] + ' min', {sticky: true}).addTo(layer); });
    lg = '<b>' + (v === 'walk' ? 'Walking' : 'Driving') + ' time from the centre</b><br>' +
      TIME.map((c, i) => '<i style="background:' + c + '"></i>' + (i === 0 ? 'up to 5' : i === 5 ? 'over 30' : TB[i - 1] + ' to ' + TB[i]) + ' min').join('<br>') +
      (v === 'walk' ? '<br><span style="color:#777">4.8 km/h; motorways excluded</span>' : '<br><span style="color:#777">typical speeds by road class</span>');
  } else if (v === 'route') {
    faint();
    D.links.filter(l => l[5]).forEach(l => L.polyline(pl(l[0]), {color: '#d7191c', weight: 5}).addTo(layer));
    if (D.station) L.marker(D.station.ll).bindTooltip(D.station.name + ' station').addTo(layer);
    lg = D.station ? '<b>Shortest walk to the nearest station</b><br>' + D.station.name + ': ' + D.station.min + ' min' : 'no station nearby';
  } else if (v === 'junc') {
    faint();
    const P = D.junctions.pts;
    D.junctions.edges.forEach(e => L.polyline([P[e[0]], P[e[1]]], {color: '#2b6cb0', weight: 2.5}).addTo(layer));
    P.forEach(p => L.circleMarker(p, {radius: 3.5, color: '#08306b', fillColor: '#08306b', fillOpacity: 1, weight: 0}).addTo(layer));
    lg = '<b>Major road junction network</b><br>' + P.length + ' junctions, ' + D.junctions.edges.length + ' links<br>' +
      '<span style="color:#777">A roads and motorways; nodes that only join two road pieces are removed, so each line is a road between two junctions</span>';
  } else {
    const N = D.lsoa; if (!N) { note.textContent = 'no LSOAs here (England and Wales only)'; layer.addTo(map); return; }
    N.edges.forEach(e => L.polyline([N.pts[e[0]], N.pts[e[1]]], e[2] ? {color: '#d7191c', weight: 2.5, dashArray: '6 4'} : {color: '#2b8a3e', weight: 1.2, opacity: 0.8}).addTo(layer));
    N.pts.forEach((p, i) => { const r = N.ring[i], c = v === 'rings' ? (r >= 0 ? RING[r] : '#bbb') : '#1d4e89';
      L.circleMarker(p, {radius: v === 'rings' && r === 0 ? 7 : 4, color: '#fff', weight: 1, fillColor: c, fillOpacity: 1})
        .bindTooltip(N.names[i] + (v === 'rings' && r >= 0 ? ': ' + (r === 0 ? 'centre' : 'within ' + r + ' boundar' + (r > 1 ? 'ies' : 'y')) : ''), {sticky: true}).addTo(layer); });
    const nb = N.edges.filter(e => e[2]).length;
    lg = v === 'rings' ? '<b>Within n boundaries of the centre LSOA</b><br>' + ['centre', 'within 1', 'within 2', 'within 3'].map((t, i) => '<b class="dot" style="background:' + RING[i] + '"></b>' + t).join('<br>') +
      '<br><span style="color:#777">the neighbourhoods used for spatial smoothing (bridges included)</span>'
      : '<b>LSOA neighbour network</b><br><i style="background:#2b8a3e"></i>areas that touch<br><i style="background:#d7191c"></i>bridge: joined by a road across a gap (' + nb + ')';
  }
  L.polygon(D.box, {color: '#111', weight: 1.2, fill: false, dashArray: '6 4', interactive: false}).addTo(layer);
  L.circleMarker(D.centre, {radius: 5, color: '#000', fillColor: '#fff', fillOpacity: 1, weight: 2}).bindTooltip('centre').addTo(layer);
  layer.addTo(map);
  document.getElementById('legend').innerHTML = lg;
}
function show(name) {
  cur = name; note.textContent = '';
  const go = () => { map.fitBounds(window.NET[name].box); draw(); };
  if (window.NET[name]) return go();
  const s = document.createElement('script'); s.src = 'network_data/' + name + '.js'; s.onload = go;
  s.onerror = () => alert('Could not load network_data/' + name + '.js - run 04_visualise.py'); document.body.appendChild(s);
}
document.getElementById('bg').onchange = e => { if (tiles) e.target.checked ? tiles.addTo(map) : map.removeLayer(tiles); };
locSel.onchange = () => show(locSel.value); view.onchange = draw;
const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=').map(decodeURIComponent)));
if (H.loc && M.locations.some(l => l.name === H.loc)) locSel.value = H.loc;
if (H.view) view.value = H.view;
show(locSel.value);
</script></body></html>"""
