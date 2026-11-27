"""Geographic choropleths on a street map: features averaged to ONS boundaries
(constituencies, local authorities, MSOAs) and to Ordnance Survey national grid
squares (5, 10, 25, 50 km). One Leaflet page, geography and variable drop-downs."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from . import PAGE_CSS, SCALE_JS, tile_js
from .cartograms import _aggregate, _postcode_table
from .postcode_data import _num

log = logging.getLogger(__name__)

GRID_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"   # OS grid: 5x5 letters, no I


def os_gridref(e: float, n: float) -> str:
    """British National Grid reference of a point, to 1 km (e.g. 'SD 30 90')."""
    e100, n100 = int(e // 100000), int(n // 100000)
    if not (0 <= e100 < 7 and 0 <= n100 < 13):
        return f"{e:.0f} E {n:.0f} N"
    l1 = (19 - n100) - (19 - n100) % 5 + (e100 + 10) // 5
    l2 = (19 - n100) * 5 % 25 + e100 % 5
    return f"{GRID_LETTERS[l1]}{GRID_LETTERS[l2]} {int(e % 100000 // 1000):02d} {int(n % 100000 // 1000):02d}"


def _to_ll_rings(geom, to_ll, tol: float):
    import shapely
    g = shapely.simplify(geom, tol, preserve_topology=True) if tol else geom
    out = []
    for part in shapely.get_parts(g):
        xy = shapely.get_coordinates(shapely.get_exterior_ring(part))
        if len(xy) < 4:
            continue
        lon, lat = to_ll.transform(xy[:, 0], xy[:, 1])
        out.append(np.round(np.column_stack([lat, lon]), 5).ravel().tolist())
    return out


def _boundary_geography(data_dir, spec, t, variables, to_ll):
    import geopandas as gpd
    f = data_dir / "1_extract" / spec["file"]
    if not f.exists():
        log.warning("choropleths: %s skipped - %s missing (run 01_extract.py)", spec["label"], spec["file"])
        return None
    gdf = gpd.read_parquet(f, columns=[spec["key"], spec.get("name_col", spec["key"]), "geometry"])
    codes = gdf[spec["key"]].astype(str).tolist()
    agg = _aggregate(t, spec["code"], codes, variables)
    tol = float(spec.get("simplify_m", 0))
    return {"names": gdf[spec.get("name_col", spec["key"])].astype(str).tolist(),
            "rings": [_to_ll_rings(g, to_ll, tol) for g in gdf.geometry.values], "agg": agg}


def _grid_geography(t, size_m, variables, to_ll):
    e = t["easting"].to_numpy(zero_copy_only=False).astype("f8")
    n = t["northing"].to_numpy(zero_copy_only=False).astype("f8")
    ok = np.isfinite(e) & np.isfinite(n)
    ix = np.where(ok, np.floor(e / size_m), -1).astype(np.int64)
    iy = np.where(ok, np.floor(n / size_m), -1).astype(np.int64)
    cell = np.where(ok, ix * 100000 + iy, -1)
    cells = np.unique(cell[ok])
    col = f"_grid{int(size_m)}"
    tt = t.append_column(col, pa.array(np.where(ok, cell, -1).astype(str)))
    codes = cells.astype(str).tolist()
    agg = _aggregate(tt, col, codes, variables)
    x0 = (cells // 100000) * size_m
    y0 = (cells % 100000) * size_m
    rings, names = [], []
    km = f"{size_m / 1000:g} km"
    for x, y in zip(x0, y0):
        xs = np.array([x, x + size_m, x + size_m, x, x])
        ys = np.array([y, y, y + size_m, y + size_m, y])
        lon, lat = to_ll.transform(xs, ys)
        rings.append([np.round(np.column_stack([lat, lon]), 5).ravel().tolist()])
        names.append(f"{os_gridref(x, y)} ({km} square)")
    return {"names": names, "rings": rings, "agg": agg}


def _h3_geographies(t, resolutions, variables):
    """Uber H3 hexagons (h3geo.org): postcodes -> cell at the finest resolution asked for, coarser
    resolutions via the cell's parents. One geography per resolution, aggregated like the grid squares."""
    import h3
    import pyarrow.compute as pc
    res = sorted({int(r) for r in resolutions})
    lat = pc.cast(t["lat"], "float64").to_numpy(zero_copy_only=False)
    lon = pc.cast(t["long"], "float64").to_numpy(zero_copy_only=False)
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
    fine = [h3.latlng_to_cell(a, b, res[-1]) if k else None for a, b, k in zip(lat, lon, ok)]
    out = []
    for r in res:
        if r == res[-1]:
            cells = fine
        else:
            par = {c: h3.cell_to_parent(c, r) for c in set(fine) - {None}}
            cells = [par.get(c) for c in fine]
        col = f"_h3_{r}"
        tt = t.append_column(col, pa.array(cells, pa.string()))
        codes = sorted({c for c in cells if c})
        agg = _aggregate(tt, col, codes, variables)
        km2 = h3.average_hexagon_area(r, "km^2")
        rings = [[np.round(np.asarray(h3.cell_to_boundary(c)), 5).ravel().tolist()] for c in codes]
        names = [f"H3 hexagon {c} (resolution {r}, about {km2:,.0f} km²)" if km2 >= 1 else
                 f"H3 hexagon {c} (resolution {r}, about {km2:.2f} km²)" for c in codes]
        out.append({"label": f"H3 hexagons, resolution {r} (about {km2:,.1f} km² each)", "id": f"h3_r{r}",
                    "names": names, "rings": rings, "agg": agg})
    return out


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoData choropleth maps</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>__CSS__
html,body{height:100%} body{display:flex;flex-direction:column}
#bar{display:flex;gap:18px;align-items:center;padding:10px 26px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
#bar select{font-size:14px;padding:4px;max-width:520px}
#map{flex:1}
.legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px #0003;font-size:12px;line-height:18px}
.legend i{display:inline-block;width:16px;height:14px;margin-right:6px;vertical-align:-2px;opacity:.85}
</style></head><body>
<header><h1>Choropleth maps</h1><p>Postcode features averaged to real geographies (weighted by the people living in each postcode), and rates per head: ONS boundaries,
Ordnance Survey national grid squares and <a href="https://h3geo.org">H3</a> hexagons (Uber's hierarchical hexagonal grid,
Apache 2.0). Boundaries: ONS Open Geography (OGL v3), simplified for the web.
<a href="index.html">All visualisations</a></p></header>
<div id="bar"><label>Geography <select id="geo"></select></label><label>Colour by <select id="var"></select></label>
<label>Compare with <select id="cmp"><option value="-1">(nothing: one variable)</option></select></label>
<label><input type="checkbox" id="empty"> show areas with no value</label>
<label><input type="checkbox" id="bg" checked> background map</label>
<label><input type="checkbox" id="lines" checked> boundary lines</label></div>
<div id="map"></div>
<script>
const D = __DATA__;
__SCALE__
const map = L.map('map', {preferCanvas: true}).setView([54.5, -3], 6);
__TILES__
const geo = document.getElementById('geo'), sel = document.getElementById('var'), empty = document.getElementById('empty');
const cmp = document.getElementById('cmp');
D.geos.forEach((g, i) => geo.add(new Option(g.label + ' (' + g.names.length.toLocaleString('en-GB') + ')', i)));
D.vars.forEach((v, i) => { sel.add(new Option(v.label, i)); if (!v.categorical) cmp.add(new Option(v.label, i)); });
let layer = null, fitOnBuild = false;
const metaFor = (k, G) => D.vars[k].categorical ? {cats: G.cats || []} : D.vars[k];
const show = (k, G, i) => { const v = G.values[k][i]; return D.vars[k].categorical && typeof v === 'number' ? (G.cats || [])[v] : fmt(v); };
const legend = L.control({position: 'bottomright'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.id = 'legend'; return d; };
legend.addTo(map);
function build() {
  const G = D.geos[+geo.value];
  if (layer) map.removeLayer(layer);
  layer = L.featureGroup();
  G.rings.forEach((rs, i) => {
    const latlngs = rs.map(r => { const a = []; for (let k = 0; k < r.length; k += 2) a.push([r[k], r[k + 1]]); return a; });
    L.polygon(latlngs, {weight: 0.4, color: '#555', fillOpacity: 0.75, _i: i})
      .bindTooltip(() => { const k = +sel.value, k2 = +cmp.value;
        return '<b>' + G.names[i] + '</b><br>' + D.vars[k].label + ': <b>' + show(k, G, i) + '</b>' +
          (k2 >= 0 ? '<br>' + D.vars[k2].label + ': <b>' + show(k2, G, i) + '</b>' : '') +
          '<br><span style="color:#777">' + fmt(G.residents[i]) + ' residents, ' + fmt(G.postcodes[i]) + ' postcodes, ' + fmt(G.addresses[i]) + ' addresses</span>'; },
        {sticky: true})
      .addTo(layer);
  });
  layer.addTo(map);
  if (fitOnBuild && layer.getLayers().length) {   // zoom to the chosen geography (not on first load: keep the UK view)
    map.invalidateSize(); map.fitBounds(layer.getBounds(), {padding: [10, 10]});
  }
  fitOnBuild = true;
  recolour();
}
function recolour() {
  const G = D.geos[+geo.value], k = +sel.value, k2 = D.vars[k].categorical ? -1 : +cmp.value;
  const V = G.values[k], label = D.vars[k].label;
  let colourAt, legendH;
  if (k2 >= 0) {   // bivariate: 3x3 tertiles of the two variables
    const B = bivScale(V, G.values[k2]);
    colourAt = i => B.col(i); legendH = bivLegendHtml(label, D.vars[k2].label);
  } else {
    const S = makeScale(V, metaFor(k, G));
    colourAt = i => S.col(V[i]);
    legendH = legendHtml(label, S, D.vars[k].categorical && !(G.cats || []).length ? '<br>not computed for this geography' : '');
  }
  const w = document.getElementById('lines').checked ? 0.4 : 0;   // boundary lines on/off
  layer.eachLayer(p => { const c = colourAt(p.options._i);
    p.setStyle(c ? {fillColor: c, fillOpacity: 0.75, opacity: 1, weight: w}
                 : {fillColor: '#bbb', fillOpacity: empty.checked ? 0.35 : 0, opacity: empty.checked ? 1 : 0, weight: w}); });
  document.getElementById('legend').innerHTML = legendH;
}
let tiles = null; map.eachLayer(l => { if (l instanceof L.TileLayer) tiles = l; });
document.getElementById('bg').onchange = e => {   // background map on/off (white page behind the areas when off)
  if (!tiles) return; if (e.target.checked) tiles.addTo(map); else map.removeLayer(tiles);
  document.getElementById('map').style.background = e.target.checked ? '' : '#fff'; };
geo.onchange = build; sel.onchange = recolour; empty.onchange = recolour; cmp.onchange = recolour;
document.getElementById('lines').onchange = recolour;
// optional start view from the URL, e.g. choropleths.html#geo=msoa&var=hp_price_1000m_w0&cmp=...&bg=0&lines=0
const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=').map(decodeURIComponent)));
if (H.geo) { const gi = /^[0-9]+$/.test(H.geo) ? +H.geo : D.geos.findIndex(g => g.id === H.geo); if (geo.options[gi]) geo.value = gi; }
{ const vi = varIndex(D.vars, H.var); if (vi !== null && sel.options[vi]) sel.value = vi; }
{ const ci = varIndex(D.vars, H.cmp); if (ci !== null) cmp.value = ci; }
if (H.lines === '0') document.getElementById('lines').checked = false;
build();
if (H.bg === '0') { const b = document.getElementById('bg'); b.checked = false; b.onchange({target: b}); }
</script></body></html>"""


def _clusters(agg: dict, cvars: dict, k: int, label: str):
    """k-means on standardised area averages -> (cluster code per area or None, labels).
    Clusters are numbered by size; each label names its two most distinctive variables."""
    from sklearn.cluster import KMeans
    keys = [c for c in cvars if c in agg]
    X = np.column_stack([agg[c] for c in keys]).astype("f8")
    ok = (agg["postcodes"] > 0) & (np.isfinite(X).sum(axis=1) >= len(keys) / 2)
    if ok.sum() < k * 3:
        return [None] * len(X), []
    Xo = X[ok]
    med = np.nanmedian(Xo, axis=0)
    Xo = np.where(np.isfinite(Xo), Xo, med)          # gaps (e.g. England-only measures) -> median
    sd = Xo.std(axis=0)
    Z = (Xo - Xo.mean(axis=0)) / np.where(sd > 0, sd, 1)
    import warnings
    with warnings.catch_warnings():   # sklearn's Windows/MKL thread warning is irrelevant for a few thousand rows
        warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
    order = np.argsort(-np.bincount(lab, minlength=k))          # biggest cluster first
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    lab = remap[lab]
    cats = []
    for c in range(k):
        zc = Z[lab == c].mean(axis=0)
        top = np.argsort(-np.abs(zc))[:2]
        words = [f"{'high' if zc[j] > 0 else 'low'} {cvars[keys[j]].get('short', cvars[keys[j]]['label'])}" for j in top]
        cats.append(f"{c + 1}: {', '.join(words)} ({int((lab == c).sum()):,} areas)")
    codes = [None] * len(X)
    for i, v in zip(np.flatnonzero(ok), lab):
        codes[i] = int(v)
    log.info("choropleths: %s - %d clusters from %d variables", label, k, len(keys))
    return codes, cats


def build_choropleths(data_dir: Path, out_dir: Path, cfg: dict) -> Path | None:
    from pyproj import Transformer
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    variables = {c: (v if isinstance(v, dict) else {"label": v}) for c, v in cfg.get("variables", {}).items()}
    bounds = cfg.get("boundaries", {})
    grids = cfg.get("grid_km", [])
    ccfg = cfg.get("clusters") or {}
    cvars = {c: (v if isinstance(v, dict) else {"label": v}) for c, v in (ccfg.get("variables") or {}).items()}
    agg_vars = {**cvars, **variables}   # cluster inputs are averaged too, even if not shown
    h3_res = cfg.get("h3_res") or []
    t = _postcode_table(data_dir, [*(b["code"] for b in bounds.values()), "easting", "northing",
                                   *(["lat", "long"] if h3_res else [])], agg_vars)
    missing = [c for c in agg_vars if c not in t.column_names]
    variables, agg_vars = ({c: v for c, v in d.items() if c in t.column_names} for d in (variables, agg_vars))
    found = list(variables)
    if missing:
        log.warning("choropleths: columns not found, skipped: %s", missing)
    geos = []
    for name, spec in bounds.items():
        if spec["code"] not in t.column_names:
            log.warning("choropleths: %s skipped - %s not in the postcode lookup", name, spec["code"])
            continue
        g = _boundary_geography(data_dir, spec, t, agg_vars, to_ll)
        if g:
            g["label"], g["id"] = spec["label"], name
            geos.append(g)
    for km in grids:
        g = _grid_geography(t, float(km) * 1000, agg_vars, to_ll)
        g["label"], g["id"] = f"OS national grid, {km} km squares", f"grid_{km:g}km"
        geos.append(g)
    if h3_res:
        try:
            geos.extend(_h3_geographies(t, h3_res, agg_vars))
        except ImportError:
            log.warning("choropleths: H3 hexagons skipped - install h3 (conda install -c conda-forge h3-py)")
    if not geos:
        log.warning("choropleths: no geographies - nothing written")
        return None
    from .cartograms import var_meta
    data = {"vars": [var_meta(c, variables[c]) for c in found], "geos": []}
    if cvars:
        data["vars"].append({"key": "cluster", "label": ccfg.get("label", "Area type: clusters of similar areas"),
                             "categorical": True})
    for g in geos:
        agg = g["agg"]
        values = [_num(pa.chunked_array([pa.array(agg[c], from_pandas=True)])) for c in found]
        cats = []
        if cvars:
            if g["id"] in ccfg.get("geographies", []):
                codes, cats = _clusters(agg, cvars, int(ccfg.get("k", 8)), g["label"])
                values.append(codes)
            else:
                values.append([None] * len(g["names"]))
        data["geos"].append({
            "id": g["id"], "label": g["label"], "names": g["names"], "rings": g["rings"], "cats": cats,
            "postcodes": [int(v) for v in agg["postcodes"]], "addresses": [int(round(v)) for v in agg["addresses"]],
            "residents": [int(round(v)) for v in agg["residents"]],
            "values": values})
        log.info("choropleths: %s - %d areas", g["label"], len(g["names"]))
    page = (TEMPLATE.replace("__CSS__", PAGE_CSS).replace("__SCALE__", SCALE_JS)
            .replace("__TILES__", tile_js("light", "boundaries: ONS (OGL)"))
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    out = out_dir / "choropleths.html"
    out.write_text(page, encoding="utf-8")
    log.info("wrote   %s (%d geographies x %d variables, %.1f MB)", out.name, len(geos), len(found),
             out.stat().st_size / 1e6)
    return out
