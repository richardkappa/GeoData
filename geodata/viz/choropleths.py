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

from . import PAGE_CSS, tile_js
from .cartograms import _aggregate, _postcode_table
from .interactive import _num

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
    f = data_dir / "extract" / spec["file"]
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
<header><h1>Choropleth maps</h1><p>Postcode features averaged to real geographies (weighted by addresses): ONS boundaries
and Ordnance Survey national grid squares. Boundaries: ONS Open Geography (OGL v3), simplified for the web.
<a href="index.html">All visualisations</a></p></header>
<div id="bar"><label>Geography <select id="geo"></select></label><label>Colour by <select id="var"></select></label>
<label><input type="checkbox" id="empty"> show areas with no value</label>
<label><input type="checkbox" id="bg" checked> background map</label>
<label><input type="checkbox" id="lines" checked> boundary lines</label></div>
<div id="map"></div>
<script>
const D = __DATA__;
const PAL = ["#440154","#46327e","#365c8d","#277f8e","#1fa187","#4ac16d","#a0da39","#fde725"];
const map = L.map('map', {preferCanvas: true}).setView([54.5, -3], 6);
__TILES__
const geo = document.getElementById('geo'), sel = document.getElementById('var'), empty = document.getElementById('empty');
D.geos.forEach((g, i) => geo.add(new Option(g.label + ' (' + g.names.length.toLocaleString('en-GB') + ')', i)));
D.vars.forEach((v, i) => sel.add(new Option(v.label, i)));
const fmt = v => v === null || v === undefined ? '–' : (typeof v === 'number' ? v.toLocaleString('en-GB') : v);
let layer = null, breaks = [];
const colour = v => { if (typeof v !== 'number') return null; let k = 0; while (k < breaks.length && v > breaks[k]) k++;
  return PAL[Math.round(k * (PAL.length - 1) / Math.max(breaks.length, 1))]; };
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
      .bindTooltip(() => { const V = G.values[+sel.value]; return '<b>' + G.names[i] + '</b><br>' + D.vars[+sel.value].label +
        ': <b>' + fmt(V[i]) + '</b><br><span style="color:#777">' + fmt(G.postcodes[i]) + ' postcodes, ' + fmt(G.addresses[i]) + ' addresses</span>'; },
        {sticky: true})
      .addTo(layer);
  });
  layer.addTo(map);
  recolour();
}
function recolour() {
  const G = D.geos[+geo.value], V = G.values[+sel.value], label = D.vars[+sel.value].label;
  const nums = V.filter(v => typeof v === 'number').sort((a, b) => a - b);
  const q = p => nums[Math.min(nums.length - 1, Math.floor(p * nums.length))];
  breaks = nums.length ? [...new Set([1, 2, 3, 4, 5, 6, 7].map(k => q(k / 8)))] : [];
  const w = document.getElementById('lines').checked ? 0.4 : 0;   // boundary lines on/off
  layer.eachLayer(p => { const c = colour(V[p.options._i]);
    p.setStyle(c ? {fillColor: c, fillOpacity: 0.75, opacity: 1, weight: w}
                 : {fillColor: '#bbb', fillOpacity: empty.checked ? 0.35 : 0, opacity: empty.checked ? 1 : 0, weight: w}); });
  let h = '<b>' + label + '</b><br>';
  if (nums.length) { const edges = [nums[0], ...breaks, nums[nums.length - 1]];
    for (let k = 0; k <= breaks.length; k++) h += '<i style="background:' + PAL[Math.round(k * (PAL.length - 1) / Math.max(breaks.length, 1))] + '"></i>' + fmt(edges[k]) + ' to ' + fmt(edges[k + 1]) + '<br>'; }
  h += '<span style="color:#777">eighths (quantiles) of the areas shown</span>';
  document.getElementById('legend').innerHTML = h;
}
let tiles = null; map.eachLayer(l => { if (l instanceof L.TileLayer) tiles = l; });
document.getElementById('bg').onchange = e => {   // background map on/off (white page behind the areas when off)
  if (!tiles) return; if (e.target.checked) tiles.addTo(map); else map.removeLayer(tiles);
  document.getElementById('map').style.background = e.target.checked ? '' : '#fff'; };
geo.onchange = build; sel.onchange = recolour; empty.onchange = recolour;
document.getElementById('lines').onchange = recolour;
// optional start view from the URL, e.g. choropleths.html#geo=2&var=0&bg=0&lines=0
const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=')));
if (H.geo && geo.options[+H.geo]) geo.value = H.geo;
if (H.var && sel.options[+H.var]) sel.value = H.var;
if (H.lines === '0') document.getElementById('lines').checked = false;
build();
if (H.bg === '0') { const b = document.getElementById('bg'); b.checked = false; b.onchange({target: b}); }
</script></body></html>"""


def build_choropleths(data_dir: Path, out_dir: Path, cfg: dict) -> Path | None:
    from pyproj import Transformer
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    variables = {c: (v if isinstance(v, dict) else {"label": v}) for c, v in cfg.get("variables", {}).items()}
    bounds = cfg.get("boundaries", {})
    grids = cfg.get("grid_km", [])
    t = _postcode_table(data_dir, [*(b["code"] for b in bounds.values()), "easting", "northing"], variables)
    found = [c for c in variables if c in t.column_names]
    missing = [c for c in variables if c not in found]
    if missing:
        log.warning("choropleths: columns not found, skipped: %s", missing)
    geos = []
    for name, spec in bounds.items():
        if spec["code"] not in t.column_names:
            log.warning("choropleths: %s skipped - %s not in the postcode lookup", name, spec["code"])
            continue
        g = _boundary_geography(data_dir, spec, t, variables, to_ll)
        if g:
            g["label"] = spec["label"]
            geos.append(g)
    for km in grids:
        g = _grid_geography(t, float(km) * 1000, variables, to_ll)
        g["label"] = f"OS national grid, {km} km squares"
        geos.append(g)
    if not geos:
        log.warning("choropleths: no geographies - nothing written")
        return None
    data = {"vars": [{"label": variables[c]["label"]} for c in found], "geos": []}
    for g in geos:
        agg = g["agg"]
        data["geos"].append({
            "label": g["label"], "names": g["names"], "rings": g["rings"],
            "postcodes": [int(v) for v in agg["postcodes"]], "addresses": [int(round(v)) for v in agg["addresses"]],
            "values": [_num(pa.chunked_array([pa.array(agg[c], from_pandas=True)])) for c in found]})
        log.info("choropleths: %s - %d areas", g["label"], len(g["names"]))
    page = (TEMPLATE.replace("__CSS__", PAGE_CSS).replace("__TILES__", tile_js("light", "boundaries: ONS (OGL)"))
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    out = out_dir / "choropleths.html"
    out.write_text(page, encoding="utf-8")
    log.info("wrote   %s (%d geographies x %d variables, %.1f MB)", out.name, len(geos), len(found),
             out.stat().st_size / 1e6)
    return out
