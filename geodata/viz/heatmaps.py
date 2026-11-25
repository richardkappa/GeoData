"""Heat maps: where point data is concentrated (crimes, collisions, listed buildings,
food outlets, house sales ...). Points are counted in national grid cells and drawn
as a zoomable heat layer (Leaflet.heat). One data file per dataset, loaded on demand."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml

from .. import io
from ..config import REPO_ROOT
from . import PAGE_CSS, tile_js

log = logging.getLogger(__name__)

TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoData heat maps</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>__CSS__
html,body{height:100%} body{display:flex;flex-direction:column}
#bar{display:flex;gap:18px;align-items:center;padding:10px 26px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
#bar select{font-size:14px;padding:4px} #map{flex:1}
#help{padding:6px 26px 8px;background:#fafaf7;border-bottom:1px solid #e5e5e5;font-size:13px;color:#444;line-height:1.45}
.legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px #0003;font-size:12px;width:200px}
.bar{height:10px;border-radius:3px;background:linear-gradient(90deg,#2b83ba,#abdda4,#ffffbf,#fdae61,#d7191c)}
</style></head><body>
<header><h1>Heat maps</h1><p>Where things are concentrated. Points are counted in __CELL__ m national grid cells;
colour shows how many fall nearby, relative to the busiest places in view on the scale chosen.
<a href="index.html">All visualisations</a></p></header>
<div id="bar"><label>Show <select id="ds"></select></label>
<label title="Which cells count as full heat (red). Lower settings make more places look hot.">Hot spot level
<select id="mx"><option value="0.99">top 1% of cells</option><option value="0.95" selected>top 5%</option>
<option value="0.8">top 20%</option></select></label>
<label title="How far each cell's heat is smeared on screen. Wider merges nearby places into bigger blobs.">Spread
<select id="rad"><option value="10">tight</option><option value="18" selected>medium</option><option value="30">wide</option></select></label>
<label><input type="checkbox" id="bg" checked> background map</label><span id="note"></span></div>
<div id="help"></div>
<div id="map"></div>
<script>
const M = __META__;
window.HEAT = {};
const map = L.map('map').setView([54.3, -2.5], 6);
__TILES__
let tiles = null; map.eachLayer(l => { if (l instanceof L.TileLayer) tiles = l; });
const ds = document.getElementById('ds'), mx = document.getElementById('mx'), rad = document.getElementById('rad');
M.datasets.forEach(d => ds.add(new Option(d.label + ' (' + d.points.toLocaleString('en-GB') + ')', d.name)));
let heat = null;
const legend = L.control({position: 'bottomright'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.id = 'legend'; return d; };
legend.addTo(map);
function draw() {
  const D = window.HEAT[ds.value]; if (!D) return;
  if (heat) map.removeLayer(heat);
  const w = D.pts.map(p => p[2]).sort((a, b) => a - b);
  const top = w[Math.min(w.length - 1, Math.floor(+mx.value * w.length))] || 1;
  heat = L.heatLayer(D.pts, {radius: +rad.value, blur: +rad.value * 0.8, max: top, minOpacity: 0.25,
    gradient: {0.2: '#2b83ba', 0.4: '#abdda4', 0.6: '#ffffbf', 0.8: '#fdae61', 1: '#d7191c'}}).addTo(map);
  const meta = M.datasets.find(d => d.name === ds.value);
  document.getElementById('legend').innerHTML = '<b>' + meta.label + '</b><div class="bar"></div>' +
    '<div style="display:flex;justify-content:space-between"><span>few</span><span>' + top.toLocaleString('en-GB') +
    '+ per cell</span></div><span style="color:#777">' + meta.note + '</span>';
  document.getElementById('note').textContent = D.pts.length.toLocaleString('en-GB') + ' occupied cells';
}
function help() {   // plain-English explanation of the two controls, kept in step with the choices
  const px = +rad.value, mPerPx = 40075016 * Math.cos(map.getCenter().lat * Math.PI / 180) / Math.pow(2, map.getZoom() + 8);
  const km = px * mPerPx / 1000, kmTxt = km >= 10 ? Math.round(km) : km >= 1 ? km.toFixed(1) : km.toFixed(2);
  const lvl = {0.99: 'the busiest 1%', 0.95: 'the busiest 5%', 0.8: 'the busiest 20%'}[mx.value];
  document.getElementById('help').innerHTML =
    '<b>Spread</b> is how far each cell\\'s heat is smeared out: every __CELL__ m cell is drawn as a soft blob reaching ' + px +
    ' pixels from its centre, which at this zoom is about <b>' + kmTxt + ' km</b> on the ground. Places closer together than that ' +
    'merge into one hot spot, so <i>tight</i> shows individual streets and estates (zoom in), <i>wide</i> shows the broad ' +
    'pattern across a region. Because the blob is a fixed size on screen, it covers less ground as you zoom in. ' +
    '<b>Hot spot level</b> sets what counts as full heat (red): cells in ' + lvl + ' and anything nearby adding up to as ' +
    'much; a lower level makes more places look hot.';
}
map.on('zoomend', help);
function load() {
  if (window.HEAT[ds.value]) return draw();
  const s = document.createElement('script'); s.src = 'heat_data/' + ds.value + '.js'; s.onload = draw;
  s.onerror = () => alert('Could not load heat_data/' + ds.value + '.js - run 04_visualise.py'); document.body.appendChild(s);
}
ds.onchange = load; mx.onchange = () => { draw(); help(); }; rad.onchange = () => { draw(); help(); };
help();
document.getElementById('bg').onchange = e => { if (tiles) e.target.checked ? tiles.addTo(map) : map.removeLayer(tiles); };
const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=')));
if (H.ds && M.datasets.some(d => d.name === H.ds)) ds.value = H.ds;
load();
</script></body></html>"""


def build_heatmaps(data_dir: Path, out_dir: Path, cfg: dict) -> Path | None:
    from pyproj import Transformer
    from ..features.data import load_points, load_spine
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    fcfg = yaml.safe_load(open(REPO_ROOT / "config" / "features.yaml", encoding="utf-8"))
    points = fcfg.get("points", {})
    cell = float(cfg.get("cell_m", 500))
    _, full = load_spine(data_dir, fcfg)
    hd = out_dir / "heat_data"
    hd.mkdir(parents=True, exist_ok=True)
    meta = []
    for name, spec in cfg.get("datasets", {}).items():
        spec = spec if isinstance(spec, dict) else {"label": spec}
        if name not in points:
            log.warning("heatmaps: %s is not a named point dataset in config/features.yaml - skipped", name)
            continue
        try:
            P = load_points(data_dir, points[name], full)
        except FileNotFoundError as e:
            log.warning("heatmaps: %s skipped: %s", name, e)
            continue
        if len(P.x) == 0:
            continue
        ix = np.floor(P.x / cell).astype(np.int64)
        iy = np.floor(P.y / cell).astype(np.int64)
        key = ix * 10_000_000 + iy
        cells, counts = np.unique(key, return_counts=True)
        cx = (cells // 10_000_000 + 0.5) * cell
        cy = (cells % 10_000_000 + 0.5) * cell
        lon, lat = to_ll.transform(cx, cy)
        pts = np.column_stack([np.round(lat, 4), np.round(lon, 4), counts]).tolist()
        pts = [[a, b, int(c)] for a, b, c in pts]
        (hd / f"{name}.js").write_text(f"window.HEAT[{json.dumps(name)}]=" +
                                       json.dumps({"pts": pts}, separators=(",", ":")) + ";", encoding="utf-8")
        meta.append({"name": name, "label": spec["label"], "points": int(len(P.x)), "note": spec.get("note", "")})
        log.info("heatmaps: %s - %s points in %s cells", name, f"{len(P.x):,}", f"{len(cells):,}")
    if not meta:
        log.warning("heatmaps: no datasets - nothing written")
        return None
    page = (TEMPLATE.replace("__CSS__", PAGE_CSS).replace("__CELL__", f"{cell:g}")
            .replace("__TILES__", tile_js("light")).replace("__META__", json.dumps({"datasets": meta})))
    out = out_dir / "heatmaps.html"
    out.write_text(page, encoding="utf-8")
    size = sum(p.stat().st_size for p in hd.glob("*.js")) / 1e6
    log.info("wrote   %s (%d datasets, %.1f MB of data files)", out.name, len(meta), size)
    return out
