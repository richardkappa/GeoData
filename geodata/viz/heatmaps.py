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
{ const g1 = document.createElement('optgroup'), g2 = document.createElement('optgroup');
  g1.label = 'Where things are (counts)'; g2.label = 'Rates per head of population';
  M.datasets.forEach(d => (d.kind === 'rate' ? g2 : g1).appendChild(new Option(d.label + (d.kind === 'rate' ? '' : ' (' + d.points.toLocaleString('en-GB') + ')'), d.name)));
  ds.appendChild(g1); if (g2.children.length) ds.appendChild(g2); }
const RPAL = ["#ffffb2","#fed976","#feb24c","#fd8d3c","#fc4e2a","#e31a1c","#bd0026","#800026"];
let heat = null;
const metaOf = () => M.datasets.find(d => d.name === ds.value);
const legend = L.control({position: 'bottomright'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.id = 'legend'; return d; };
legend.addTo(map);
function drawRate(D, meta) {   // smoothed rate per head: coloured squares, eighths of the cells
  const v = D.cells.map(c => c[4]).sort((a, b) => a - b), q = p => v[Math.min(v.length - 1, Math.floor(p * v.length))];
  const br = [...new Set([1, 2, 3, 4, 5, 6, 7].map(k => q(k / 8)))];
  const idx = k => Math.round(k * (RPAL.length - 1) / Math.max(br.length, 1));
  const col = x => { let k = 0; while (k < br.length && x > br[k]) k++; return RPAL[idx(k)]; };
  heat = L.layerGroup();
  const r = L.canvas();
  D.cells.forEach(c => { const k = col(c[4]);   // hairline outline in the same colour closes seams between squares
    L.rectangle([[c[0], c[1]], [c[2], c[3]]], {renderer: r, color: k, weight: 0.6, opacity: 0.75, fillColor: k, fillOpacity: 0.75})
    .bindTooltip(meta.label + ': <b>' + c[4].toLocaleString('en-GB') + '</b>', {sticky: true}).addTo(heat); });
  heat.addTo(map);
  const e = [v[0], ...br, v[v.length - 1]];
  let h = '<b>' + meta.label + '</b><br>';
  for (let k = 0; k <= br.length; k++) h += '<i style="display:inline-block;width:14px;height:12px;margin-right:6px;background:' + RPAL[idx(k)] +
    '"></i>' + (+e[k].toPrecision(3)).toLocaleString('en-GB') + ' to ' + (+e[k + 1].toPrecision(3)).toLocaleString('en-GB') + '<br>';
  document.getElementById('legend').innerHTML = h + '<span style="color:#777">eighths of the populated squares. ' + meta.note + '</span>';
  document.getElementById('note').textContent = D.cells.length.toLocaleString('en-GB') + ' populated squares';
}
function draw() {
  const D = window.HEAT[ds.value]; if (!D) return;
  if (heat) map.removeLayer(heat);
  const meta0 = metaOf(), isRate = meta0.kind === 'rate';
  mx.disabled = isRate; rad.disabled = isRate; help();
  if (isRate) return drawRate(D, meta0);
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
  const m = metaOf();
  if (m && m.kind === 'rate') {
    document.getElementById('help').innerHTML = '<b>Rate per head</b>: instead of where things happen, this shows how often ' +
      'they happen for the number of people living there, so busy city centres no longer dominate just because many people ' +
      'live or pass through. ' + m.note + '. Spread and hot spot level do not apply to rates.';
    return;
  }
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


def _rate_maps(rcfg: dict, points: dict, data_dir: Path, full, hd: Path, to_ll) -> list[dict]:
    """Rates per head: points and people counted in national grid cells, both summed over a
    window of cells (smoothing), rate = points per `scale` people where enough people live."""
    if not rcfg.get("datasets"):
        return []
    from scipy.ndimage import uniform_filter
    from ..features.build import postcode_population
    from ..features.data import load_points
    cell = float(rcfg.get("cell_m", 2000))
    win = int(rcfg.get("smooth_cells", 3))
    min_pop = float(rcfg.get("min_population", 1000))
    pop = postcode_population(full)
    ok = np.isfinite(full.x) & np.isfinite(full.y)
    x0, y0 = np.floor(np.nanmin(full.x[ok]) / cell), np.floor(np.nanmin(full.y[ok]) / cell)
    nx_ = int(np.floor(np.nanmax(full.x[ok]) / cell) - x0) + 1
    ny_ = int(np.floor(np.nanmax(full.y[ok]) / cell) - y0) + 1

    def grid(x, y, w=None):
        ix = (np.floor(x / cell) - x0).astype(np.int64)
        iy = (np.floor(y / cell) - y0).astype(np.int64)
        m = (ix >= 0) & (ix < nx_) & (iy >= 0) & (iy < ny_)
        g = np.zeros(nx_ * ny_)
        np.add.at(g, ix[m] * ny_ + iy[m], 1.0 if w is None else w[m])
        return g.reshape(nx_, ny_)

    def window(g):   # sum over win x win cells (clipped: the filter leaves tiny negative rounding errors)
        return np.maximum(uniform_filter(g, size=win, mode="constant") * win * win, 0.0)

    denoms = {}
    out = []
    for name, spec in rcfg["datasets"].items():
        spec = spec if isinstance(spec, dict) else {"label": spec}
        src = spec.get("points", name)
        if src not in points:
            log.warning("heatmaps: rate %s - %s is not a point dataset in config/features.yaml", name, src)
            continue
        per = spec.get("per", "residents")
        if per not in pop:
            log.warning("heatmaps: rate %s - no %s denominator", name, per)
            continue
        try:
            P = load_points(data_dir, points[src], full)
        except FileNotFoundError as e:
            log.warning("heatmaps: rate %s skipped: %s", name, e)
            continue
        if per not in denoms:
            w = np.where(ok, pop[per], 0.0)
            d = grid(np.where(ok, full.x, -1e9), np.where(ok, full.y, -1e9), w)
            denoms[per] = (d, window(d))
        d_cell, d_win = denoms[per]
        n_win = window(grid(P.x, P.y))
        scale = float(spec.get("scale", 1000))
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(d_win >= min_pop, scale * n_win / d_win, np.nan)
        show = np.argwhere((d_cell > 0) & np.isfinite(rate))      # populated cells only
        sx = (show[:, 0] + x0) * cell
        sy = (show[:, 1] + y0) * cell
        lon0, lat0 = to_ll.transform(sx, sy)
        lon1, lat1 = to_ll.transform(sx + cell, sy + cell)
        vals = rate[show[:, 0], show[:, 1]]
        cells = [[round(a, 4), round(b, 4), round(c, 4), round(e, 4), float(f"{v:.3g}")]
                 for a, b, c, e, v in zip(lat0, lon0, lat1, lon1, vals)]
        key = f"rate_{name}"
        (hd / f"{key}.js").write_text(f"window.HEAT[{json.dumps(key)}]=" + json.dumps({"cells": cells}, separators=(",", ":"))
                                      + ";", encoding="utf-8")
        km, wkm = cell / 1000, cell * win / 1000
        out.append({"name": key, "label": spec["label"], "kind": "rate", "points": int(len(P.x)),
                    "note": spec.get("note", "") + (" " if spec.get("note") else "") +
                    f"{km:g} km squares; points and {per} summed over the surrounding {wkm:g} x {wkm:g} km "
                    f"to steady the rate; shown where at least {min_pop:,.0f} {per} live in that area"})
        log.info("heatmaps: rate %s - %s cells", name, f"{len(cells):,}")
    return out


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
    rates = _rate_maps(cfg.get("rates") or {}, points, data_dir, full, hd, to_ll)
    meta += rates
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
