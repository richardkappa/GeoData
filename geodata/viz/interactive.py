"""Interactive feature explorer: one Leaflet page per area, every postcode in the
square as a dot coloured by the feature chosen in a drop-down (quantile colours
within the area); hover for all values. Data embedded once, column-wise."""
from __future__ import annotations

import html
import json
import logging
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..postcode import POSTCODE_KEY
from . import PAGE_CSS, label, square, tile_js

log = logging.getLogger(__name__)


def var_label(spec) -> str:
    """A variable's display label: config entries are a label or {label, countries, ...}."""
    return str(spec.get("label", "")) if isinstance(spec, dict) else str(spec)


def _area_table(data_dir: Path, box, variables: dict) -> tuple[pa.Table, dict]:
    """Postcodes in the box with the variables; returns (table, {column: label}).
    Variables with `countries:` are blanked outside the countries their source covers."""
    lookup_f = data_dir / "collate" / "postcode_lookup.parquet"
    feat_f = data_dir / "features" / "postcode_features.parquet"
    lnames = set(pq.read_schema(lookup_f).names)
    fnames = set(pq.read_schema(feat_f).names) if feat_f.exists() else set()
    x0, y0, x1, y1 = box
    base = ["pcds", "easting", "northing", "lat", "long"]
    specs = {c: (v if isinstance(v, dict) else {"label": v}) for c, v in variables.items()}
    if any(s.get("countries") for s in specs.values()) and "ctry26cd" in lnames:
        base.append("ctry26cd")
    lt = pq.read_table(lookup_f, columns=[POSTCODE_KEY, *base, *(c for c in variables if c in lnames and c not in fnames)],
                       filters=[("easting", ">=", x0), ("easting", "<=", x1),
                                ("northing", ">=", y0), ("northing", "<=", y1)])
    if fnames:
        ft = pq.read_table(feat_f, columns=[POSTCODE_KEY, *(c for c in variables if c in fnames)])
        idx = pc.index_in(lt[POSTCODE_KEY], value_set=ft[POSTCODE_KEY])
        lt = lt.filter(pc.is_valid(idx))      # live postcodes (the features spine)
        idx = pc.index_in(lt[POSTCODE_KEY], value_set=ft[POSTCODE_KEY])
        for c in ft.column_names[1:]:
            lt = lt.append_column(c, ft[c].take(idx))
    from .cartograms import _mask_countries   # local import: cartograms imports this module
    lt = _mask_countries(lt, specs)
    found = {c: var_label(s) for c, s in specs.items() if c in lt.column_names}
    return lt, found


def _num(col: pa.ChunkedArray) -> list:
    """Column -> JSON-friendly list (numbers rounded to 3 significant figures, None for missing)."""
    if pa.types.is_boolean(col.type):
        col = pc.cast(col, pa.int8())
    if not (pa.types.is_integer(col.type) or pa.types.is_floating(col.type)):
        return [None if v is None else str(v) for v in col.to_pylist()]
    out = []
    for v in col.to_pylist():
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            out.append(None)
        elif v == 0:
            out.append(0)
        else:
            d = 3 - int(math.floor(math.log10(abs(v)))) - 1
            out.append(round(v, max(d, 0)) if d > 0 else int(round(v, d)))
    return out


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>__CSS__
html,body{height:100%} body{display:flex;flex-direction:column}
#bar{display:flex;gap:14px;align-items:center;padding:10px 26px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
#bar select{font-size:14px;padding:4px;max-width:520px}
#map{flex:1}
.legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px #0003;font-size:12px;line-height:18px}
.legend i{display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:-2px;border-radius:50%}
.tip table{border-collapse:collapse;font-size:12px} .tip td{padding:1px 6px} .tip td:last-child{text-align:right;font-weight:600}
.tip tr.on td{background:#fff3c4}
</style></head><body>
<header><h1>__H1__</h1><p>__SUB__ <a href="index.html">All visualisations</a></p></header>
<div id="bar"><label>Colour postcodes by <select id="var"></select></label><span id="stats"></span></div>
<div id="map"></div>
<script>
const D = __DATA__;
const PAL = ["#440154","#46327e","#365c8d","#277f8e","#1fa187","#4ac16d","#a0da39","#fde725"];
const map = L.map('map', {preferCanvas: true});
__TILES__
L.polygon(D.box, {color: '#222', weight: 1.5, fill: false, dashArray: '6 4'}).addTo(map);
map.fitBounds(D.box);
const sel = document.getElementById('var');
D.vars.forEach((v, i) => { const o = document.createElement('option'); o.value = i; o.textContent = v.label; sel.appendChild(o); });
const fmt = v => v === null ? '–' : (typeof v === 'number' ? v.toLocaleString('en-GB') : v);
let cur = 0, breaks = [];
const markers = D.pc.map((p, i) => L.circleMarker([D.lat[i], D.lon[i]], {radius: 5, weight: 0.6, color: '#333', fillOpacity: 0.85})
  .bindTooltip(() => tip(i), {sticky: true, className: 'tip'}).addTo(map));
function tip(i) {
  let r = '<b>' + D.pc[i] + '</b><table>';
  D.vars.forEach((v, k) => { r += '<tr' + (k == cur ? ' class="on"' : '') + '><td>' + v.label + '</td><td>' + fmt(v.values[i]) + '</td></tr>'; });
  return r + '</table>';
}
function colour(v) {
  if (v === null || typeof v !== 'number') return null;
  let k = 0; while (k < breaks.length && v > breaks[k]) k++;
  return PAL[Math.round(k * (PAL.length - 1) / Math.max(breaks.length, 1))];
}
const legend = L.control({position: 'bottomright'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.id = 'legend'; return d; };
legend.addTo(map);
function draw() {
  cur = +sel.value; const vals = D.vars[cur].values;
  const nums = vals.filter(v => typeof v === 'number').sort((a, b) => a - b);
  const q = p => nums[Math.min(nums.length - 1, Math.floor(p * nums.length))];
  breaks = nums.length ? [...new Set([1, 2, 3, 4, 5, 6, 7].map(k => q(k / 8)))] : [];
  markers.forEach((m, i) => { const c = colour(vals[i]);
    m.setStyle(c ? {fillColor: c, fillOpacity: 0.85, radius: 5} : {fillColor: '#fff', fillOpacity: 0.3, radius: 3}); });
  let h = '<b>' + D.vars[cur].label + '</b><br>';
  if (!nums.length) h += 'no values in this area';
  else { const edges = [nums[0], ...breaks, nums[nums.length - 1]];
    for (let k = 0; k < breaks.length + 1; k++) h += '<i style="background:' + PAL[Math.round(k * (PAL.length - 1) / Math.max(breaks.length, 1))] + '"></i>' + fmt(edges[k]) + ' to ' + fmt(edges[k + 1]) + '<br>'; }
  h += '<i style="background:#fff;border:1px solid #333"></i>no value';
  document.getElementById('legend').innerHTML = h;
  document.getElementById('stats').textContent = nums.length.toLocaleString('en-GB') + ' of ' + vals.length.toLocaleString('en-GB') +
    ' postcodes have a value; colours are eighths (quantiles) within this area';
}
sel.addEventListener('change', draw); draw();
</script></body></html>"""


def build_interactive(data_dir: Path, out_dir: Path, locs: dict, cfg: dict) -> list[Path]:
    from pyproj import Transformer
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    size = float(cfg.get("size_m", 5000))
    variables = cfg.get("variables", {})
    written = []
    for area in cfg.get("areas", []):
        if area not in locs:
            log.warning("interactive: area %s is not a resolved location - skipped", area)
            continue
        box = square(locs[area], size)
        t, found = _area_table(data_dir, box, variables)
        missing = [c for c in variables if c not in found]
        if missing:
            log.warning("interactive %s: columns not found, skipped: %s", area, missing)
        if t.num_rows == 0:
            log.warning("interactive %s: no postcodes in the square - skipped", area)
            continue
        lon, lat = to_ll.transform([box[0], box[2], box[2], box[0]], [box[1], box[1], box[3], box[3]])
        data = {
            "box": [[la, lo] for la, lo in zip(lat, lon)],
            "pc": t["pcds"].to_pylist(),
            "lat": [round(v, 6) for v in t["lat"].to_pylist()],
            "lon": [round(v, 6) for v in t["long"].to_pylist()],
            "vars": [{"label": lab, "values": _num(t[c])} for c, lab in found.items()],
        }
        km = f"{size / 1000:g} km"
        note = locs[area].get("note") or ""
        page = (TEMPLATE.replace("__TITLE__", f"{label(area)} - postcode features")
                .replace("__CSS__", PAGE_CSS)
                .replace("__TILES__", tile_js("streets", "data: ONS, OS, Land Registry, Police, Ofcom, Met Office and others"))
                .replace("__H1__", f"{label(area)}: postcode features")
                .replace("__SUB__", html.escape(f"{t.num_rows:,} live postcodes in a {km} square"
                                                f"{' around ' + note if note else ''}. Pick a feature to colour the "
                                                "dots; hover a dot for every value."))
                .replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False)))
        out = out_dir / f"interactive_{area}.html"
        out.write_text(page, encoding="utf-8")
        log.info("wrote   %s (%s postcodes, %d features, %.1f MB)", out.name, f"{t.num_rows:,}", len(found),
                 out.stat().st_size / 1e6)
        written.append(out)
    return written
