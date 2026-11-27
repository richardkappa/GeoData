"""Section 4: visualisations written as self-contained HTML pages."""
from __future__ import annotations

import base64
import html
import io as _io
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..postcode import POSTCODE_KEY

log = logging.getLogger(__name__)


def resolve_locations(data_dir: Path, locations: dict) -> dict[str, dict]:
    """{name: {x, y, note}} from config entries given as `xy` or `postcode`."""
    out, want = {}, {}
    for name, spec in locations.items():
        spec = spec or {}
        if "xy" in spec:
            out[name] = {"x": float(spec["xy"][0]), "y": float(spec["xy"][1]), "note": spec.get("note", "")}
        elif "postcode" in spec:
            want[name] = "".join(str(spec["postcode"]).upper().split())
        else:
            log.warning("location %s has neither xy nor postcode - skipped", name)
    if want:
        t = pq.read_table(data_dir / "1_extract" / "ons_nspl" / "nspl.parquet",
                          columns=[POSTCODE_KEY, "easting", "northing"])
        t = t.filter(pc.is_in(t[POSTCODE_KEY], value_set=pa.array(list(want.values()))))
        xy = {p: (e, n) for p, e, n in zip(*(t[c].to_pylist() for c in (POSTCODE_KEY, "easting", "northing")))}
        for name, p in want.items():
            if p in xy and xy[p][0] is not None:
                spec = locations[name]
                out[name] = {"x": float(xy[p][0]), "y": float(xy[p][1]),
                             "note": spec.get("note") or spec["postcode"]}
            else:
                log.warning("location %s: postcode %s not found in the NSPL - skipped", name, p)
    return {k: out[k] for k in locations if k in out}   # keep config order


def square(loc: dict, size_m: float):
    h = size_m / 2
    return loc["x"] - h, loc["y"] - h, loc["x"] + h, loc["y"] + h


def png_data_uri(fig, dpi: int = 110) -> str:
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def label(name: str) -> str:
    return html.escape(name.replace("_", " "))


# Background map tiles. The pages are opened as local files, which OpenStreetMap's own
# servers refuse (no referrer) and CARTO only serves with an API key; Esri's tiles accept
# them without one. Overridden from `global.basemaps` in config/4_visualise.yaml.
ESRI = {
    "streets": {"url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                "attribution": "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors", "max_zoom": 19},
    "light": {"url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
              "attribution": "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors", "max_zoom": 16},
}
BASEMAPS = {**ESRI, **{f"esri_{k}": v for k, v in ESRI.items()}}


def _basemap(kind: str) -> dict:
    """Resolve a basemap: a `key_env` basemap needs that environment variable (its value fills `{key}` in the
    url) and otherwise falls back to `fallback` (default: the built-in Esri tiles)."""
    import os
    b = dict(BASEMAPS.get(kind) or next(iter(BASEMAPS.values())))
    env = b.get("key_env")
    if env:
        key = os.environ.get(env)
        if not key:
            log.warning("basemap %s needs the environment variable %s - using %s instead", kind, env,
                        b.get("fallback", "esri_" + kind))
            return dict(BASEMAPS.get(b.get("fallback", "esri_" + kind)) or ESRI[kind if kind in ESRI else "streets"])
        b["url"] = b["url"].replace("{key}", key)
    return b


def tile_js(kind: str = "streets", extra_attribution: str = "") -> str:
    """JavaScript that adds the configured tile layer to `map`."""
    b = _basemap(kind)
    attr = b["attribution"] + (" | " + extra_attribution if extra_attribution else "")
    return (f"L.tileLayer({json.dumps(b['url'])}, {{maxZoom: {int(b.get('max_zoom', 19))}, "
            f"attribution: {json.dumps(attr)}}}).addTo(map);")


# Derived postcode variables {name: {expr, uses, fill}}, set from `derived:` in
# config/4_visualise.yaml by 04_visualise.py; computed before area averages.
DERIVED: dict = {}

# Colour scales shared by the cartogram, choropleth and small-multiple pages.
# meta = {scale: 'quantile' (default) | 'diverging', cats: [labels] for categories}
SCALE_JS = r"""
const PAL = ["#440154","#46327e","#365c8d","#277f8e","#1fa187","#4ac16d","#a0da39","#fde725"];
const DIV = ["#2166ac","#4393c3","#92c5de","#d1e5f0","#fddbc7","#f4a582","#d6604d","#b2182b"];
const CAT = ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#1f78b4","#b15928","#666666"];
const BIV = ["#e8e8e8","#ace4e4","#5ac8c8","#dfb0d6","#a5add3","#5698b9","#be64ac","#8c62aa","#3b4994"];
const fmt = v => v === null || v === undefined ? '–' : (typeof v === 'number' ? v.toLocaleString('en-GB') : v);
function makeScale(vals, meta) {
  meta = meta || {};
  if (meta.cats) {   // categories (e.g. clusters): integer codes -> labels
    return {col: v => (typeof v === 'number') ? CAT[v % CAT.length] : null, n: vals.filter(v => typeof v === 'number').length,
            rows: meta.cats.map((c, i) => [CAT[i % CAT.length], c]), note: ''};
  }
  const nums = vals.filter(v => typeof v === 'number').sort((a, b) => a - b);
  const q = p => nums[Math.min(nums.length - 1, Math.max(0, Math.floor(p * nums.length)))];
  if (!nums.length) return {col: () => null, rows: [], n: 0, note: 'no values'};
  if (meta.scale === 'diverging') {   // symmetric around 0: blue = fall, red = rise
    const m = Math.max(Math.abs(q(0.05)), Math.abs(q(0.95))) || 1;
    const br = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75].map(f => f * m);
    const col = v => { if (typeof v !== 'number') return null; let k = 0; while (k < br.length && v > br[k]) k++; return DIV[k]; };
    const r = x => +x.toPrecision(2);
    const rows = DIV.map((c, k) => [c, k === 0 ? 'below ' + fmt(r(br[0])) : k === 7 ? 'above ' + fmt(r(br[6])) : fmt(r(br[k - 1])) + ' to ' + fmt(r(br[k]))]);
    return {col, rows, n: nums.length, note: 'blue = fall, red = rise; centred on zero'};
  }
  const br = [...new Set([1, 2, 3, 4, 5, 6, 7].map(k => q(k / 8)))];
  const idx = k => Math.round(k * (PAL.length - 1) / Math.max(br.length, 1));
  const col = v => { if (typeof v !== 'number') return null; let k = 0; while (k < br.length && v > br[k]) k++; return PAL[idx(k)]; };
  const e = [nums[0], ...br, nums[nums.length - 1]];
  const rows = []; for (let k = 0; k <= br.length; k++) rows.push([PAL[idx(k)], fmt(e[k]) + ' to ' + fmt(e[k + 1])]);
  return {col, rows, n: nums.length, note: 'eighths (quantiles) of the areas'};
}
function bivScale(a, b) {   // two variables -> 3x3 tertile colours
  const t = vals => { const n = vals.filter(v => typeof v === 'number').sort((x, y) => x - y);
    return n.length ? [n[Math.floor(n.length / 3)], n[Math.floor(2 * n.length / 3)]] : [0, 0]; };
  const ta = t(a), tb = t(b), cls = (v, tt) => v <= tt[0] ? 0 : v <= tt[1] ? 1 : 2;
  return {col: i => (typeof a[i] === 'number' && typeof b[i] === 'number') ? BIV[cls(a[i], ta) + 3 * cls(b[i], tb)] : null};
}
function legendHtml(title, s, extra) {
  let h = '<b>' + title + '</b><br>';
  s.rows.forEach(r => h += '<i style="background:' + r[0] + '"></i>' + r[1] + '<br>');
  return h + (s.note ? '<span style="color:#777">' + s.note + '</span>' : '') + (extra || '');
}
function bivLegendHtml(la, lb) {
  let g = '<div style="display:grid;grid-template-columns:repeat(3,18px);gap:1px;transform:rotate(0deg)">';
  for (let y = 2; y >= 0; y--) for (let x = 0; x < 3; x++) g += '<div style="width:18px;height:18px;background:' + BIV[x + 3 * y] + '"></div>';
  return '<b>Two variables</b><div style="display:flex;gap:8px;align-items:center;margin:4px 0">' + g + '</div>' +
    '<div style="font-size:11px">&rarr; ' + la + ' (low to high)<br>&uarr; ' + lb + ' (low to high)</div>' +
    '<span style="color:#777">thirds (tertiles) of each</span>';
}
function varIndex(list, v) {   // URL hash value: an index or a variable key
  if (v === undefined) return null; if (/^\d+$/.test(v)) return +v;
  const i = list.findIndex(x => x.key === v); return i < 0 ? null : i;
}
"""

PAGE_CSS = """
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;margin:0;background:#f6f6f3;color:#222}
header{padding:18px 26px;background:#24343f;color:#fff}
header h1{margin:0 0 4px;font-size:22px;font-weight:600} header p{margin:0;opacity:.8;font-size:14px}
header a{color:#cde3f5}
main{padding:20px 26px}
"""
