"""Hex cartograms: features averaged to Westminster constituencies and local
authorities, each area drawn as one equal hexagon using Open Innovations'
HexJSON layouts (https://hexmap.uk, MIT licence)."""
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

from .. import download as dl
from ..postcode import POSTCODE_KEY
from . import DERIVED, PAGE_CSS, SCALE_JS
from .interactive import _num

log = logging.getLogger(__name__)


def _postcode_table(data_dir: Path, codes: list[str], variables: dict) -> pa.Table:
    """Live postcodes with area codes, address weights and the requested variables
    (derived variables are computed from their `uses` columns)."""
    derived = {c: DERIVED[c] for c in variables if c in DERIVED}
    if derived:
        base = {c: v for c, v in variables.items() if c not in derived}
        for spec in derived.values():
            for u in spec.get("uses", []):
                base.setdefault(u, {"label": u})
        t = _postcode_table(data_dir, codes, base)
        return _add_derived(t, derived)
    lookup_f = data_dir / "collate" / "postcode_lookup.parquet"
    feat_f = data_dir / "features" / "postcode_features.parquet"
    lnames = set(pq.read_schema(lookup_f).names)
    if any(v.get("countries") for v in variables.values()) and "ctry26cd" in lnames:
        codes = [*codes, "ctry26cd"]
    fnames = set(pq.read_schema(feat_f).names) if feat_f.exists() else set()
    lcols = [POSTCODE_KEY, *(c for c in codes if c in lnames), *(c for c in ["uprn_count"] if c in lnames),
             *(c for c in variables if c in lnames and c not in fnames)]
    lt = pq.read_table(lookup_f, columns=list(dict.fromkeys(lcols)))
    if fnames:
        fcols = [*(c for c in variables if c in fnames), *(c for c in codes if c in fnames and c not in lnames)]
        ft = pq.read_table(feat_f, columns=[POSTCODE_KEY, *dict.fromkeys(fcols)])
        idx = pc.index_in(lt[POSTCODE_KEY], value_set=ft[POSTCODE_KEY])
        lt = lt.filter(pc.is_valid(idx))   # live postcodes only (the features spine)
        idx = pc.index_in(lt[POSTCODE_KEY], value_set=ft[POSTCODE_KEY])
        for c in ft.column_names[1:]:
            lt = lt.append_column(c, ft[c].take(idx))
    return _mask_countries(lt, variables)


def _mask_countries(t: pa.Table, variables: dict) -> pa.Table:
    """Blank variables outside the countries their source covers (`countries: [E, W]`), e.g. drive time
    to A&E uses CQC data for England only, so elsewhere the 'nearest' A&E would be a false long trip."""
    if "ctry26cd" not in t.column_names:
        return t
    first = pc.utf8_slice_codeunits(t["ctry26cd"], 0, 1)
    for c, spec in variables.items():
        if spec.get("countries") and c in t.column_names:
            keep = pc.fill_null(pc.is_in(first, value_set=pa.array([str(x) for x in spec["countries"]])), False)
            nulls = pa.nulls(t.num_rows, t[c].type)
            t = t.set_column(t.schema.get_field_index(c), c, pc.if_else(keep, t[c], nulls))
    return t


def _add_derived(t: pa.Table, derived: dict) -> pa.Table:
    """Append derived columns: a pandas expression over `uses` columns (cast to numbers,
    booleans to 0/1, missing values optionally filled per column with `fill`)."""
    import pandas as pd
    for name, spec in derived.items():
        uses = [u for u in spec.get("uses", []) if u in t.column_names]
        if len(uses) < len(spec.get("uses", [])):
            log.warning("derived %s skipped: missing %s", name, sorted(set(spec["uses"]) - set(uses)))
            continue
        def num(col):
            if pa.types.is_boolean(col.type) or pa.types.is_integer(col.type) or pa.types.is_floating(col.type):
                return pc.cast(col, pa.float64()).to_pandas()
            return pd.to_numeric(col.to_pandas(), errors="coerce").astype("float64")   # e.g. '1'..'3' classes
        df = pd.DataFrame({u: num(t[u]) for u in uses})
        fill = spec.get("fill") or {}
        df = df.fillna({c: v for c, v in fill.items() if c in df}) if isinstance(fill, dict) else df.fillna(fill)
        vals = df.eval(spec["expr"]).astype("float64")
        t = t.append_column(name, pa.array(vals.to_numpy(), from_pandas=True))
    return t


def _aggregate(t: pa.Table, code_col: str, area_codes: list[str], variables: dict) -> dict:
    """Address-weighted mean of each variable per area ('sum' variables are summed)."""
    pos = {c: i for i, c in enumerate(area_codes)}
    codes = t[code_col].to_pylist()
    g = np.array([pos.get(c, -1) for c in codes])
    ok = g >= 0
    g = g[ok]
    n = len(area_codes)
    w = t["uprn_count"].to_numpy(zero_copy_only=False).astype("f8")[ok] if "uprn_count" in t.column_names \
        else np.ones(int(ok.sum()))
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)   # postcodes without an address count still count once
    out = {"postcodes": np.bincount(g, minlength=n).astype("f8"),
           "addresses": np.bincount(g, weights=w, minlength=n)}
    for c, spec in variables.items():
        if c not in t.column_names:
            continue
        col = t[c]
        if pa.types.is_boolean(col.type):
            col = pc.cast(col, pa.float64())
        if not (pa.types.is_integer(col.type) or pa.types.is_floating(col.type)):
            continue
        v = col.to_numpy(zero_copy_only=False).astype("f8")[ok]
        has = np.isfinite(v)
        if spec.get("agg") == "sum":
            out[c] = np.bincount(g[has], weights=v[has], minlength=n)
        else:
            sw = np.bincount(g[has], weights=w[has], minlength=n)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[c] = np.where(sw > 0, np.bincount(g[has], weights=(v * w)[has], minlength=n) / sw, np.nan)
    return out


def _square_layout(f: Path, spec: dict):
    """OS equal-area cartogram GeoPackage -> one unit square per area plus outline paths."""
    import pyogrio
    import shapely
    sq = pyogrio.read_dataframe(f, layer=spec.get("layer", "square"))
    key, nm = spec.get("key", "lad_code"), spec.get("name_col", "lad_name")
    b = shapely.bounds(sq.geometry.values)
    shape = {"kind": "square", "x": b[:, 0].tolist(), "y": b[:, 1].tolist(), "outlines": []}
    for layer in spec.get("outlines", ["region_outline", "country_outline"]):
        try:
            g = pyogrio.read_dataframe(f, layer=layer).geometry.values
        except Exception:
            continue
        for line in shapely.get_parts(shapely.get_parts(g)):
            xy = shapely.get_coordinates(line)
            shape["outlines"].append({"w": 2.4 if "country" in layer else 1.2,
                                      "pts": [[float(x), float(y)] for x, y in xy]})
    return shape, sq[key].astype(str).tolist(), sq[nm].astype(str).tolist()


def _rings(geom, scale: float, digits: int = 2):
    """Polygon/MultiPolygon -> list of exterior rings as flat [x0,y0,x1,y1...] lists (scaled, rounded)."""
    import shapely
    out = []
    for part in shapely.get_parts(geom):
        xy = shapely.get_coordinates(shapely.get_exterior_ring(part)) / scale
        out.append(np.round(xy, digits).ravel().tolist())
    return out


def _geojson_layout(f: Path, spec: dict):
    """GeoJSON polygon cartogram (one feature per area, e.g. hexmap.uk MSOA 2021 hexes)."""
    import shapely
    j = json.loads(f.read_text(encoding="utf-8"))
    feats = [x for x in j["features"]
             if not spec.get("feature_type") or x["properties"].get("type") == spec["feature_type"]]
    scale = float(spec.get("scale", 1))
    key, nm = spec.get("key", "id"), spec.get("name_col", "n")
    rings = [_rings(shapely.geometry.shape(x["geometry"]), scale, spec.get("digits", 2)) for x in feats]
    shape = {"kind": "poly", "rings": rings, "outlines": [], "labels": []}
    return shape, [str(x["properties"][key]) for x in feats], [str(x["properties"].get(nm, "")) for x in feats]


def _polygon_layout(f: Path, spec: dict):
    """Polygon cartogram in a GeoPackage (e.g. House of Commons Library MSOA hexes) -> one shape per area,
    plus outline and label layers. Coordinates are scaled to km so the page stays small."""
    import pyogrio
    import shapely
    scale = float(spec.get("scale", 1000))
    g = pyogrio.read_dataframe(f, layer=spec["layer"])
    key, nm = spec["key"], spec.get("name_col", spec["key"])
    shape = {"kind": "poly", "rings": [_rings(x, scale) for x in g.geometry.values], "outlines": [], "labels": []}
    for layer in spec.get("outlines", []):
        try:
            o = pyogrio.read_dataframe(f, layer=layer).geometry.values
        except Exception as e:
            log.warning("cartograms: outline layer %s not read (%s)", layer, e)
            continue
        for line in shapely.get_parts(shapely.get_parts(shapely.boundary(o))):
            xy = np.round(shapely.get_coordinates(line) / scale, 2)
            shape["outlines"].append({"w": 1, "pts": xy.tolist()})
    if spec.get("labels"):
        lab = spec["labels"]
        try:
            p = pyogrio.read_dataframe(f, layer=lab["layer"])
            for geom, text in zip(p.geometry.values, p[lab["text"]].astype(str)):
                shape["labels"].append([round(geom.x / scale, 2), round(geom.y / scale, 2), text])
        except Exception as e:
            log.warning("cartograms: label layer %s not read (%s)", lab.get("layer"), e)
    return shape, g[key].astype(str).tolist(), g[nm].astype(str).tolist()


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoData hex cartograms</title><style>__CSS__
#bar{display:flex;gap:18px;align-items:center;padding:10px 26px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
#bar select{font-size:14px;padding:4px;max-width:560px}
#wrap{display:flex;gap:20px;padding:16px 26px;align-items:flex-start;flex-wrap:wrap}
#svg{background:#fff;border-radius:6px;box-shadow:0 1px 3px #0002;flex:1;min-width:420px;max-width:880px}
#side{width:300px;font-size:13px}
#side .box{background:#fff;border-radius:6px;padding:10px 12px;box-shadow:0 1px 3px #0002;margin-bottom:12px}
.legend i{display:inline-block;width:16px;height:14px;margin-right:6px;vertical-align:-2px}
path{vector-effect:non-scaling-stroke}
path.hex{stroke:#fff;stroke-width:0.5;cursor:pointer} path.hex:hover{stroke:#111;stroke-width:2}
#tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #bbb;border-radius:5px;padding:6px 8px;font-size:12px;box-shadow:0 2px 6px #0003;display:none;max-width:300px}
ol{padding-left:20px;margin:4px 0} li{margin:1px 0}
</style></head><body>
<header><h1>Cartograms</h1><p>Every area drawn the same size, so densely populated places are as visible as large rural ones.
Values are averages of the postcodes in each area, weighted by their number of addresses.
Layouts: <a href="https://hexmap.uk">Open Innovations hexmaps</a> (MIT) and the
<a href="https://github.com/OrdnanceSurvey/equal-area-cartogram">Ordnance Survey equal-area cartogram</a>
(OGL v3; contains OS and National Statistics data © Crown copyright and database right 2019).
<a href="index.html">All visualisations</a></p></header>
<div id="bar"><label>Areas <select id="lay"></select></label><label>Colour by <select id="var"></select></label></div>
<div id="wrap"><svg id="svg"></svg><div id="side"><div class="box legend" id="legend"></div>
<div class="box"><b>Highest</b><ol id="top"></ol><b>Lowest</b><ol id="bottom"></ol></div></div></div>
<div id="tip"></div>
<script>
const D = __DATA__;
__SCALE__
const lay = document.getElementById('lay'), sel = document.getElementById('var'), svg = document.getElementById('svg'), tip = document.getElementById('tip');
D.layouts.forEach((l, i) => lay.add(new Option(l.label, i)));
D.vars.forEach((v, i) => sel.add(new Option(v.label, i)));
function centres(L) {
  const s = 10, w = Math.sqrt(3) * s, h = 2 * s, pts = [];
  L.q.forEach((q, i) => { const r = L.r[i], odd = Math.abs(r) % 2 == 1;
    if (L.layout.endsWith('-r')) { const shift = (L.layout == 'odd-r') == odd ? w / 2 : 0; pts.push([q * w + shift, -r * 0.75 * h]); }
    else { const oddq = Math.abs(q) % 2 == 1, shift = (L.layout == 'odd-q') == oddq ? -w / 2 : 0; pts.push([q * 0.75 * h, -r * w + shift]); } });
  return {pts, s, pointy: L.layout.endsWith('-r')};
}
function hexPath(x, y, s, pointy) {
  let d = '';
  for (let k = 0; k < 6; k++) { const a = Math.PI / 180 * (60 * k + (pointy ? 30 : 0));
    d += (k ? 'L' : 'M') + (x + s * Math.cos(a)).toFixed(2) + ',' + (y + s * Math.sin(a)).toFixed(2); }
  return d + 'Z';
}
const SQ = 20;
function shapes(L) {   // [path d per area], centre points and extra outline paths
  if (L.kind === 'square') {
    const ds = L.x.map((x, i) => { const X = x * SQ, Y = -(L.y[i] + 1) * SQ, a = SQ * 0.03, w = SQ * 0.94;
      return 'M' + (X + a) + ',' + (Y + a) + 'h' + w + 'v' + w + 'h' + (-w) + 'Z'; });
    const pts = L.x.map((x, i) => [(x + 0.5) * SQ, -(L.y[i] + 0.5) * SQ]);
    const lines = (L.outlines || []).map(o => '<path d="M' + o.pts.map(p => (p[0] * SQ) + ',' + (-p[1] * SQ)).join('L') +
      '" fill="none" stroke="#222" stroke-width="' + o.w + '" pointer-events="none"/>').join('');
    return {ds, pts, s: SQ / 2, lines};
  }
  if (L.kind === 'poly') {   // projected polygons in km: draw at 1 unit per km, y flipped
    const ds = L.rings.map(rs => rs.map(r => { let d = 'M' + r[0] + ',' + (-r[1]);
      for (let k = 2; k < r.length; k += 2) d += 'L' + r[k] + ',' + (-r[k + 1]); return d + 'Z'; }).join(''));
    const pts = L.rings.map(rs => [rs[0][0], -rs[0][1]]);
    let lines = (L.outlines || []).map(o => '<path d="M' + o.pts.map(p => p[0] + ',' + (-p[1])).join('L') +
      '" fill="none" stroke="#222" stroke-width="' + o.w + '" pointer-events="none"/>').join('');
    lines += (L.labels || []).map(l => '<text x="' + l[0] + '" y="' + (-l[1]) + '" font-size="4" text-anchor="middle" ' +
      'paint-order="stroke" stroke="#fff" stroke-width="0.8" fill="#111" pointer-events="none">' + l[2] + '</text>').join('');
    return {ds, pts, s: 2, lines};
  }
  const {pts, s, pointy} = centres(L);
  return {ds: pts.map(p => hexPath(p[0], p[1], s * 0.98, pointy)), pts, s, lines: ''};
}
function draw() {
  const L = D.layouts[+lay.value], V = L.values[+sel.value], label = D.vars[+sel.value].label;
  const {ds, pts, s, lines} = shapes(L);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), pad = s * 1.5;
  const x0 = Math.min(...xs) - pad, y0 = Math.min(...ys) - pad, x1 = Math.max(...xs) + pad, y1 = Math.max(...ys) + pad;
  svg.setAttribute('viewBox', [x0, y0, x1 - x0, y1 - y0].join(' '));
  const S = makeScale(V, D.vars[+sel.value]), col = v => S.col(v) || '#e3e3e3';
  svg.innerHTML = ds.map((d, i) => '<path class="hex" data-i="' + i + '" d="' + d + '" fill="' + col(V[i]) + '"/>').join('') + lines;
  svg.querySelectorAll('path.hex').forEach(el => {
    el.onmousemove = e => { const i = +el.dataset.i; tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 10) + 'px';
      tip.innerHTML = '<b>' + L.name[i] + '</b><br>' + label + ': <b>' + fmt(V[i]) + '</b><br><span style="color:#777">' +
        fmt(L.postcodes[i]) + ' postcodes, ' + fmt(L.addresses[i]) + ' addresses</span>'; };
    el.onmouseleave = () => tip.style.display = 'none'; });
  document.getElementById('legend').innerHTML = legendHtml(label, S, '<br><i style="background:#e3e3e3"></i>no data');
  const ranked = V.map((v, i) => [v, i]).filter(a => typeof a[0] === 'number').sort((a, b) => b[0] - a[0]);
  const li = a => '<li>' + L.name[a[1]] + ': ' + fmt(a[0]) + '</li>';
  document.getElementById('top').innerHTML = ranked.slice(0, 5).map(li).join('');
  document.getElementById('bottom').innerHTML = ranked.slice(-5).reverse().map(li).join('');
}
// optional start view from the URL, e.g. cartograms.html#lay=3&var=0
const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=')));
if (H.lay && lay.options[+H.lay]) lay.value = H.lay;
{ const k = varIndex(D.vars, H.var); if (k !== null && sel.options[k]) sel.value = k; }
lay.onchange = draw; sel.onchange = draw; draw();
</script></body></html>"""


def build_cartograms(data_dir: Path, out_dir: Path, cfg: dict) -> Path | None:
    layouts = cfg.get("layouts", {})
    variables = {c: (v if isinstance(v, dict) else {"label": v}) for c, v in cfg.get("variables", {}).items()}
    raw = data_dir / "raw" / "hexmaps"
    t = _postcode_table(data_dir, [spec["code"] for spec in layouts.values()], variables)
    found = [c for c in variables if c in t.column_names]
    missing = [c for c in variables if c not in found]
    if missing:
        log.warning("cartograms: columns not found, skipped: %s", missing)
    data = {"vars": [var_meta(c, variables[c]) for c in found], "layouts": []}
    for name, spec in layouts.items():
        if spec["code"] not in t.column_names:
            log.warning("cartograms: layout %s skipped - column %s not in the postcode lookup (run 02_collate.py)",
                        name, spec["code"])
            continue
        f = raw / spec["url"].rsplit("/", 1)[-1]
        try:
            dl.download(spec["url"], f)
        except Exception as e:
            log.warning("cartograms: layout %s skipped - could not download %s (%s)", name, spec["url"], e)
            continue
        if spec.get("type", "hexjson") == "hexjson":
            hj = json.loads(f.read_text(encoding="utf-8"))
            codes = list(hj["hexes"])
            hexes = [hj["hexes"][c] for c in codes]
            shape = {"kind": "hex", "layout": hj.get("layout", "odd-r"),
                     "q": [h["q"] for h in hexes], "r": [h["r"] for h in hexes]}
            nk = spec.get("name_key", "n")
            names = [str(h.get(nk, c)) for h, c in zip(hexes, codes)]
        elif spec["type"] == "geojson_polygons":   # polygon cartogram as GeoJSON (hexmap.uk MSOA hexes)
            shape, codes, names = _geojson_layout(f, spec)
        elif spec["type"] == "gpkg_polygons":   # polygon cartogram in a GeoPackage
            shape, codes, names = _polygon_layout(f, spec)
        else:   # unit-square grid in a GeoPackage (OS equal-area cartogram)
            shape, codes, names = _square_layout(f, spec)
        agg = _aggregate(t, spec["code"], codes, variables)
        matched = int((agg["postcodes"] > 0).sum())
        log.info("cartograms: %s - %d areas, %d with postcodes", name, len(codes), matched)
        if matched < len(codes):
            log.info("cartograms: %s areas without postcodes (code changes?): %s", name,
                     [n for n, p in zip(names, agg["postcodes"]) if p == 0][:10])
        data["layouts"].append({
            "id": name, "label": spec.get("label", name), **shape, "name": names,
            "postcodes": [int(v) for v in agg["postcodes"]], "addresses": [int(round(v)) for v in agg["addresses"]],
            "values": [_num(pa.chunked_array([pa.array(agg[c], from_pandas=True)])) for c in found],
        })
    if not data["layouts"]:
        log.warning("cartograms: no layouts available - nothing written")
        return None
    page = (TEMPLATE.replace("__CSS__", PAGE_CSS).replace("__SCALE__", SCALE_JS)
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    out = out_dir / "cartograms.html"
    out.write_text(page, encoding="utf-8")
    log.info("wrote   %s (%d layouts x %d variables)", out.name, len(data["layouts"]), len(found))
    if cfg.get("small_multiples"):
        _small_multiples(out_dir, data, cfg["small_multiples"])
    return out


def var_meta(col: str, spec: dict) -> dict:
    """What the pages need to know about a variable: key, label, colour scale."""
    m = {"key": col, "label": spec["label"]}
    if spec.get("scale"):
        m["scale"] = spec["scale"]
    return m


SM_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoData small multiples</title><style>__CSS__
#bar{display:flex;gap:18px;align-items:center;padding:10px 26px;background:#fff;border-bottom:1px solid #ddd}
#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:16px 26px}
.panel{background:#fff;border-radius:6px;box-shadow:0 1px 3px #0002;padding:8px}
.panel h3{font-size:13px;margin:0 0 4px} .panel svg{width:100%;height:420px}
.legend{font-size:11px;line-height:15px} .legend i{display:inline-block;width:12px;height:10px;margin-right:4px}
path{vector-effect:non-scaling-stroke} path.hex{stroke:#fff;stroke-width:0.3} path.hex:hover{stroke:#111;stroke-width:1.5}
#tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #bbb;border-radius:5px;padding:5px 7px;font-size:12px;display:none}
@media (max-width:1100px){#grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><h1>Small multiples</h1><p>The same cartogram repeated for several variables, so patterns can be compared at a glance.
Each panel has its own colour scale. Layouts: hexmap.uk (MIT), Ordnance Survey equal-area cartogram (OGL).
<a href="index.html">All visualisations</a></p></header>
<div id="bar"><label>Areas <select id="lay"></select></label></div>
<div id="grid"></div><div id="tip"></div>
<script>
const D = __DATA__;
__SCALE__
__SHAPES__
const lay = document.getElementById('lay'), tip = document.getElementById('tip');
D.layouts.forEach((l, i) => lay.add(new Option(l.label, i)));
function draw() {
  const L = D.layouts[+lay.value], {ds, pts, s, lines} = shapes(L);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), pad = s * 1.5;
  const vb = [Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) - Math.min(...xs) + 2 * pad, Math.max(...ys) - Math.min(...ys) + 2 * pad].join(' ');
  const grid = document.getElementById('grid'); grid.innerHTML = '';
  D.vars.forEach((v, k) => {
    const V = L.values[k], S = makeScale(V, v), p = document.createElement('div'); p.className = 'panel';
    p.innerHTML = '<h3>' + v.label + '</h3><svg viewBox="' + vb + '">' + ds.map((d, i) => '<path class="hex" data-i="' + i +
      '" d="' + d + '" fill="' + (S.col(V[i]) || '#e3e3e3') + '"/>').join('') + lines + '</svg><div class="legend">' +
      S.rows.map(r => '<i style="background:' + r[0] + '"></i>' + r[1]).join(' &nbsp; ') + '</div>';
    p.querySelectorAll('path.hex').forEach(el => {
      el.onmousemove = e => { const i = +el.dataset.i; tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px';
        tip.style.top = (e.clientY + 8) + 'px'; tip.innerHTML = '<b>' + L.name[i] + '</b><br>' + v.label + ': <b>' + fmt(V[i]) + '</b>'; };
      el.onmouseleave = () => tip.style.display = 'none'; });
    grid.appendChild(p);
  });
}
const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=')));
if (H.lay && lay.options[+H.lay]) lay.value = H.lay;
lay.onchange = draw; draw();
</script></body></html>"""


def _small_multiples(out_dir: Path, data: dict, cfg: dict) -> Path | None:
    """Several variables side by side on the same cartogram (shared shapes, own colour scales)."""
    keys = [v["key"] for v in data["vars"]]
    want = [k for k in cfg.get("variables", []) if k in keys]
    if not want:
        log.warning("small multiples: none of the configured variables are available")
        return None
    idx = [keys.index(k) for k in want]
    layouts = [dict(L, values=[L["values"][i] for i in idx]) for L in data["layouts"]
               if not cfg.get("layouts") or L["id"] in cfg["layouts"]]
    sm = {"vars": [data["vars"][i] for i in idx], "layouts": layouts}
    shapes_js = TEMPLATE[TEMPLATE.index("function centres"):TEMPLATE.index("function draw()")]
    page = (SM_TEMPLATE.replace("__CSS__", PAGE_CSS).replace("__SCALE__", SCALE_JS).replace("__SHAPES__", shapes_js)
            .replace("__DATA__", json.dumps(sm, separators=(",", ":"))))
    out = out_dir / "small_multiples.html"
    out.write_text(page, encoding="utf-8")
    log.info("wrote   %s (%d layouts x %d variables)", out.name, len(layouts), len(want))
    return out
