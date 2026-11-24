"""Static mini maps: a size_m square around each key location, every layer drawn,
collected into one HTML gallery (images embedded) plus a PNG per location."""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import numpy as np

from .. import io
from . import PAGE_CSS, label, png_data_uri, square

log = logging.getLogger(__name__)


def _load_layer(data_dir: Path, spec: dict):
    import geopandas as gpd
    import pyarrow.parquet as pq
    path = data_dir / "extract" / spec["file"]
    if not path.exists():
        log.warning("minimaps: layer %s skipped - %s missing", spec["name"], spec["file"])
        return None
    names = pq.read_schema(path).names
    filt = spec.get("filter") or {}
    extra = [c for c in {*filt, spec.get("by")} if c and c in names]
    if "geometry" in names:
        gdf = gpd.read_parquet(path, columns=[*extra, "geometry"])
    else:   # plain easting/northing tables (e.g. NaPTAN)
        t = pq.read_table(path, columns=[*extra, "easting", "northing"]).to_pandas()
        t = t[t["easting"].notna() & t["northing"].notna()]
        gdf = gpd.GeoDataFrame(t.drop(columns=["easting", "northing"]),
                               geometry=gpd.points_from_xy(t["easting"], t["northing"]), crs=27700)
    for c, v in filt.items():
        vals = v if isinstance(v, list) else [v]
        gdf = gdf[gdf[c].astype(str).isin([str(x) for x in vals])]
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf.sindex   # build the spatial index once, reused for every location
    return gdf


def _draw(ax, gdf, kind: str, style: dict):
    style = dict(style)
    z = style.pop("zorder", None)
    if kind == "point":
        xy = np.column_stack([gdf.geometry.x, gdf.geometry.y]) if gdf.geom_type.eq("Point").all() else \
            np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
        ax.scatter(xy[:, 0], xy[:, 1], c=style.get("color", "k"), s=style.get("s", 4), marker=style.get("marker", "o"),
                   linewidths=0, zorder=z or 7)
    elif kind == "line":
        gdf.plot(ax=ax, color=style.get("color", "k"), linewidth=style.get("linewidth", 1), zorder=z or 3)
    else:
        gdf.plot(ax=ax, color=style.get("color", "#ddd"), edgecolor=style.get("edgecolor", "none"),
                 linewidth=style.get("linewidth", 0), hatch=style.get("hatch"), alpha=style.get("alpha", 1),
                 zorder=z or 1)


def _legend_handles(layers):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = []
    for spec in layers:
        if spec["name"] == "Land":
            continue
        items = spec.get("styles", {}).items() if spec.get("by") else [(spec["name"], spec.get("style", {}))]
        seen = set()
        for name, st in items:
            key = (st.get("color"), st.get("linewidth"))
            if key in seen:      # e.g. the several minor road classes share a style
                continue
            seen.add(key)
            if spec["kind"] == "point":
                hs.append(Line2D([], [], marker=st.get("marker", "o"), color=st.get("color", "k"), linestyle="",
                                 markersize=4 if st.get("s", 4) < 10 else 6, label=name))
            elif spec["kind"] == "line":
                nice = {"Classified Unnumbered": "Minor road", "inlandRiver": "River", "tidalRiver": "Tidal river",
                        "canal": "Canal", "lake": "Lake (centre-line)"}
                hs.append(Line2D([], [], color=st.get("color", "k"), linewidth=min(max(st.get("linewidth", 1), 1), 4),
                                 label=nice.get(name, name)))
            else:
                c = st.get("color", "#ddd")
                hs.append(Patch(facecolor="none" if c == "none" else c, edgecolor=st.get("edgecolor", c),
                                hatch=st.get("hatch"), label=name))
    return hs


EXPLORER = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini map explorer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>__CSS__
html,body{height:100%} body{display:flex;flex-direction:column}
#bar{display:flex;gap:18px;align-items:center;padding:10px 26px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
#bar select{font-size:14px;padding:4px;max-width:460px}
#main{flex:1;display:flex;min-height:0} #map{flex:1}
#side{width:250px;overflow:auto;background:#fff;border-left:1px solid #ddd;padding:10px 14px;font-size:13px}
#side label{display:block;margin:3px 0;cursor:pointer} #side h3{font-size:13px;margin:12px 0 4px}
.sw{display:inline-block;width:18px;height:10px;margin:0 6px 0 4px;vertical-align:middle;border-radius:2px}
.legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px #0003;font-size:12px;line-height:18px}
.legend i{display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:-2px;border-radius:50%}
.tip table{border-collapse:collapse;font-size:12px} .tip td{padding:1px 6px} .tip td:last-child{text-align:right;font-weight:600}
</style></head><body>
<header><h1>Mini map explorer</h1><p>Pick a location, then choose which map layers and which postcode feature to show.
__KM__ squares. Open data: OS, Historic England, Natural England, NaPTAN, ONS and others. <a href="index.html">All visualisations</a></p></header>
<div id="bar"><label>Location <select id="loc"></select></label>
<label>Overlay <select id="mode"><option value="none">none</option><option value="dots">postcode dots</option>
<option value="lsoa">LSOA choropleth</option></select></label>
<label>Colour by <select id="var"></select></label><span id="note"></span></div>
<div id="main"><div id="map"></div><div id="side"><b>Map layers</b><div id="layers"></div>
<h3>Background</h3><label><input type="checkbox" id="tiles" checked> street map</label></div></div>
<script>
const M = __META__;
window.MINIMAP = {};
const PAL = ["#440154","#46327e","#365c8d","#277f8e","#1fa187","#4ac16d","#a0da39","#fde725"];
const map = L.map('map', {preferCanvas: true});
const tiles = (function () { __TILES__ return null; })();
let tileLayer = null; map.eachLayer(l => { if (l instanceof L.TileLayer) tileLayer = l; });
const locSel = document.getElementById('loc'), varSel = document.getElementById('var');
M.locations.forEach(l => locSel.add(new Option(l.label, l.name)));
const modeSel = document.getElementById('mode');
function fillVars() {   // postcode dots and the LSOA choropleth have their own variable lists
  const list = modeSel.value === 'lsoa' ? M.areaVars : M.vars;
  varSel.innerHTML = ''; list.forEach((v, i) => varSel.add(new Option(v, i))); varSel.disabled = modeSel.value === 'none';
}
fillVars();
const checks = {};
M.layers.forEach((ly, i) => {
  const st = ly.styles ? Object.values(ly.styles)[0] : ly.style;
  const c = st.color === 'none' ? (st.edgecolor || '#888') : (st.color || '#888');
  const lab = document.createElement('label');
  lab.innerHTML = '<input type="checkbox" ' + (ly.on ? 'checked' : '') + '><span class="sw" style="background:' + c + '"></span>' + ly.name;
  checks[i] = lab.querySelector('input'); checks[i].onchange = drawLayers;
  document.getElementById('layers').appendChild(lab);
});
let groups = [], dots = null, cur = null;
const fmt = v => v === null || v === undefined ? 'â€“' : (typeof v === 'number' ? v.toLocaleString('en-GB') : v);
function lineStyle(st) { return {color: st.color, weight: Math.max(0.6, (st.linewidth || 1) * 1.3), opacity: 0.95}; }
function polyStyle(st) { const fill = st.color && st.color !== 'none';
  return {color: st.edgecolor && st.edgecolor !== 'none' ? st.edgecolor : (fill ? st.color : '#888'), weight: st.linewidth || 0.5,
          fillColor: fill ? st.color : 'none', fillOpacity: fill ? Math.min(st.alpha || 0.6, 0.6) : 0, opacity: st.edgecolor && st.edgecolor !== 'none' ? 1 : 0.3,
          dashArray: st.hatch ? '4 3' : null}; }
function drawLayers() {
  groups.forEach(g => map.removeLayer(g)); groups = [];
  const D = window.MINIMAP[cur]; if (!D) return;
  M.layers.forEach((ly, i) => {
    if (!checks[i].checked || !D.layers[i]) return;
    const g = L.layerGroup(), data = D.layers[i];
    if (ly.kind === 'point') {
      const st = ly.style || {};
      data.forEach(p => L.circleMarker(p, {radius: (st.s || 4) > 10 ? 5 : 2.5, color: st.color, fillColor: st.color, fillOpacity: 1,
        weight: (st.marker === 's') ? 2 : 0}).addTo(g));
    } else {
      L.geoJSON(data, {style: f => { const st = ly.styles ? (ly.styles[f.properties.s] || {}) : ly.style;
        return ly.kind === 'line' ? lineStyle(st) : polyStyle(st); }, interactive: false}).addTo(g);
    }
    g.addTo(map); groups.push(g);
  });
  drawDots();
}
const legend = L.control({position: 'bottomright'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.id = 'legend'; d.style.display = 'none'; return d; };
legend.addTo(map);
function scale(vals) {
  const nums = vals.filter(v => typeof v === 'number').sort((a, b) => a - b);
  const q = p => nums[Math.min(nums.length - 1, Math.floor(p * nums.length))];
  const br = nums.length ? [...new Set([1, 2, 3, 4, 5, 6, 7].map(j => q(j / 8)))] : [];
  const col = v => { let j = 0; while (j < br.length && v > br[j]) j++; return PAL[Math.round(j * (PAL.length - 1) / Math.max(br.length, 1))]; };
  return {nums, br, col};
}
function legendHtml(label, s, note) {
  let h = '<b>' + label + '</b><br>';
  if (s.nums.length) { const e = [s.nums[0], ...s.br, s.nums[s.nums.length - 1]];
    for (let j = 0; j <= s.br.length; j++) h += '<i style="background:' + PAL[Math.round(j * (PAL.length - 1) / Math.max(s.br.length, 1))] + '"></i>' + fmt(e[j]) + ' to ' + fmt(e[j + 1]) + '<br>'; }
  return h + '<span style="color:#777">' + note + '</span>';
}
function drawDots() {
  if (dots) { map.removeLayer(dots); dots = null; }
  const lg = document.getElementById('legend'), k = +varSel.value, D = window.MINIMAP[cur], note = document.getElementById('note');
  lg.style.display = 'none'; note.textContent = '';
  if (!D || modeSel.value === 'none') return;
  if (modeSel.value === 'lsoa') {
    if (!D.lsoa) { note.textContent = 'no LSOA boundaries here (England and Wales only)'; return; }
    const vals = D.lsoa.vals[k], s = scale(vals);
    dots = L.geoJSON(D.lsoa.features, {style: f => { const v = vals[f.properties.i], has = typeof v === 'number';
        return {color: '#333', weight: 0.6, fillColor: has ? s.col(v) : '#fff', fillOpacity: has ? 0.6 : 0.1}; },
      onEachFeature: (f, l) => l.bindTooltip(() => '<b>' + D.lsoa.names[f.properties.i] + '</b><br>' + M.areaVars[k] + ': <b>' +
        fmt(vals[f.properties.i]) + '</b>', {sticky: true})});
    dots.addTo(map);
    lg.innerHTML = legendHtml(M.areaVars[k], s, 'LSOA averages (weighted by addresses); eighths of the LSOAs shown');
    lg.style.display = 'block';
    note.textContent = s.nums.length + ' of ' + vals.length + ' LSOAs have a value';
    return;
  }
  if (!D.pc) return;
  const vals = D.pc.vars[k], sc = scale(vals), nums = sc.nums, br = sc.br, col = sc.col;
  dots = L.layerGroup();
  D.pc.pc.forEach((p, i) => { const v = vals[i], has = typeof v === 'number';
    L.circleMarker([D.pc.lat[i], D.pc.lon[i]], {radius: has ? 4.5 : 2.5, weight: 0.5, color: '#222', fillColor: has ? col(v) : '#fff', fillOpacity: has ? 0.9 : 0.4})
      .bindTooltip(() => { let r = '<b>' + p + '</b><table>'; M.vars.forEach((lab, j) => r += '<tr' + (j == k ? ' style="background:#fff3c4"' : '') +
        '><td>' + lab + '</td><td>' + fmt(D.pc.vars[j][i]) + '</td></tr>'); return r + '</table>'; }, {sticky: true, className: 'tip'})
      .addTo(dots); });
  dots.addTo(map);
  let h = '<b>' + M.vars[k] + '</b><br>';
  if (nums.length) { const e = [nums[0], ...br, nums[nums.length - 1]];
    for (let j = 0; j <= br.length; j++) h += '<i style="background:' + PAL[Math.round(j * (PAL.length - 1) / Math.max(br.length, 1))] + '"></i>' + fmt(e[j]) + ' to ' + fmt(e[j + 1]) + '<br>'; }
  lg.innerHTML = h + '<span style="color:#777">eighths within this square</span>'; lg.style.display = 'block';
  document.getElementById('note').textContent = nums.length.toLocaleString('en-GB') + ' of ' + vals.length.toLocaleString('en-GB') + ' postcodes have a value';
}
function show(name) {
  cur = name;
  const go = () => { const D = window.MINIMAP[name]; map.fitBounds(D.box);
    if (window.boxLayer) map.removeLayer(window.boxLayer);
    window.boxLayer = L.polygon(D.box, {color: '#111', weight: 1.5, fill: false, dashArray: '6 4', interactive: false}).addTo(map);
    drawLayers(); };
  if (window.MINIMAP[name]) return go();
  const s = document.createElement('script'); s.src = 'minimap_data/' + name + '.js'; s.onload = go;
  s.onerror = () => alert('Could not load minimap_data/' + name + '.js - run 04_visualise.py'); document.body.appendChild(s);
}
document.getElementById('tiles').onchange = e => { if (tileLayer) e.target.checked ? tileLayer.addTo(map) : map.removeLayer(tileLayer); };
locSel.onchange = () => show(locSel.value); varSel.onchange = drawDots;
modeSel.onchange = () => { fillVars(); drawDots(); };
// optional start state from the URL, e.g. minimap_explorer.html#loc=Bath&mode=lsoa&var=0
const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=').map(decodeURIComponent)));
if (H.loc && M.locations.some(l => l.name === H.loc)) locSel.value = H.loc;
if (H.mode) { modeSel.value = H.mode; fillVars(); }
if (H.var) varSel.value = H.var;
show(locSel.value);
</script></body></html>"""


def _geojson_features(gdf, box, to_ll, by=None, simplify=1.0):
    """Clip a layer to the box, simplify slightly, reproject to lat/lon -> compact GeoJSON features."""
    import shapely
    g = shapely.clip_by_rect(gdf.geometry.values, *box)
    if simplify:
        g = shapely.simplify(g, simplify, preserve_topology=True)
    keep = ~(shapely.is_empty(g) | shapely.is_missing(g))
    g = g[keep]
    if not len(g):
        return []
    g = shapely.transform(g, lambda xy: np.column_stack(to_ll.transform(xy[:, 0], xy[:, 1])))
    g = shapely.set_precision(g, 1e-5, mode="pointwise")   # display only: round coordinates, no topology checks
    props = gdf[by].astype(str).to_numpy()[keep] if by else [None] * len(g)
    feats = []
    for geom, s in zip(g, props):
        if geom is None or geom.is_empty:
            continue
        feats.append({"type": "Feature", "properties": {"s": s} if s else {},
                      "geometry": json.loads(shapely.to_geojson(geom))})
    return feats


def _lsoa_source(data_dir: Path, lcfg: dict, area_variables: dict):
    """LSOA polygons plus address-weighted LSOA averages of the area variables (computed once, nationally)."""
    import geopandas as gpd
    import pyarrow as pa
    from .cartograms import _aggregate, _postcode_table
    from .interactive import _num
    f = next((data_dir / "extract" / p for p in [lcfg.get("file"), lcfg.get("fallback_file")]
              if p and (data_dir / "extract" / p).exists()), None)
    if f is None:
        log.warning("minimap explorer: no LSOA boundaries (%s) - LSOA choropleth disabled", lcfg.get("file"))
        return None
    key, nm = lcfg.get("key", "lsoa21cd"), lcfg.get("name_col", "lsoa21nm")
    gdf = gpd.read_parquet(f, columns=[key, nm, "geometry"])
    gdf.sindex
    variables = {c: (v if isinstance(v, dict) else {"label": v}) for c, v in area_variables.items()}
    t = _postcode_table(data_dir, [key], variables)
    codes = gdf[key].astype(str).tolist()
    agg = _aggregate(t, key, codes, variables)
    found = [c for c in variables if c in agg]
    vals = [_num(pa.chunked_array([pa.array(agg[c], from_pandas=True)])) for c in found]
    log.info("minimap explorer: LSOA choropleth from %s (%d LSOAs, %d variables)", f.name, len(codes), len(found))
    return {"gdf": gdf, "names": gdf[nm].astype(str).tolist(), "vals": vals,
            "labels": [variables[c]["label"] for c in found]}


def build_minimap_explorer(data_dir: Path, out_dir: Path, locs: dict, cfg: dict, loaded=None,
                           variables: dict | None = None, area_variables: dict | None = None) -> Path:
    """Interactive mini maps: location picker, layer checkboxes, and an overlay of postcode feature dots
    or an LSOA choropleth."""
    from pyproj import Transformer
    from . import tile_js
    from .interactive import _area_table, _num
    lsoa = _lsoa_source(data_dir, cfg.get("lsoa") or {}, area_variables) if area_variables else None
    to_ll = Transformer.from_crs(27700, 4326, always_xy=True)
    size = float(cfg.get("size_m", 5000))
    layers = cfg.get("layers", [])
    if loaded is None:
        with io.Timer("minimap explorer: load layers"):
            loaded = [(spec, _load_layer(data_dir, spec)) for spec in layers]
    variables = variables or {}
    data_dir_out = out_dir / "minimap_data"
    data_dir_out.mkdir(parents=True, exist_ok=True)
    found_vars = None
    with io.Timer(f"minimap explorer: {len(locs)} locations"):
        for name, loc in locs.items():
            box = square(loc, size)
            pad = size * 0.1   # a little context beyond the square
            big = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
            out_layers = []
            for spec, gdf in loaded:
                if gdf is None:
                    out_layers.append(None)
                    continue
                sub = gdf.cx[big[0]:big[2], big[1]:big[3]]
                if spec["kind"] == "point":
                    pts = []
                    if len(sub):
                        c = sub.geometry.centroid if not sub.geom_type.eq("Point").all() else sub.geometry
                        lon, lat = to_ll.transform(c.x.to_numpy(), c.y.to_numpy())
                        pts = np.round(np.column_stack([lat, lon]), 5).tolist()
                    out_layers.append(pts)
                else:
                    by = spec.get("by")
                    if by:
                        sub = sub[sub[by].astype(str).isin(list(spec.get("styles", {})))]
                    out_layers.append(_geojson_features(sub, big, to_ll, by, spec.get("simplify_m", 1.0)))
            lon, lat = to_ll.transform([box[0], box[2], box[2], box[0]], [box[1], box[1], box[3], box[3]])
            d = {"box": [[round(a, 6), round(b, 6)] for a, b in zip(lat, lon)], "layers": out_layers}
            if lsoa is not None:
                sub = lsoa["gdf"].cx[big[0]:big[2], big[1]:big[3]]
                if len(sub):
                    idx = [int(i) for i in lsoa["gdf"].index.get_indexer(sub.index)]
                    sub = sub.assign(_i=list(range(len(idx))))
                    feats = _geojson_features(sub, big, to_ll, "_i", 2.0)
                    for ft in feats:
                        ft["properties"] = {"i": int(ft["properties"]["s"])}
                    d["lsoa"] = {"features": feats, "names": [lsoa["names"][i] for i in idx],
                                 "vals": [[v[i] for i in idx] for v in lsoa["vals"]]}
            if variables:
                t, found = _area_table(data_dir, box, variables)
                found_vars = found_vars or found
                if t.num_rows:
                    d["pc"] = {"pc": t["pcds"].to_pylist(), "lat": [round(v, 6) for v in t["lat"].to_pylist()],
                               "lon": [round(v, 6) for v in t["long"].to_pylist()],
                               "vars": [_num(t[c]) if c in t.column_names else [None] * t.num_rows for c in variables]}
            js = f"window.MINIMAP[{json.dumps(name)}]=" + json.dumps(d, separators=(",", ":")) + ";"
            (data_dir_out / f"{name}.js").write_text(js, encoding="utf-8")
    meta = {"locations": [{"name": n, "label": n.replace("_", " ") + (f" - {l['note']}" if l.get("note") else "")}
                          for n, l in locs.items()],
            "layers": [{"name": s["name"], "kind": s["kind"], "style": s.get("style"), "styles": s.get("styles"),
                        "on": s.get("show", True)} for s in layers],
            "vars": list(variables.values()), "areaVars": lsoa["labels"] if lsoa else []}
    page = (EXPLORER.replace("__CSS__", PAGE_CSS).replace("__KM__", f"{size / 1000:g} km")
            .replace("__TILES__", tile_js("streets"))
            .replace("__META__", json.dumps(meta, separators=(",", ":"))))
    out = out_dir / "minimap_explorer.html"
    out.write_text(page, encoding="utf-8")
    total = sum(p.stat().st_size for p in data_dir_out.glob("*.js")) / 1e6
    log.info("wrote   %s (+ %d location files, %.1f MB)", out.name, len(locs), total)
    return out


def build_minimaps(data_dir: Path, out_dir: Path, locs: dict, cfg: dict, variables: dict | None = None,
                   area_variables: dict | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = float(cfg.get("size_m", 5000))
    dpi = int(cfg.get("dpi", 110))
    layers = cfg.get("layers", [])
    with io.Timer("minimaps: load layers"):
        loaded = [(spec, _load_layer(data_dir, spec)) for spec in layers]
    png_dir = out_dir / "minimaps"
    png_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    with io.Timer(f"minimaps: draw {len(locs)} maps"):
        for name, loc in locs.items():
            x0, y0, x1, y1 = square(loc, size)
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            ax.set_facecolor(cfg.get("sea_colour", "#cfe3f3"))
            for spec, gdf in loaded:
                if gdf is None:
                    continue
                sub = gdf.cx[x0:x1, y0:y1]
                if sub.empty:
                    continue
                if spec.get("by"):
                    for val, st in spec.get("styles", {}).items():
                        part = sub[sub[spec["by"]].astype(str) == val]
                        if not part.empty:
                            _draw(ax, part, spec["kind"], st)
                else:
                    _draw(ax, sub, spec["kind"], spec.get("style", {}))
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            # 1 km scale bar, bottom left
            sx, sy = x0 + size * 0.05, y0 + size * 0.05
            ax.plot([sx, sx + 1000], [sy, sy], color="k", linewidth=2.5, solid_capstyle="butt", zorder=20)
            ax.text(sx + 500, sy + size * 0.015, "1 km", ha="center", va="bottom", fontsize=7, zorder=20)
            ax.set_title(name.replace("_", " "), fontsize=11, loc="left")
            fig.savefig(png_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
            uri = png_data_uri(fig, dpi)
            plt.close(fig)
            note = html.escape(loc.get("note") or "")
            cards.append(f'<figure><img src="{uri}" alt="{label(name)}"><figcaption><b>{label(name)}</b>'
                         f'{" - " + note if note else ""}<br><span>centre {loc["x"]:,.0f} E, {loc["y"]:,.0f} N'
                         f'</span></figcaption></figure>')

    fig = plt.figure(figsize=(9, 1.1))
    fig.legend(handles=_legend_handles(layers), loc="center", ncol=6, frameon=False, fontsize=8)
    legend_uri = png_data_uri(fig, dpi)
    plt.close(fig)

    km = f"{size / 1000:g} km"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>GeoData mini maps</title><style>{PAGE_CSS}
.legend{{background:#fff;border-radius:6px;padding:6px;margin-bottom:16px;text-align:center}}
.grid{{display:grid;grid-template-columns:repeat({int(cfg.get('columns', 4))},1fr);gap:16px}}
figure{{margin:0;background:#fff;border-radius:6px;padding:8px;box-shadow:0 1px 3px #0002}}
figure img{{width:100%;display:block}} figcaption{{font-size:13px;padding:4px 2px 0}}
figcaption span{{color:#777;font-size:12px}}
@media (max-width:1100px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<header><h1>Mini maps: {km} squares around key locations</h1>
<p>{len(cards)} locations. Open data: OS Open Roads, Open Rivers, Open Greenspace and Built Up Areas;
Historic England; Natural England; NaPTAN; ONS boundaries. <a href="minimap_explorer.html">Interactive version</a> |
<a href="index.html">All visualisations</a></p></header>
<main><div class="legend"><img src="{legend_uri}" alt="legend" style="max-width:100%"></div>
<div class="grid">{''.join(cards)}</div></main></body></html>"""
    out = out_dir / "minimaps.html"
    out.write_text(page, encoding="utf-8")
    log.info("wrote   %s (%d maps)", out.name, len(cards))
    if cfg.get("explorer", True):
        build_minimap_explorer(data_dir, out_dir, locs, cfg, loaded=loaded, variables=variables,
                               area_variables=area_variables)
    return out
