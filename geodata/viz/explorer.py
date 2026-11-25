"""Map explorer: map layers, postcode features and road network views for each key location, on one page.

For each key location (a 5 km square) the page combines:

* map layers (roads, water, greenspace, heritage, stations...) with All / None, drawn at the bottom;
* a postcode overlay: dots, a heat surface or an LSOA choropleth of a chosen feature;
* a road network view on top: road centrality, walking / driving times, the route to the nearest
  station, collision hotspots, collisions within a 30-minute drive, the junction graph and the LSOA
  neighbour network;
* a choice of background map (street map, light grey, none).

The data comes from the per-location files the two builds write (``minimap_data/<name>.js`` from the
mini maps step, ``network_data/<name>.js`` from the networks step); each also writes a small
``meta.json`` that this page reads. The old ``minimap_explorer.html`` address forwards here with the
same settings, so existing links to it keep working.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from . import PAGE_CSS, _basemap

log = logging.getLogger(__name__)


def _tiles_js() -> str:
    """The background choices as Leaflet tile layers (not added to the map yet)."""
    out = []
    for key, kind in (("streets", "streets"), ("light", "light")):
        b = _basemap(kind)
        out.append(f"{key}: L.tileLayer({json.dumps(b['url'])}, {{maxZoom: {int(b.get('max_zoom', 19))}, "
                   f"attribution: {json.dumps(b['attribution'])}}})")
    return "{" + ", ".join(out) + "}"


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Map explorer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>__CSS__
html,body{height:100%} body{display:flex;flex-direction:column}
#bar{display:flex;gap:10px 14px;align-items:stretch;padding:8px 20px 10px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
#bar select{font-size:14px;padding:4px;max-width:440px}
/* each group of controls is a titled box; its colour matches the legend and panel it drives */
.loc{--c:#555} .net{--c:#c0392b} .pc{--c:#6a3d9a} .lay{--c:#2e7d4f}
#bar fieldset.grp{border:1px solid #d9d9d6;border-top:3px solid var(--c);border-radius:6px;margin:0;padding:2px 12px 8px;
  display:flex;gap:6px 14px;align-items:center;flex-wrap:wrap;background:#fbfbfa;min-width:0}
.grp>legend,.sideh{font-size:11px;font-weight:700;color:var(--c);text-transform:uppercase;letter-spacing:.05em}
.grp>legend{padding:0 4px}
.gnote{flex-basis:100%;font-size:12px;color:#666} .gnote:empty{display:none}
#side .sideh{font-size:11px;margin:2px 0 4px} #side b.sideh{display:inline-block} #side h3.sideh{margin-top:14px}
#nlegend,#olegend{position:relative;padding-top:21px !important;border-top:3px solid var(--c)}
#nlegend::after,#olegend::after{position:absolute;top:5px;left:10px;font-size:10px;font-weight:700;color:var(--c);
  text-transform:uppercase;letter-spacing:.05em}
#nlegend{--c:#c0392b} #nlegend::after{content:'Road network'} #olegend{--c:#6a3d9a} #olegend::after{content:'Postcode overlay'}
#main{flex:1;display:flex;min-height:0} #map{flex:1}
#side{width:250px;overflow:auto;background:#fff;border-left:1px solid #ddd;padding:10px 14px;font-size:13px}
#side label{display:block;margin:3px 0;cursor:pointer} #side h3{font-size:13px;margin:12px 0 4px}
.allnone{float:right} .allnone button{font-size:11px;padding:1px 7px;margin-left:4px;cursor:pointer;border:1px solid #bbb;border-radius:3px;background:#f4f4f4}
.allnone button:hover{background:#e6e6e6}
.sw{display:inline-block;width:18px;height:10px;margin:0 6px 0 4px;vertical-align:middle;border-radius:2px}
.legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px #0003;font-size:12px;line-height:18px;max-width:280px}
#olegend i{display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:-2px;border-radius:50%}
#nlegend i{display:inline-block;width:22px;height:4px;margin-right:6px;vertical-align:middle}
.legend b.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.tip table{border-collapse:collapse;font-size:12px} .tip td{padding:1px 6px} .tip td:last-child{text-align:right;font-weight:600}
</style></head><body>
<header><h1>Map explorer</h1><p>__KM__ squares around key locations. Choose the map layers, a postcode feature to plot
(dots, a smooth surface or LSOA areas) and a road network view: through-routes, walking and driving times, the route to the
nearest station, collision hotspots and collisions within a 30-minute drive, the junction graph and the LSOA neighbour network.
Open data: OS, ONS, DfT, Historic England, Natural England, NaPTAN and others. <a href="index.html">All visualisations</a></p></header>
<div id="bar"><fieldset class="grp loc"><legend>Location</legend><select id="loc" aria-label="Location"></select></fieldset>
<fieldset class="grp net"><legend>Road network</legend><label>View <select id="view"><option value="none">none</option>
<option value="btw">Road centrality: the through-routes</option>
<option value="walk">Walking time from the centre</option>
<option value="drive">Driving time from the centre</option>
<option value="route">Shortest walk to the nearest railway station</option>
<option value="walkst">Walking time to the nearest railway station (every road)</option>
<option value="drivest">Driving time to the nearest railway station (every road)</option>
<option value="coll">Road collision hotspots (all injury collisions, 5 years)</option>
<option value="collf">Fatal road collision hotspots (5 years)</option>
<option value="reach">Road collisions within a 30-minute drive</option>
<option value="reachf">Fatal road collisions within a 30-minute drive</option>
<option value="junc">Major road junction network (A roads and motorways, simplified)</option>
<option value="lsoa">LSOA neighbour network with bridges</option>
<option value="rings">LSOAs within 1, 2 and 3 boundaries of the centre</option>
</select></label>
<span class="gnote" id="nnote"></span></fieldset>
<fieldset class="grp pc"><legend>Postcode overlay</legend><label>Show <select id="mode"><option value="none">none</option><option value="dots">postcode dots</option>
<option value="heat">heat surface (postcode values)</option><option value="lsoa">LSOA choropleth</option></select></label>
<label>Colour by <select id="var"></select></label>
<label id="smoothL" title="How far each postcode's value spreads: the surface is a distance-weighted average of nearby postcodes">Smoothing
<select id="smooth"><option value="150">tight (150 m)</option><option value="300" selected>medium (300 m)</option>
<option value="600">wide (600 m)</option></select></label>
<label title="Transparency of the postcode overlay (dots, heat surface or LSOA fills); the road network and map layers are not affected">Opacity <input type="range" id="op" min="10" max="100" value="75"
style="vertical-align:middle;width:110px"></label>
<span class="gnote" id="note"></span></fieldset></div>
<div id="main"><div id="map"></div><div id="side" class="lay"><b class="sideh">Map layers</b>
<span class="allnone"><button type="button" id="all">All</button><button type="button" id="none">None</button></span>
<div id="layers"></div>
<h3 class="sideh">Background map</h3><label><input type="radio" name="bgm" value="streets" checked> street map</label>
<label><input type="radio" name="bgm" value="light"> light grey</label><label><input type="radio" name="bgm" value="none"> none</label></div></div>
<script>
const M = __META__, VER = '?v=__VER__';   // VER: build stamp so browsers refetch data after a rebuild
window.MINIMAP = {}; window.NET = {};
const PAL = ["#440154","#46327e","#365c8d","#277f8e","#1fa187","#4ac16d","#a0da39","#fde725"];
const map = L.map('map', {preferCanvas: true});
const TILES = __TILES__;
let tileLayer = null;
function setBg(k) { if (tileLayer) map.removeLayer(tileLayer); tileLayer = TILES[k] ? TILES[k].addTo(map) : null;
  document.querySelectorAll('input[name=bgm]').forEach(r => r.checked = r.value === k); }
// stacking: map layers (bgPane) < heat surface (heatPane) < postcode dots / LSOAs, then the network (overlay pane, drawn last)
map.createPane('bgPane').style.zIndex = 350;
map.createPane('heatPane').style.zIndex = 390;
const bgR = L.canvas({pane: 'bgPane'});
const locSel = document.getElementById('loc'), varSel = document.getElementById('var'), view = document.getElementById('view');
const modeSel = document.getElementById('mode'), opSl = document.getElementById('op'), smoothSel = document.getElementById('smooth');
const nnote = document.getElementById('nnote');
M.locations.forEach(l => locSel.add(new Option(l.label, l.name)));
const op = () => +opSl.value / 100;   // overlay opacity 0.1 - 1
function fillVars() {   // postcode dots / heat surface and the LSOA choropleth have their own variable lists
  const keep = varSel.value, list = modeSel.value === 'lsoa' ? M.areaVars : M.vars;
  varSel.innerHTML = ''; list.forEach((v, i) => varSel.add(new Option(v, i))); varSel.disabled = modeSel.value === 'none';
  if (keep && varSel.options[+keep]) varSel.value = keep;
  document.getElementById('smoothL').style.display = modeSel.value === 'heat' ? '' : 'none';
  opSl.parentNode.style.display = varSel.parentNode.style.display = modeSel.value === 'none' ? 'none' : '';
}
// --- map layers
const checks = {};
M.layers.forEach((ly, i) => {
  const st = ly.styles ? Object.values(ly.styles)[0] : ly.style;
  const c = st.color === 'none' ? (st.edgecolor || '#888') : (st.color || '#888');
  const lab = document.createElement('label');
  lab.innerHTML = '<input type="checkbox" ' + (ly.on ? 'checked' : '') + '><span class="sw" style="background:' + c + '"></span>' + ly.name;
  checks[i] = lab.querySelector('input'); checks[i].onchange = drawBg;
  document.getElementById('layers').appendChild(lab);
});
let bgGroups = [], dots = null, layer = null, cur = null, boxLayer = null;
const fmt = v => v === null || v === undefined ? '–' : (typeof v === 'number' ? v.toLocaleString('en-GB') : v);
function lineStyle(st) { return {color: st.color, weight: Math.max(0.6, (st.linewidth || 1) * 1.3), opacity: 0.95, interactive: false, renderer: bgR}; }
function polyStyle(st) { const fill = st.color && st.color !== 'none';
  return {color: st.edgecolor && st.edgecolor !== 'none' ? st.edgecolor : (fill ? st.color : '#888'), weight: st.linewidth || 0.5,
          fillColor: fill ? st.color : 'none', fillOpacity: fill ? Math.min(st.alpha || 0.6, 0.6) : 0,
          opacity: st.edgecolor && st.edgecolor !== 'none' ? 1 : 0.3, dashArray: st.hatch ? '4 3' : null, interactive: false,
          renderer: bgR}; }
function drawBg() {
  bgGroups.forEach(g => map.removeLayer(g)); bgGroups = [];
  const D = window.MINIMAP[cur]; if (!D) return;
  M.layers.forEach((ly, i) => {
    if (!checks[i].checked || !D.layers[i]) return;
    const g = L.layerGroup(), data = D.layers[i];
    if (ly.kind === 'point') {
      const st = ly.style || {};
      data.forEach(p => L.circleMarker(p, {radius: (st.s || 4) > 10 ? 5 : 2.5, color: st.color, fillColor: st.color, fillOpacity: 1,
        weight: (st.marker === 's') ? 2 : 0, interactive: false, renderer: bgR}).addTo(g));
    } else {
      L.geoJSON(data, {style: f => { const st = ly.styles ? (ly.styles[f.properties.s] || {}) : ly.style;
        return ly.kind === 'line' ? lineStyle(st) : polyStyle(st); }, interactive: false, pane: 'bgPane'}).addTo(g);
    }
    g.addTo(map); bgGroups.push(g);
  });
}
const setAll = on => { Object.values(checks).forEach(c => c.checked = on); drawBg(); };
document.getElementById('all').onclick = () => setAll(true);
document.getElementById('none').onclick = () => setAll(false);
// --- legends: postcode overlay and network view
['nlegend', 'olegend'].forEach(id => { const c = L.control({position: 'bottomright'});
  c.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.id = id; d.style.display = 'none'; return d; }; c.addTo(map); });
// --- postcode overlay
__OVERLAY__
// --- road network views
__NETWORK__
// overlay first, then the network so it stays on top (one canvas: last drawn is uppermost)
function redraw() { drawDots(); drawNet(); }
function show(name) {
  cur = name; nnote.textContent = ''; document.getElementById('note').textContent = '';
  const loc = M.locations.find(l => l.name === name) || {};
  const go = () => { const D = window.MINIMAP[name] || window.NET[name]; if (!D) return;
    map.fitBounds(D.box);
    if (boxLayer) map.removeLayer(boxLayer);
    boxLayer = L.polygon(D.box, {color: '#111', weight: 1.5, fill: false, dashArray: '6 4', interactive: false, pane: 'heatPane'}).addTo(map);
    view.disabled = !window.NET[name];
    drawBg(); redraw(); };
  const load = (dir, ok) => { const s = document.createElement('script'); s.src = dir + '/' + name + '.js' + VER; s.onload = ok;
    s.onerror = ok; document.body.appendChild(s); };
  const net = () => window.NET[name] || !loc.net ? go() : load('network_data', go);
  window.MINIMAP[name] || !loc.mini ? net() : load('minimap_data', net);
}
document.querySelectorAll('input[name=bgm]').forEach(r => r.onchange = () => setBg(r.value));
locSel.onchange = () => show(locSel.value); varSel.onchange = redraw; view.onchange = drawNet;
modeSel.onchange = () => { fillVars(); redraw(); };
smoothSel.onchange = redraw;
// opacity: the heat surface is one image (fast); dots and LSOA fills are restyled by redrawing
opSl.oninput = () => { if (dots && dots.setOpacity) dots.setOpacity(0.9 * op()); else redraw(); };
// start state from the address, e.g. explorer.html#loc=Bath&view=reach&mode=lsoa&var=0&layers=none&bg=light
const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=').map(decodeURIComponent)));
if (H.loc && M.locations.some(l => l.name === H.loc)) locSel.value = H.loc;
if (H.view && view.querySelector('option[value="' + H.view + '"]')) view.value = H.view;
if (H.mode) modeSel.value = H.mode;
fillVars();
if (H.var) varSel.value = H.var;
if (H.op) opSl.value = H.op;
if (H.smooth) smoothSel.value = H.smooth;
if (H.layers === 'all' || H.layers === 'none') Object.values(checks).forEach(c => c.checked = H.layers === 'all');
setBg(H.bg in TILES || H.bg === 'none' ? H.bg : 'streets');
show(locSel.value);
</script></body></html>"""

# postcode overlay: dots, heat surface, LSOA choropleth (uses M, varSel, modeSel, smoothSel, op)
OVERLAY_JS = r"""function scale(vals) {
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
function heatSurface(D, vals, col, sigma) {
  // Smooth surface of a postcode variable: each pixel = Gaussian-weighted average of nearby postcodes
  // (weights exp(-d²/2σ²) out to 3σ); pixels with no postcode within ~2σ are left transparent.
  const b = L.latLngBounds(D.box), s = b.getSouth(), n = b.getNorth(), w = b.getWest(), e = b.getEast();
  const lat0 = (s + n) / 2, mx = 111320 * Math.cos(lat0 * Math.PI / 180), my = 110540;
  const W = 220, Hh = Math.round(W * (n - s) * my / ((e - w) * mx));
  const px = (e - w) * mx / W, cut = 3 * sigma, cs = cut;   // pixel size (m), cut-off, bucket size
  const pts = [], buckets = new Map();
  D.pc.pc.forEach((_, i) => { const v = vals[i]; if (typeof v !== 'number') return;
    const x = (D.pc.lon[i] - w) * mx, y = (n - D.pc.lat[i]) * my, key = Math.floor(x / cs) + ',' + Math.floor(y / cs);
    pts.push([x, y, v]); (buckets.get(key) || buckets.set(key, []).get(key)).push(pts.length - 1); });
  if (!pts.length) return null;
  const cv = document.createElement('canvas'); cv.width = W; cv.height = Hh;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(W, Hh), k2 = 1 / (2 * sigma * sigma), minW = Math.exp(-2);
  const hex = c => [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
  const cache = {};
  for (let r = 0; r < Hh; r++) for (let c = 0; c < W; c++) {
    const x = (c + 0.5) * px, y = (r + 0.5) * px, bx = Math.floor(x / cs), by = Math.floor(y / cs);
    let sw = 0, sv = 0, best = 0;
    for (let i = bx - 1; i <= bx + 1; i++) for (let j = by - 1; j <= by + 1; j++) {
      const bk = buckets.get(i + ',' + j); if (!bk) continue;
      for (const q of bk) { const p = pts[q], dx = p[0] - x, dy = p[1] - y, d2 = dx * dx + dy * dy;
        if (d2 > cut * cut) continue; const wt = Math.exp(-d2 * k2); sw += wt; sv += wt * p[2]; if (wt > best) best = wt; } }
    if (best < minW) continue;   // nothing within ~2 sigma: leave transparent
    const colr = col(sv / sw), rgb = cache[colr] || (cache[colr] = hex(colr)), o = 4 * (r * W + c);
    img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return L.imageOverlay(cv.toDataURL(), b, {opacity: 0.9 * op(), interactive: false, pane: 'heatPane'});
}
function drawDots() {
  if (dots) { map.removeLayer(dots); dots = null; }
  const lg = document.getElementById('olegend'), k = +varSel.value, D = window.MINIMAP[cur], note = document.getElementById('note');
  lg.style.display = 'none'; note.textContent = '';
  if (!D || modeSel.value === 'none') return;
  if (modeSel.value === 'lsoa') {
    if (!D.lsoa) { note.textContent = 'no LSOA boundaries here (England and Wales only)'; return; }
    const vals = D.lsoa.vals[k], s = scale(vals);
    dots = L.geoJSON(D.lsoa.features, {style: f => { const v = vals[f.properties.i], has = typeof v === 'number';
        return {color: '#333', weight: 0.6, opacity: Math.min(1, op() + 0.2), fillColor: has ? s.col(v) : '#fff',
                fillOpacity: has ? 0.8 * op() : 0.1 * op()}; },
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
  if (modeSel.value === 'heat') {
    dots = heatSurface(D, vals, col, +smoothSel.value);
    if (dots) dots.addTo(map);
    lg.innerHTML = legendHtml(M.vars[k], sc, 'eighths of the postcodes in this square; each point of the surface is a ' +
      'distance-weighted average of nearby postcodes (Gaussian, ' + smoothSel.value + ' m); blank where no postcodes are near');
    lg.style.display = 'block';
    note.textContent = nums.length.toLocaleString('en-GB') + ' of ' + vals.length.toLocaleString('en-GB') + ' postcodes have a value';
    return;
  }
  dots = L.layerGroup();
  D.pc.pc.forEach((p, i) => { const v = vals[i], has = typeof v === 'number';
    L.circleMarker([D.pc.lat[i], D.pc.lon[i]], {radius: has ? 4.5 : 2.5, weight: 0.5, color: '#222', opacity: op(),
                                                fillColor: has ? col(v) : '#fff', fillOpacity: has ? 0.95 * op() : 0.4 * op()})
      .bindTooltip(() => { let r = '<b>' + p + '</b><table>'; M.vars.forEach((lab, j) => r += '<tr' + (j == k ? ' style="background:#fff3c4"' : '') +
        '><td>' + lab + '</td><td>' + fmt(D.pc.vars[j][i]) + '</td></tr>'); return r + '</table>'; }, {sticky: true, className: 'tip'})
      .addTo(dots); });
  dots.addTo(map);
  let h = '<b>' + M.vars[k] + '</b><br>';
  if (nums.length) { const e = [nums[0], ...br, nums[nums.length - 1]];
    for (let j = 0; j <= br.length; j++) h += '<i style="background:' + PAL[Math.round(j * (PAL.length - 1) / Math.max(br.length, 1))] + '"></i>' + fmt(e[j]) + ' to ' + fmt(e[j + 1]) + '<br>'; }
  lg.innerHTML = h + '<span style="color:#777">eighths within this square</span>'; lg.style.display = 'block';
  document.getElementById('note').textContent = nums.length.toLocaleString('en-GB') + ' of ' + vals.length.toLocaleString('en-GB') + ' postcodes have a value';
}"""

# road network views (uses view, layer, cur, nnote)
NETWORK_JS = r"""const HEAT = ['#ffffcc','#fed976','#fd8d3c','#e31a1c','#800026'];
const TIME = ['#1a9850','#91cf60','#d9ef8b','#fee08b','#fc8d59','#d73027'], TB = [5, 10, 15, 20, 30];
const RING = ['#111111', '#1b9e77', '#d95f02', '#7570b3'];
const pl = c => { const a = []; for (let k = 0; k < c.length; k += 2) a.push([c[k], c[k + 1]]); return a; };
function tband(t) { if (t === null) return null; let k = 0; while (k < TB.length && t > TB[k]) k++; return TIME[k]; }
function drawNet() {
  if (layer) { map.removeLayer(layer); layer = null; }
  const v = view.value, nl = document.getElementById('nlegend');
  map.getPane('tilePane').style.opacity = (v === 'coll' || v === 'collf') ? 0.45 : 1;   // fade the street map under the hotspots
  nnote.textContent = '';
  if (v === 'none') { nl.style.display = 'none'; return; }
  const D = window.NET[cur];
  if (!D) { nl.style.display = 'none'; nnote.textContent = 'no road network for this location'; return; }
  nl.style.display = '';
  layer = L.layerGroup();
  let lg = '';
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
  } else if (v === 'walkst' || v === 'drivest') {
    const idx = v === 'walkst' ? 8 : 9, B = v === 'walkst' ? TB : [3, 5, 10, 15, 20];
    const band = t => { if (t === null || t === undefined) return null; let k = 0; while (k < B.length && t > B[k]) k++; return TIME[k]; };
    D.links.forEach(l => { const c = band(l[idx]); if (c) L.polyline(pl(l[0]), {color: c, weight: 2.2, opacity: 0.9})
      .bindTooltip(l[idx] + ' min to the nearest station', {sticky: true}).addTo(layer); });
    lg = '<b>' + (v === 'walkst' ? 'Walking' : 'Driving') + ' time to the nearest railway station</b><br>' +
      TIME.map((c, i) => '<i style="background:' + c + '"></i>' + (i === 0 ? 'up to ' + B[0] : i === B.length ? 'over ' + B[B.length - 1] : B[i - 1] + ' to ' + B[i]) + ' min').join('<br>') +
      '<br><span style="color:#777">from every road, over the road network to the nearest of all stations (searched up to 45 km beyond the square)' +
      (v === 'walkst' ? '; 4.8 km/h, motorways excluded' : '; typical speeds by road class') + '</span>';
  } else if (v === 'coll' || v === 'collf') {
    // hotspots: dark outline + strong colours, the street map faded, rings on fatal / busiest sites so they show when zoomed out
    const idx = v === 'coll' ? 6 : 7, CB = [1, 2, 4, 9], W = [1.8, 2.6, 3.6, 5, 7],
      CP = v === 'coll' ? ['#fdae6b', '#fd8d3c', '#e6550d', '#b30000', '#49006a'] : ['#d7191c', '#a50026', '#67000d', '#49006a', '#2d004b'];
    const cband = n => { let k = 0; while (k < CB.length && n > CB[k]) k++; return k; };
    faint();
    let tot = 0;
    D.links.filter(l => l[idx] > 0).sort((a, b) => a[idx] - b[idx]).forEach(l => { const k = cband(l[idx]), ll = pl(l[0]); tot += l[idx];
      const tip = l[idx] + (v === 'coll' ? ' injury collision' : ' fatal collision') + (l[idx] > 1 ? 's' : '') + ' on this road section (5 years)';
      const fat = v === 'collf';   // low counts thin and light so the hotspots stand out; a dark outline from 3 up
      if (fat || k >= 2) L.polyline(ll, {color: '#1a0000', weight: W[k] + (fat ? 4 : 2.5), opacity: 0.5, interactive: false}).addTo(layer);
      L.polyline(ll, {color: CP[k], weight: fat ? W[k] + 2 : W[k], opacity: k || fat ? 1 : 0.85}).bindTooltip(tip, {sticky: true}).addTo(layer);
      if (fat || k === 4) L.circleMarker(ll[Math.floor(ll.length / 2)], {radius: fat ? 11 : 13, color: CP[k], weight: 3,
        fillColor: CP[k], fillOpacity: 0.2}).bindTooltip(tip).addTo(layer); });
    lg = '<b>' + (v === 'coll' ? 'Road collision hotspots' : 'Fatal road collisions') + '</b><br>' +
      ['1', '2', '3 to 4', '5 to 9', '10 or more'].map((t, i) => '<i style="background:' + CP[i] + ';height:' + Math.round(W[i]) + 'px"></i>' + t).join('<br>') +
      '<br><span style="color:#777">collisions per road section; ' + (v === 'collf' ? 'every fatal site is ringed' : 'sections with 10 or more are ringed') +
      '. ' + tot + ' collision' + (tot === 1 ? '' : 's') + ' on the roads shown; each one placed on the nearest road within 30 m (DfT STATS19, last 5 years)</span>';
  } else if (v === 'reach' || v === 'reachf') {
    // each road coloured by the collisions reachable within N minutes' drive (eighths of the roads shown)
    const idx = v === 'reach' ? 10 : 11, vals = D.links.map(l => l[idx]).filter(x => typeof x === 'number').sort((a, b) => a - b);
    const q = p => vals[Math.min(vals.length - 1, Math.floor(p * vals.length))];
    const br = vals.length ? [...new Set([1, 2, 3, 4, 5, 6, 7].map(k => q(k / 8)))] : [];
    const PAL8 = ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#b10026'];
    const idc = k => PAL8[Math.round(k * (PAL8.length - 1) / Math.max(br.length, 1))];
    const col = x => { let k = 0; while (k < br.length && x > br[k]) k++; return idc(k); };
    D.links.forEach(l => { const x = l[idx]; if (typeof x !== 'number') return;
      L.polyline(pl(l[0]), {color: col(x), weight: 2.4, opacity: 0.9})
        .bindTooltip((v === 'reach' ? 'Injury' : 'Fatal') + ' collisions within a ' + D.reach_min + '-minute drive of this road: <b>' +
          x.toLocaleString('en-GB') + '</b>', {sticky: true}).addTo(layer); });
    const e = [vals[0], ...br, vals[vals.length - 1]];
    lg = '<b>' + (v === 'reach' ? 'Road collisions' : 'Fatal road collisions') + ' within a ' + (D.reach_min || 30) + '-minute drive</b><br>';
    for (let k = 0; k <= br.length && vals.length; k++) lg += '<i style="background:' + idc(k) + '"></i>' + e[k].toLocaleString('en-GB') + ' to ' + e[k + 1].toLocaleString('en-GB') + '<br>';
    lg += '<span style="color:#777">from each road: collisions over the last 5 years on all roads reachable within ' +
      (D.reach_min || 30) + ' minutes at typical (free-flow) speeds; eighths of the roads shown' +
      (v === 'reachf' ? '; fatal = collisions in which someone died (STATS19 severity 1)' : '') + '</span>';
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
    const N = D.lsoa; if (!N) { nnote.textContent = 'no LSOAs here (England and Wales only)'; layer.addTo(map); return; }
    N.edges.forEach(e => L.polyline([N.pts[e[0]], N.pts[e[1]]], e[2] ? {color: '#d7191c', weight: 2.5, dashArray: '6 4'} : {color: '#2b8a3e', weight: 1.2, opacity: 0.8}).addTo(layer));
    N.pts.forEach((p, i) => { const r = N.ring[i], c = v === 'rings' ? (r >= 0 ? RING[r] : '#bbb') : '#1d4e89';
      L.circleMarker(p, {radius: v === 'rings' && r === 0 ? 7 : 4, color: '#fff', weight: 1, fillColor: c, fillOpacity: 1})
        .bindTooltip(N.names[i] + (v === 'rings' && r >= 0 ? ': ' + (r === 0 ? 'centre' : 'within ' + r + ' boundar' + (r > 1 ? 'ies' : 'y')) : ''), {sticky: true}).addTo(layer); });
    const nb = N.edges.filter(e => e[2]).length;
    lg = v === 'rings' ? '<b>Within n boundaries of the centre LSOA</b><br>' + ['centre', 'within 1', 'within 2', 'within 3'].map((t, i) => '<b class="dot" style="background:' + RING[i] + '"></b>' + t).join('<br>') +
      '<br><span style="color:#777">the neighbourhoods used for spatial smoothing (bridges included)</span>'
      : '<b>LSOA neighbour network</b><br><i style="background:#2b8a3e"></i>areas that touch<br><i style="background:#d7191c"></i>bridge: joined by a road across a gap (' + nb + ')';
  }
  L.circleMarker(D.centre, {radius: 5, color: '#000', fillColor: '#fff', fillOpacity: 1, weight: 2}).bindTooltip('centre').addTo(layer);
  layer.addTo(map);
  document.getElementById('nlegend').innerHTML = lg;
}"""

# the old pages forward here, translating their settings
REDIRECT = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Map explorer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>
(function () {
  const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=')));
  const q = Object.entries(H).map(([k, v]) => k + '=' + v).join('&');
  location.replace('explorer.html' + (q ? '#' + q : ''));
})();
</script></head><body><p>This page has moved to the <a href="explorer.html">map explorer</a>.</p></body></html>"""


def build_explorer(out_dir: Path, size_m: float = 5000) -> Path | None:
    """Write explorer.html from the meta.json the mini map and network builds leave in their data folders,
    and the two forwarding pages."""
    mf, nf = out_dir / "minimap_data" / "meta.json", out_dir / "network_data" / "meta.json"
    mini = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else None
    net = json.loads(nf.read_text(encoding="utf-8")) if nf.exists() else None
    if not mini and not net:
        log.warning("explorer skipped: build the mini maps or networks first")
        return None
    locs, seen = [], {}
    for src, key in ((mini, "mini"), (net, "net")):
        for l in (src or {}).get("locations", []):
            if l["name"] not in seen:
                seen[l["name"]] = {"name": l["name"], "label": l["label"], "mini": False, "net": False}
                locs.append(seen[l["name"]])
            seen[l["name"]][key] = True
    meta = {"locations": locs, "layers": (mini or {}).get("layers", []), "vars": (mini or {}).get("vars", []),
            "areaVars": (mini or {}).get("areaVars", [])}
    page = (TEMPLATE.replace("__CSS__", PAGE_CSS).replace("__KM__", f"{size_m / 1000:g} km")
            .replace("__TILES__", _tiles_js())
            .replace("__VER__", str(int(time.time())))
            .replace("__OVERLAY__", OVERLAY_JS).replace("__NETWORK__", NETWORK_JS)
            .replace("__META__", json.dumps(meta, separators=(",", ":"))))
    out = out_dir / "explorer.html"
    out.write_text(page, encoding="utf-8")
    # the old mini map explorer address forwards here (its settings carry over as they are)
    (out_dir / "minimap_explorer.html").write_text(REDIRECT, encoding="utf-8")
    (out_dir / "network_explorer.html").unlink(missing_ok=True)   # retired: its views are part of this page
    log.info("wrote   %s (%d locations; mini map data for %d, networks for %d)", out.name, len(locs),
             sum(l["mini"] for l in locs), sum(l["net"] for l in locs))
    return out
