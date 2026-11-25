"""Dynamic cartograms: areas resized by the people who live there, with animation.

Two styles on one page (``dynamic_cartograms.html``), for Westminster constituencies and local authorities:

* **Morphing cartogram** - a contiguous, shape-keeping cartogram: every area keeps its neighbours and
  roughly its shape but is stretched so its size matches its population. Worked out here with the
  Gastner-Newman diffusion method (2004): the population density is spread out like heat on a grid
  until it is even everywhere, and every boundary point is carried along by the flow. A slider (and
  a play button) morphs between the real map and the cartogram.
* **Dorling cartogram** - one circle per area, sized by residents, households, addresses or land area,
  starting at the area's real position and pushed apart by a force simulation in the browser until no
  circles overlap; changing the sizing re-runs it, animated.

Both are coloured by any of the area variables (the cartogram / choropleth list), averaged with the same
population weighting as the other area pages.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from . import PAGE_CSS, SCALE_JS
from .cartograms import _aggregate, _postcode_table, var_meta
from .postcode_data import _num

log = logging.getLogger(__name__)


# --- geometry -------------------------------------------------------------------------------------
def _area_rings(geoms, tol: float, min_part_m2: float):
    """Coverage-simplify the areas together (shared borders stay shared) -> per area a list of exterior
    rings as (n, 2) arrays in metres. Tiny islands are dropped (an area always keeps its largest part)."""
    import shapely
    g = shapely.coverage_simplify(np.asarray(geoms), tol) if tol else np.asarray(geoms)
    out = []
    for geom in g:
        parts = [p for p in shapely.get_parts(geom) if not p.is_empty]
        if not parts:
            out.append([])
            continue
        areas = np.array([p.area for p in parts])
        keep = [p for p, a in zip(parts, areas) if a >= min_part_m2 or a == areas.max()]
        out.append([shapely.get_coordinates(shapely.get_exterior_ring(p))[:-1] for p in keep])
    return out


def _poly_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# --- Gastner-Newman diffusion cartogram -----------------------------------------------------------
def _rasterise_density(rings, pops, nx, ny):
    """Population density on the grid (grid units): each area's people spread evenly over its cells;
    cells outside every area (the sea) get the mean density so the coast is not pulled about."""
    import shapely
    yy, xx = np.mgrid[0:ny, 0:nx] + 0.5
    rho = np.full((ny, nx), np.nan)
    land_pop = land_area = 0.0
    for rs, pop in zip(rings, pops):
        if not rs or not np.isfinite(pop) or pop <= 0:
            continue
        area = sum(_poly_area(r) for r in rs)
        if area <= 0:
            continue
        dens = pop / area
        land_pop += pop
        land_area += area
        for r in rs:
            x0, y0 = np.floor(r.min(axis=0)).astype(int)
            x1, y1 = np.ceil(r.max(axis=0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, nx), min(y1, ny)
            if x1 <= x0 or y1 <= y0:
                continue
            poly = shapely.Polygon(r)
            sx, sy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
            inside = shapely.contains_xy(poly, sx, sy)
            block = rho[y0:y1, x0:x1]
            block[inside] = np.where(np.isnan(block[inside]), dens, block[inside] + dens)
    mean = land_pop / land_area
    return np.where(np.isnan(rho), mean, rho)


def _diffuse(points: np.ndarray, rho0: np.ndarray, max_steps: int = 6000) -> np.ndarray:
    """Carry points (grid units, x right / y down) along the diffusion flow until the density is even."""
    from scipy.fft import dctn, idctn
    from scipy.ndimage import gaussian_filter, map_coordinates
    ny, nx = rho0.shape
    a = dctn(gaussian_filter(rho0, 1.0), type=2, norm="ortho")
    ky, kx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    lam = np.pi ** 2 * ((kx / nx) ** 2 + (ky / ny) ** 2)       # decay rate of each cosine mode

    def velocity(t, p):
        r = idctn(a * np.exp(-lam * t), type=2, norm="ortho")
        gy, gx = np.gradient(r)
        coords = [p[:, 1] - 0.5, p[:, 0] - 0.5]                # map_coordinates wants (row, col)
        rr = map_coordinates(r, coords, order=1, mode="nearest")
        vx = -map_coordinates(gx, coords, order=1, mode="nearest") / rr
        vy = -map_coordinates(gy, coords, order=1, mode="nearest") / rr
        return np.column_stack([vx, vy])

    t, p = 0.0, points.astype("f8").copy()
    t_end = 8 * max(nx, ny) ** 2 / np.pi ** 2                  # the slowest mode has decayed by e^-8
    for _ in range(max_steps):
        v1 = velocity(t, p)
        vmax = np.abs(v1).max()
        if vmax < 1e-6:
            break
        dt = min(0.4 / vmax, max(t, 1.0))                      # move under half a cell per step
        mid = p + 0.5 * dt * v1                                # second-order (midpoint) step
        p = p + dt * velocity(t + 0.5 * dt, mid)
        p[:, 0] = np.clip(p[:, 0], 0, nx)
        p[:, 1] = np.clip(p[:, 1], 0, ny)
        t += dt
        if t >= t_end:
            break
    return p


def diffusion_cartogram(rings, pops, cells: int = 512, passes: int = 3):
    """Rings (metres) and populations -> cartogram rings (metres). Several passes: each re-measures the
    density of the shapes it has so far, which sharpens the result."""
    allxy = np.vstack([r for rs in rings for r in rs])
    (x0, y0), (x1, y1) = allxy.min(axis=0), allxy.max(axis=0)
    pad = 0.35 * max(x1 - x0, y1 - y0)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    h = max(x1 - x0, y1 - y0) / cells
    nx, ny = int(np.ceil((x1 - x0) / h)), int(np.ceil((y1 - y0) / h))
    to_g = lambda xy: np.column_stack([(xy[:, 0] - x0) / h, (y1 - xy[:, 1]) / h])
    from_g = lambda g: np.column_stack([g[:, 0] * h + x0, y1 - g[:, 1] * h])
    cur = [[to_g(r) for r in rs] for rs in rings]
    sizes = [len(r) for rs in cur for r in rs]
    for k in range(passes):
        rho = _rasterise_density(cur, pops, nx, ny)
        flat = np.vstack([r for rs in cur for r in rs])
        moved = _diffuse(flat, rho)
        out, i = [], 0
        it = iter(sizes)
        for rs in cur:
            new = []
            for _ in rs:
                n = next(it)
                new.append(moved[i:i + n])
                i += n
            out.append(new)
        cur = out
    return [[from_g(r) for r in rs] for rs in cur]


def _fit_error(rings, pops) -> float:
    """Mean absolute error of area size vs population share (0 = perfect)."""
    a = np.array([sum(_poly_area(r) for r in rs) for rs in rings])
    p = np.asarray(pops, float)
    ok = (a > 0) & (p > 0)
    return float(np.mean(np.abs(a[ok] / a[ok].sum() - p[ok] / p[ok].sum()) / (p[ok] / p[ok].sum())))


# --- page -----------------------------------------------------------------------------------------
TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dynamic cartograms</title>
<style>__CSS__
#bar{display:flex;gap:10px 14px;align-items:stretch;padding:8px 20px 10px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}
#bar select{font-size:14px;padding:4px;max-width:420px}
.grp{border:1px solid #d9d9d6;border-top:3px solid var(--c);border-radius:6px;margin:0;padding:2px 12px 8px;display:flex;gap:6px 14px;
  align-items:center;flex-wrap:wrap;background:#fbfbfa;min-width:0}
.grp>legend{font-size:11px;font-weight:700;color:var(--c);text-transform:uppercase;letter-spacing:.05em;padding:0 4px}
.g1{--c:#555} .g2{--c:#1d5f8a} .g3{--c:#6a3d9a}
button{font-size:13px;padding:4px 12px;cursor:pointer;border:1px solid #bbb;border-radius:4px;background:#f4f4f4}
#wrap{display:flex;gap:16px;padding:12px 20px;align-items:flex-start;flex-wrap:wrap}
#svg{background:#fff;border-radius:6px;box-shadow:0 1px 3px #0002;flex:1;min-width:320px;max-width:900px;height:calc(100vh - 300px);min-height:420px}
#side{width:300px;font-size:13px}
.box{background:#fff;border-radius:6px;box-shadow:0 1px 3px #0002;padding:10px 12px;margin-bottom:12px}
.legend{line-height:18px} .legend i{display:inline-block;width:16px;height:14px;margin-right:6px;vertical-align:-2px}
path.a,circle.a{stroke:#fff;stroke-width:0.25;vector-effect:non-scaling-stroke} path.a:hover,circle.a:hover{stroke:#111;stroke-width:1.5}
#tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #bbb;border-radius:5px;padding:6px 8px;font-size:12px;
  box-shadow:0 2px 6px #0003;display:none;max-width:280px}
#mrow{display:flex;align-items:center;gap:8px} #morph{width:180px}
</style></head><body>
<header><h1>Dynamic cartograms</h1><p>Areas resized by the people who live there. The <b>morphing cartogram</b> keeps every area
touching its neighbours but stretches it until its size matches its population (Gastner-Newman diffusion); slide or press play to
move between the real map and the cartogram. The <b>Dorling cartogram</b> turns each area into a circle sized by its people (or
households, addresses or land), pushed apart from its real position until none overlap. Colours: averages weighted by residents.
<a href="index.html">All visualisations</a></p></header>
<div id="bar">
<fieldset class="grp g1"><legend>Areas</legend><select id="geo" aria-label="Areas"></select></fieldset>
<fieldset class="grp g2"><legend>Style</legend><select id="style" aria-label="Style">
<option value="morph">Morphing cartogram (area = population)</option><option value="dorling">Dorling circles</option></select>
<span id="mrow"><label for="morph">Real map</label><input type="range" id="morph" min="0" max="100" value="100"><label for="morph">Cartogram</label>
<button type="button" id="play">Play</button></span>
<label id="sizeL">Circle size <select id="size"><option value="residents">residents</option><option value="households">households</option>
<option value="addresses">addresses</option><option value="land">land area</option></select></label>
<button type="button" id="replay">Re-run</button></fieldset>
<fieldset class="grp g3"><legend>Colour</legend><select id="var" aria-label="Colour by"></select></fieldset></div>
<div id="wrap"><svg id="svg" xmlns="http://www.w3.org/2000/svg"></svg><div id="side"><div class="box legend" id="legend"></div>
<div class="box" id="info"></div></div></div>
<div id="tip"></div>
<script>
const D = __DATA__;
__SCALE__
const svg = document.getElementById('svg'), geoSel = document.getElementById('geo'), styleSel = document.getElementById('style');
const varSel = document.getElementById('var'), morph = document.getElementById('morph'), sizeSel = document.getElementById('size');
const tip = document.getElementById('tip'), NS = 'http://www.w3.org/2000/svg';
D.geos.forEach((g, i) => geoSel.add(new Option(g.label, i)));
D.vars.forEach((v, i) => varSel.add(new Option(v.label, i)));
let G = null, els = [], sim = null;
const ease = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
function pathAt(rg, rc, f) {   // one area at morph fraction f: every boundary point moves in a straight line
  let d = '';
  for (let k = 0; k < rg.length; k++) { const a = rg[k], b = rc[k];
    for (let i = 0; i < a.length; i += 2) d += (i ? 'L' : 'M') + (a[i] + (b[i] - a[i]) * f).toFixed(2) + ' ' + (a[i + 1] + (b[i + 1] - a[i + 1]) * f).toFixed(2);
    d += 'Z'; }
  return d;
}
function colours() {
  const v = D.vars[+varSel.value], vals = G.values[+varSel.value], S = makeScale(vals, v);
  els.forEach((e, i) => e.setAttribute('fill', S.col(vals[i]) || '#e3e3e3'));
  document.getElementById('legend').innerHTML = legendHtml(v.label, S, '<br><i style="background:#e3e3e3"></i>no data');
}
function tipFor(i) {
  const v = D.vars[+varSel.value];
  return '<b>' + G.names[i] + '</b><br>' + v.label + ': <b>' + fmt(G.values[+varSel.value][i]) + '</b><br>' +
    fmt(G.residents[i]) + ' residents, ' + fmt(G.addresses[i]) + ' addresses';
}
function hook(e, i) {
  e.addEventListener('mousemove', ev => { tip.innerHTML = tipFor(i); tip.style.display = 'block';
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - 300) + 'px'; tip.style.top = (ev.clientY + 14) + 'px'; });
  e.addEventListener('mouseleave', () => tip.style.display = 'none');
}
function drawMorph() {
  const f = ease(+morph.value / 100);
  els.forEach((e, i) => e.setAttribute('d', pathAt(G.rg[i], G.rc[i], f)));
}
function build() {
  if (sim) { cancelAnimationFrame(sim); sim = null; }
  G = D.geos[+geoSel.value]; svg.innerHTML = ''; els = [];
  svg.setAttribute('viewBox', G.view.join(' '));
  const dorling = styleSel.value === 'dorling';
  document.getElementById('mrow').style.display = dorling ? 'none' : '';
  document.getElementById('sizeL').style.display = document.getElementById('replay').style.display = dorling ? '' : 'none';
  if (dorling) {   // faint real map underneath for orientation
    G.rg.forEach(rg => { const p = document.createElementNS(NS, 'path'); p.setAttribute('d', pathAt(rg, rg, 0));
      p.setAttribute('fill', '#f1f1ee'); p.setAttribute('stroke', '#ddd'); p.setAttribute('stroke-width', '0.3');
      p.setAttribute('vector-effect', 'non-scaling-stroke'); svg.appendChild(p); });
  }
  G.names.forEach((n, i) => {
    const e = document.createElementNS(NS, dorling ? 'circle' : 'path'); e.setAttribute('class', 'a'); hook(e, i);
    svg.appendChild(e); els.push(e); });
  colours();
  if (dorling) runDorling(); else drawMorph();
  info();
}
function runDorling() {   // force simulation: circles start at their real centres, collide, and are pulled home
  if (sim) cancelAnimationFrame(sim);
  const key = sizeSel.value, w = key === 'land' ? G.land : G[key];
  const total = w.reduce((a, b) => a + (b > 0 ? b : 0), 0);
  const k = Math.sqrt(G.landTotal * 0.45 / (Math.PI * total));   // circles cover ~45% of the land
  const n = G.names.length, x = G.cx.slice(), y = G.cy.slice(), r = w.map(v => k * Math.sqrt(Math.max(v, 0)));
  let tick = 0;
  const step = () => {
    for (let rep = 0; rep < 3; rep++) {
      for (let i = 0; i < n; i++) { x[i] += (G.cx[i] - x[i]) * 0.02; y[i] += (G.cy[i] - y[i]) * 0.02; }
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const dx = x[j] - x[i], dy = y[j] - y[i], d = Math.hypot(dx, dy) || 1e-6, o = r[i] + r[j] + 0.3 - d;
        if (o > 0) { const s = o / d / 2, wi = r[j] / (r[i] + r[j]), wj = 1 - wi;
          x[i] -= dx * s * wi; y[i] -= dy * s * wi; x[j] += dx * s * wj; y[j] += dy * s * wj; } }
    }
    els.forEach((e, i) => { e.setAttribute('cx', x[i].toFixed(2)); e.setAttribute('cy', y[i].toFixed(2)); e.setAttribute('r', r[i].toFixed(2)); });
    if (++tick < 120) sim = requestAnimationFrame(step); else sim = null;
  };
  step();
}
function info() {
  const dorling = styleSel.value === 'dorling';
  document.getElementById('info').innerHTML = dorling
    ? '<b>Dorling cartogram</b><br>One circle per area, its size set by the ' + sizeSel.value.replace('land', 'land area') +
      ' in it. Circles start where the area really is and are pushed apart until they no longer overlap, so places keep ' +
      'roughly their real positions. The faint map underneath is the real geography.'
    : '<b>Morphing cartogram</b><br>Each area\'s size is in proportion to its residents, while it keeps touching the same ' +
      'neighbours. Cities swell and thinly populated places shrink. Fit: area sizes are within ' + G.fit + '% of their population ' +
      'share on average.';
}
let playing = null;
document.getElementById('play').onclick = () => {
  if (playing) return;
  const from = +morph.value >= 50 ? 100 : 0, to = 100 - from, t0 = performance.now();
  const go = now => { const f = Math.min(1, (now - t0) / 2500); morph.value = from + (to - from) * f; drawMorph();
    playing = f < 1 ? requestAnimationFrame(go) : null; };
  playing = requestAnimationFrame(go);
};
morph.oninput = drawMorph;
geoSel.onchange = build; styleSel.onchange = build; varSel.onchange = colours;
sizeSel.onchange = runDorling; document.getElementById('replay').onclick = runDorling;
const H = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=')));
if (H.geo && geoSel.options[+H.geo]) geoSel.value = H.geo;
if (H.style) styleSel.value = H.style;
if (H.var) { const i = varIndex(D.vars, H.var); if (i !== null) varSel.value = i; }
build();
</script></body></html>"""


def build_dynamic(data_dir: Path, out_dir: Path, cfg: dict) -> Path | None:
    import geopandas as gpd
    t0 = time.perf_counter()
    variables = {c: (v if isinstance(v, dict) else {"label": v}) for c, v in (cfg.get("variables") or {}).items()}
    geos_cfg = cfg.get("geographies") or {}
    t = _postcode_table(data_dir, [g["code"] for g in geos_cfg.values()], variables)
    found = [c for c in variables if c in t.column_names]
    data = {"vars": [var_meta(c, variables[c]) for c in found], "geos": []}
    for name, g in geos_cfg.items():
        f = data_dir / "extract" / g["file"]
        if not f.exists():
            log.warning("dynamic cartograms: %s skipped - %s missing", g["label"], g["file"])
            continue
        gdf = gpd.read_parquet(f, columns=[g["key"], g.get("name_col", g["key"]), "geometry"])
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
        codes = gdf[g["key"]].astype(str).tolist()
        agg = _aggregate(t, g["code"], codes, variables)
        pops = np.asarray(agg["residents"], float)
        rings = _area_rings(gdf.geometry.values, float(g.get("simplify_m", 800)), float(g.get("min_part_km2", 15)) * 1e6)
        keep = [i for i, rs in enumerate(rings) if rs and pops[i] > 0]
        rings = [rings[i] for i in keep]
        pops = pops[keep]
        tc = time.perf_counter()
        carto = diffusion_cartogram(rings, pops, int(g.get("cells", 512)), int(g.get("passes", 3)))
        fit = _fit_error(carto, pops)
        log.info("dynamic cartograms: %s - %d areas, %d points, cartogram in %.0fs, mean size error %.0f%% (real map %.0f%%)",
                 g["label"], len(keep), sum(len(r) for rs in rings for r in rs), time.perf_counter() - tc,
                 100 * fit, 100 * _fit_error(rings, pops))
        allxy = np.vstack([r for rs in rings + carto for r in rs])
        (x0, y0), (x1, y1) = allxy.min(axis=0), allxy.max(axis=0)
        km = lambda xy: np.round(np.column_stack([(xy[:, 0] - x0) / 1000, (y1 - xy[:, 1]) / 1000]), 2).ravel().tolist()
        land = [sum(_poly_area(r) for r in rs) / 1e6 for rs in rings]
        cent = []
        for rs in rings:     # area-weighted centroid of the parts' vertices (good enough for a starting point)
            big = max(rs, key=_poly_area)
            c = big.mean(axis=0)
            cent.append(((c[0] - x0) / 1000, (y1 - c[1]) / 1000))
        pick = lambda arr: [arr[i] for i in keep]
        data["geos"].append({
            "label": g["label"], "names": pick(gdf[g.get("name_col", g["key"])].astype(str).tolist()),
            "view": [-10, -10, round((x1 - x0) / 1000 + 20, 1), round((y1 - y0) / 1000 + 20, 1)],
            "rg": [[km(r) for r in rs] for rs in rings], "rc": [[km(r) for r in rs] for rs in carto],
            "cx": [round(c[0], 2) for c in cent], "cy": [round(c[1], 2) for c in cent],
            "residents": [int(round(v)) for v in pops], "households": [int(round(v)) for v in pick(agg["households"])]
            if "households" in agg else [int(round(v)) for v in pops],
            "addresses": [int(round(v)) for v in pick(agg["addresses"])],
            "land": [round(v, 1) for v in land], "landTotal": round(sum(land), 1), "fit": round(100 * fit),
            "values": [_num(pa.chunked_array([pa.array(np.asarray(agg[c])[keep], from_pandas=True)])) for c in found]})
    if not data["geos"]:
        return None
    page = (TEMPLATE.replace("__CSS__", PAGE_CSS).replace("__SCALE__", SCALE_JS)
            .replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False)))
    out = out_dir / "dynamic_cartograms.html"
    out.write_text(page, encoding="utf-8")
    log.info("wrote   %s (%.1f MB) in %.0fs", out.name, out.stat().st_size / 1e6, time.perf_counter() - t0)
    return out
