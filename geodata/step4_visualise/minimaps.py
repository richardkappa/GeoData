"""Static mini maps: a size_m square around each key location, every layer drawn,
collected into one HTML gallery (images embedded) plus a PNG per location."""
from __future__ import annotations

import html
import json
import time
import logging
from pathlib import Path

import numpy as np

from .. import io
from . import PAGE_CSS, label, png_data_uri, square

log = logging.getLogger(__name__)


class _BBoxLayer:
    """A layer too big to load whole (e.g. trees outside woodland, millions of polygons): read square by
    square through the file's spatial index. Supports the ``.cx[x0:x1, y0:y1]`` slicing the builds use."""

    def __init__(self, path: Path, spec: dict):
        import pyogrio
        self.path = str(path)
        self.layers = [spec["layer"]] if spec.get("layer") else [l[0] for l in pyogrio.list_layers(self.path)]
        self.min_area = float(spec.get("min_area_m2", 0))

    @property
    def cx(self):
        return self

    def __getitem__(self, key):
        import pyogrio
        xs, ys = key
        import pandas as pd
        bbox = (xs.start, ys.start, xs.stop, ys.stop)
        parts = [pyogrio.read_dataframe(self.path, layer=l, columns=[], bbox=bbox) for l in self.layers]
        parts = [p for p in parts if len(p)]
        if not parts:
            return pyogrio.read_dataframe(self.path, layer=self.layers[0], columns=[], max_features=0)
        g = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        if self.min_area:
            g = g[g.geometry.area >= self.min_area]
        return g


def _load_layer(data_dir: Path, spec: dict):
    import geopandas as gpd
    import pyarrow.parquet as pq
    if spec.get("raw"):   # a large file read square by square (path under <data_dir>/1_raw)
        path = data_dir / "1_raw" / spec["raw"]
        if not path.exists():
            log.warning("minimaps: layer %s skipped - %s missing (run 01_extract.py)", spec["name"], spec["raw"])
            return None
        return _BBoxLayer(path, spec)
    path = data_dir / "1_extract" / spec["file"]
    if not path.exists():
        log.warning("minimaps: layer %s skipped - %s missing", spec["name"], spec["file"])
        return None
    names = pq.read_schema(path).names
    filt = spec.get("filter") or {}
    extra = [c for c in {*filt, spec.get("by")} if c and c in names]
    if "geometry" in names:
        gdf = gpd.read_parquet(path, columns=[*extra, "geometry"])
    else:   # plain easting/northing tables (section 1 gives every point table these: geodata/geocode.py)
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
    from .postcode_data import _num
    f = next((data_dir / "1_extract" / p for p in [lcfg.get("file"), lcfg.get("fallback_file")]
              if p and (data_dir / "1_extract" / p).exists()), None)
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
    from .postcode_data import _area_table, _num, var_label
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
                    feats = _geojson_features(sub, big, to_ll, by, spec.get("simplify_m", 1.0))
                    if spec.get("separate"):   # heavy layer: its own file, loaded only when the layer is ticked
                        li = len(out_layers)
                        key = f"{name}__{li}"
                        (data_dir_out / f"{key}.js").write_text(
                            f"window.MINIMAP_LAYER[{json.dumps(key)}]=" + json.dumps(feats, separators=(",", ":")) + ";",
                            encoding="utf-8")
                        out_layers.append({"file": key} if feats else None)
                    else:
                        out_layers.append(feats)
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
            "vars": [var_label(v) for v in variables.values()], "areaVars": lsoa["labels"] if lsoa else []}
    # the page itself is the map explorer (explorer.py), which reads this
    (data_dir_out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    out = data_dir_out / "meta.json"
    total = sum(p.stat().st_size for p in data_dir_out.glob("*.js")) / 1e6
    log.info("wrote   minimap_data (%d location files, %.1f MB)", len(locs), total)
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
                if gdf is None or spec.get("static") is False:   # static: false = explorer only
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
    fig.legend(handles=_legend_handles([l for l in layers if l.get("static") is not False]), loc="center", ncol=6, frameon=False, fontsize=8)
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
Historic England; Natural England; NaPTAN; ONS boundaries. <a href="explorer.html">Map explorer</a> (interactive: layers, postcode features, road networks) |
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
