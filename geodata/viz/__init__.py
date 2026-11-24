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
        t = pq.read_table(data_dir / "extract" / "ons_nspl" / "nspl.parquet",
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
# them without one. Overridden from `global.basemaps` in config/visualise.yaml.
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


PAGE_CSS = """
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;margin:0;background:#f6f6f3;color:#222}
header{padding:18px 26px;background:#24343f;color:#fff}
header h1{margin:0 0 4px;font-size:22px;font-weight:600} header p{margin:0;opacity:.8;font-size:14px}
header a{color:#cde3f5}
main{padding:20px 26px}
"""
