"""ArcGIS Online helpers.

ONS Geography, Natural England, Historic England and the Met Office all
publish on ArcGIS Online. Two access patterns are supported:

* **Items** (``/sharing/rest/content/items/<id>/data``) for bulk files such as
  the NSPL zip. :func:`find_latest_item` searches an owner's items and picks the
  newest whose title matches a regex, so the code follows new releases
  automatically.
* **Feature services** (``.../FeatureServer/<layer>``) for boundaries and
  designations. :func:`fetch_layer` pages through a layer in parallel and
  returns a GeoDataFrame (parsed by GDAL via pyogrio, not Python loops).
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import download as dl

log = logging.getLogger(__name__)

AGOL = "https://www.arcgis.com/sharing/rest"
ONS_SERVICES = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"


# ------------------------------------------------------------------------ items
def search_items(query: str, *, num: int = 100) -> list[dict]:
    out, start = [], 1
    while start > 0 and len(out) < 1000:
        d = dl.get_json(f"{AGOL}/search", params={
            "q": query, "f": "json", "num": num, "start": start,
            "sortField": "modified", "sortOrder": "desc"})
        out += d.get("results", [])
        start = d.get("nextStart", -1)
    return out


def find_latest_item(query: str, title_regex: str, item_type: str | None = None,
                     exclude_regex: str | None = None) -> dict:
    rx = re.compile(title_regex, re.I)
    ex = re.compile(exclude_regex, re.I) if exclude_regex else None
    cands = [r for r in search_items(query)
             if rx.search(r["title"]) and not (ex and ex.search(r["title"]))
             and (item_type is None or r["type"] == item_type)]
    if not cands:
        raise LookupError(f"No ArcGIS item matches {title_regex!r} for query {query!r}")
    cands.sort(key=lambda r: r.get("created", 0), reverse=True)
    best = cands[0]
    log.info("item    %s  (%s)", best["title"], best["id"])
    return best


def item_data_url(item_id: str) -> str:
    return f"{AGOL}/content/items/{item_id}/data"


# ------------------------------------------------------------------ services
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _vintage(name: str) -> tuple[int, int]:
    y = re.findall(r"(?<!\d)(19|20)(\d{2})(?!\d)", name)
    year = int("".join(y[0])) if y else 0
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", name, re.I)
    month = _MONTHS[m.group(1).lower()] if m else 0
    return year, month


def list_services(root: str = ONS_SERVICES) -> list[str]:
    d = dl.get_json(f"{root}?f=json")
    return [s["name"] for s in d.get("services", []) if s.get("type") == "FeatureServer"]


def resolve_service(spec: dict[str, Any], root: str = ONS_SERVICES) -> str:
    """Return a FeatureServer layer URL from a config spec.

    Spec keys: ``url`` (full layer URL) or ``service`` (exact name) or
    ``service_regex`` (newest matching name by the year/month in its name);
    plus optional ``layer`` (default 0) and ``root``.
    """
    if spec.get("url"):
        return spec["url"].rstrip("/")
    root = spec.get("root", root).rstrip("/")
    layer = spec.get("layer")
    name = spec.get("service")
    if not name:
        rx = re.compile(spec["service_regex"], re.I)
        names = [n for n in list_services(root) if rx.search(n)]
        if not names:
            raise LookupError(f"No service matches {spec['service_regex']!r} under {root}")
        names.sort(key=_vintage)
        name = names[-1]
        log.info("service %s", name)
    if layer is None:  # layer ids are not always 0: take the service's first layer/table
        svc = dl.get_json(f"{root}/{name}/FeatureServer", params={"f": "json"})
        if "error" in svc:
            raise LookupError(f"{name}: {svc['error'].get('message')}")
        first = (svc.get("layers") or []) + (svc.get("tables") or [])
        if not first:
            raise LookupError(f"{name} has no layers")
        layer = first[0]["id"]
    return f"{root}/{name}/FeatureServer/{layer}"


def layer_info(layer_url: str) -> dict:
    info = dl.get_json(layer_url, params={"f": "json"})
    if "error" in info:
        raise LookupError(f"{layer_url}: {info['error'].get('message')}")
    return info


def fetch_layer(layer_url: str, *, fields: list[str] | str = "*", where: str = "1=1",
                threads: int = 4, out_sr: int = 27700, max_allowable_offset: float | None = None):
    """Download every feature of a layer into a GeoDataFrame (EPSG:27700).

    Object IDs are fetched first, then pulled in ``maxRecordCount`` batches in
    parallel threads. Tables without geometry come back as a DataFrame.
    """
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    info = layer_info(layer_url)
    page = int(info.get("maxRecordCount") or 1000)
    page = min(page, 2000)
    oid_field = info.get("objectIdField") or "OBJECTID"
    has_geom = bool(info.get("geometryType"))

    ids = dl.get_json(f"{layer_url}/query", params={"where": where, "returnIdsOnly": "true", "f": "json"})
    oids = sorted(ids.get("objectIds") or [])
    log.info("layer   %s: %s features", info.get("name"), f"{len(oids):,}")
    if not oids:
        return gpd.GeoDataFrame() if has_geom else pd.DataFrame()

    out_fields = fields if isinstance(fields, str) else ",".join(dict.fromkeys([oid_field, *fields]))
    batches = [oids[i:i + page] for i in range(0, len(oids), page)]

    def get(batch):
        data = {"objectIds": ",".join(map(str, batch)), "outFields": out_fields,
                "returnGeometry": "true" if has_geom else "false", "f": "geojson",
                "outSR": "4326"}
        if max_allowable_offset:
            data["maxAllowableOffset"] = str(max_allowable_offset)
        raw = dl.get_bytes(f"{layer_url}/query", data=data, timeout=300)
        return pyogrio.read_dataframe(raw, use_arrow=True) if has_geom else \
            pd.json_normalize([f["properties"] for f in __import__("json").loads(raw)["features"]])

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        parts = list(ex.map(get, batches))

    if not has_geom:
        return pd.concat(parts, ignore_index=True)
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    if out_sr:
        gdf = gdf.to_crs(epsg=out_sr)
    return gdf
