"""Ordnance Survey OpenData (and BGS layers distributed through it).

Uses the OS Downloads API (no key needed for open products):
``https://api.os.uk/downloads/v1/products/<id>/downloads``. Configure which
products to fetch, their format, and (for GeoPackages) which layers to keep.

Special CSV handling:
* ``CodePointOpen`` -> ``codepoint_open.parquet`` keyed on ``postcode_ns``
  (an alternative postcode centroid source to the NSPL, GB only)
* ``OpenUPRN``      -> ``open_uprn.parquet`` (UPRN -> BNG coordinates)
* ``OpenNames``     -> ``open_names.parquet`` (places, roads, postcodes)
GeoPackage products are streamed to GeoParquet layer by layer.
"""
from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .. import download as dl
from .. import io, vector
from ..postcode import add_postcode_ns
from . import source

log = logging.getLogger(__name__)

API = "https://api.os.uk/downloads/v1/products"

CODEPOINT_NAMES = ["postcode", "positional_quality_indicator", "easting", "northing", "country_code",
                   "nhs_regional_ha_code", "nhs_ha_code", "admin_county_code", "admin_district_code",
                   "admin_ward_code"]
OPENNAMES_NAMES = ["id", "names_uri", "name1", "name1_lang", "name2", "name2_lang", "type", "local_type",
                   "geometry_x", "geometry_y", "most_detail_view_res", "least_detail_view_res",
                   "mbr_xmin", "mbr_ymin", "mbr_xmax", "mbr_ymax", "postcode_district",
                   "postcode_district_uri", "populated_place", "populated_place_uri",
                   "populated_place_type", "district_borough", "district_borough_uri",
                   "district_borough_type", "county_unitary", "county_unitary_uri",
                   "county_unitary_type", "region", "region_uri", "country", "country_uri",
                   "related_spatial_object", "same_as_dbpedia", "same_as_geonames"]


def _pick(product: str, fmt: str, area: str) -> dict:
    opts = dl.get_json(f"{API}/{product}/downloads")
    fmt_l = fmt.lower()
    for o in opts:
        if o["format"].lower().startswith(fmt_l) and o["area"] == area:
            import html
            return {**o, "url": html.unescape(o["url"])}  # some BGS links contain "&amp;"
    have = sorted({(o["format"], o["area"]) for o in opts})
    raise LookupError(f"{product}: no {fmt}/{area} download. Available: {have}")


def _stream_zip_csv_to_parquet(zpath: Path, members: list[str], out: Path, *, names, columns, transform):
    tmp = out.with_suffix(".parquet.tmp")
    writer, rows = None, 0
    with zipfile.ZipFile(zpath) as z:
        for m in members:
            with z.open(m) as f:
                peek = f.peek(64)[:64] if hasattr(f, "peek") else b""
                has_header = names is None or peek.upper().startswith(names[0].upper().encode())
                ro = pacsv.ReadOptions(column_names=None if (has_header and names is None) else names,
                                       skip_rows=1 if (has_header and names is not None) else 0,
                                       block_size=1 << 24)
                conv = pacsv.ConvertOptions(include_columns=columns,
                                            column_types={c: pa.string() for c in (columns or names or [])})
                reader = pacsv.open_csv(f, read_options=ro, convert_options=conv)
                for batch in reader:
                    t = transform(pa.Table.from_batches([batch]))
                    if writer is None:
                        schema = t.schema
                        writer = pq.ParquetWriter(tmp, schema, **io.PARQUET_KW)
                    writer.write_table(t.cast(schema))
                    rows += t.num_rows
    if writer:
        writer.close()
        tmp.replace(out)
        log.info("wrote   %s  (%s rows)", out.name, f"{rows:,}")
    return out


def _codepoint(zpath, out):
    members = io.zip_members(zpath, r"Data/CSV/.*\.csv$")

    def tf(t):
        t = add_postcode_ns(t, "postcode")
        t = io.to_numeric(t, ["easting", "northing"], pa.float64())
        return io.to_numeric(t, ["positional_quality_indicator"], pa.int8())
    return _stream_zip_csv_to_parquet(zpath, members, out, names=CODEPOINT_NAMES, columns=None, transform=tf)


def _open_uprn(zpath, out):
    members = io.zip_members(zpath, r"\.csv$")

    def tf(t):
        t = io.snake_columns(t)
        t = io.to_numeric(t, ["uprn"], pa.int64())
        t = io.to_numeric(t, ["x_coordinate", "y_coordinate", "latitude", "longitude"], pa.float64())
        return t.rename_columns([{"x_coordinate": "easting", "y_coordinate": "northing"}.get(c, c)
                                 for c in t.column_names])
    return _stream_zip_csv_to_parquet(zpath, members, out, names=None,
                                      columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"], transform=tf)


def _open_names(zpath, out, local_types=None):
    members = io.zip_members(zpath, r"DATA/.*\.csv$")
    cols = ["id", "name1", "type", "local_type", "geometry_x", "geometry_y", "postcode_district",
            "populated_place", "district_borough", "county_unitary", "region", "country"]
    import pyarrow.compute as pc

    def tf(t):
        t = io.to_numeric(t, ["geometry_x", "geometry_y"], pa.float64())
        t = t.rename_columns([{"geometry_x": "easting", "geometry_y": "northing"}.get(c, c) for c in t.column_names])
        if local_types:
            t = t.filter(pc.is_in(t["local_type"], pa.array(local_types)))
        return t
    return _stream_zip_csv_to_parquet(zpath, members, out, names=OPENNAMES_NAMES, columns=cols, transform=tf)


def _extract(zpath: Path, pattern: str) -> list[Path]:
    dest = zpath.with_suffix("")
    out = []
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if re.search(pattern, m, re.I):
                target = dest / Path(m).name
                if not target.exists() or target.stat().st_size != z.getinfo(m).file_size:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as s, open(target, "wb") as d:
                        while chunk := s.read(1 << 22):
                            d.write(chunk)
                out.append(target)
    return out


@source("os_open", order=40, keys=["postcode_ns", "uprn", "geometry"],
        description="Ordnance Survey / BGS OpenData via OS Downloads API (configurable products)")
def os_open(settings, params):
    outs = []
    out_dir = settings.out("os_open")
    for product, spec in (params.get("products") or {}).items():
        spec = spec or {}
        if spec.get("enabled", True) is False:
            continue
        fmt = spec.get("format", "GeoPackage")
        area = spec.get("area", "GB")
        stem = io.snake(f"{product}_{area}") if area != "GB" else io.snake(product)
        try:
            opt = _pick(product, fmt, area)
            zpath = settings.raw("os_open") / opt["fileName"]
            dl.download(opt["url"], zpath, force=settings.force, timeout=max(settings.timeout, 900),
                        expected_size=opt.get("size"))
        except Exception as e:
            log.error("os_open %s: %s", product, e)
            continue
        if spec.get("process", True) is False:
            log.info("os_open %s downloaded only (process: false) -> %s", product, zpath)
            continue
        rp = settings.reprocess or settings.force
        try:
            if product == "CodePointOpen":
                out = out_dir / "codepoint_open.parquet"
                outs.append(out if io.is_fresh(out, rp) else _codepoint(zpath, out))
            elif product == "OpenUPRN":
                out = out_dir / "open_uprn.parquet"
                outs.append(out if io.is_fresh(out, rp) else _open_uprn(zpath, out))
            elif product == "OpenNames":
                out = out_dir / "open_names.parquet"
                outs.append(out if io.is_fresh(out, rp) else _open_names(zpath, out, spec.get("local_types")))
            elif fmt.lower().startswith("geopackage"):
                gpkgs = _extract(zpath, r"\.gpkg$")
                for gpkg in gpkgs:
                    wanted = spec.get("layers")
                    for layer in vector.list_layers(gpkg):
                        if (wanted and layer not in wanted) or layer.lower() == "layer_styles":
                            continue
                        name = layer if len(gpkgs) == 1 else f"{gpkg.stem}_{layer}"
                        out = out_dir / f"{stem}__{io.snake(name)}.parquet"
                        if io.is_fresh(out, rp):
                            outs.append(out)
                            continue
                        with io.Timer(f"{product}/{layer}"):
                            outs.append(vector.to_geoparquet(gpkg, out, layer=layer,
                                                             columns=spec.get("columns"),
                                                             target_crs=settings.crs))
                    if not settings.keep_raw:
                        gpkg.unlink(missing_ok=True)
            else:
                log.info("os_open %s: format %s kept as raw download only", product, fmt)
        except Exception as e:
            log.exception("os_open %s processing failed: %s", product, e)
    return outs
