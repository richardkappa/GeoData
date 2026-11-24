"""ONS postcode and UPRN lookups (the spine every other dataset joins to).

* ``ons_nspl``  National Statistics Postcode Lookup -> one row per postcode
  with every statistical geography code (OA/LSOA/MSOA 2021, LAD, ward,
  constituency, region, ...), the 2025 IMD rank, rural-urban class and an
  easting/northing on the British National Grid for the *whole* UK
  (Northern Ireland is converted from Irish Grid).
* ``ons_nsul``  National Statistics UPRN Lookup -> one row per UPRN
  (Unique Property Reference Number) with its postcode and geographies.
  Note: UDPRN (Royal Mail) is not open data; UPRN is the open address key.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .. import arcgis, io
from .. import download as dl
from ..postcode import POSTCODE_KEY, add_postcode_ns
from . import source

log = logging.getLogger(__name__)


def _download_item(settings, name, query, title_regex, exclude=r"user guide|2011 census"):
    item = arcgis.find_latest_item(query, title_regex, item_type="CSV Collection", exclude_regex=exclude)
    raw = settings.raw(name)
    dest = raw / f"{re.sub(r'[^0-9A-Za-z]+', '_', item['title']).strip('_')}.zip"
    dl.download(arcgis.item_data_url(item["id"]), dest, force=settings.force, timeout=settings.timeout)
    return item, dest


@source("ons_nspl", order=1, keys=["postcode_ns"],
        description="ONS National Statistics Postcode Lookup (latest): postcode -> geographies + BNG coords")
def nspl(settings, params):
    out = settings.out("ons_nspl") / "nspl.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        log.info("fresh   %s", out)
        return [out]
    item, zpath = _download_item(settings, "ons_nspl",
                                 '"National Statistics Postcode Lookup" owner:ONSGeography_data',
                                 r"^National Statistics Postcode Lookup \(")
    member = next(m for m in io.zip_members(zpath, r"^Data/NSPL[^/]*_UK\.csv$"))
    with io.Timer("nspl read"):
        t = io.read_csv_from_zip(zpath, member, strings=True)
    t = io.snake_columns(t)

    keep = params.get("columns")
    if keep:
        base = ["pcds", "dointr", "doterm", "usrtypind", "east1m", "north1m", "gridind", "lat", "long"]
        t = t.select([c for c in dict.fromkeys(base + list(keep)) if c in t.column_names])

    if not params.get("include_terminated", True):
        t = t.filter(pc.is_null(t["doterm"]))
    # live flag + standard key
    t = t.append_column("live", pc.is_null(t["doterm"]))
    t = add_postcode_ns(t, "pcds")
    for c in ("pcd7", "pcd8"):
        if c in t.column_names:
            t = t.drop_columns([c])

    # ONS uses "<letter>99999999" as a 'not applicable' pseudo code: make it null
    if params.get("null_pseudo_codes", True):
        for c in t.column_names:
            if c.endswith("cd"):
                col = t[c]
                t = t.set_column(t.schema.get_field_index(c), c,
                                 pc.if_else(pc.ends_with(col, "99999999"), pa.scalar(None, pa.string()), col))

    # numeric columns
    t = io.to_numeric(t, ["lat", "long"], pa.float64())
    t = io.to_numeric(t, ["east1m", "north1m"], pa.float64())
    t = io.to_numeric(t, ["imd25ind", "imd19ind", "usrtypind", "gridind"], pa.int32())

    # one BNG easting/northing for the whole UK (NI is on Irish grid in source)
    e = t["east1m"].to_numpy(zero_copy_only=False).astype("f8")
    n = t["north1m"].to_numpy(zero_copy_only=False).astype("f8")
    ni = pc.starts_with(t[POSTCODE_KEY], "BT").to_numpy(zero_copy_only=False)
    ni = ni & np.isfinite(e) & np.isfinite(n)
    if ni.any():
        e[ni], n[ni] = io.bng_from_irish(e[ni], n[ni])
    # gridind 9 / missing coords are placeholders, not real locations
    nogrid = ~np.isfinite(e) | (t["gridind"].to_numpy(zero_copy_only=False) == 9) if "gridind" in t.column_names else ~np.isfinite(e)
    e[nogrid] = np.nan
    n[nogrid] = np.nan
    t = t.append_column("easting", pa.array(np.round(e, 1), mask=np.isnan(e)))
    t = t.append_column("northing", pa.array(np.round(n, 1), mask=np.isnan(n)))

    # postcode_ns must be unique; keep the live record if a postcode recurs
    t = t.sort_by([(POSTCODE_KEY, "ascending"), ("live", "descending")])
    pcs = t[POSTCODE_KEY].to_numpy(zero_copy_only=False)
    first = np.ones(len(pcs), dtype=bool)
    first[1:] = pcs[1:] != pcs[:-1]
    if not first.all():
        log.info("dropped %d duplicate postcode rows", (~first).sum())
        t = t.filter(pa.array(first))

    io.write_table(t, out, meta={"source": item["title"], "item": item["id"]})
    return [out]


@source("ons_nsul", order=2, keys=["uprn", "postcode_ns"],
        description="ONS National Statistics UPRN Lookup (latest): UPRN -> postcode + geographies (large)")
def nsul(settings, params):
    out_dir = settings.out("ons_nsul")
    out = out_dir / "nsul.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    item, zpath = _download_item(settings, "ons_nsul",
                                 '"National Statistics UPRN Lookup" owner:ONSGeography_data',
                                 r"^National Statistics UPRN Lookup \(")
    members = io.zip_members(zpath, r"\.csv$")
    members = [m for m in members if re.search(r"/Data/|^Data/", m)] or members
    cols = params.get("columns")  # None = everything
    regions = params.get("regions")  # NSUL is split per region, e.g. ["EE","LN"]
    if regions:
        members = [m for m in members if any(re.search(rf"_{r}\.csv$", m, re.I) for r in regions)]
    import pyarrow.parquet as pq
    writer = None
    tmp = out.with_suffix(".parquet.tmp")
    rows = 0
    try:
        for m in members:
            t = io.read_csv_from_zip(zpath, m, strings=True)
            t = io.snake_columns(t)
            if cols:
                t = t.select([c for c in dict.fromkeys(["uprn", "pcds", "gridgb1e", "gridgb1n", *cols]) if c in t.column_names])
            t = add_postcode_ns(t, "pcds")
            t = io.to_numeric(t, ["uprn"], pa.int64())
            t = io.to_numeric(t, ["gridgb1e", "gridgb1n"], pa.float64())
            if writer is None:
                schema = t.schema
                writer = pq.ParquetWriter(tmp, schema, **io.PARQUET_KW)
            writer.write_table(t.cast(schema))
            rows += t.num_rows
            log.info("  nsul %s: %s rows", m.rsplit("/", 1)[-1], f"{t.num_rows:,}")
    finally:
        if writer:
            writer.close()
    tmp.replace(out)
    log.info("wrote   %s  (%s rows)", out.name, f"{rows:,}")
    return [out]
