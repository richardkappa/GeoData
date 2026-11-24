"""data.police.uk street-level crime, outcomes and stop & search.

The monthly archive (``/data/archive/<YYYY-MM>.zip``) holds the previous 36
months for every force (~1.7 GB). We stream it once and write:

* ``crimes.parquet``            one row per crime: month, type, LSOA, lon/lat,
                                easting/northing
* ``crimes_lsoa_month.parquet``  counts per LSOA x month x crime type (small,
                                joins to NSPL ``lsoa21cd``)
* ``stop_and_search.parquet``   one row per stop (points only, no LSOA)
* ``outcomes.parquet``          optional

Locations are anonymised to snap points, so treat them as approximate.
"""
from __future__ import annotations

import io as _io
import logging
import zipfile

import pyarrow as pa
import pyarrow.compute as pc

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

ARCHIVE = "https://data.police.uk/data/archive/{v}.zip"
DATES = "https://data.police.uk/api/crimes-street-dates"

CRIME_COLS = ["Crime ID", "Month", "Falls within", "Longitude", "Latitude", "LSOA code",
              "Crime type", "Last outcome category"]
SAS_COLS = ["Type", "Date", "Latitude", "Longitude", "Gender", "Age range",
            "Officer-defined ethnicity", "Legislation", "Object of search", "Outcome"]
OUT_COLS = ["Crime ID", "Month", "Falls within", "Longitude", "Latitude", "LSOA code", "Outcome type"]


def _read_members(z: zipfile.ZipFile, members, cols, months):
    parts = []
    for m in members:
        if months and m.split("/", 1)[0] not in months:
            continue
        data = z.read(m)
        if len(data) < 50:
            continue
        t = io.read_csv(_io.BytesIO(data), strings=True)
        t = t.select([c for c in cols if c in t.column_names])
        for c in cols:  # some months miss optional columns
            if c not in t.column_names:
                t = t.append_column(c, pa.nulls(t.num_rows, pa.string()))
        parts.append(t.select(cols))
    return pa.concat_tables(parts) if parts else None


def _finish_points(t: pa.Table) -> pa.Table:
    t = io.snake_columns(t)
    # police.uk now uses 2021 LSOAs: name the column like the NSPL so it joins directly
    t = t.rename_columns(["lsoa21cd" if c == "lsoa_code" else c for c in t.column_names])
    t = io.to_numeric(t, ["longitude", "latitude"], pa.float64())
    t = t.filter(pc.is_valid(t["longitude"]))
    t = io.add_bng_from_lonlat(t, "longitude", "latitude")
    for c in t.column_names:  # low-cardinality text -> dictionary (much smaller, faster)
        if c not in ("crime_id", "longitude", "latitude", "easting", "northing") and pa.types.is_string(t[c].type):
            t = t.set_column(t.schema.get_field_index(c), c, pc.dictionary_encode(t[c]))
    return t


@source("police", order=20, keys=["lsoa21cd", "easting/northing"],
        description="data.police.uk street crime + stop & search (last N months, all forces)")
def police(settings, params):
    out_dir = settings.out("police")
    version = params.get("archive", "latest")
    if version == "latest":
        version = dl.get_json(DATES)[0]["date"]
    n_months = int(params.get("months", 12))
    y, m = map(int, version.split("-"))
    months = set()
    for i in range(n_months):
        mm = m - i
        yy = y + (mm - 1) // 12
        mm = (mm - 1) % 12 + 1
        months.add(f"{yy:04d}-{mm:02d}")

    datasets = params.get("datasets", ["street", "stop-and-search"])
    targets = {"street": out_dir / "crimes.parquet", "stop-and-search": out_dir / "stop_and_search.parquet",
               "outcomes": out_dir / "outcomes.parquet"}
    targets = {k: v for k, v in targets.items() if k in datasets}
    if all(io.is_fresh(p, settings.reprocess or settings.force) for p in targets.values()):
        return list(targets.values()) + ([out_dir / "crimes_lsoa_month.parquet"] if "street" in targets else [])

    zpath = dl.download(ARCHIVE.format(v=version), settings.raw("police") / f"police_{version}.zip",
                        force=settings.force, timeout=max(settings.timeout, 600))
    outs = []
    meta = {"archive": version, "months": sorted(months)}
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        spec = {"street": CRIME_COLS, "stop-and-search": SAS_COLS, "outcomes": OUT_COLS}
        for ds, out in targets.items():
            members = [n for n in names if n.endswith(f"-{ds}.csv")]
            with io.Timer(f"police {ds}"):
                t = _read_members(z, members, spec[ds], months)
            if t is None:
                log.warning("police: no %s files for %s", ds, sorted(months))
                continue
            t = _finish_points(t)
            io.write_table(t, out, meta=meta)
            outs.append(out)
            if ds == "street":
                slim = pa.table({"lsoa21cd": pc.cast(t["lsoa21cd"], pa.string()),
                                 "month": pc.cast(t["month"], pa.string()),
                                 "crime_type": pc.cast(t["crime_type"], pa.string())})
                slim = slim.filter(pc.is_valid(slim["lsoa21cd"]))
                agg = (slim.group_by(["lsoa21cd", "month", "crime_type"]).aggregate([([], "count_all")])
                       .rename_columns(["lsoa21cd", "month", "crime_type", "crimes"]))
                p = out_dir / "crimes_lsoa_month.parquet"
                io.write_table(agg, p, meta=meta)
                outs.append(p)
    if not settings.keep_raw:
        zpath.unlink(missing_ok=True)
    return outs
