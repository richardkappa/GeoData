"""data.police.uk street-level crime, outcomes and stop & search.

Set the period with ``start``/``end`` (``YYYY-MM``; ``end: latest``). Each
monthly archive (``/data/archive/<YYYY-MM>.zip``, ~1.7 GB) holds 36 months for
every force, so a period inside the last three years needs one download and
longer histories need one extra archive per further 36 months. Outputs:

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
from ..periods import add_months, month_range, month_str, parse_date
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


def _plan_archives(months: list[str], available: list[str], latest: str) -> dict[str, list[str]]:
    """Choose archives covering ``months`` (each archive holds the 36 months up
    to its own month). Months in the newest archive's window come from it (most
    revised data); older months use as few extra archives as possible."""
    avail = sorted(available)
    todo = sorted(m for m in months if m <= latest)
    plan: dict[str, list[str]] = {}
    while todo:
        last = todo[-1]
        if not plan and add_months(latest, -35) <= last:
            arc = latest
        else:
            later = [a for a in avail if a >= last]
            if not later:
                break
            arc = later[0]
        lo = add_months(arc, -35)
        cover = [m for m in todo if lo <= m <= arc]
        if not cover:
            break
        plan[arc] = cover
        todo = [m for m in todo if m not in cover]
    if todo:
        log.warning("police: no archive covers %s .. %s (data starts 2010-12)", todo[0], todo[-1])
    return plan


def _period(params, latest: str) -> list[str]:
    start, end = params.get("start"), params.get("end", "latest")
    end_m = latest if parse_date(end, end=True) is None else min(month_str(parse_date(end, end=True)), latest)
    if start is None:   # backwards compatible: last N months up to `end`
        start_m = add_months(end_m, -(int(params.get("months", 12)) - 1))
    else:
        start_m = month_str(parse_date(start))
    if start_m > end_m:
        raise ValueError(f"police start {start_m} is after end {end_m}")
    return month_range(start_m, end_m)


def _fresh(path, months) -> bool:
    import json
    import pyarrow.parquet as pq
    if not path.exists():
        return False
    md = pq.read_schema(path).metadata or {}
    try:
        return json.loads(md.get(b"geodata", b"{}")).get("months") == months
    except ValueError:
        return False


@source("police", order=20, keys=["lsoa21cd", "easting/northing"],
        description="data.police.uk street crime + stop & search for a start/end period (all forces)")
def police(settings, params):
    out_dir = settings.out("police")
    latest = dl.get_json(DATES)[0]["date"]
    months = _period(params, latest)
    datasets = params.get("datasets", ["street", "stop-and-search"])
    targets = {"street": out_dir / "crimes.parquet", "stop-and-search": out_dir / "stop_and_search.parquet",
               "outcomes": out_dir / "outcomes.parquet"}
    targets = {k: v for k, v in targets.items() if k in datasets}
    agg_path = out_dir / "crimes_lsoa_month.parquet"
    if not (settings.reprocess or settings.force) and all(_fresh(p, months) for p in targets.values()):
        log.info("police outputs already cover %s .. %s", months[0], months[-1])
        return list(targets.values()) + ([agg_path] if "street" in targets else [])

    import re
    html = dl.get_text("https://data.police.uk/data/archive/")
    available = sorted(set(re.findall(r"archive/(\d{4}-\d{2})\.zip", html)))
    plan = _plan_archives(months, available, latest)
    log.info("police  %s .. %s (%d months) from archive(s) %s", months[0], months[-1], len(months),
             ", ".join(plan))
    spec = {"street": CRIME_COLS, "stop-and-search": SAS_COLS, "outcomes": OUT_COLS}
    parts: dict[str, list[pa.Table]] = {ds: [] for ds in targets}
    for arc, arc_months in plan.items():
        zpath = dl.download(ARCHIVE.format(v=arc), settings.raw("police") / f"police_{arc}.zip",
                            force=settings.force, timeout=max(settings.timeout, 600))
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            for ds in targets:
                members = [n for n in names if n.endswith(f"-{ds}.csv")]
                with io.Timer(f"police {ds} {arc}"):
                    t = _read_members(z, members, spec[ds], set(arc_months))
                if t is not None:
                    parts[ds].append(t)
        if not settings.keep_raw:
            zpath.unlink(missing_ok=True)

    outs = []
    meta = {"months": months, "start": months[0], "end": months[-1], "archives": list(plan)}
    for ds, out in targets.items():
        if not parts[ds]:
            log.warning("police: no %s data for %s .. %s", ds, months[0], months[-1])
            continue
        t = _finish_points(pa.concat_tables(parts[ds]))
        io.write_table(t, out, meta=meta)
        outs.append(out)
        if ds == "street":
            slim = pa.table({"lsoa21cd": pc.cast(t["lsoa21cd"], pa.string()),
                             "month": pc.cast(t["month"], pa.string()),
                             "crime_type": pc.cast(t["crime_type"], pa.string())})
            slim = slim.filter(pc.is_valid(slim["lsoa21cd"]))
            agg = (slim.group_by(["lsoa21cd", "month", "crime_type"]).aggregate([([], "count_all")])
                   .rename_columns(["lsoa21cd", "month", "crime_type", "crimes"]))
            io.write_table(agg, agg_path, meta=meta)
            outs.append(agg_path)
    return outs
