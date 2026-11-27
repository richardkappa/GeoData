"""Datasets published directly at postcode level.

* ``land_registry``   HM Land Registry Price Paid (England & Wales sales)
* ``schools``         DfE Get Information about Schools (all establishments)
* ``energy_postcode`` DESNZ postcode-level domestic electricity & gas use
"""
from __future__ import annotations

import datetime as dt
import logging

import pyarrow as pa
import pyarrow.compute as pc

from .. import download as dl
from .. import govuk, io
from ..periods import parse_date
from ..postcode import add_postcode_ns
from . import source

log = logging.getLogger(__name__)

# Plain http: the S3 website endpoint has no TLS certificate for this name
PP = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/{f}"
PP_NAMES = ["transaction_id", "price", "date_of_transfer", "postcode", "property_type", "old_new",
            "duration", "paon", "saon", "street", "locality", "town_city", "district", "county",
            "ppd_category_type", "record_status"]


@source("land_registry", order=30, keys=["postcode_ns"],
        description="HM Land Registry Price Paid data (E&W house sales) for a start/end period")
def land_registry(settings, params):
    """Sales with ``start <= date_of_transfer <= end``.

    ``start``/``end`` take a year, ``YYYY-MM`` or ``YYYY-MM-DD``; ``end: latest``.
    Yearly files are downloaded for the years in range; periods longer than
    ``complete_file_years`` (default 8) use the single complete file instead.
    """
    today = dt.date.today()
    start = parse_date(params.get("start", today.year - 2)) or dt.date(1995, 1, 1)
    end = parse_date(params.get("end", "latest"), end=True) or today
    if start > end:
        raise ValueError(f"land_registry start {start} is after end {end}")
    start = max(start, dt.date(1995, 1, 1))          # Price Paid starts in 1995
    years = list(range(start.year, min(end.year, today.year) + 1))
    files = ["pp-complete.csv"] if len(years) > int(params.get("complete_file_years", 8)) \
        else [f"pp-{y}.csv" for y in years]
    cols = params.get("columns", ["transaction_id", "price", "date_of_transfer", "postcode",
                                  "property_type", "old_new", "duration", "ppd_category_type"])
    if "date_of_transfer" not in cols:
        cols = [*cols, "date_of_transfer"]
    out = settings.out("land_registry") / "price_paid.parquet"
    meta = {"start": str(start), "end": str(end), "files": files}
    if not (settings.reprocess or settings.force) and _meta_matches(out, meta, ("start", "end")):
        return [out]
    parts = []
    for f in files:
        dest = settings.raw("land_registry") / f
        # the current year's file and the complete file are updated monthly: refresh if old
        stale = f in (f"pp-{today.year}.csv", "pp-complete.csv") and dest.exists() and \
            (dt.datetime.now().timestamp() - dest.stat().st_mtime) > 20 * 86400
        try:
            dl.download(PP.format(f=f), dest, force=settings.force or stale, timeout=max(settings.timeout, 900))
        except Exception as e:
            log.error("land_registry %s: %s", f, e)
            continue
        parts.append(io.read_csv(dest, names=PP_NAMES, columns=cols, strings=True))
    if not parts:
        return []
    t = pa.concat_tables(parts)
    t = io.to_numeric(t, ["price"], pa.int64())
    d = pc.utf8_slice_codeunits(t["date_of_transfer"], 0, 10)
    t = t.set_column(t.schema.get_field_index("date_of_transfer"), "date_of_transfer",
                     pc.cast(pc.strptime(d, "%Y-%m-%d", "s"), pa.date32()))
    keep = pc.and_(pc.greater_equal(t["date_of_transfer"], pa.scalar(start, pa.date32())),
                   pc.less_equal(t["date_of_transfer"], pa.scalar(end, pa.date32())))
    t = t.filter(keep)
    for c in ("property_type", "old_new", "duration", "ppd_category_type", "record_status", "county", "district"):
        if c in t.column_names:
            t = t.set_column(t.schema.get_field_index(c), c, pc.dictionary_encode(t[c]))
    t = add_postcode_ns(t, "postcode")
    log.info("land_registry %s .. %s: %s sales", start, end, f"{t.num_rows:,}")
    io.write_table(t, out, meta=meta)
    return [out]


def _meta_matches(path, meta: dict, keys) -> bool:
    """True when ``path`` exists and was built with the same ``keys`` in its metadata."""
    import json
    import pyarrow.parquet as pq
    if not path.exists():
        return False
    try:
        old = json.loads((pq.read_schema(path).metadata or {}).get(b"geodata", b"{}"))
    except ValueError:
        return False
    return all(old.get(k) == meta.get(k) for k in keys)


GIAS = "https://ea-edubase-api-prod.azurewebsites.net/edubase/downloads/public/edubasealldata{d}.csv"


@source("schools", order=31, keys=["postcode_ns", "urn", "easting/northing"],
        description="DfE Get Information about Schools: every establishment with postcode & BNG coords")
def schools(settings, params):
    out = settings.out("schools") / "schools.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    today = dt.date.today()
    src = None
    for back in range(0, 10):  # file is published daily but not always for today yet
        d = (today - dt.timedelta(days=back)).strftime("%Y%m%d")
        dest = settings.raw("schools") / f"edubasealldata{d}.csv"
        if dest.exists() or dl.head_ok(GIAS.format(d=d)):
            src = dl.download(GIAS.format(d=d), dest, force=settings.force, timeout=settings.timeout)
            break
    if src is None:
        raise RuntimeError("Could not find a GIAS extract in the last 10 days")
    t = io.read_csv(src, encoding="cp1252", strings=True)
    t = io.snake_columns(t)
    if params.get("open_only", True):
        t = t.filter(pc.equal(t["establishmentstatus_name"], "Open"))
    keep = params.get("columns")
    if keep:
        t = t.select([c for c in dict.fromkeys(["urn", "postcode", "easting", "northing", *keep]) if c in t.column_names])
    t = io.to_numeric(t, ["easting", "northing", "percentagefsm"], pa.float64())
    t = io.auto_numeric(t, ["urn", "numberofpupils", "schoolcapacity", "statutorylowage", "statutoryhighage"])
    t = add_postcode_ns(t, "postcode")
    io.write_table(t, out, meta={"file": src.name})
    return [out]


@source("energy_postcode", order=32, keys=["postcode_ns"],
        description="DESNZ postcode-level domestic electricity and gas consumption (latest year)")
def energy(settings, params):
    year = params.get("year", "latest")
    fuels = params.get("fuels", {"electricity": r"all (domestic )?meters electricity",
                                 "gas": r"domestic gas"})
    paths = {"electricity": "government/statistics/postcode-level-electricity-statistics-{year}",
             "gas": "government/statistics/postcode-level-gas-statistics-{year}"}
    outs = []
    this = dt.date.today().year
    for fuel, rx in fuels.items():
        out = settings.out("energy_postcode") / f"{fuel}.parquet"
        if io.is_fresh(out, settings.reprocess or settings.force):
            outs.append(out)
            continue
        yrs = range(this - 4, this + 1) if year == "latest" else range(int(year), int(year) + 1)
        y, att = govuk.latest_in_series(paths[fuel], yrs, rx)
        f = dl.download(att["url"], settings.raw("energy_postcode") / att["url"].rsplit("/", 1)[-1],
                        force=settings.force, timeout=settings.timeout)
        t = io.read_csv(f, strings=True)
        t = io.snake_columns(t)
        # drop the outcode/"All postcodes" summary rows
        t = t.filter(pc.invert(pc.match_substring_regex(t["postcode"], "(?i)all postcodes")))
        num = [c for c in t.column_names if c not in ("outcode", "postcode")]
        t = io.to_numeric(t, num, pa.float64())
        t = t.rename_columns([c if c in ("outcode", "postcode") else f"{fuel}_{c}" for c in t.column_names])
        t = add_postcode_ns(t, "postcode")
        io.write_table(t, out, meta={"year": y, "source": att["url"]})
        outs.append(out)
    return outs
