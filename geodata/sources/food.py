"""Food Standards Agency food hygiene ratings (FHRS / Scottish FHIS).

Every rated food business in the UK with postcode, coordinates, business type
and rating. Uses the single all-establishments CSV the FSA publishes; if that
is unavailable it falls back to the per-local-authority XML files.

``rating_numeric`` holds 0-5 for FHRS ratings (null for "Exempt",
"AwaitingInspection", Scottish "Pass"/"Improvement Required" etc., which stay
in ``ratingvalue``).
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.compute as pc

from .. import download as dl
from .. import io
from ..postcode import add_postcode_ns
from . import source

log = logging.getLogger(__name__)

ALL_CSV = "https://safhrsprodstorage.blob.core.windows.net/opendatafileblobstorage/FHRS_All_en-GB.csv"
OPEN_DATA = "https://ratings.food.gov.uk/open-data/"
KEEP = ["fhrsid", "businessname", "businesstype", "businesstypeid", "postcode", "ratingvalue", "ratingdate",
        "hygiene", "structural", "confidenceinmanagement", "schemetype", "newratingpending",
        "localauthoritycode", "localauthorityname", "latitude", "longitude"]


def _from_xml(settings) -> pa.Table:
    html = dl.get_text(OPEN_DATA)
    files = sorted(set(re.findall(r"/api/open-data-files/FHRS\d+en-GB\.xml", html)))
    log.info("fsa: CSV unavailable, reading %d local authority XML files", len(files))
    rows = {k: [] for k in KEEP}
    for rel in files:
        f = dl.download("https://ratings.food.gov.uk" + rel, settings.raw("fsa") / rel.rsplit("/", 1)[-1],
                        force=settings.force, timeout=settings.timeout)
        for _, el in ET.iterparse(f):
            if el.tag != "EstablishmentDetail":
                continue
            flat = {c.tag.lower(): (c.text or "") for c in el.iter() if c is not el and len(c) == 0}
            for k in KEEP:
                rows[k].append(flat.get(k) or None)
            el.clear()
    return pa.table({k: pa.array(v, pa.string()) for k, v in rows.items()})


@source("fsa", order=62, keys=["postcode_ns", "fhrsid", "easting/northing"],
        description="Food Standards Agency food hygiene ratings: every rated food business (UK)")
def fsa(settings, params):
    out = settings.out("fsa") / "food_hygiene.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    try:
        f = dl.download(ALL_CSV, settings.raw("fsa") / "FHRS_All_en-GB.csv", force=settings.force,
                        timeout=max(settings.timeout, 600))
        t = io.snake_columns(io.read_csv(f, strings=True, encoding="utf-8-sig"))
        t = t.select([c for c in KEEP if c in t.column_names])
    except Exception as e:
        log.warning("fsa: all-establishments CSV failed (%s)", e)
        t = _from_xml(settings)
    t = io.to_numeric(t, ["latitude", "longitude"], pa.float64())
    t = io.auto_numeric(t, ["fhrsid", "businesstypeid", "hygiene", "structural", "confidenceinmanagement"])
    t = t.append_column("rating_numeric", io.to_numeric(
        pa.table({"r": t["ratingvalue"]}), ["r"], pa.int8())["r"])
    if "ratingdate" in t.column_names:
        d = pc.utf8_slice_codeunits(t["ratingdate"], 0, 10)
        t = t.set_column(t.schema.get_field_index("ratingdate"), "ratingdate",
                         pc.cast(pc.strptime(d, "%Y-%m-%d", "s", error_is_null=True), pa.date32()))
    t = io.add_bng_from_lonlat(t, "longitude", "latitude")
    for c in ("businesstype", "schemetype", "localauthorityname", "ratingvalue"):
        if c in t.column_names:
            t = t.set_column(t.schema.get_field_index(c), c, pc.dictionary_encode(t[c]))
    t = add_postcode_ns(t, "postcode", validate=True)
    io.write_table(t, out)
    return [out]
