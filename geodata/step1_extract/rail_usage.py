"""Railway station usage: ORR "Estimates of station usage" (latest year, Great Britain).

Table 1410 gives entries and exits (all ticket types) and interchanges per station for the latest year;
table 6329 gives each station's Ordnance Survey grid reference. Output ``rail_usage/station_usage.parquet``:
one row per station with ``tlc`` (three-letter code), name, easting / northing, ``entries_exits`` and
``interchanges`` a year. Open Government Licence.
"""
from __future__ import annotations

import logging

import pyarrow as pa

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

USAGE = "https://dataportal.orr.gov.uk/media/1907/table-1410-passenger-entries-and-exits-and-interchanges-by-station.ods"
ATTRS = "https://dataportal.orr.gov.uk/media/ootlf0cn/table-6329-station-attributes-for-all-mainline-stations.ods"


def _table(f, prefix: str):
    import pandas as pd
    x = pd.ExcelFile(f, engine="calamine")
    sheet = next(s for s in x.sheet_names if s.startswith(prefix))
    raw = pd.read_excel(x, sheet, header=None)
    hdr = next(i for i in range(20) if str(raw.iat[i, 0]).strip().lower() == "station name")
    d = raw.iloc[hdr + 1:].copy()
    d.columns = [" ".join(str(c).split()) for c in raw.iloc[hdr]]
    return d


@source("rail_usage", order=23, keys=["tlc", "easting/northing"],
        description="ORR estimates of station usage: entries and exits per station a year (GB)")
def rail_usage(settings, params):
    import pandas as pd
    out = settings.out("rail_usage") / "station_usage.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    fu = dl.download(params.get("usage_url", USAGE), settings.raw("rail_usage") / "table-1410.ods",
                     force=settings.force, timeout=settings.timeout)
    fa = dl.download(params.get("attributes_url", ATTRS), settings.raw("rail_usage") / "table-6329.ods",
                     force=settings.force, timeout=settings.timeout)
    u, a = _table(fu, "1410"), _table(fa, "6329")
    col = lambda d, *words: next(c for c in d.columns if all(w.lower() in c.lower() for w in words))
    u = pd.DataFrame({"tlc": u[col(u, "three letter")], "name": u[col(u, "station name")],
                      "entries_exits": pd.to_numeric(u[col(u, "entries and exits", "all tickets")], errors="coerce"),
                      "interchanges": pd.to_numeric(u[col(u, "interchanges")], errors="coerce")})
    a = pd.DataFrame({"tlc": a[col(a, "three letter")],
                      "easting": pd.to_numeric(a[col(a, "easting")], errors="coerce"),
                      "northing": pd.to_numeric(a[col(a, "northing")], errors="coerce")})
    d = u.merge(a, on="tlc", how="left")
    log.info("rail_usage: %d stations, %d with a grid reference, %s entries and exits a year", len(d),
             int(d["easting"].notna().sum()), f"{d['entries_exits'].sum():,.0f}")
    io.write_table(pa.Table.from_pandas(d, preserve_index=False), out)
    return [out]
