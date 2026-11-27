"""MOT tests: DVSA anonymised MOT test results (Great Britain), aggregated for vehicle-value work.

Every MOT test of the year (about 40 million) with make, model, fuel, engine size, date of first use,
mileage, result and the postcode area of the test station. Only normal (initial) tests of cars and light
vans (class 4) are kept; each is close to one vehicle a year. The file is streamed from the zip and
aggregated to postcode area x make x model x fuel x year of first use x engine-size band:
``tests``, ``mileage_sum`` (tests with a reading), ``mileage_n``, ``fails`` (failed initial tests) and
``prs`` (passed after rectification at the station). Output ``mot/mot_cars_<year>.parquet``.

Caveats: the postcode area is the test station's (about 120 areas), not the keeper's; cars under three years
old have no MOT and are missing. Open Government Licence.
"""
from __future__ import annotations

import logging
import re
import time
import zipfile

import numpy as np
import pyarrow as pa

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

PAGE = "https://open.data.dvsa.gov.uk/mot-anonymised/index.html"
COLS = {"testclassid": "cls", "testtype": "type", "testresult": "result", "testmileage": "mileage",
        "postcoderegion": "area", "postcodearea": "area", "make": "make", "model": "model", "fueltype": "fuel",
        "cylcpcty": "cc", "cylindercapacity": "cc", "firstusedate": "first_use"}
CC_BANDS = [0, 1, 1000, 1400, 1700, 2000, 2500, 3000, 4000, 100000]     # 0 = none recorded (e.g. electric)


_bad = {"rows": 0}


def _skip(row) -> str:
    """Malformed rows (a handful have stray quotes) are skipped and counted."""
    _bad["rows"] += 1
    return "skip"


def _norm(c: str) -> str:
    return re.sub(r"[^a-z]", "", c.lower())


@source("mot", order=26, keys=["postcode area", "make / model"],
        description="DVSA anonymised MOT results: cars by test postcode area, make, model, age, mileage, outcome")
def mot(settings, params):
    import pandas as pd
    import pyarrow.csv as pacsv
    year = int(params.get("year", 2025))
    out = settings.out("mot") / f"mot_cars_{year}.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    html = dl.get_text(PAGE)
    url = next((u for u in re.findall(r'https://[^"\']+\.zip', html)
                if re.search(rf"test_result[a-z_]*_{year}\.zip$", u) or f"results+({year})" in u), None)
    if not url:
        raise RuntimeError(f"mot: no {year} results file on {PAGE}")
    f = dl.download(url, settings.raw("mot") / url.rsplit("/", 1)[-1].replace("+", "_"), force=settings.force,
                    timeout=max(settings.timeout, 7200))
    parts, t0, rows = [], time.perf_counter(), 0
    with zipfile.ZipFile(f) as z:
        members = [m for m in z.namelist() if m.lower().endswith((".csv", ".txt"))]
        for m in members:
            with z.open(m) as h:
                first = h.readline().decode("utf-8", "replace")
            sep = "|" if first.count("|") > first.count(",") else ","
            header = [c.strip() for c in first.strip().split(sep)]
            keep = {c: COLS[_norm(c)] for c in header if _norm(c) in COLS}
            with z.open(m) as h:
                reader = pacsv.open_csv(h, read_options=pacsv.ReadOptions(block_size=64 << 20),
                                        parse_options=pacsv.ParseOptions(delimiter=sep, escape_char="\\",
                                                                         invalid_row_handler=_skip),
                                        convert_options=pacsv.ConvertOptions(include_columns=list(keep),
                                                                             column_types={c: pa.string() for c in keep}))
                for batch in reader:
                    d = batch.to_pandas().rename(columns=keep)
                    rows += len(d)
                    d = d[(d["cls"].str.strip() == "4") & (d["type"].str.strip() == "NT")]
                    if d.empty:
                        continue
                    mil = pd.to_numeric(d["mileage"], errors="coerce")
                    cc = pd.to_numeric(d["cc"], errors="coerce").fillna(0)
                    fu = pd.to_datetime(d["first_use"], errors="coerce", dayfirst=False)
                    g = pd.DataFrame({
                        "postcode_area": d["area"].str.strip().str.upper(), "make": d["make"].str.strip().str.upper(),
                        "model": d["model"].str.strip().str.upper(), "fuel": d["fuel"].str.strip(),
                        "first_use_year": fu.dt.year.fillna(0).astype("int16"),
                        "cc_band": pd.cut(cc, CC_BANDS, right=False, labels=False).fillna(0).astype("int8"),
                        "tests": 1, "mileage_sum": mil.where(mil > 0).fillna(0),
                        "mileage_n": (mil > 0).astype("int32"),
                        "fails": d["result"].str.strip().eq("F").astype("int32"),
                        "prs": d["result"].str.strip().eq("PRS").astype("int32")})
                    keys = ["postcode_area", "make", "model", "fuel", "first_use_year", "cc_band"]
                    parts.append(g.groupby(keys, as_index=False, observed=True)[
                        ["tests", "mileage_sum", "mileage_n", "fails", "prs"]].sum())
                    if len(parts) >= 40:   # keep memory flat: fold the partial sums together
                        parts = [pd.concat(parts).groupby(keys, as_index=False).sum()]
                    log.info("mot: %s rows read (%.0fs)", f"{rows:,}", time.perf_counter() - t0)
    keys = ["postcode_area", "make", "model", "fuel", "first_use_year", "cc_band"]
    agg = pd.concat(parts).groupby(keys, as_index=False).sum()
    agg["cc_band_from"] = np.array(CC_BANDS)[agg["cc_band"].to_numpy()]
    log.info("mot: %d - %s car tests in %s groups, %d postcode areas", year, f"{int(agg['tests'].sum()):,}",
             f"{len(agg):,}", agg["postcode_area"].nunique())
    if _bad["rows"]:
        log.warning("mot: %s malformed rows skipped", f"{_bad['rows']:,}")
    io.write_table(pa.Table.from_pandas(agg.assign(year=year), preserve_index=False), out)
    return [out]
