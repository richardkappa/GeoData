"""Air quality: Defra Pollution Climate Mapping (PCM) modelled annual mean background concentrations.

One CSV per pollutant and year on a 1 km national grid covering the UK (uk-air.defra.gov.uk/data/pcm-data):
nitrogen dioxide (NO2), fine particles (PM2.5) and PM10, in micrograms per cubic metre. Output
``air_quality/pcm_<year>.parquet``: one row per 1 km square (easting / northing of its centre) with a
column per pollutant. Section 3 gives each postcode the value of the square it is in. Open Government
Licence.
"""
from __future__ import annotations

import io as _io
import logging

import pyarrow as pa

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

BASE = "https://uk-air.defra.gov.uk/datastore/pcm/map{code}{year}{suffix}.csv"
POLLUTANTS = {"no2": ("no2", ""), "pm25": ("pm25", "g"), "pm10": ("pm10", "g")}   # PM files are gravimetric ("g")


@source("air_quality", order=56, keys=["easting/northing (1 km grid)"],
        description="Defra modelled annual mean background NO2, PM2.5 and PM10 on a 1 km grid (UK)")
def air_quality(settings, params):
    import pandas as pd
    year = int(params.get("year", 2024))
    out = settings.out("air_quality") / f"pcm_{year}.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    merged = None
    for name in params.get("pollutants", list(POLLUTANTS)):
        code, suffix = POLLUTANTS[name]
        url = BASE.format(code=code, year=year, suffix=suffix)
        f = dl.download(url, settings.raw("air_quality") / url.rsplit("/", 1)[-1], force=settings.force,
                        timeout=max(settings.timeout, 600))
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        start = next(i for i, l in enumerate(lines) if l.lower().startswith("gridcode"))
        d = pd.read_csv(_io.StringIO("\n".join(lines[start:])), na_values=["MISSING", "missing", ""])
        d = d.rename(columns={d.columns[-1]: name, "x": "easting", "y": "northing"})[["easting", "northing", name]]
        merged = d if merged is None else merged.merge(d, on=["easting", "northing"], how="outer")
        log.info("air_quality: %s %d - %s squares", name, year, f"{d[name].notna().sum():,}")
    io.write_table(pa.Table.from_pandas(merged.assign(year=year), preserve_index=False), out)
    return [out]
