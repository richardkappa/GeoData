"""Recorded crime in Scotland by council area (Scottish Government "Recorded Crimes and Offences").

Police Scotland does not publish street-level crime (police.uk covers England, Wales and Northern
Ireland only), so for Scotland the finest open figures are per council area. Output
``scotland_crime/council_crime.parquet``: one row per council (``lad_code``, the same S12 codes as the
NSPL's lad26cd) with crimes recorded per 10,000 people in the latest year (``year``) for a few groups.
Open Government Licence.
"""
from __future__ import annotations

import logging

import pyarrow as pa

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

URL = "https://data.gov.scot/dataset/recorded_crimes_and_offences/resource/3716a560-80b0-4860-b11f-6a53d24deb21/download"

# output column -> crime categories summed (rates per 10,000 add up because they share a population)
GROUPS = {
    "all_crimes": ["All Crimes"],
    "violence": ["All Group 1: Non-sexual crimes of violence"],
    "dishonesty": ["All Group 3: Crimes of dishonesty"],
    "housebreaking": ["Crimes: Group 3: Housebreaking"],
    "vehicle_crime": ["Crimes: Group 3: Theft of a motor vehicle", "Crimes: Group 3: Theft from a motor vehicle"],
    "shoplifting": ["Crimes: Group 3: Shoplifting"],
    "vandalism": ["Crimes: Group 4: Vandalism"],
    "fire_raising": ["Crimes: Group 4: Fire-raising"],
    "drugs": ["Crimes: Group 5: Drugs - Possession", "Crimes: Group 5: Drugs - Supply"],
}


@source("scotland_crime", order=24, keys=["lad26cd (Scottish councils)"],
        description="Recorded crime per 10,000 people by Scottish council area (Scottish Government)")
def scotland_crime(settings, params):
    import pandas as pd
    out = settings.out("scotland_crime") / "council_crime.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    f = dl.download(URL, settings.raw("scotland_crime") / "recorded_crime.csv", force=settings.force,
                    timeout=max(settings.timeout, 600))
    d = pd.read_csv(f, dtype={"GeographyCode": str})
    d = d[(d["GeographyType"] == "Council Areas") & (d["Measurement"] == "Ratio")]
    year = params.get("year") or sorted(d["DateCode"].astype(str).unique())[-1]
    d = d[d["DateCode"].astype(str) == str(year)]
    res = pd.DataFrame(index=sorted(d["GeographyCode"].unique()))
    for col, cats in GROUPS.items():
        sub = d[d["Crime Or Offence"].isin(cats)]
        if sub["Crime Or Offence"].nunique() < len(cats):
            log.warning("scotland_crime: %s - categories missing in %s: %s", col, year,
                        sorted(set(cats) - set(sub["Crime Or Offence"])))
        res[col] = sub.groupby("GeographyCode")["Value"].sum()
    res.index.name = "lad_code"
    res = res.reset_index().assign(year=str(year))
    log.info("scotland_crime: %d councils, %s", len(res), year)
    io.write_table(pa.Table.from_pandas(res, preserve_index=False), out)
    return [out]
