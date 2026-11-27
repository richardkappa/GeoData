"""English Indices of Deprivation (IoD) at LSOA level, resolved via GOV.UK.

Default is the 2025 release (LSOA 2021 codes, matches NSPL ``lsoa21cd``).
The 2019 release (LSOA 2011 codes) can be added for time comparisons.
Note the NSPL already carries the overall IMD rank (``imd25ind``); this
source adds every domain score, rank and decile.
"""
from __future__ import annotations

import logging


from .. import download as dl
from .. import govuk, io
from . import source

log = logging.getLogger(__name__)

RELEASES = {
    2025: ("government/statistics/english-indices-of-deprivation-2025", r"^File 7", "lsoa21cd"),
    2019: ("government/statistics/english-indices-of-deprivation-2019", r"^File 7", "lsoa11cd"),
}


@source("imd_england", order=11, keys=["lsoa21cd"],
        description="English Indices of Deprivation (2025 default; 2019 optional) - all domains, LSOA")
def imd(settings, params):
    outs = []
    for year in params.get("years", [2025]):
        page, rx, key = RELEASES[int(year)]
        out = settings.out("imd_england") / f"iod{year}_lsoa.parquet"
        if io.is_fresh(out, settings.reprocess or settings.force):
            outs.append(out)
            continue
        att = govuk.find_attachment(page, rx)
        src = dl.download(att["url"], settings.raw("imd_england") / att["url"].rsplit("/", 1)[-1],
                          force=settings.force, timeout=settings.timeout)
        t = io.read_csv(src, encoding="utf8")
        t = io.snake_columns(t)
        first = t.column_names[0]  # "lsoa_code_2021" / "lsoa_code_2011"
        t = t.rename_columns([key if c == first else c for c in t.column_names])
        # Shorter names: drop the "(where 1 is most deprived ...)" explanations
        import re
        t = t.rename_columns([re.sub(r"_where_1_is_most_deprived.*$", "", c) for c in t.column_names])
        io.write_table(t, out, meta={"source": att["url"], "release": year})
        outs.append(out)
    return outs
