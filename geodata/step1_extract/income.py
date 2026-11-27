"""Household income: ONS "Income estimates for small areas, England and Wales" (MSOA, 2021 boundaries).

Modelled mean annual household income per MSOA, latest edition (financial year ending 2023): total income,
net (disposable) income, and net income equivalised before and after housing costs. Output
``income/msoa_income.parquet`` keyed on ``msoa21cd``. Open Government Licence.
"""
from __future__ import annotations

import logging

import pyarrow as pa

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

URL = ("https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/"
       "smallareaincomeestimatesformiddlelayersuperoutputareasenglandandwales/financialyearending2023/datasetfinal.xlsx")
SHEETS = {"Total annual income": "total_annual", "Net annual income": "net_annual",
          "Net income before housing costs": "net_bhc", "Net income after housing costs": "net_ahc"}


@source("income", order=12, keys=["msoa21cd"],
        description="ONS modelled mean household income by MSOA (England and Wales)")
def income(settings, params):
    import pandas as pd
    out = settings.out("income") / "msoa_income.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    url = params.get("url", URL)
    f = dl.download(url, settings.raw("income") / url.rsplit("/", 1)[-1], force=settings.force,
                    timeout=max(settings.timeout, 600))
    res = None
    for sheet, col in SHEETS.items():
        raw = pd.read_excel(f, sheet_name=sheet, header=None, engine="calamine")
        hdr = next(i for i in range(15) if str(raw.iat[i, 0]).strip().lower() == "msoa code")
        d = raw.iloc[hdr + 1:, [0, 6]]
        d.columns = ["msoa21cd", col]
        d = d[d["msoa21cd"].astype(str).str.match(r"^[EW]02")]
        d[col] = pd.to_numeric(d[col], errors="coerce")
        res = d if res is None else res.merge(d, on="msoa21cd", how="outer")
    log.info("income: %d MSOAs", len(res))
    io.write_table(pa.Table.from_pandas(res, preserve_index=False), out)
    return [out]
