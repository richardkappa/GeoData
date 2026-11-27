"""2021 Census (England & Wales) topic summaries from the Nomis bulk service.

Each topic summary table (TS001 ... TS0xx) is a zip with one CSV per
geography level. We keep the requested levels, prefix columns with the table
id and write one Parquet per table per level, keyed on the 2021 area code
(``oa21cd``, ``lsoa21cd``, ``msoa21cd``, ``ltla22cd`` ...). These join to
postcodes through the NSPL columns of the same name.

A ``census_2021_catalogue.parquet`` lists every table id, title and output
column so you can choose what to keep in section 2.
"""
from __future__ import annotations

import logging
import re

import pyarrow as pa

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

BULK = "https://www.nomisweb.co.uk/census/2021/bulk"
ZIP = "https://www.nomisweb.co.uk/output/census/2021/census2021-{t}.zip"
LEVEL_KEY = {"oa": "oa21cd", "lsoa": "lsoa21cd", "msoa": "msoa21cd", "ltla": "ltla22cd",
             "utla": "utla22cd", "rgn": "rgn22cd", "ctry": "ctry22cd"}


def catalogue() -> dict[str, str]:
    html = dl.get_text(BULK)
    rows = re.findall(r"<strong>(TS\d{3}[A-Z]?)</strong></td>\s*<td[^>]*>([^<]+)</td>", html)
    return {t.upper(): title.strip() for t, title in rows}


def _clean_col(col: str, table: str) -> str:
    c = re.sub(r";\s*measures:\s*Value$", "", col)
    c = re.sub(r"^[^:]+:\s*", "", c, count=1)  # drop "Residence type: "
    return f"{table.lower()}_{io.snake(c)}"[:120]


@source("census_2021", order=10, keys=["oa21cd", "lsoa21cd", "msoa21cd"],
        description="Census 2021 topic summaries (England & Wales) from Nomis, at OA/LSOA/MSOA")
def census(settings, params):
    cat = catalogue()
    tables = params.get("tables", "all")
    tables = sorted(cat) if tables in ("all", None) else [t.upper() for t in tables]
    levels = params.get("levels", ["oa"])
    out_dir = settings.out("census_2021")
    outs, cat_rows = [], []
    for t in tables:
        tl = t.lower()
        wanted = {lvl: out_dir / f"{tl}_{lvl}.parquet" for lvl in levels}
        zpath = settings.raw("census_2021") / f"census2021-{tl}.zip"
        try:
            if all(io.is_fresh(p, settings.reprocess or settings.force) for p in wanted.values()):
                outs += list(wanted.values())
                continue
            dl.download(ZIP.format(t=tl), zpath, force=settings.force, timeout=settings.timeout)
        except Exception as e:
            log.error("census %s download failed: %s", t, e)
            continue
        for lvl, out in wanted.items():
            members = io.zip_members(zpath, rf"-{lvl}\.csv$")
            if not members:
                log.warning("census %s has no %s level", t, lvl)
                continue
            tab = io.read_csv_from_zip(zpath, members[0])
            names = []
            for c in tab.column_names:
                if c == "geography code":
                    names.append(LEVEL_KEY.get(lvl, f"{lvl}_code"))
                elif c in ("date", "geography"):
                    names.append(f"{lvl}_name" if c == "geography" else "census_year")
                else:
                    names.append(_clean_col(c, t))
                    cat_rows.append((t, cat.get(t, ""), lvl, names[-1], c))
            # de-duplicate truncated names
            seen = {}
            for i, n in enumerate(names):
                if n in seen:
                    seen[n] += 1
                    names[i] = f"{n}_{seen[n]}"
                else:
                    seen[n] = 0
            tab = tab.rename_columns(names).drop_columns(["census_year"])
            io.write_table(tab, out, meta={"table": t, "title": cat.get(t), "level": lvl})
            outs.append(out)
        if not settings.keep_raw:
            zpath.unlink(missing_ok=True)
    if cat_rows:
        catp = out_dir / "census_2021_catalogue.parquet"
        io.write_table(pa.table(list(zip(*cat_rows)), names=["table", "title", "level", "column", "source_column"]), catp)
        outs.append(catp)
    return outs
