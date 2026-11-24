"""Environment Agency flood risk at postcode level (England).

Files are found through the data.gov.uk catalogue (CKAN API) so new releases
are picked up automatically; a fallback URL is kept for each.

* ``rofrs_postcodes``  Risk of Flooding from Rivers and Sea - postcodes in
  areas at risk: counts of residential / non-residential / unclassified
  properties in each likelihood band (very low, low, medium, high).
* ``postcode_risk``    Flood risk postcode search tool data: counts of
  properties at high / medium / low risk and the groundwater risk class.
"""
from __future__ import annotations

import logging
import re

import pyarrow as pa
import pyarrow.compute as pc

from .. import download as dl
from .. import io
from ..postcode import POSTCODE_KEY, add_postcode_ns
from . import source

log = logging.getLogger(__name__)

CKAN = "https://ckan.publishing.service.gov.uk/api/action/package_show"
EA_FILE = "https://environment.data.gov.uk/api/file/download?fileDataSetId={id}&fileName={name}"

DATASETS = {
    "rofrs_postcodes": {
        "package": "risk-of-flooding-from-rivers-and-sea-postcodes-in-areas-at-risk2",
        "resource_regex": r"RoFRS_Postcodes.*\.zip",
        "fallback": EA_FILE.format(id="97781741-2982-4802-af2e-313fe3fd8f7e",
                                   name="RoFRS_Postcodes_in_Areas_at_Risk.zip"),
        "postcode_col": "PC", "prefix": "rofrs_",
        "drop": ["SORTOFF", "DISTRICT", "SECTOR", "UNIT"],
    },
    "postcode_risk": {
        "package": "flood-risk-postcode-search-tool-data",
        "resource_regex": r"Postcodes_Risk_Assessment.*\.csv",
        "fallback": EA_FILE.format(id="fb921496-1788-4fc2-b469-7b51e2a45553",
                                   name="Postcodes_Risk_Assessment_All.csv"),
        "postcode_col": "Postcode", "prefix": "flood_",
        "drop": [],
    },
}


def _resolve(spec) -> str:
    try:
        pkg = dl.get_json(CKAN, params={"id": spec["package"]})["result"]
        for r in pkg.get("resources", []):
            if re.search(spec["resource_regex"], r.get("url", ""), re.I):
                return r["url"]
    except Exception as e:
        log.warning("CKAN lookup for %s failed (%s); using fallback URL", spec["package"], e)
    return spec["fallback"]


def _postcode_table(t: pa.Table, pc_col: str, prefix: str, drop) -> pa.Table:
    t = t.drop_columns([c for c in drop if c in t.column_names])
    t = add_postcode_ns(t, pc_col, validate=True)
    t = t.filter(pc.is_valid(t[POSTCODE_KEY])).drop_columns([pc_col])
    t = io.snake_columns(t)
    t = io.auto_numeric(t, [c for c in t.column_names if c != POSTCODE_KEY])
    t = t.rename_columns([c if c == POSTCODE_KEY else f"{prefix}{c}" for c in t.column_names])
    # one row per postcode
    first = pc.indices_nonzero(_first(t[POSTCODE_KEY]))
    return t.take(first)


def _first(col):
    import numpy as np
    g = pa.table({"k": col, "i": pa.array(np.arange(len(col)))}).group_by("k").aggregate([("i", "min")])
    mask = np.zeros(len(col), dtype=bool)
    mask[g["i_min"].to_numpy()] = True
    return pa.array(mask)


@source("environment_agency", order=54, keys=["postcode_ns"],
        description="Environment Agency flood risk by postcode (rivers & sea, surface water, groundwater)")
def environment_agency(settings, params):
    outs = []
    for name in params.get("datasets", list(DATASETS)):
        spec = DATASETS[name]
        out = settings.out("environment_agency") / f"{name}.parquet"
        if io.is_fresh(out, settings.reprocess or settings.force):
            outs.append(out)
            continue
        url = _resolve(spec)
        fname = re.search(r"fileName=([^&]+)", url)
        fname = fname.group(1) if fname else url.rsplit("/", 1)[-1]
        f = dl.download(url, settings.raw("environment_agency") / fname, force=settings.force,
                        timeout=max(settings.timeout, 600))
        if f.suffix.lower() == ".zip":
            member = io.zip_members(f, r"\.csv$")[0]
            t = io.read_csv_from_zip(f, member, strings=True, encoding="utf-8-sig")
        else:
            t = io.read_csv(f, strings=True, encoding="utf-8-sig")
        t = _postcode_table(t, spec["postcode_col"], spec["prefix"], spec["drop"])
        io.write_table(t, out, meta={"source": url})
        outs.append(out)
    return outs
