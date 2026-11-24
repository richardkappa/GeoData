"""Health and care services.

* ``nhs_ods``  NHS Organisation Data Service reports via the ODS Data Search
  & Export API (no key): GP practices, branch surgeries, pharmacies, dental
  practices, NHS trust sites (hospitals and clinics) ... England & Wales.
  One Parquet per report, active organisations only, keyed on ``postcode_ns``.
* ``cqc``      Care Quality Commission active registered locations (care homes
  with beds, home care, GP surgeries, dentists, hospitals, clinics) with
  overall rating, UPRN and coordinates. England.
"""
from __future__ import annotations

import logging
import re

import pyarrow as pa
import pyarrow.compute as pc

from .. import download as dl
from .. import io
from ..postcode import add_postcode_ns
from . import source

log = logging.getLogger(__name__)

ODS = "https://www.odsdatasearchandexport.nhs.uk/api/getReport?report={r}"
# Standard ODS 27-column layout (the reports have no header row)
ODS_COLS = ["org_code", "name", "national_grouping", "health_geography", "address_1", "address_2",
            "address_3", "address_4", "address_5", "postcode", "open_date", "close_date", "status_code",
            "org_sub_type", "parent_code", "join_parent_date", "left_parent_date", "phone", "c18", "c19",
            "c20", "amended_record", "c22", "current_care_org", "c24", "roles", "c26"]
ODS_KEEP = ["org_code", "name", "postcode", "open_date", "close_date", "status_code", "org_sub_type",
            "parent_code", "national_grouping", "health_geography", "roles"]
REPORT_NAMES = {"epraccur": "gp_practices", "ebranchs": "gp_branch_surgeries", "edispensary": "pharmacies",
                "egdpprac": "dental_practices", "ets": "nhs_trust_sites", "ephp": "independent_healthcare_sites"}


@source("nhs_ods", order=60, keys=["postcode_ns", "org_code"],
        description="NHS ODS: GP practices, pharmacies, dentists, NHS trust sites (active, by postcode)")
def nhs_ods(settings, params):
    outs = []
    for report in params.get("reports", ["epraccur", "edispensary", "egdpprac", "ets"]):
        name = REPORT_NAMES.get(report, report)
        out = settings.out("nhs_ods") / f"{name}.parquet"
        if io.is_fresh(out, settings.reprocess or settings.force):
            outs.append(out)
            continue
        f = dl.download(ODS.format(r=report), settings.raw("nhs_ods") / f"{report}.csv",
                        force=settings.force, timeout=max(settings.timeout, 300))
        t = io.read_csv(f, names=ODS_COLS, strings=True, encoding="latin1")
        t = t.select([c for c in ODS_KEEP if c in t.column_names])
        if params.get("active_only", True):
            t = t.filter(pc.is_null(t["close_date"]))
        # epraccur: ODS role RO76 = GP practice (drops out-of-hours RO80, walk-in RO87,
        # other prescribing cost centres RO72 ...). `roles` can hold several codes split by "|".
        if report == "epraccur" and params.get("gp_roles"):
            rx = "(^|\\|)(" + "|".join(map(str, params["gp_roles"])) + ")(\\||$)"
            t = t.filter(pc.fill_null(pc.match_substring_regex(t["roles"], rx), False))
        for c in ("open_date", "close_date"):
            t = t.set_column(t.schema.get_field_index(c), c,
                             pc.cast(pc.strptime(t[c], "%Y%m%d", "s", error_is_null=True), pa.date32()))
        t = add_postcode_ns(t, "postcode")
        io.write_table(t, out, meta={"report": report})
        outs.append(out)
    return outs


CQC_PAGE = "https://www.cqc.org.uk/about-us/transparency/using-cqc-data"
CQC_COLS = {
    "Location ID": "location_id", "Location Name": "name", "Care home?": "care_home",
    "Care homes beds": "care_home_beds", "Location Type/Sector": "type_sector",
    "Location Primary Inspection Category": "primary_inspection_category",
    "Location Latest Overall Rating": "overall_rating", "Publication Date": "rating_published",
    "Dormant (Y/N)": "dormant", "Location Local Authority": "local_authority",
    "Location Postal Code": "postcode", "Location UPRN ID": "uprn",
    "Location Latitude": "latitude", "Location Longitude": "longitude",
    "Location ODS Code": "ods_code", "Provider ID": "provider_id", "Provider Name": "provider_name",
}


@source("cqc", order=61, keys=["postcode_ns", "uprn", "easting/northing"],
        description="CQC active registered care locations (care homes, beds, ratings, GP/dentist/hospital)")
def cqc(settings, params):
    out = settings.out("cqc") / "cqc_locations.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    html = dl.get_text(CQC_PAGE)
    links = re.findall(r'href="([^"]*HSCA_Active_Locations\.ods)"', html, re.I)
    if not links:
        raise LookupError("HSCA_Active_Locations.ods link not found on the CQC 'using CQC data' page")
    url = links[0] if links[0].startswith("http") else "https://www.cqc.org.uk" + links[0]
    f = dl.download(url, settings.raw("cqc") / url.rsplit("/", 1)[-1], force=settings.force,
                    timeout=max(settings.timeout, 600))
    import pandas as pd
    try:  # calamine (Rust) reads the ~25 MB ODS in seconds; odfpy would take minutes
        df = pd.read_excel(f, sheet_name="HSCA_Active_Locations", engine="calamine",
                           usecols=lambda c: c in CQC_COLS, dtype=str)
    except ImportError:
        log.warning("python-calamine not installed: falling back to the slow odf reader")
        df = pd.read_excel(f, sheet_name="HSCA_Active_Locations", engine="odf",
                           usecols=lambda c: c in CQC_COLS, dtype=str)
    df = df.rename(columns=CQC_COLS)
    t = pa.Table.from_pandas(df, preserve_index=False)
    t = t.cast(pa.schema([pa.field(c, pa.string()) for c in t.column_names]))
    t = io.to_numeric(t, ["latitude", "longitude"], pa.float64())
    t = io.to_numeric(t, ["care_home_beds"], pa.int32())
    t = io.to_numeric(t, ["uprn"], pa.int64())
    t = io.add_bng_from_lonlat(t, "longitude", "latitude")
    if params.get("exclude_dormant", True) and "dormant" in t.column_names:
        t = t.filter(pc.invert(pc.fill_null(pc.equal(t["dormant"], "Y"), False)))
    t = add_postcode_ns(t, "postcode")
    io.write_table(t, out, meta={"source": url})
    return [out]
