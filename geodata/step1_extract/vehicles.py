"""Licensed vehicles by small area: DfT vehicle licensing statistics (VEH0125, VEH0135), latest quarter.

VEH0125: licensed vehicles by 2021 LSOA (Scottish 2022 data zones, Northern Ireland super data zones too),
body type and keeper; VEH0135: plug-in vehicles by fuel type. Vehicles are counted at the registered keeper's postcode;
counts of 1 to 4 are suppressed ("[c]") and taken as 2 here. Output ``vehicles/lsoa_vehicles.parquet``:
one row per area (``lsoa21cd``) with easting / northing (mean of the area's live postcodes), counts of
private cars, all licensed vehicles, plug-in vehicles (battery electric, plug-in hybrid, range extended)
and battery-electric vehicles, company-kept cars, plug-in, battery-electric and company-car shares (% of all vehicles), and private cars
per household (England and Wales, Census 2021 households). Open Government Licence.
"""
from __future__ import annotations

import logging

import pyarrow as pa
import pyarrow.parquet as pq

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

PAGE = "https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-files"
PLUGIN = ["Battery electric", "Plug-in hybrid electric (petrol)", "Plug-in hybrid electric (diesel)",
          "Range extended electric"]


def _file(settings, name: str):
    import re
    html = dl.get_text(PAGE)
    m = re.search(r"https://assets\.publishing\.service\.gov\.uk/media/[^\"'<> ]+/df_%s\.csv" % name, html)
    if not m:
        raise RuntimeError(f"vehicles: df_{name}.csv not found on {PAGE}")
    return dl.download(m.group(0), settings.raw("vehicles") / f"df_{name}.csv", force=settings.force,
                       timeout=max(settings.timeout, 1800))


def _num(s):
    import pandas as pd
    return pd.to_numeric(s.astype(str).str.replace(",", "").replace({"[c]": "2", "[x]": None, "[z]": None}),
                         errors="coerce")


@source("vehicles", order=25, keys=["lsoa21cd", "easting/northing"],
        description="DfT licensed vehicles by LSOA / data zone: cars, plug-in and battery-electric vehicles (UK)")
def vehicles(settings, params):
    import pandas as pd
    out = settings.out("vehicles") / "lsoa_vehicles.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    f125, f135 = _file(settings, "VEH0125"), _file(settings, "VEH0135")
    head = pd.read_csv(f125, nrows=0).columns
    q = params.get("quarter") or next(c for c in head if c[:4].isdigit())       # newest quarter first
    a = pd.read_csv(f125, usecols=["LSOA21CD", "BodyType", "Keepership", "LicenceStatus", q], dtype=str)
    a = a[a["LicenceStatus"] == "Licensed"]
    a[q] = _num(a[q])
    cars = a[(a["BodyType"] == "Cars") & (a["Keepership"] == "Private")].groupby("LSOA21CD")[q].sum()
    allv = a[(a["BodyType"] == "Total") & (a["Keepership"] == "Total")].groupby("LSOA21CD")[q].sum()
    b = pd.read_csv(f135, usecols=["LSOA21CD", "Fuel", "Keepership", q], dtype=str)
    b = b[b["Keepership"] == "Total"]
    b[q] = _num(b[q])
    plug = b[b["Fuel"].isin(PLUGIN)].groupby("LSOA21CD")[q].sum()
    bev = b[b["Fuel"] == "Battery electric"].groupby("LSOA21CD")[q].sum()
    cars_all = a[(a["BodyType"] == "Cars") & (a["Keepership"] == "Total")].groupby("LSOA21CD")[q].sum()
    cars_co = a[(a["BodyType"] == "Cars") & (a["Keepership"] == "Company")].groupby("LSOA21CD")[q].sum()
    d = pd.DataFrame({"cars_private": cars, "cars": cars_all, "cars_company": cars_co, "vehicles": allv,
                      "plugin_vehicles": plug, "bev_vehicles": bev})
    d["company_car_share_pct"] = (100 * cars_co / cars_all).round(2)   # newer, pricier cars: a value signal
    # VEH0135 lists plug-in vehicles only (its "Total" is all plug-ins), so shares are of all licensed vehicles
    d["plugin_share_pct"] = (100 * plug / allv).round(2)
    d["bev_share_pct"] = (100 * bev / allv).round(2)
    d.index.name = "lsoa21cd"
    d = d[~d.index.str.startswith("Mis")].reset_index()
    # households (England and Wales census) and position: from the output areas and postcodes of each area
    ns = pq.read_table(settings.out_dir / "ons_nspl" / "nspl.parquet",
                       columns=["oa21cd", "lsoa21cd", "easting", "northing", "doterm"]).to_pandas()
    live = ns[ns["doterm"].isna() & ns["easting"].notna()]
    d = d.merge(live.groupby("lsoa21cd")[["easting", "northing"]].mean().round(1), on="lsoa21cd", how="left")
    hf = settings.out_dir / "census_2021" / "ts017_oa.parquet"
    if hf.exists():
        hh = pq.read_table(hf, columns=["oa21cd", "ts017_total_all_household_spaces"]).to_pandas()
        oa = ns[["oa21cd", "lsoa21cd"]].drop_duplicates("oa21cd")
        hh = hh.merge(oa, on="oa21cd").groupby("lsoa21cd")["ts017_total_all_household_spaces"].sum()
        d["households"] = d["lsoa21cd"].map(hh)
        d["cars_per_household"] = (d["cars_private"] / d["households"]).round(3)
    log.info("vehicles: %s - %d areas, %s private cars, %s plug-in vehicles", q, len(d),
             f"{d['cars_private'].sum():,.0f}", f"{d['plugin_vehicles'].sum():,.0f}")
    io.write_table(pa.Table.from_pandas(d.assign(quarter=q), preserve_index=False), out)
    return [out]
