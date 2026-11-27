"""Fire and rescue incidents as points: England by LSOA, Scotland by data zone.

There is no open, national, point-level fire incident dataset, so incidents are counted per small area
and placed at the area's centre:

* **England** - MHCLG / Home Office "Fire statistics incident level datasets: low level geography
  dataset": every incident attended in England, one row each, with its 2021 LSOA and one of 67
  incident types (fires, false alarms, non-fire incidents). The newest file (``files``) holds the last
  three financial years. LSOAs are placed at the mean position of their live postcodes (NSPL).
* **Scotland** - Scottish Fire and Rescue Service "Total Fires" by 2011 data zone (fires only,
  accidental / not accidental), placed at the Scottish Government's 2011 data zone population-weighted
  centroids.

Output ``fire/fire_points.parquet``: one row per area (lsoa21cd or 2011 data zone code), easting /
northing, country, and yearly averages over the years kept: ``fires`` (all fires; both countries),
``deliberate_fires`` (England: fires recorded as deliberate - secondary and outdoor fires only, as the
other primary fires are not split by cause; Scotland: all fires not recorded as accidental), and for
England only ``dwelling_fires``, ``vehicle_fires``, ``fire_false_alarms``, ``rtc_attended`` (road traffic
collisions attended) and ``all_incidents``. Open Government Licence.
"""
from __future__ import annotations

import logging
import re

import pyarrow as pa
import pyarrow.parquet as pq

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

ENG_PAGE = "https://www.gov.uk/government/statistical-data-sets/fire-statistics-incident-level-datasets"
SCOT_FIRES = "https://data.gov.scot/dataset/total_fires/resource/73789dd2-7508-4eb5-9410-1d9102db5217/download"
SCOT_DZ_CENTROIDS = "https://maps.gov.scot/ATOM/shapefiles/SG_DataZoneCent_2011.zip"


def _england(settings, params) -> "pd.DataFrame":
    import pandas as pd
    want = params.get("england_files", ["to_present"])
    html = dl.get_text(ENG_PAGE)
    links = sorted(set(re.findall(r"https://assets\.publishing\.service\.gov\.uk/media/[^\"'<> ]+?/"
                                  r"Low_level_geography_dataset_[^\"'<> ]+?\.ods", html)))
    links = [u for u in links if any(w in u for w in want)]
    if not links:
        raise RuntimeError(f"fire: no low level geography file matching {want} on {ENG_PAGE}")
    frames = []
    for url in links:
        f = dl.download(url, settings.raw("fire") / url.rsplit("/", 1)[-1], force=settings.force,
                        timeout=max(settings.timeout, 600))
        sheets = [s for s in pd.ExcelFile(f, engine="calamine").sheet_names if re.fullmatch(r"\d{6}", s)]
        for s in sheets:
            frames.append(pd.read_excel(f, sheet_name=s, engine="calamine",
                                        usecols=["LSOA_CODE", "INCIDENT_TYPE", "FINANCIAL_YEAR"]))
        log.info("fire: %s - sheets %s", f.name, ", ".join(sheets))
    d = pd.concat(frames, ignore_index=True)
    years = sorted(d["FINANCIAL_YEAR"].astype(str).unique())[-int(params.get("years", 3)):]
    d = d[d["FINANCIAL_YEAR"].astype(str).isin(years) & d["LSOA_CODE"].astype(str).str.startswith("E01")]
    t = d["INCIDENT_TYPE"].astype(str)
    fire = t.str.startswith(("Primary fire", "Secondary Fire", "Chimney fire"))
    cats = {"fires": fire, "deliberate_fires": fire & t.str.contains("deliberate", case=False),
            "dwelling_fires": t.eq("Primary fire - dwelling"), "vehicle_fires": t.eq("Primary fire - road vehicle"),
            "fire_false_alarms": t.str.startswith("Fire false alarm"),
            "rtc_attended": t.str.contains("Road Traffic Collision", case=False),
            "all_incidents": pd.Series(True, index=d.index)}
    out = pd.DataFrame({k: v.groupby(d["LSOA_CODE"]).sum() / len(years) for k, v in cats.items()})
    out.index.name = "area_code"
    log.info("fire: England %s incidents in %d LSOAs, years %s", f"{len(d):,}", len(out), ", ".join(years))
    return out.reset_index().assign(country="E", years=", ".join(years))


def _scotland(settings, params) -> "pd.DataFrame":
    import geopandas as gpd
    import pandas as pd
    f = dl.download(SCOT_FIRES, settings.raw("fire") / "scotland_total_fires.csv", force=settings.force,
                    timeout=max(settings.timeout, 600))
    c = dl.download(SCOT_DZ_CENTROIDS, settings.raw("fire") / "SG_DataZoneCent_2011.zip", force=settings.force,
                    timeout=settings.timeout)
    d = pd.read_csv(f, dtype={"GeographyCode": str})
    d = d[(d["GeographyType"] == "2011 Data Zones") & (d["Measurement"] == "Count")]
    years = sorted(d["DateCode"].astype(str).unique())[-int(params.get("years", 3)):]
    d = d[d["DateCode"].astype(str).isin(years)]
    piv = d.pivot_table(index="GeographyCode", columns="Accident Status", values="Value", aggfunc="sum") / len(years)
    out = pd.DataFrame({"fires": piv.get("All"), "deliberate_fires": piv.get("Not Accidental")})
    out.index.name = "area_code"
    cent = gpd.read_file(f"zip://{c}")[["DataZone", "Easting", "Northing"]].rename(
        columns={"DataZone": "area_code", "Easting": "easting", "Northing": "northing"})
    out = out.reset_index().merge(cent, on="area_code", how="left")
    log.info("fire: Scotland %d data zones, years %s", len(out), ", ".join(years))
    return out.assign(country="S", years=", ".join(years))


@source("fire", order=63, keys=["lsoa21cd / 2011 data zone", "easting/northing"],
        description="Fire and rescue incidents per small area (England LSOAs, Scotland data zones) as points")
def fire(settings, params):
    import pandas as pd
    out = settings.out("fire") / "fire_points.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    parts = []
    if params.get("england", True):
        eng = _england(settings, params)
        # LSOA position: the mean of its live postcodes
        ns = pq.read_table(settings.out_dir / "ons_nspl" / "nspl.parquet",
                           columns=["lsoa21cd", "easting", "northing", "doterm"]).to_pandas()
        ns = ns[ns["doterm"].isna() & ns["easting"].notna()]
        xy = ns.groupby("lsoa21cd")[["easting", "northing"]].mean().round(1)
        eng = eng.merge(xy, left_on="area_code", right_index=True, how="left")
        parts.append(eng)
    if params.get("scotland", True):
        parts.append(_scotland(settings, params))
    d = pd.concat(parts, ignore_index=True)
    missing = int(d["easting"].isna().sum())
    if missing:
        log.warning("fire: %d areas without a position (dropped)", missing)
    d = d[d["easting"].notna()]
    io.write_table(pa.Table.from_pandas(d, preserve_index=False), out)
    return [out]
