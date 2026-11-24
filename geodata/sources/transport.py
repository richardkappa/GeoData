"""Department for Transport: road collisions (STATS19) and NaPTAN stops."""
from __future__ import annotations

import logging

import pyarrow as pa
import pyarrow.compute as pc

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

STATS19 = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-{table}-{period}.csv"
NAPTAN = "https://naptan.api.dft.gov.uk/v1/access-nodes?dataFormat=csv"


@source("road_safety", order=21, keys=["easting/northing", "lsoa_of_collision"],
        description="DfT STATS19 road collisions (+ optional vehicles/casualties) by year")
def road_safety(settings, params):
    """``years`` may be a list of years, ``"last-5-years"`` or ``"1979-latest-published-year"``."""
    years = params.get("years", "last-5-years")
    periods = [years] if isinstance(years, str) else [str(y) for y in years]
    tables = params.get("tables", ["collision"])
    outs = []
    for table in tables:
        out = settings.out("road_safety") / f"{table}.parquet"
        if io.is_fresh(out, settings.reprocess or settings.force):
            outs.append(out)
            continue
        parts = []
        for p in periods:
            url = STATS19.format(table=table, period=p)
            try:
                f = dl.download(url, settings.raw("road_safety") / url.rsplit("/", 1)[-1],
                                force=settings.force, timeout=max(settings.timeout, 600))
            except Exception as e:
                log.error("road_safety %s %s: %s", table, p, e)
                continue
            parts.append(io.read_csv(f, strings=True))
        if not parts:
            continue
        t = pa.concat_tables(parts, promote_options="permissive")
        t = io.snake_columns(t)
        # everything is coded as integers apart from ids, dates, times and LSOA codes
        text = {"collision_index", "collision_ref_no", "date", "time", "lsoa_of_collision",
                "local_authority_ons_district", "local_authority_highway", "local_authority_highway_current",
                "lsoa_of_casualty", "lsoa_of_driver", "generic_make_model", "accident_index",
                "accident_reference", "collision_reference"}
        num_f = ["location_easting_osgr", "location_northing_osgr", "longitude", "latitude"]
        t = io.to_numeric(t, num_f, pa.float64())
        t = io.auto_numeric(t, [c for c in t.column_names if c not in text and c not in num_f])
        if "date" in t.column_names:  # dd/mm/yyyy -> date
            t = t.set_column(t.schema.get_field_index("date"), "date",
                             pc.cast(pc.strptime(t["date"], "%d/%m/%Y", "s", error_is_null=True), pa.date32()))
        if "location_easting_osgr" in t.column_names:
            t = t.rename_columns([{"location_easting_osgr": "easting", "location_northing_osgr": "northing"}.get(c, c)
                                  for c in t.column_names])
        io.write_table(t, out, meta={"periods": periods})
        outs.append(out)
    return outs


@source("naptan", order=22, keys=["atcocode", "easting/northing"],
        description="DfT NaPTAN public transport access nodes (bus, rail, tram, ferry, air)")
def naptan(settings, params):
    out = settings.out("naptan") / "naptan_stops.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    f = dl.download(NAPTAN, settings.raw("naptan") / "Stops.csv", force=settings.force,
                    timeout=max(settings.timeout, 600))
    t = io.read_csv(f, strings=True, encoding="utf8")
    t = io.snake_columns(t)
    if params.get("active_only", True) and "status" in t.column_names:
        t = t.filter(pc.equal(t["status"], "active"))
    types = params.get("stop_types")  # e.g. ["RLY","BCT","FER","TMU","AIR"]
    if types:
        t = t.filter(pc.is_in(t["stoptype"], pa.array(types)))
    t = io.to_numeric(t, ["easting", "northing", "longitude", "latitude"], pa.float64())
    keep = params.get("columns")
    if keep:
        t = t.select([c for c in keep if c in t.column_names])
    # Friendly grouping of NaPTAN stop types
    groups = {"BCT": "bus", "BCS": "bus", "BCQ": "bus", "BST": "bus", "BCE": "bus_station",
              "RLY": "rail", "RSE": "rail", "RPL": "rail", "MET": "metro_tram", "PLT": "metro_tram",
              "TMU": "metro_tram", "FER": "ferry", "FTD": "ferry", "FBT": "ferry", "AIR": "air",
              "GAT": "air", "TXR": "taxi", "STR": "taxi", "LCE": "lift_cable", "LPL": "lift_cable",
              "LSE": "lift_cable"}
    codes = pa.array(list(groups))
    labels = pa.array(list(groups.values()) + ["other"])
    idx = pc.fill_null(pc.index_in(t["stoptype"], value_set=codes), len(groups))
    t = t.append_column("mode", pc.dictionary_encode(labels.take(idx)))
    io.write_table(t, out)
    return [out]
