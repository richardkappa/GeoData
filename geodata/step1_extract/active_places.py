"""Sports facilities: Sport England Active Places (England).

Every swimming pool, sports hall, gym, pitch, court and so on, from Sport England's Active Places ArcGIS
feature services (one per facility type), with easting / northing, status (operational or closed), access
(public, private, sports club...) and size (pool lanes, gym stations...). Output
``active_places/<facility>.parquet``, one file per facility type.

Licence: CC BY 4.0. Attribution required wherever the data is used: "Contains Data © Sport England".
"""
from __future__ import annotations

import logging

import pyarrow as pa

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

BASE = "https://services-eu1.arcgis.com/s9MgJChYyPlPX2Nk/arcgis/rest/services/GIS_Active_Places_Power_{}/FeatureServer"
DEFAULT = {"swimming_pools": "Swimming_Pools", "sports_halls": "Sports_Hall", "gyms": "Health_and_Fitness",
           "grass_pitches": "Grass_Pitches", "artificial_pitches": "Artificial_Grass_Pitches",
           "tennis_courts": "Outdoor_Tennis_Courts", "athletics": "Athletics", "golf": "Golf"}
KEEP = ["sitename", "facilitytype", "facilitysubtype", "facstatus", "accessibilitygroupstr", "accessibilitytypestr",
        "managementtypestr", "lanes", "stations", "courts", "pitches", "yearbuilt", "easting", "northing"]


@source("active_places", order=64, keys=["easting/northing"],
        description="Sport England Active Places: sports facilities by type (England; CC BY 4.0)")
def active_places(settings, params):
    import pandas as pd
    outs = []
    for name, service in (params.get("facilities") or DEFAULT).items():
        out = settings.out("active_places") / f"{name}.parquet"
        if io.is_fresh(out, settings.reprocess or settings.force):
            outs.append(out)
            continue
        base = BASE.format(service)
        layer = (dl.get_json(base, params={"f": "json"}).get("layers") or [{}])[0].get("id")
        if layer is None:
            log.warning("active_places: %s - no layer found", name)
            continue
        fields = [f["name"] for f in dl.get_json(f"{base}/{layer}", params={"f": "json"})["fields"]]
        cols = [c for c in KEEP if c in fields]
        rows, offset = [], 0
        while True:
            page = dl.get_json(f"{base}/{layer}/query", params={
                "where": "1=1", "outFields": ",".join(cols), "returnGeometry": "false", "f": "json",
                "resultOffset": offset, "resultRecordCount": 2000, "orderByFields": "OBJECTID"})
            feats = page.get("features", [])
            rows += [f["attributes"] for f in feats]
            if not feats or not page.get("exceededTransferLimit"):
                break
            offset += len(feats)
        d = pd.DataFrame(rows, columns=cols)
        log.info("active_places: %s - %s facilities (%s operational)", name, f"{len(d):,}",
                 f"{(d['facstatus'] == 'Operational').sum():,}" if "facstatus" in d else "?")
        io.write_table(pa.Table.from_pandas(d, preserve_index=False), out, meta={"licence": "CC BY 4.0",
                                                                                 "attribution": "Contains Data © Sport England"})
        outs.append(out)
    return outs
