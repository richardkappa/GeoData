"""Trees outside woodland: Forest Research National Trees Outside Woodland map (England).

Tree canopy over 3 m tall and 5 m² outside the National Forest Inventory woodland: lone trees, groups of trees
and small woodlands, mapped from LiDAR (2016-22) and satellite imagery (2021-24). About 30% of England's
tree cover. The national file is a 6.7 GB geodatabase of millions of polygons, so it is read in chunks and
each polygon's canopy area is added to the 100 m national grid square holding its centre. Output
``trees/tow_canopy_100m.parquet``: one row per 100 m square with canopy (easting / northing of the square's
centre, ``canopy_m2`` and the area by class). Open Government Licence.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pyarrow as pa

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

URL = ("https://environment.data.gov.uk/api/file/download?fileDataSetId=b01e513d-8aae-4098-98b1-6524a33f0767"
       "&fileName=National_Trees_Outside_Woodland_Map.gdb.zip")


@source("trees", order=58, keys=["easting/northing (100 m grid)"],
        description="Forest Research trees outside woodland (England): canopy area per 100 m square")
def trees(settings, params):
    import pandas as pd
    import pyogrio
    import shapely
    out = settings.out("trees") / "tow_canopy_100m.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    f = dl.download(params.get("url", URL), settings.raw("trees") / "National_Trees_Outside_Woodland_Map.gdb.zip",
                    force=settings.force, timeout=max(settings.timeout, 7200))
    # unpack once: reading the geodatabase from a folder is much faster, and the map explorer reads
    # its trees layer from it square by square (spatial index)
    import zipfile
    gdb_dir = settings.raw("trees")
    with zipfile.ZipFile(f) as z:
        names = z.namelist()
        top = sorted({n.split("/")[0] for n in names if n.split("/")[0].endswith(".gdb")})
        # the geodatabase's files may sit at the root of the zip: unpack them into <name>.gdb
        gdb = gdb_dir / (top[0] if top else f.name.removesuffix(".zip"))
        if not (gdb / "gdb").exists() and not any(gdb.glob("*.gdbtable")):
            log.info("trees: unpacking to %s", gdb.name)
            z.extractall(gdb_dir if top else gdb)
    path = str(gdb)
    # one layer per English region: read them all, chunk by chunk
    layers = [l[0] for l in pyogrio.list_layers(path) if l[1] and "polygon" in str(l[1]).lower()] or         [l[0] for l in pyogrio.list_layers(path)]
    cell = float(params.get("cell_m", 100))
    parts, t0, chunk = [], time.perf_counter(), int(params.get("chunk", 1_000_000))
    total = 0
    for layer in layers:
        info = pyogrio.read_info(path, layer=layer)
        n = info["features"]
        cls_col = next((c for c in info["fields"] if c.lower() in ("woodland_type", "type", "class", "category")), None)
        log.info("trees: layer %s - %s polygons (class: %s)", layer, f"{n:,}", cls_col)
        for start in range(0, n, chunk):
            g = pyogrio.read_dataframe(path, layer=layer, columns=[cls_col] if cls_col else [], skip_features=start,
                                       max_features=chunk)
            geoms = g.geometry.values
            area = shapely.area(geoms)
            c = shapely.get_coordinates(shapely.centroid(geoms))
            part = pd.DataFrame({"ix": np.floor(c[:, 0] / cell).astype(np.int64),
                                 "iy": np.floor(c[:, 1] / cell).astype(np.int64), "canopy_m2": area,
                                 "cls": g[cls_col].astype(str).to_numpy() if cls_col else "all"})
            parts.append(part.groupby(["ix", "iy", "cls"], as_index=False)["canopy_m2"].sum())
            if len(parts) >= 30:   # fold partial sums to keep memory flat
                parts = [pd.concat(parts).groupby(["ix", "iy", "cls"], as_index=False)["canopy_m2"].sum()]
        total += n
        log.info("trees: %s polygons done (%.0fs)", f"{total:,}", time.perf_counter() - t0)
    d = pd.concat(parts).groupby(["ix", "iy", "cls"], as_index=False)["canopy_m2"].sum()
    wide = d.pivot_table(index=["ix", "iy"], columns="cls", values="canopy_m2", aggfunc="sum", fill_value=0)
    wide.columns = [f"canopy_m2_{str(c).lower().replace(' ', '_')}" for c in wide.columns]
    wide["canopy_m2"] = wide.sum(axis=1)
    wide = wide.reset_index()
    wide["easting"] = (wide.pop("ix") + 0.5) * cell
    wide["northing"] = (wide.pop("iy") + 0.5) * cell
    log.info("trees: %s squares with canopy, %.0f km² of canopy", f"{len(wide):,}", wide["canopy_m2"].sum() / 1e6)
    io.write_table(pa.Table.from_pandas(wide, preserve_index=False), out)
    return [out]
