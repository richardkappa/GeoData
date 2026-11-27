"""Road and rail noise: Defra strategic noise mapping, England, round 3 (2017).

Polygons of the day-evening-night noise level (Lden, an annual average with evening and night weighted as
louder) in 5 dB bands from 55 dB, around major roads and railways and across large urban areas. Round 3 is
used because round 4 (2022) is only offered through an interactive download tool and a map service, not a
file. Output ``noise/road_lden.parquet`` and ``noise/rail_lden.parquet``: GeoParquet (EPSG:27700) with
``noiseclass`` and ``lden_db`` (the band's lower bound). Places outside every polygon are quieter than
55 dB or were not modelled. Open Government Licence.
"""
from __future__ import annotations

import logging
import re

from .. import download as dl
from .. import io
from . import source

log = logging.getLogger(__name__)

FILES = {
    "road": ("13921cc4-abc0-4651-a3f6-1850613cea5e", "Road_Noise_Lden_England_Round_3.gpkg.zip"),
    "rail": ("aac9cd75-46e3-4faa-b878-1a2512fa0878", "Rail_Noise_Lden_England_Round_3.gpkg.zip"),
}
URL = "https://environment.data.gov.uk/api/file/download?fileDataSetId={id}&fileName={name}"


def _lower_db(cls: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", str(cls))
    return float(m.group(1)) if m else None


@source("noise", order=57, keys=["geometry"],
        description="Defra strategic noise maps (England, round 3): road and rail Lden bands")
def noise(settings, params):
    import pyogrio
    outs = []
    for kind in params.get("sources", ["road", "rail"]):
        out = settings.out("noise") / f"{kind}_lden.parquet"
        if io.is_fresh(out, settings.reprocess or settings.force):
            outs.append(out)
            continue
        fid, name = FILES[kind]
        f = dl.download(URL.format(id=fid, name=name), settings.raw("noise") / name, force=settings.force,
                        timeout=max(settings.timeout, 1800))
        path = f"/vsizip/{f}/{name.removesuffix('.zip')}"
        layer = pyogrio.list_layers(path)[0][0]
        g = pyogrio.read_dataframe(path, layer=layer, columns=["noiseclass"])
        g["lden_db"] = g["noiseclass"].map(_lower_db)
        g = g[g.geometry.notna()]
        log.info("noise: %s - %s polygons, bands %s", kind, f"{len(g):,}", sorted(g["noiseclass"].unique()))
        outs.append(io.write_gdf(g, out))
    return outs
