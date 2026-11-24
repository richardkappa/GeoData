"""Environmental designations, heritage and climate.

* ``natural_england``  Natural England designations (SSSI, AONB/National
  Landscapes, ancient woodland, nature reserves ...) from its ArcGIS org.
* ``historic_england`` National Heritage List for England (listed buildings,
  scheduled monuments, parks & gardens, battlefields, WHS) + conservation areas.
* ``metoffice_cdp``    Met Office Climate Data Portal 1991-2020 averages
  (open, no login) - frost/summer/icing/rain days and 12 km climate grids.
* ``metoffice_haduk``  HadUK-Grid 1 km annual/seasonal/monthly grids from CEDA
  for chosen years. CEDA now requires a (free) account: set the environment
  variable ``CEDA_TOKEN`` to an access token from https://services.ceda.ac.uk.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pyarrow as pa

from .. import download as dl
from .. import io
from . import source
from .ons_boundaries import fetch_arcgis_layers

log = logging.getLogger(__name__)

NE_ROOT = "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services"
HE_ROOT = "https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/arcgis/rest/services"
MO_ROOT = "https://services.arcgis.com/Lq3V5RFuTBC9I7kv/arcgis/rest/services"


@source("natural_england", order=50, keys=["geometry"],
        description="Natural England designations (SSSI, AONB, ancient woodland, nature reserves...)")
def natural_england(settings, params):
    return fetch_arcgis_layers(settings, "natural_england", params.get("layers", {}), NE_ROOT,
                               simplify_m=params.get("simplify_m"))


@source("historic_england", order=51, keys=["geometry"],
        description="Historic England NHLE (listed buildings, monuments...) and conservation areas")
def historic_england(settings, params):
    return fetch_arcgis_layers(settings, "historic_england", params.get("layers", {}), HE_ROOT,
                               simplify_m=params.get("simplify_m"))


@source("metoffice_cdp", order=52, keys=["geometry"],
        description="Met Office Climate Data Portal 1991-2020 climate averages (open, no login)")
def metoffice_cdp(settings, params):
    return fetch_arcgis_layers(settings, "metoffice_cdp", params.get("layers", {}), MO_ROOT)


CEDA = ("https://dap.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/"
        "{version}/{res}/{var}/{freq}/{dv}/{var}_hadukgrid_uk_{res}_{freq}_{period}.nc")


def _latest_dir(url: str, prefix: str) -> str:
    import re
    html = dl.get_text(url)
    found = sorted(set(re.findall(rf'href="({prefix}[^"/?]*)/"', html)))
    if not found:
        raise LookupError(f"nothing like {prefix}* at {url}")
    return found[-1]


@source("metoffice_haduk", order=53, keys=["easting/northing (1km grid)"],
        description="Met Office HadUK-Grid 1km climate grids from CEDA (needs CEDA_TOKEN)")
def metoffice_haduk(settings, params):
    token = os.environ.get("CEDA_TOKEN")
    if not token:
        log.warning("metoffice_haduk skipped: set CEDA_TOKEN (free CEDA account) to download HadUK-Grid. "
                    "metoffice_cdp provides open 1991-2020 averages without a login.")
        return []
    import xarray as xr

    base = "https://dap.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid"
    version = params.get("version", "latest")
    if version == "latest":
        version = _latest_dir(base + "/", "v")
    res = params.get("resolution", "1km")
    freq = params.get("frequency", "ann")
    variables = params.get("variables", ["rainfall", "tas", "sun", "groundfrost", "snowLying"])
    years = params.get("years", [])
    out = settings.out("metoffice_haduk") / f"haduk_{res}_{freq}.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    headers = {"Authorization": f"Bearer {token}"}
    cols: dict[str, np.ndarray] = {}
    grid_x = grid_y = None
    for var in variables:
        vdir = f"{base}/{version}/{res}/{var}/{freq}/"
        dv = _latest_dir(vdir, "v")
        for y in years:
            period = f"{y}01-{y}12" if freq in ("ann", "mon", "seas") else str(y)
            url = CEDA.format(version=version, res=res, var=var, freq=freq, dv=dv, period=period)
            f = dl.download(url, settings.raw("metoffice_haduk") / url.rsplit("/", 1)[-1],
                            force=settings.force, timeout=settings.timeout, headers=headers)
            with xr.open_dataset(f) as ds:
                da = ds[var]
                if grid_x is None:
                    gx, gy = np.meshgrid(ds["projection_x_coordinate"].values,
                                         ds["projection_y_coordinate"].values)
                    grid_x, grid_y = gx.ravel(), gy.ravel()
                vals = da.values  # (time, y, x)
                for ti in range(vals.shape[0]):
                    suffix = f"{y}" if vals.shape[0] == 1 else f"{y}_{ti + 1:02d}"
                    cols[f"{var}_{suffix}"] = vals[ti].ravel().astype("f4")
    if not cols:
        return []
    stack = np.vstack(list(cols.values()))
    land = ~np.all(np.isnan(stack), axis=0)       # drop sea cells
    t = pa.table({"easting": grid_x[land], "northing": grid_y[land],
                  **{k: v[land] for k, v in cols.items()}})
    io.write_table(t, out, meta={"version": version, "resolution": res, "frequency": freq})
    return [out]
