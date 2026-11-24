"""Environmental designations, heritage and climate.

* ``natural_england``  Natural England designations (SSSI, AONB/National
  Landscapes, ancient woodland, nature reserves ...) from its ArcGIS org.
* ``historic_england`` National Heritage List for England (listed buildings,
  scheduled monuments, parks & gardens, battlefields, WHS) + conservation areas.
* ``metoffice_cdp``    Met Office Climate Data Portal 1991-2020 averages
  (open, no login) - frost/summer/icing/rain days and 12 km climate grids.
* ``metoffice_haduk``  HadUK-Grid 1 km annual/seasonal/monthly grids from CEDA
  for chosen years. CEDA now requires a (free) account: set the environment
  variables ``CEDA_USERNAME`` and ``CEDA_PASSWORD`` and a short-lived access
  token is created through CEDA's token API on each run, or set ``CEDA_TOKEN``
  to a token from https://services.ceda.ac.uk.
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


CEDA_TOKEN_API = "https://services.ceda.ac.uk/api/token/create/"


def _jwt_seconds_left(token: str) -> float | None:
    """Seconds until a JWT access token expires (None if it can't be read)."""
    import base64, json, time
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))["exp"] - time.time()
    except Exception:
        return None


def _ceda_token() -> str | None:
    """A usable CEDA access token: ``CEDA_TOKEN`` if it has more than an hour left,
    otherwise a new one from CEDA's token API using ``CEDA_USERNAME`` / ``CEDA_PASSWORD``."""
    import requests
    token = os.environ.get("CEDA_TOKEN")
    left = _jwt_seconds_left(token) if token else None
    if token and (left is None or left > 3600):
        return token
    user, pw = os.environ.get("CEDA_USERNAME"), os.environ.get("CEDA_PASSWORD")
    if not (user and pw):
        if token:
            log.warning("CEDA_TOKEN has expired or expires within the hour; set CEDA_USERNAME and "
                        "CEDA_PASSWORD to have a new one created automatically")
        return token
    r = requests.post(CEDA_TOKEN_API, auth=(user, pw), timeout=60)
    r.raise_for_status()
    data = r.json()
    new = data.get("access_token") or data.get("token") or (data.get("data") or {}).get("access_token")
    if not new:
        raise ValueError(f"CEDA token API returned no token (keys: {sorted(data)})")
    log.info("created a new CEDA access token (%.0f h validity)", (_jwt_seconds_left(new) or 0) / 3600)
    os.environ["CEDA_TOKEN"] = new   # reuse within this run
    return new


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
    res = params.get("resolution", "1km")
    freq = params.get("frequency", "ann")
    out = settings.out("metoffice_haduk") / f"haduk_{res}_{freq}.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    token = _ceda_token()
    if not token:
        log.warning("metoffice_haduk skipped: set CEDA_TOKEN, or CEDA_USERNAME and CEDA_PASSWORD (free CEDA "
                    "account), to download HadUK-Grid. metoffice_cdp provides open 1991-2020 averages without a login.")
        return []
    import xarray as xr

    base = "https://dap.ceda.ac.uk/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid"
    version = params.get("version", "latest")
    if version == "latest":
        version = _latest_dir(base + "/", "v")
    variables = params.get("variables", ["rainfall", "tas", "sun", "groundfrost", "snowLying"])
    years = params.get("years", [])
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
