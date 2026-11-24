"""OS Terrain 50 elevation features.

The OS download is ~2,860 zipped ASCII-grid tiles (10 km x 10 km at 50 m).
:func:`build_mosaic` writes them once into a single tiled, compressed GeoTIFF
(``extract/os_open/terrain50.tif``) tile by tile, so memory stays small.

:func:`sample` then works through the country in blocks: for each block it
reads the raster window (plus a margin) and, for the postcodes inside it,
computes with vectorised numpy indexing:

* ``elevation_m``           bilinear height
* ``slope_deg``/``aspect_deg``  Horn's 3x3 method (aspect: 0 = north-facing)
* ``relative_height_<r>m``  height minus the mean height on a ring of 16
                            points at radius r (+ hilltop, - valley)
"""
from __future__ import annotations

import io as _io
import logging
import warnings
import zipfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RES = 50.0
GB_BOUNDS = (0.0, 0.0, 700000.0, 1300000.0)
NODATA = -9999.0


def build_mosaic(zip_path: Path, out: Path) -> Path:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    x0, y0, x1, y1 = GB_BOUNDS
    w, h = int((x1 - x0) / RES), int((y1 - y0) / RES)
    profile = dict(driver="GTiff", width=w, height=h, count=1, dtype="float32", nodata=NODATA,
                   crs="EPSG:27700", transform=from_origin(x0, y1, RES, RES), tiled=True,
                   blockxsize=512, blockysize=512, compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
    tmp = out.with_suffix(".tif.tmp")
    n = 0
    with zipfile.ZipFile(zip_path) as outer, rasterio.open(tmp, "w", **profile) as dst:
        for name in outer.namelist():
            if not name.lower().endswith(".zip"):
                continue
            with zipfile.ZipFile(_io.BytesIO(outer.read(name))) as inner:
                asc = next((m for m in inner.namelist() if m.lower().endswith(".asc")), None)
                if asc is None:
                    continue
                text = inner.read(asc).decode("ascii")
            head, body = _split_header(text)
            nc, nr = int(head["ncols"]), int(head["nrows"])
            arr = np.array(body.split(), dtype="float32").reshape(nr, nc)  # C parser
            col = int((float(head["xllcorner"]) - x0) / RES)
            row = int((y1 - (float(head["yllcorner"]) + nr * RES)) / RES)
            dst.write(arr, 1, window=Window(col, row, nc, nr))
            n += 1
    tmp.replace(out)
    log.info("terrain mosaic written from %d tiles -> %s", n, out)
    return out


def _split_header(text: str):
    head, lines = {}, text.split("\n", 6)
    body_start = 0
    for i, line in enumerate(lines[:6]):
        k, *v = line.split()
        if k.lower() in ("ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value",
                         "xllcenter", "yllcenter"):
            head[k.lower()] = v[0]
            body_start = i + 1
    body = "\n".join(lines[body_start:])
    return head, body


def _bilinear(z, fx, fy):
    ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    ix = np.clip(ix, 0, z.shape[1] - 2)
    iy = np.clip(iy, 0, z.shape[0] - 2)
    tx, ty = fx - ix, fy - iy
    a, b = z[iy, ix], z[iy, ix + 1]
    c, d = z[iy + 1, ix], z[iy + 1, ix + 1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def sample(tif: Path, px, py, radii=(500, 2000), block: float = 100000.0):
    """Elevation features for points (BNG metres). Returns dict of arrays."""
    import rasterio
    from rasterio.windows import from_bounds

    n = len(px)
    out = {"elevation_m": np.full(n, np.nan), "slope_deg": np.full(n, np.nan),
           "aspect_deg": np.full(n, np.nan)}
    for r in radii:
        out[f"relative_height_{int(r)}m"] = np.full(n, np.nan)
    margin = max(radii) + 2 * RES
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    with rasterio.open(tif) as src:
        bx = np.floor(np.asarray(px) / block).astype(int)
        by = np.floor(np.asarray(py) / block).astype(int)
        for kx, ky in set(zip(bx.tolist(), by.tolist())):
            sel = np.flatnonzero((bx == kx) & (by == ky))
            xmin, ymin = kx * block - margin, ky * block - margin
            xmax, ymax = (kx + 1) * block + margin, (ky + 1) * block + margin
            win = from_bounds(xmin, ymin, xmax, ymax, src.transform).round_offsets().round_lengths()
            z = src.read(1, window=win, boundless=True, fill_value=NODATA).astype("f8")
            z[z == NODATA] = np.nan
            z = z[::-1]                       # row 0 = south
            wy0 = ymax - z.shape[0] * RES     # south edge of the window
            # fractional cell coordinates of cell centres
            fx = (np.asarray(px)[sel] - xmin) / RES - 0.5
            fy = (np.asarray(py)[sel] - wy0) / RES - 0.5
            e = _bilinear(z, fx, fy)
            out["elevation_m"][sel] = e
            ix = np.clip(np.round(fx).astype(int), 1, z.shape[1] - 2)
            iy = np.clip(np.round(fy).astype(int), 1, z.shape[0] - 2)
            # Horn: rows increase northwards here
            a, b, c = z[iy + 1, ix - 1], z[iy + 1, ix], z[iy + 1, ix + 1]
            d, f = z[iy, ix - 1], z[iy, ix + 1]
            g, h, i = z[iy - 1, ix - 1], z[iy - 1, ix], z[iy - 1, ix + 1]
            dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * RES)
            dzdy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * RES)
            out["slope_deg"][sel] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
            # downslope direction: 0 = facing north, 90 = east ...
            out["aspect_deg"][sel] = (np.degrees(np.arctan2(-dzdx, -dzdy)) + 360) % 360
            for r in radii:
                ring = np.stack([_bilinear(z, fx + np.cos(t) * r / RES, fy + np.sin(t) * r / RES) for t in ang])
                with warnings.catch_warnings():   # all-NaN rings (e.g. off the coast) are expected -> NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out[f"relative_height_{int(r)}m"][sel] = e - np.nanmean(ring, axis=0)
    return out


def heights_at(tif: Path, x, y):
    """Bilinear elevation at arbitrary points (used for height above river)."""
    return sample(tif, x, y, radii=(RES,))["elevation_m"]
