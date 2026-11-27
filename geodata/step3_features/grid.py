"""Raster grid tools for "within radius" features.

Points (or line samples, or polygon coverage) are binned onto a regular grid
in British National Grid metres. For each grid row we keep a running (prefix)
sum, so the total inside a disc of radius R cells around any cell is

    sum over dy in [-R, R] of  P[y+dy, x+w(dy)+1] - P[y+dy, x-w(dy)]

with ``w(dy) = floor(sqrt(R^2 - dy^2))``. That is 2R+1 vectorised numpy gathers
for *all* postcodes at once, whatever the number of points in the disc - a 5 km
crime count in central London costs the same as one on a Scottish island.
Accuracy: postcodes and points are snapped to cell centres, so counts are
exact to within about half a cell (50 m at the default 100 m grid).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Grid:
    x0: float          # west edge (m)
    y0: float          # south edge (m)
    res: float         # cell size (m)
    nx: int
    ny: int

    @classmethod
    def around(cls, x: np.ndarray, y: np.ndarray, res: float, pad: float) -> "Grid":
        x0 = np.floor((np.nanmin(x) - pad) / res) * res
        y0 = np.floor((np.nanmin(y) - pad) / res) * res
        x1 = np.ceil((np.nanmax(x) + pad) / res) * res
        y1 = np.ceil((np.nanmax(y) + pad) / res) * res
        g = cls(float(x0), float(y0), float(res), int((x1 - x0) / res), int((y1 - y0) / res))
        log.info("grid    %d x %d cells at %g m (%.0f MB per float64 layer)", g.nx, g.ny, res,
                 g.nx * g.ny * 8 / 1e6)
        return g

    # -- cell indices ---------------------------------------------------------
    def cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.x0) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.y0) / self.res).astype(np.int64)
        return ix, iy

    def inside(self, ix, iy):
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def transform(self):
        """rasterio Affine for row 0 = north edge."""
        from rasterio.transform import from_origin
        return from_origin(self.x0, self.y0 + self.ny * self.res, self.res, self.res)

    # -- layers ---------------------------------------------------------------
    def bin_points(self, x, y, weights=None, dtype="float64") -> np.ndarray:
        """Sum of weights (or count) per cell; array shape (ny, nx), row 0 = south."""
        ix, iy = self.cell(x, y)
        ok = self.inside(ix, iy) & np.isfinite(np.asarray(x, dtype="f8")) & np.isfinite(np.asarray(y, dtype="f8"))
        flat = iy[ok] * self.nx + ix[ok]
        w = None if weights is None else np.asarray(weights, dtype="f8")[ok]
        if w is not None:
            good = np.isfinite(w)
            flat, w = flat[good], w[good]
        out = np.bincount(flat, weights=w, minlength=self.nx * self.ny)
        return out.reshape(self.ny, self.nx).astype(dtype, copy=False)

    def coverage(self, geoms) -> np.ndarray:
        """1 where a cell centre falls inside any polygon (GDAL rasterize, C)."""
        from rasterio.features import rasterize
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros((self.ny, self.nx), dtype="uint8")
        r = rasterize(((g, 1) for g in geoms), out_shape=(self.ny, self.nx), transform=self.transform(),
                      fill=0, dtype="uint8", all_touched=False)
        return r[::-1]  # rasterio row 0 is north; ours is south

    def line_length(self, geoms, step: float = 10.0) -> np.ndarray:
        """Metres of line per cell (lines sampled every ``step`` m)."""
        import shapely
        pts = shapely.get_coordinates(shapely.segmentize(np.asarray(geoms), step))
        # each vertex stands for ~step metres of line (approximation, fine at 100 m cells)
        return self.bin_points(pts[:, 0], pts[:, 1], np.full(len(pts), step))


def disc_offsets(radius_m: float, res: float):
    R = int(round(radius_m / res))
    dy = np.arange(-R, R + 1)
    w = np.floor(np.sqrt(np.maximum(R * R - dy * dy, 0))).astype(np.int64)
    return dy, w


def disc_sums(layer: np.ndarray, grid: Grid, px, py, radii) -> dict[float, np.ndarray]:
    """Sum of ``layer`` within each radius (m) of every point (px, py).

    The row prefix sums are built once and reused for all radii.
    """
    ix, iy = grid.cell(px, py)
    ok = grid.inside(ix, iy)
    integer = np.issubdtype(layer.dtype, np.integer) or layer.dtype == bool
    pref = np.zeros((layer.shape[0], layer.shape[1] + 1), dtype="int64" if integer else "float64")
    np.cumsum(layer, axis=1, out=pref[:, 1:])
    ixc, iyc = np.where(ok, ix, 0), np.where(ok, iy, 0)
    res = {}
    for r in radii:
        out = np.zeros(len(ix), dtype=pref.dtype)
        dy, w = disc_offsets(r, grid.res)
        for d, half in zip(dy, w):
            yy = iyc + d
            valid = (yy >= 0) & (yy < grid.ny)
            rows = np.clip(yy, 0, grid.ny - 1)
            lo = np.clip(ixc - half, 0, grid.nx)
            hi = np.clip(ixc + half + 1, 0, grid.nx)
            out += np.where(valid, pref[rows, hi] - pref[rows, lo], 0)
        out = out.astype("float64")
        out[~ok] = np.nan
        res[r] = out
    return res


def disc_sum(layer, grid, px, py, radius_m):
    return disc_sums(layer, grid, px, py, [radius_m])[radius_m]


def disc_cells(radius_m: float, res: float) -> int:
    dy, w = disc_offsets(radius_m, res)
    return int((2 * w + 1).sum())
