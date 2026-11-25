"""Give every point table from section 1 British National Grid coordinates (easting / northing).

Run on each table a source writes (``01_extract.py``; ``--geocode-only`` re-runs it on the tables
already extracted). For each row, in order of preference:

1. the source's own easting / northing, when present and inside Great Britain + Northern Ireland;
2. the source's latitude / longitude, converted to the national grid;
3. the centre of the row's postcode (``postcode_ns``) from the ONS postcode directory (NSPL, already
   on the national grid for the whole UK, Northern Ireland included).

Tables that only carry a postcode get easting / northing columns added. Tables with one row per
postcode (energy, broadband, flood risk...) are postcode attributes rather than points and are left
alone, as are tables with geometry and the sources listed under ``geocode.skip``. When any row is
filled, an ``xy_source`` column records where each row's coordinates came from (source, latlong,
postcode; empty = still none). Re-running is safe: rows filled earlier are filled again from the
current postcode directory, rows with the source's own coordinates are untouched, and a table that
needs nothing is not rewritten.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import io
from .postcode import POSTCODE_KEY

log = logging.getLogger(__name__)

# national grid box covering GB and Northern Ireland (NI converted to the GB grid); 0 / negative = missing
BOX = (0, 0, 700_000, 1_250_000)
LATLON = (("latitude", "longitude"), ("lat", "long"), ("lat", "lon"))


class PostcodeXY:
    """Postcode -> (easting, northing) from the NSPL, loaded once."""

    def __init__(self, nspl_path: Path):
        self.path = Path(nspl_path)
        self._t = None

    def lookup(self, postcodes: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
        if self._t is None:
            if not self.path.exists():
                raise FileNotFoundError(f"{self.path} missing - extract ons_nspl first")
            self._t = pq.read_table(self.path, columns=[POSTCODE_KEY, "easting", "northing"])
        idx = pc.index_in(pc.cast(postcodes, pa.string()), value_set=self._t[POSTCODE_KEY])
        ok = pc.is_valid(idx).to_numpy(zero_copy_only=False)
        i = idx.to_numpy(zero_copy_only=False)
        e = np.full(len(i), np.nan)
        n = np.full(len(i), np.nan)
        e[ok] = self._t["easting"].to_numpy()[i[ok].astype(np.int64)]
        n[ok] = self._t["northing"].to_numpy()[i[ok].astype(np.int64)]
        return e, n


def _f64(t: pa.Table, c: str) -> np.ndarray:
    return pc.cast(t[c], pa.float64(), safe=False).to_numpy(zero_copy_only=False).astype("f8")


def _valid(e, n) -> np.ndarray:
    return (np.isfinite(e) & np.isfinite(n) & (e > BOX[0]) & (e < BOX[2]) & (n > BOX[1]) & (n < BOX[3])
            & ~((e == 0) & (n == 0)))


def geocode_file(path: Path, postcodes: PostcodeXY | None) -> dict | None:
    """Fill / add easting and northing in one parquet table. Returns a summary, or None if the table is
    not a point table or already complete."""
    path = Path(path)
    if path.suffix != ".parquet" or not path.exists():
        return None
    names = pq.read_schema(path).names
    if "geometry" in names:
        return None
    has_en = "easting" in names and "northing" in names
    ll = next(((a, b) for a, b in LATLON if a in names and b in names), None)
    pcol = POSTCODE_KEY if POSTCODE_KEY in names else None
    if not (has_en or ll or pcol):
        return None
    if not has_en and not ll:
        # postcode only: a point table (several rows can share a postcode), or a postcode attribute table?
        p = pq.read_table(path, columns=[pcol])[pcol]
        if pc.count_distinct(p).as_py() == len(p):
            return None
    cols = [c for c in ("easting", "northing", "xy_source", *(ll or ()), pcol) if c and c in names]
    t = pq.read_table(path, columns=cols)
    rows = t.num_rows
    if has_en:
        e, n = _f64(t, "easting"), _f64(t, "northing")
    else:
        e, n = np.full(rows, np.nan), np.full(rows, np.nan)
    if "xy_source" in names:      # rows we filled last time are filled again (the NSPL may have changed)
        prev = pc.fill_null(pc.cast(t["xy_source"], pa.string()), "").to_numpy(zero_copy_only=False)
        redo = (prev == "latlong") | (prev == "postcode") | (prev == "")
        e[redo], n[redo] = np.nan, np.nan
    src = np.full(rows, "", dtype=object)
    own = _valid(e, n)
    src[own] = "source"
    miss = ~own
    n_ll = n_pc = 0
    if miss.any() and ll:
        from pyproj import Transformer
        la, lo = _f64(t, ll[0]), _f64(t, ll[1])
        ok = miss & np.isfinite(la) & np.isfinite(lo) & (np.abs(la) > 1e-9) & (la > 49) & (la < 61.5) & (lo > -9) & (lo < 3)
        if ok.any():
            x, y = Transformer.from_crs(4326, 27700, always_xy=True).transform(lo[ok], la[ok])
            e[ok], n[ok] = x, y
            good = ok.copy()
            good[ok] = _valid(np.asarray(x), np.asarray(y))
            src[good] = "latlong"
            n_ll = int(good.sum())
            miss = miss & ~good
    if miss.any() and pcol and postcodes is not None:
        pe, pn = postcodes.lookup(t[pcol].take(pa.array(np.flatnonzero(miss))))
        ok = _valid(pe, pn)
        idx = np.flatnonzero(miss)[ok]
        e[idx], n[idx] = pe[ok], pn[ok]
        src[idx] = "postcode"
        n_pc = int(ok.sum())
        miss[idx] = False
    e[miss], n[miss] = np.nan, np.nan
    summary = {"file": path.name, "rows": rows, "source": int(own.sum()), "latlong": n_ll, "postcode": n_pc,
               "missing": int(miss.sum())}
    if n_ll == 0 and n_pc == 0 and "xy_source" not in names and has_en:
        return None if summary["missing"] == 0 else summary     # nothing could be added: leave the file alone
    full = pq.read_table(path)
    md = full.schema.metadata
    ea = pa.array(np.round(e, 1), mask=~np.isfinite(e))
    na = pa.array(np.round(n, 1), mask=~np.isfinite(n))
    sa = pa.array([s or None for s in src], pa.string()).dictionary_encode()
    for c, a in (("easting", ea), ("northing", na), ("xy_source", sa)):
        if c in full.column_names:
            full = full.set_column(full.column_names.index(c), c, a)
        else:
            full = full.append_column(c, a)
    full = full.replace_schema_metadata(md)
    io.write_table(full, path)
    return summary


def geocode_files(paths, nspl_path: Path) -> list[dict]:
    postcodes = PostcodeXY(nspl_path)
    out = []
    for p in paths:
        try:
            s = geocode_file(p, postcodes)
        except FileNotFoundError as e:
            log.warning("geocode %s: %s - postcode filling skipped", Path(p).name, e)
            s = geocode_file(p, None)
        if s:
            log.info("geocode %-28s %s rows: %s own, %s from lat/long, %s from postcode, %s still without",
                     s["file"], f"{s['rows']:,}", f"{s['source']:,}", f"{s['latlong']:,}", f"{s['postcode']:,}",
                     f"{s['missing']:,}")
            out.append(s)
    return out
