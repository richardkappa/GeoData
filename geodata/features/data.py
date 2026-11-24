"""Loading inputs for section 3 as plain numpy arrays.

Point datasets are described in ``config/features.yaml`` under ``points:``::

    crimes_burglary: {file: police/crimes.parquet, filter: {crime_type: Burglary}}
    gp_practices:    {file: nhs_ods/gp_practices.parquet}        # no coords -> via postcode
    sales:           {file: land_registry/price_paid.parquet, weight: price}

Coordinates come from ``easting``/``northing`` when the file has them, otherwise
from the record's ``postcode_ns`` looked up in the postcode spine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..postcode import POSTCODE_KEY

log = logging.getLogger(__name__)


@dataclass
class Spine:
    """The postcodes features are computed for (aligned arrays)."""
    postcode: pa.Array
    x: np.ndarray
    y: np.ndarray
    table: pa.Table                     # lookup columns used by features (lsoa21cd, uprn_count ...)
    _xy_lookup: dict = field(default_factory=dict, repr=False)

    @property
    def n(self):
        return len(self.x)

    def xy_for_postcodes(self, pcs: pa.ChunkedArray | pa.Array, full: "Spine | None" = None):
        """Coordinates for arbitrary postcodes via the full spine (incl. terminated)."""
        ref = full or self
        idx = pc.fill_null(pc.index_in(pcs, value_set=ref.postcode), -1)
        i = np.asarray(idx.to_numpy(zero_copy_only=False), dtype=np.int64)
        valid = i >= 0
        i = np.where(valid, i, 0)
        x = np.where(valid, ref.x[i], np.nan)
        y = np.where(valid, ref.y[i], np.nan)
        return x, y


def load_spine(data_dir: Path, cfg: dict) -> tuple[Spine, Spine]:
    """Returns (target postcodes, full reference spine for coordinate lookups)."""
    sp = cfg.get("spine", {})
    lookup = data_dir / "collate" / "postcode_lookup.parquet"
    src = lookup if lookup.exists() else data_dir / "extract" / "ons_nspl" / "nspl.parquet"
    want = [POSTCODE_KEY, "easting", "northing", "live", "ctry26cd", "oa21cd", "lsoa21cd", "msoa21cd",
            "uprn_count", "ts001_total", "ts017_total_all_household_spaces", "ts003_total",
            "imd_total_population_mid_2022"]
    have = pq.read_schema(src).names
    t = pq.read_table(src, columns=[c for c in want if c in have])
    t = t.filter(pc.is_valid(t["easting"]))
    full = Spine(t[POSTCODE_KEY].combine_chunks(), t["easting"].to_numpy(), t["northing"].to_numpy(), t)
    if not sp.get("include_terminated", False) and "live" in t.column_names:
        t = t.filter(t["live"])
    if sp.get("countries"):
        t = t.filter(pc.is_in(pc.utf8_slice_codeunits(t["ctry26cd"], 0, 1), pa.array(sp["countries"])))
    bbox = sp.get("bbox")
    if bbox:
        x, y = t["easting"], t["northing"]
        m = pc.and_(pc.and_(pc.greater_equal(x, bbox[0]), pc.less_equal(x, bbox[2])),
                    pc.and_(pc.greater_equal(y, bbox[1]), pc.less_equal(y, bbox[3])))
        t = t.filter(m)
    t = t.sort_by(POSTCODE_KEY)
    log.info("spine   %s postcodes (from %s)", f"{t.num_rows:,}", src.name)
    return Spine(t[POSTCODE_KEY].combine_chunks(), t["easting"].to_numpy(), t["northing"].to_numpy(), t), full


@dataclass
class Points:
    x: np.ndarray
    y: np.ndarray
    table: pa.Table   # any extra columns that were asked for


def load_points(data_dir: Path, spec: dict, full: Spine, columns=(), bbox=None) -> Points:
    path = data_dir / "extract" / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"{spec['file']} missing - run 01_extract.py")
    names = pq.read_schema(path).names
    has_xy = "easting" in names and "northing" in names
    if not has_xy and "geometry" in names and POSTCODE_KEY not in names:
        return _points_from_geometry(path, spec, columns, bbox)
    filt = spec.get("filter") or {}
    cols = set(columns) | set(filt) | ({"easting", "northing"} if has_xy else {POSTCODE_KEY})
    for k in ("weight",):
        if spec.get(k):
            cols.add(spec[k])
    t = pq.read_table(path, columns=[c for c in cols if c in names])
    for i, f in enumerate(t.schema):
        if pa.types.is_dictionary(f.type):
            t = t.set_column(i, f.name, pc.cast(t[f.name], f.type.value_type))
    for c, v in filt.items():
        vals = v if isinstance(v, list) else [v]
        col = pc.cast(t[c], pa.string())
        if spec.get("filter_mode") == "contains":
            m = None
            for val in vals:
                hit = pc.fill_null(pc.match_substring(col, str(val), ignore_case=True), False)
                m = hit if m is None else pc.or_(m, hit)
            t = t.filter(m)
        else:
            t = t.filter(pc.is_in(col, pa.array([str(x) for x in vals])))
    if has_xy:
        x = pc.cast(t["easting"], pa.float64()).to_numpy(zero_copy_only=False)
        y = pc.cast(t["northing"], pa.float64()).to_numpy(zero_copy_only=False)
    else:
        x, y = full.xy_for_postcodes(t[POSTCODE_KEY].combine_chunks())
    ok = np.isfinite(x) & np.isfinite(y)
    if bbox is not None:
        ok &= (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])
    t = t.filter(pa.array(ok))
    return Points(x[ok], y[ok], t)


def _points_from_geometry(path, spec, columns, bbox):
    """Point GeoParquet (e.g. motorway junctions) -> x/y via vectorised GEOS."""
    import shapely
    filt = spec.get("filter") or {}
    names = pq.read_schema(path).names
    t = pq.read_table(path, columns=[c for c in {*columns, *filt, "geometry"} if c in names])
    for c, v in filt.items():
        vals = v if isinstance(v, list) else [v]
        t = t.filter(pc.is_in(pc.cast(t[c], pa.string()), pa.array([str(x) for x in vals])))
    g = shapely.from_wkb(t["geometry"].to_numpy(zero_copy_only=False))
    xy = shapely.get_coordinates(shapely.centroid(g))
    x, y = xy[:, 0], xy[:, 1]
    ok = np.isfinite(x)
    if bbox is not None:
        ok &= (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])
    return Points(x[ok], y[ok], t.drop_columns(["geometry"]).filter(pa.array(ok)))


def load_geoms(data_dir: Path, spec: dict, bbox=None):
    """Geometries of a GeoParquet layer (optionally filtered / clipped to bbox)."""
    import geopandas as gpd
    path = data_dir / "extract" / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"{spec['file']} missing - run 01_extract.py")
    filt = spec.get("filter") or {}
    cols = list(filt) + ["geometry"]
    gdf = gpd.read_parquet(path, columns=cols)
    if bbox is not None:  # vectorised bounds test (GEOS), keeps anything touching the box
        import shapely
        b = shapely.bounds(gdf.geometry.values)
        keep = (b[:, 2] >= bbox[0]) & (b[:, 0] <= bbox[2]) & (b[:, 3] >= bbox[1]) & (b[:, 1] <= bbox[3])
        gdf = gdf[keep]
    for c, v in filt.items():
        vals = v if isinstance(v, list) else [v]
        gdf = gdf[gdf[c].astype(str).isin([str(x) for x in vals])]
    g = gdf.geometry.values
    return np.asarray(g[~(g.isna() | g.is_empty)])
