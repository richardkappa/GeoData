"""Section 2 - collation: one flat table per postcode.

Starts from the NSPL spine (one row per ``postcode_ns``) and attaches the
section 1 outputs with four kinds of join, all set in ``config/2_collate.yaml``:

``postcode``  tables already keyed on ``postcode_ns`` (energy use, ...)
``area``      tables keyed on an area code the spine carries (census by
              ``oa21cd``, IMD by ``lsoa21cd``, names by ``lad26cd`` ...)
``polygon``   attribute of the polygon each postcode centroid falls in
              (AONB, conservation area, radon class, climate grid cell ...)
``summary``   simple within-postcode / within-area summaries of record
              tables (sales per postcode, addresses per postcode, crimes per
              LSOA). No distances, radii or neighbourhoods: those are section 3.

Joins are Arrow hash joins (C++); polygon lookups are one vectorised GEOS
STRtree query per layer over all postcode points.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .. import io
from ..postcode import POSTCODE_KEY

log = logging.getLogger(__name__)


@dataclass
class Dictionary:
    rows: list[tuple] = field(default_factory=list)
    extract_dir: Path | None = None   # for description catalogues (e.g. the census catalogue)

    def add(self, cols, source: str, join: str, key: str, note: str = ""):
        for c in cols:
            self.rows.append((c, source, join, key, note))

    def table(self) -> pa.Table:
        from ..descriptions import fill
        cols = list(zip(*self.rows)) if self.rows else [[]] * 5
        if self.rows:   # blank or generic join notes -> specific descriptions (config/column_descriptions.yaml)
            cols[4] = fill(list(cols[0]), list(cols[4]), self.extract_dir)
        return pa.table(cols, names=["column", "source", "join", "key", "note"])


# --------------------------------------------------------------------- helpers
def _read(extract_dir: Path, rel: str, columns=None) -> pa.Table:
    path = extract_dir / rel
    if not path.exists():
        raise FileNotFoundError(f"{rel} not found - run 01_extract.py for that source first")
    t = pq.read_table(path, columns=columns)
    # dictionary columns -> plain (joins and concatenation are simpler)
    for i, f in enumerate(t.schema):
        if pa.types.is_dictionary(f.type):
            t = t.set_column(i, f.name, pc.cast(t[f.name], f.type.value_type))
    md = t.schema.metadata or {}
    return t.replace_schema_metadata({k: v for k, v in md.items() if k != b"geo"})


def _prefixed(t: pa.Table, keep_key: str, prefix: str | None, rename: dict | None = None) -> pa.Table:
    rename = rename or {}
    names = []
    for c in t.column_names:
        if c == keep_key:
            names.append(c)
        elif c in rename:
            names.append(rename[c])
        else:
            names.append(f"{prefix}{c}" if prefix else c)
    return t.rename_columns(names)


def _plain_strings(t: pa.Table) -> pa.Table:
    for i, f in enumerate(t.schema):
        if pa.types.is_large_string(f.type):
            t = t.set_column(i, f.name, pc.cast(t[f.name], pa.string()))
    return t


def _left_join(spine: pa.Table, right: pa.Table, left_key: str, right_key: str) -> pa.Table:
    # guarantee a many-to-one join so the spine never gains rows
    right = _plain_strings(right)
    right = right.filter(pc.is_valid(right[right_key]))
    n_right = right.num_rows
    uniq = pc.count_distinct(right[right_key]).as_py()
    if uniq != n_right:
        log.warning("right table not unique on %s (%d rows, %d keys): keeping first", right_key, n_right, uniq)
        idx = pc.indices_nonzero(_first_mask(right[right_key]))
        right = right.take(idx)
    clash = set(spine.column_names) & set(right.column_names) - {right_key}
    if clash:
        right = right.drop_columns(sorted(clash))
        log.warning("dropped duplicate columns already on spine: %s", sorted(clash))
    # Index lookup instead of a relational join: find each spine key's row in
    # `right` (C++ hash), then `take` only the new columns and append them. The
    # existing spine columns are never copied, so memory grows only by what is
    # added, and row order is unchanged.
    idx = pc.index_in(spine[left_key], value_set=right[right_key])
    for c in right.column_names:
        if c == right_key:
            continue
        spine = spine.append_column(c, _compact(right[c]).take(idx))
    return spine


def _compact(col):
    """int64 -> int32 when values fit (census counts etc.) to halve memory."""
    if pa.types.is_int64(col.type):
        mn, mx = pc.min_max(col).values()
        if mn.as_py() is None or (mn.as_py() >= -2**31 and mx.as_py() < 2**31):
            return pc.cast(col, pa.int32())
    return col


def _first_mask(col: pa.ChunkedArray) -> pa.Array:
    idx = pa.table({"k": col, "i": pa.array(np.arange(len(col)))}).group_by("k").aggregate([("i", "min")])
    mask = np.zeros(len(col), dtype=bool)
    mask[idx["i_min"].to_numpy()] = True
    return pa.array(mask)


# ---------------------------------------------------------------- join types
def join_postcode(spine, extract_dir, spec, dd: Dictionary):
    t = _read(extract_dir, spec["file"])
    cols = spec.get("columns")
    drop = set(spec.get("drop", [])) | {"postcode", "pcds", "outcode"}
    keep = [c for c in (cols or t.column_names) if c in t.column_names and (c not in drop or c == POSTCODE_KEY)]
    t = t.select(list(dict.fromkeys([POSTCODE_KEY, *keep])))
    t = _prefixed(t, POSTCODE_KEY, spec.get("prefix"), spec.get("rename"))
    dd.add([c for c in t.column_names if c != POSTCODE_KEY], spec["file"], "postcode", POSTCODE_KEY, spec.get("note", ""))
    return _left_join(spine, t, POSTCODE_KEY, POSTCODE_KEY)


def join_area(spine, extract_dir, spec, dd: Dictionary):
    on = spec["spine_col"]             # column on the spine, e.g. oa21cd ("on" is a YAML boolean)
    files = spec.get("file") or []
    files = files if isinstance(files, list) else [files]
    if spec.get("glob"):
        files = sorted(str(p.relative_to(extract_dir)) for p in extract_dir.glob(spec["glob"]))
    key = spec.get("key", on)          # column in the file
    if not files:
        log.warning("area join %s: no files", spec.get("name"))
        return spine
    for f in files:
        t = _read(extract_dir, f)
        cols = spec.get("columns")
        drop = set(spec.get("drop", []))
        keep = [c for c in (cols or t.column_names) if c in t.column_names and c not in drop and c != key]
        if spec.get("drop_regex"):
            keep = [c for c in keep if not re.search(spec["drop_regex"], c)]
        t = t.select([key, *keep]).rename_columns([on, *keep])
        if spec.get("shares"):         # optional: counts -> share of the table total
            t = _to_shares(t, on)
        t = _prefixed(t, on, spec.get("prefix"), spec.get("rename"))
        dd.add([c for c in t.column_names if c != on], f, "area", on, spec.get("note", ""))
        spine = _left_join(spine, t, on, on)
    return spine


def _to_shares(t: pa.Table, key: str) -> pa.Table:
    """Census style: divide each count by the table's '..._total...' column."""
    tot = [c for c in t.column_names if re.search(r"(^|_)total(_|$)", c)]
    if not tot:
        return t
    denom = pc.cast(t[tot[0]], pa.float64())
    for c in t.column_names:
        if c in (key, tot[0]) or not pa.types.is_integer(t[c].type):
            continue
        t = t.set_column(t.schema.get_field_index(c), c,
                         pc.round(pc.divide(pc.cast(t[c], pa.float64()), denom), 4))
    return t


class PointIndex:
    """Shapely points for every spine postcode with coordinates (built once)."""

    def __init__(self, spine: pa.Table):
        import shapely
        e = spine["easting"].to_numpy(zero_copy_only=False).astype("f8")
        n = spine["northing"].to_numpy(zero_copy_only=False).astype("f8")
        self.ok = np.isfinite(e) & np.isfinite(n)
        self.rows = np.flatnonzero(self.ok)
        self.points = shapely.points(e[self.ok], n[self.ok])
        self.tree = shapely.STRtree(self.points)
        self.n = spine.num_rows


def join_polygon(spine, extract_dir, spec, dd: Dictionary, pts: PointIndex):
    """Attributes of the polygon containing each postcode centroid.

    ``how``: ``first`` (attributes of one containing polygon), ``flag`` (bool:
    inside any), ``count`` (how many overlapping polygons), ``max``/``min``
    (of a numeric column, e.g. radon class across overlapping tiles).
    """
    import geopandas as gpd

    path = extract_dir / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"{spec['file']} not found - run 01_extract.py first")
    cols = spec.get("columns", [])
    gdf = gpd.read_parquet(path, columns=[*cols, "geometry"])
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    geoms = gdf.geometry.values._data if hasattr(gdf.geometry.values, "_data") else np.asarray(gdf.geometry.values)
    # one query per polygon against the point tree: GEOS prepares each polygon
    # once, which is far faster than testing each point against complex shapes
    g_idx, p_idx = pts.tree.query(geoms, predicate="intersects")
    how = spec.get("how", "first")
    name = spec["name"]
    prefix = spec.get("prefix", f"{name}_")
    new: dict[str, pa.Array] = {}
    if how in ("flag", "count"):
        counts = np.bincount(p_idx, minlength=len(pts.points))
        full = np.zeros(pts.n, dtype="int32")
        full[pts.rows] = counts
        col = spec.get("column_name", f"in_{name}" if how == "flag" else f"{name}_count")
        new[col] = pa.array(full > 0) if how == "flag" else pa.array(full)
    elif how in ("max", "min"):
        for c in cols:
            v = gdf[c].to_numpy(dtype="f8")
            agg = np.full(len(pts.points), -np.inf if how == "max" else np.inf)
            (np.maximum if how == "max" else np.minimum).at(agg, p_idx, v[g_idx])
            agg[~np.isfinite(agg)] = np.nan
            full = np.full(pts.n, np.nan)
            full[pts.rows] = agg
            new[spec.get("rename", {}).get(c, f"{prefix}{c}")] = pa.array(full, mask=np.isnan(full))
    else:  # first
        order = np.lexsort((g_idx, p_idx))
        p_s, g_s = p_idx[order], g_idx[order]
        uniq_p, first = np.unique(p_s, return_index=True)
        chosen = np.full(len(pts.points), -1)
        chosen[uniq_p] = g_s[first]
        full = np.full(pts.n, -1)
        full[pts.rows] = chosen
        hit = full >= 0
        for c in cols:
            src = pa.array(gdf[c].to_numpy(), from_pandas=True)
            vals = src.take(pa.array(np.where(hit, full, 0)))
            vals = pc.if_else(pa.array(hit), vals, pa.scalar(None, vals.type))
            new[spec.get("rename", {}).get(c, f"{prefix}{c}")] = vals
        if spec.get("flag", True):
            new[f"in_{name}"] = pa.array(hit)
    for c, arr in new.items():
        if c in spine.column_names:
            spine = spine.drop_columns([c])
        spine = spine.append_column(c, arr)
    dd.add(list(new), spec["file"], f"polygon:{how}", "easting/northing", spec.get("note", ""))
    log.info("polygon %-28s %s postcodes matched", name, f"{len(np.unique(p_idx)):,}")
    return spine


# -------------------------------------------------------------- summaries
def summary_land_registry(extract_dir, spec):
    t = _read(extract_dir, spec.get("file", "land_registry/price_paid.parquet"),
              columns=[POSTCODE_KEY, "price", "date_of_transfer", "property_type", "ppd_category_type"])
    if spec.get("standard_only", True):          # category A = standard market sales
        t = t.filter(pc.equal(t["ppd_category_type"], "A"))
    t = t.filter(pc.and_(pc.is_valid(t[POSTCODE_KEY]), pc.is_valid(t["price"])))
    # integer group ids, then everything is numpy sorting / bincount (exact, low memory)
    enc = pc.dictionary_encode(t[POSTCODE_KEY]).combine_chunks()
    gid = enc.indices.to_numpy()
    keys = enc.dictionary
    ng = len(keys)
    price = t["price"].to_numpy().astype("f8")
    order = np.lexsort((price, gid))
    g_s, p_s = gid[order], price[order]
    counts = np.bincount(g_s, minlength=ng)
    start = np.concatenate(([0], np.cumsum(counts)[:-1]))
    median = (p_s[start + (counts - 1) // 2] + p_s[start + counts // 2]) / 2
    days = t["date_of_transfer"].cast(pa.int32()).to_numpy()
    last = np.full(ng, np.iinfo("int32").min, dtype="int32")
    np.maximum.at(last, gid, days)
    out = pa.table({POSTCODE_KEY: keys, "lr_sales": pa.array(counts.astype("int32")),
                    "lr_median_price": pa.array(np.round(median, -2)),
                    "lr_last_sale_date": pa.array(last).cast(pa.date32())})
    # share of sales by property type (D/S/T/F/O) per postcode
    labels = {"D": "detached", "S": "semi", "T": "terraced", "F": "flat", "O": "other"}
    ptype = t["property_type"].to_numpy(zero_copy_only=False)
    for code, label in labels.items():
        n = np.bincount(gid[ptype == code], minlength=ng)
        out = out.append_column(f"lr_share_{label}", pa.array(np.round(n / counts, 3)))
    return out, POSTCODE_KEY


def summary_uprn_count(extract_dir, spec):
    t = pq.read_table(extract_dir / spec.get("file", "ons_nsul/nsul.parquet"), columns=[POSTCODE_KEY])
    g = t.group_by(POSTCODE_KEY).aggregate([([], "count_all")]).rename_columns([POSTCODE_KEY, "uprn_count"])
    return g, POSTCODE_KEY


def summary_police_lsoa(extract_dir, spec):
    t = _read(extract_dir, spec.get("file", "police/crimes_lsoa_month.parquet"))
    months = sorted(set(t["month"].to_pylist()))
    n = spec.get("months")
    if n:
        months = months[-int(n):]
        t = t.filter(pc.is_in(t["month"], pa.array(months)))
    df = t.to_pandas()
    wide = df.pivot_table(index="lsoa21cd", columns="crime_type", values="crimes", aggfunc="sum", fill_value=0)
    wide.columns = [f"crimes_{io.snake(c)}" for c in wide.columns]
    wide["crimes_total"] = wide.sum(axis=1)
    wide = wide.astype("int32").reset_index()
    log.info("police  %d months: %s .. %s", len(months), months[0], months[-1])
    return pa.Table.from_pandas(wide, preserve_index=False), "lsoa21cd"


def summary_count(extract_dir, spec):
    """Generic per-key counts of a record table (default key ``postcode_ns``).

    spec: ``file``; ``prefix``; optional ``filter`` {column: value | [values]};
    ``category`` column with ``categories`` {label: value | [values]} for
    per-category counts; ``sum`` / ``mean`` lists of numeric columns.
    Produces ``<prefix>count``, ``<prefix>count_<label>``, ``<prefix>sum_<col>``,
    ``<prefix>mean_<col>``.
    """
    key = spec.get("key", POSTCODE_KEY)
    cat = spec.get("category")
    num = list(dict.fromkeys(spec.get("sum", []) + spec.get("mean", [])))
    filt = spec.get("filter") or {}
    cols = list(dict.fromkeys([key, *([cat] if cat else []), *num, *filt]))
    t = _read(extract_dir, spec["file"], columns=cols)
    t = t.filter(pc.is_valid(t[key]))
    for c, v in filt.items():
        vals = v if isinstance(v, list) else [v]
        t = t.filter(pc.is_in(pc.cast(t[c], pa.string()), pa.array([str(x) for x in vals])))
    prefix = spec.get("prefix", "")
    enc = pc.dictionary_encode(t[key]).combine_chunks()
    gid, keys = enc.indices.to_numpy(), enc.dictionary
    ng = len(keys)
    counts = np.bincount(gid, minlength=ng)
    out = {key: keys, f"{prefix}count": pa.array(counts.astype("int32"))}
    if cat:
        cv = pc.cast(t[cat], pa.string()).to_numpy(zero_copy_only=False)
        for label, v in (spec.get("categories") or {}).items():
            vals = v if isinstance(v, list) else [v]
            n = np.bincount(gid[np.isin(cv, [str(x) for x in vals])], minlength=ng)
            out[f"{prefix}count_{label}"] = pa.array(n.astype("int32"))
    for c in num:
        x = pc.cast(t[c], pa.float64()).to_numpy(zero_copy_only=False)
        ok = ~np.isnan(x)
        tot = np.bincount(gid[ok], weights=x[ok], minlength=ng)
        if c in spec.get("sum", []):
            out[f"{prefix}sum_{c}"] = pa.array(tot)
        if c in spec.get("mean", []):
            n = np.bincount(gid[ok], minlength=ng)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.round(tot / n, 3)
            out[f"{prefix}mean_{c}"] = pa.array(m, mask=n == 0)
    return pa.table(out), key


SUMMARIES = {
    "land_registry": summary_land_registry,
    "uprn_count": summary_uprn_count,
    "police_lsoa": summary_police_lsoa,
    "count": summary_count,
}


def join_summary(spine, extract_dir, spec, dd: Dictionary):
    fn = SUMMARIES[spec["summary"]]
    t, key = fn(extract_dir, spec)
    dd.add([c for c in t.column_names if c != key], spec.get("file", spec["summary"]), "summary", key,
           spec.get("note", ""))
    spine = _left_join(spine, t, key, key)
    # a postcode absent from a record table has zero of those records
    if spec["summary"] == "count" and spec.get("fill_zero", True):
        for c in t.column_names:
            if c != key and ("_count" in c or c.endswith("count") or "_sum_" in c):
                spine = spine.set_column(spine.schema.get_field_index(c), c, pc.fill_null(spine[c], 0))
    return spine


# ------------------------------------------------------------------- driver
def build(extract_dir: Path, cfg: dict[str, Any]) -> tuple[pa.Table, pa.Table]:
    sp = cfg.get("spine", {})
    spine = _read(extract_dir, sp.get("file", "ons_nspl/nspl.parquet"), columns=sp.get("columns"))
    if not sp.get("include_terminated", True):
        spine = spine.filter(spine["live"])
    if sp.get("countries"):  # first letter of the country code: E, W, S, N
        spine = spine.filter(pc.is_in(pc.utf8_slice_codeunits(spine["ctry26cd"], 0, 1),
                                      pa.array(sp["countries"])))
    if sp.get("require_coordinates", True):
        spine = spine.filter(pc.is_valid(spine["easting"]))
    log.info("spine   %s postcodes", f"{spine.num_rows:,}")
    dd = Dictionary(extract_dir=Path(extract_dir))
    dd.add(spine.column_names, sp.get("file", "ons_nspl/nspl.parquet"), "spine", POSTCODE_KEY)

    # row order is fixed from here: every join appends aligned columns
    spine = spine.sort_by(POSTCODE_KEY)
    pts = None
    failed = []
    for spec in cfg.get("joins", []):
        if spec.get("enabled", True) is False:
            continue
        kind = spec["type"]
        name = spec.get("name", spec.get("file"))
        try:
            with io.Timer(f"join {kind:<8} {name}"):
                if kind == "postcode":
                    spine = join_postcode(spine, extract_dir, spec, dd)
                elif kind == "area":
                    spine = join_area(spine, extract_dir, spec, dd)
                elif kind == "summary":
                    spine = join_summary(spine, extract_dir, spec, dd)
                elif kind == "polygon":
                    if pts is None:
                        pts = PointIndex(spine)
                    spine = join_polygon(spine, extract_dir, spec, dd, pts)
                else:
                    raise ValueError(f"unknown join type {kind}")
        except FileNotFoundError as e:
            if spec.get("optional"):   # an alternative input is expected to be missing (e.g. Ofcom mobile)
                log.info("skip optional join %s: %s", name, e)
            else:
                log.warning("skip %s: %s", name, e)
                failed.append(name)
        except Exception as e:
            log.exception("join %s failed: %s", name, e)
            failed.append(name)
    if failed:
        log.warning("joins skipped/failed: %s", ", ".join(map(str, failed)))
    return enforce_primary_key(spine).combine_chunks(), dd.table()


def enforce_primary_key(t: pa.Table, key: str = POSTCODE_KEY) -> pa.Table:
    """Make ``key`` the table's primary key: first column, non-null, unique,
    sorted, declared non-nullable in the schema and recorded in metadata.
    Raises if the data would violate it (never silently drops rows)."""
    if key not in t.column_names:
        raise ValueError(f"primary key column {key!r} missing")
    nulls = t[key].null_count
    if nulls:
        raise ValueError(f"primary key {key!r} has {nulls} null values")
    dupes = t.num_rows - pc.count_distinct(t[key]).as_py()
    if dupes:
        raise ValueError(f"primary key {key!r} has {dupes} duplicate values")
    t = t.select([key] + [c for c in t.column_names if c != key]).sort_by(key)
    schema = t.schema.set(0, t.schema.field(0).with_nullable(False))
    md = dict(schema.metadata or {})
    md[b"primary_key"] = key.encode()
    return t.cast(schema.with_metadata(md))
