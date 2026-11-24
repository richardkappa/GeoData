"""Fast tabular I/O built on Arrow's multi-threaded C++ CSV reader and Parquet.

Pandas is only used at the edges (Excel files, GeoDataFrames). Large CSVs are
read with ``pyarrow.csv`` which parses on all cores and only materialises the
requested columns.
"""
from __future__ import annotations

import io
import json
import logging
import re
import time
import zipfile
from pathlib import Path
from typing import IO, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

PARQUET_KW = dict(compression="zstd", compression_level=3)


# --------------------------------------------------------------------------- CSV
def read_csv(src: str | Path | IO[bytes], *, columns: Sequence[str] | None = None,
             types: dict[str, pa.DataType] | None = None, names: Sequence[str] | None = None,
             encoding: str = "utf8", delimiter: str = ",", strings: bool = False,
             skip_rows: int = 0, null_values: Sequence[str] | None = None) -> pa.Table:
    """Read a CSV into an Arrow table.

    ``strings=True`` reads every column as text (no type inference), which is
    the safe choice for files with codes that look numeric (e.g. ``"0801193"``).
    """
    read_opts = pacsv.ReadOptions(encoding=encoding, column_names=names, skip_rows=skip_rows,
                                  block_size=1 << 24, use_threads=True)
    parse_opts = pacsv.ParseOptions(delimiter=delimiter, newlines_in_values=True)
    col_types = dict(types or {})
    conv = pacsv.ConvertOptions(
        include_columns=list(columns) if columns else None,
        column_types=col_types,
        strings_can_be_null=True,
        null_values=list(null_values) if null_values is not None else ["", "NA", "NULL", "null", "N/A"],
        quoted_strings_can_be_null=True,
    )
    if strings:
        # infer nothing: read header first to type every requested column as string
        header = _peek_header(src, encoding, delimiter, names, skip_rows)
        wanted = list(columns) if columns else header
        conv.column_types = {c: col_types.get(c, pa.string()) for c in wanted}
    return pacsv.read_csv(src, read_options=read_opts, parse_options=parse_opts, convert_options=conv)


def _peek_header(src, encoding, delimiter, names, skip_rows) -> list[str]:
    if names:
        return list(names)
    if isinstance(src, (str, Path)):
        with open(src, "rb") as f:
            head = f.read(1 << 16)
    else:
        pos = src.tell()
        head = src.read(1 << 16)
        src.seek(pos)
    text = head.decode(encoding if encoding.lower() not in ("utf8", "utf-8") else "utf-8-sig",
                       errors="replace")
    lines = text.splitlines()[skip_rows:]
    import csv
    return next(csv.reader([lines[0]], delimiter=delimiter))


def read_csv_from_zip(zip_path: Path, member: str, **kw) -> pa.Table:
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as f:
            data = f.read()       # members are read fully: Arrow then parses on all cores
    return read_csv(io.BytesIO(data), **kw)


def zip_members(zip_path: Path, pattern: str) -> list[str]:
    rx = re.compile(pattern, re.I)
    with zipfile.ZipFile(zip_path) as z:
        return [n for n in z.namelist() if rx.search(n)]


# ----------------------------------------------------------------------- columns
def snake(name: str) -> str:
    """Make a column name lower_snake_case and Parquet/SQL friendly."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    return s or "col"


def snake_columns(table: pa.Table) -> pa.Table:
    seen: dict[str, int] = {}
    out = []
    for c in table.column_names:
        n = snake(c)
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        out.append(n)
    return table.rename_columns(out)


def to_numeric(table: pa.Table, cols: Iterable[str], dtype=pa.float64()) -> pa.Table:
    """Cast text columns to numbers, turning unparsable values into nulls."""
    for c in cols:
        if c not in table.column_names:
            continue
        arr = table[c]
        if pa.types.is_string(arr.type):
            arr = pc.replace_substring_regex(arr, r"[,\s£%]", "")
            ok = pc.match_substring_regex(arr, r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
            arr = pc.if_else(ok, arr, pa.scalar(None, pa.string()))
        table = table.set_column(table.schema.get_field_index(c), c, pc.cast(arr, dtype, safe=False))
    return table


_INT_RE = r"^[-+]?\d+$"
_NUM_RE = r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$"


def auto_numeric(table: pa.Table, cols: Iterable[str]) -> pa.Table:
    """Cast each text column to int32/int64 or float64 when every non-null
    value is numeric; leave it as text otherwise (all checks run in C++)."""
    for c in cols:
        if c not in table.column_names or not pa.types.is_string(table[c].type):
            continue
        arr = table[c]
        valid = pc.is_valid(arr)
        if not pc.any(valid).as_py():
            continue
        if pc.all(pc.or_kleene(pc.invert(valid), pc.match_substring_regex(arr, _INT_RE))).as_py():
            big = pc.max(pc.utf8_length(arr)).as_py() or 0
            dtype = pa.int32() if big < 10 else pa.int64()
        elif pc.all(pc.or_kleene(pc.invert(valid), pc.match_substring_regex(arr, _NUM_RE))).as_py():
            dtype = pa.float64()
        else:
            continue
        table = table.set_column(table.schema.get_field_index(c), c, pc.cast(arr, dtype))
    return table


# ------------------------------------------------------------------- coordinates
def bng_from_lonlat(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return t.transform(np.asarray(lon, dtype="f8"), np.asarray(lat, dtype="f8"))


def bng_from_irish(e: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:29903", "EPSG:27700", always_xy=True)
    return t.transform(np.asarray(e, dtype="f8"), np.asarray(n, dtype="f8"))


def add_bng_from_lonlat(table: pa.Table, lon: str, lat: str) -> pa.Table:
    """Append ``easting``/``northing`` (EPSG:27700) computed from WGS84 columns."""
    lo = table[lon].to_numpy(zero_copy_only=False).astype("f8")
    la = table[lat].to_numpy(zero_copy_only=False).astype("f8")
    e, n = bng_from_lonlat(lo, la)
    bad = ~np.isfinite(e) | ~np.isfinite(n) | np.isnan(lo) | np.isnan(la)
    e[bad] = np.nan
    n[bad] = np.nan
    for name, v in (("easting", e), ("northing", n)):
        if name in table.column_names:
            table = table.drop_columns([name])
        table = table.append_column(name, pa.array(np.round(v, 1), mask=np.isnan(v)))
    return table


# ----------------------------------------------------------------------- outputs
def write_table(table: pa.Table, path: Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta:
        md = dict(table.schema.metadata or {})
        md[b"geodata"] = json.dumps(meta, default=str).encode()
        table = table.replace_schema_metadata(md)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, **PARQUET_KW)
    tmp.replace(path)
    log.info("wrote   %s  (%s rows, %d cols)", path.name, f"{table.num_rows:,}", table.num_columns)
    return path


def write_gdf(gdf, path: Path, crs: str = "EPSG:27700") -> Path:
    """Write a GeoDataFrame as GeoParquet in the project CRS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs) != crs and gdf.crs.to_epsg() != int(crs.split(":")[1]):
        gdf = gdf.to_crs(crs)
    gdf.columns = [c if c == gdf.geometry.name else snake(c) for c in gdf.columns]
    tmp = path.with_suffix(path.suffix + ".tmp")
    gdf.to_parquet(tmp, compression="zstd", index=False)
    tmp.replace(path)
    log.info("wrote   %s  (%s features)", path.name, f"{len(gdf):,}")
    return path


def is_fresh(path: Path, reprocess: bool) -> bool:
    return Path(path).exists() and not reprocess


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        log.info("%s took %.1fs", self.label, time.perf_counter() - self.t)
