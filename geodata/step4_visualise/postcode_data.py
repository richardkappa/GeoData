"""Postcode values for a square (the map explorer's postcode overlay) and helpers shared by the
area pages: variable labels, country masks, compact JSON numbers."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..postcode import POSTCODE_KEY

log = logging.getLogger(__name__)


def var_label(spec) -> str:
    """A variable's display label: config entries are a label or {label, countries, ...}."""
    return str(spec.get("label", "")) if isinstance(spec, dict) else str(spec)


def _area_table(data_dir: Path, box, variables: dict) -> tuple[pa.Table, dict]:
    """Postcodes in the box with the variables; returns (table, {column: label}).
    Variables with `countries:` are blanked outside the countries their source covers."""
    lookup_f = data_dir / "2_collate" / "postcode_lookup.parquet"
    feat_f = data_dir / "3_features" / "postcode_features.parquet"
    lnames = set(pq.read_schema(lookup_f).names)
    fnames = set(pq.read_schema(feat_f).names) if feat_f.exists() else set()
    x0, y0, x1, y1 = box
    base = ["pcds", "easting", "northing", "lat", "long"]
    specs = {c: (v if isinstance(v, dict) else {"label": v}) for c, v in variables.items()}
    if any(s.get("countries") for s in specs.values()) and "ctry26cd" in lnames:
        base.append("ctry26cd")
    base = list(dict.fromkeys([*base, *(k for s in specs.values() for k in (s.get("blank_where") or {}) if k in lnames)]))
    lt = pq.read_table(lookup_f, columns=[POSTCODE_KEY, *base, *(c for c in variables if c in lnames and c not in fnames)],
                       filters=[("easting", ">=", x0), ("easting", "<=", x1),
                                ("northing", ">=", y0), ("northing", "<=", y1)])
    if fnames:
        ft = pq.read_table(feat_f, columns=[POSTCODE_KEY, *(c for c in variables if c in fnames)])
        idx = pc.index_in(lt[POSTCODE_KEY], value_set=ft[POSTCODE_KEY])
        lt = lt.filter(pc.is_valid(idx))      # live postcodes (the features spine)
        idx = pc.index_in(lt[POSTCODE_KEY], value_set=ft[POSTCODE_KEY])
        for c in ft.column_names[1:]:
            lt = lt.append_column(c, ft[c].take(idx))
    from .cartograms import _mask_countries   # local import: cartograms imports this module
    lt = _mask_countries(lt, specs)
    found = {c: var_label(s) for c, s in specs.items() if c in lt.column_names}
    return lt, found


def _num(col: pa.ChunkedArray) -> list:
    """Column -> JSON-friendly list (numbers rounded to 3 significant figures, None for missing)."""
    if pa.types.is_boolean(col.type):
        col = pc.cast(col, pa.int8())
    if not (pa.types.is_integer(col.type) or pa.types.is_floating(col.type)):
        return [None if v is None else str(v) for v in col.to_pylist()]
    out = []
    for v in col.to_pylist():
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            out.append(None)
        elif v == 0:
            out.append(0)
        else:
            d = 3 - int(math.floor(math.log10(abs(v)))) - 1
            out.append(round(v, max(d, 0)) if d > 0 else int(round(v, d)))
    return out
