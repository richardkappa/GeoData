"""Vectorised postcode standardisation.

``postcode_ns`` is the project-wide primary key: the postcode upper-cased with
*all* whitespace removed, e.g. ``"sw1a 1aa"`` -> ``"SW1A1AA"``. The work is
done by Arrow compute kernels (C++), never by a Python loop.
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

POSTCODE_KEY = "postcode_ns"
# Loose UK postcode shape once spaces are removed (outward 2-4, inward 3)
_PC_RE = r"^[A-Z]{1,2}[0-9][0-9A-Z]?[0-9][A-Z]{2}$"


def normalise(arr: pa.Array | pa.ChunkedArray, *, validate: bool = False) -> pa.ChunkedArray:
    """Upper-case and strip all whitespace; empty strings become null.

    With ``validate=True`` values that do not look like a UK postcode are
    set to null (useful for free-text address fields).
    """
    if not pa.types.is_string(arr.type) and not pa.types.is_large_string(arr.type):
        arr = pc.cast(arr, pa.string())
    out = pc.utf8_upper(arr)
    out = pc.replace_substring_regex(out, pattern=r"\s+", replacement="")
    empty = pc.equal(pc.utf8_length(out), 0)
    out = pc.if_else(empty, pa.scalar(None, out.type), out)
    if validate:
        ok = pc.match_substring_regex(out, _PC_RE)
        out = pc.if_else(ok, out, pa.scalar(None, out.type))
    if isinstance(out, pa.Array):
        out = pa.chunked_array([out])
    return out


def add_postcode_ns(table: pa.Table, source_col: str, *, validate: bool = False,
                    first: bool = True) -> pa.Table:
    """Return ``table`` with a ``postcode_ns`` column built from ``source_col``."""
    col = normalise(table[source_col], validate=validate)
    if POSTCODE_KEY in table.column_names:
        table = table.drop_columns([POSTCODE_KEY])
    if first:
        return table.add_column(0, POSTCODE_KEY, col)
    return table.append_column(POSTCODE_KEY, col)


def pretty(postcode_ns: pa.Array | pa.ChunkedArray) -> pa.ChunkedArray:
    """``SW1A1AA`` -> ``SW1A 1AA`` (single space before the inward code)."""
    return pc.replace_substring_regex(postcode_ns, pattern=r"^(.+)(\d[A-Z]{2})$",
                                      replacement=r"\1 \2")
