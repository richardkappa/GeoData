"""Parsing of start/end periods used by the time-based extractors."""
from __future__ import annotations

import datetime as dt


def parse_date(value, *, end: bool = False) -> dt.date | None:
    """``2024`` / ``"2024-03"`` / ``"2024-03-15"`` -> date.

    For an end bound a year or month means its *last* day, so
    ``start: 2024, end: 2024`` is the whole of 2024. ``None``/``"latest"``
    returns None (open ended).
    """
    if value is None or str(value).lower() in ("latest", "none", ""):
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    parts = [int(p) for p in s.split("-")]
    if len(parts) == 1:
        return dt.date(parts[0], 12, 31) if end else dt.date(parts[0], 1, 1)
    if len(parts) == 2:
        y, m = parts
        if end:
            nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
            return nxt - dt.timedelta(days=1)
        return dt.date(y, m, 1)
    return dt.date(*parts[:3])


def month_str(d: dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def add_months(month: str, n: int) -> str:
    y, m = map(int, month.split("-"))
    i = y * 12 + (m - 1) + n
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def month_range(start: str, end: str) -> list[str]:
    out, m = [], start
    while m <= end:
        out.append(m)
        m = add_months(m, 1)
    return out
