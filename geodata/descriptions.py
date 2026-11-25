"""Column descriptions for the data dictionaries (config/column_descriptions.yaml)."""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from .config import REPO_ROOT


@lru_cache(maxsize=1)
def _load(path: str):
    try:
        cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}, []
    pats = [(re.compile(p), text) for p, text in cfg.get("patterns") or []]
    return cfg.get("columns") or {}, pats


def describe(column: str, path: str | None = None) -> str:
    """Description of a column from config/column_descriptions.yaml ('' if none)."""
    cols, pats = _load(str(path or REPO_ROOT / "config" / "column_descriptions.yaml"))
    if column in cols:
        return str(cols[column])
    for rx, text in pats:
        m = rx.search(column)
        if m:
            out = text
            for i, g in enumerate(m.groups(), 1):
                out = out.replace("{%d}" % i, (g or "").replace("_", " "))
            return out
    return ""


def fill(columns: list[str], notes: list[str]) -> list[str]:
    """Notes with blanks filled from the descriptions file (existing notes are kept)."""
    return [n if (n or "").strip() else describe(c) for c, n in zip(columns, notes)]
