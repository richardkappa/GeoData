"""Column descriptions for the data dictionaries (config/column_descriptions.yaml).

A column's description comes from, in order:

1. a **catalogue**: a table that maps columns to their source descriptions, e.g. the
   census catalogue written by 01_extract.py (table, title and category per column);
2. **columns**: exact names;
3. **patterns**: regular expressions; ``{1}``... insert captured groups (underscores
   shown as spaces) and ``{ref:name}`` inserts another column's description.

It replaces the note a join or feature group gives when that note is blank or
generic (shared by several columns, e.g. "Census 2021 output-area counts").
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from .config import REPO_ROOT

DEFAULT = REPO_ROOT / "config" / "column_descriptions.yaml"


@lru_cache(maxsize=4)
def _load(path: str):
    try:
        cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}, [], []
    pats = [(re.compile(p), text) for p, text in cfg.get("patterns") or []]
    return cfg.get("columns") or {}, pats, cfg.get("catalogues") or []


@lru_cache(maxsize=8)
def _catalogue(file: str, key: str, template: str) -> dict:
    """{column: description} from a catalogue table; `template` uses the table's column names."""
    import pyarrow.parquet as pq
    p = Path(file)
    if not p.exists():
        return {}
    rows = pq.read_table(p).to_pylist()
    out = {}
    for r in rows:
        r = {k: ("" if v is None else str(v)) for k, v in r.items()}
        r["category"] = re.sub(r";\s*measures:\s*Value\s*$", "", r.get("source_column", "")).strip()
        try:
            out[r[key]] = template.format(**r)
        except (KeyError, IndexError):
            continue
    return out


def describe(column: str, extract_dir: Path | None = None, path: str | None = None, _depth: int = 0) -> str:
    """Description of a column ('' if none is known)."""
    cols, pats, cats = _load(str(path or DEFAULT))
    if extract_dir is not None:
        for c in cats:
            d = _catalogue(str(Path(extract_dir) / c["file"]), c.get("key", "column"), c["template"])
            if column in d:
                return d[column]
    if column in cols:
        return str(cols[column])
    for rx, text in pats:
        m = rx.search(column)
        if m:
            out = text
            for i, g in enumerate(m.groups(), 1):
                out = out.replace("{%d}" % i, (g or "").replace("_", " "))
                out = out.replace("{raw%d}" % i, g or "")
            if "{ref:" in out:   # {ref:name} -> that entry's description; unresolved -> no description
                if _depth >= 3:
                    return ""
                missing = []

                def ref(mm):
                    d = describe(mm.group(1), extract_dir, path, _depth + 1)
                    if not d:
                        missing.append(mm.group(1))
                    return d
                out = re.sub(r"\{ref:([A-Za-z0-9_]+)\}", ref, out)
                if missing:
                    return ""
            return out
    return ""


def fill(columns: list[str], notes: list[str], extract_dir: Path | None = None) -> list[str]:
    """Final notes: the curated description (config/column_descriptions.yaml or a catalogue)
    where one exists, otherwise the note the join or feature group wrote."""
    return [describe(c, extract_dir) or (n or "").strip() for c, n in zip(columns, notes)]
