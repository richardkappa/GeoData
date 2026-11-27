#!/usr/bin/env python
"""Section 2 - collation: build the flat postcode lookup.

Joins the section 1 outputs onto the NSPL postcode spine (one row per
``postcode_ns``) as configured in ``config/2_collate.yaml`` and writes
``<data_dir>/2_collate/postcode_lookup.parquet`` plus a column dictionary.

    python 02_collate.py
    python 02_collate.py --config my_collate.yaml --data-dir /path/to/GeoData
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pyarrow.csv as pacsv
import yaml

from geodata import io
from geodata.step2_collate import build
from geodata.config import REPO_ROOT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "2_collate.yaml"))
    ap.add_argument("--data-dir")
    ap.add_argument("--csv", action="store_true", help="also write CSV")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    g = cfg.get("global", {})
    data_dir = Path(a.data_dir or os.environ.get("GEODATA_DIR") or g.get("data_dir", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    out_dir = data_dir / "2_collate"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(data_dir / "collate.log")])
    log = logging.getLogger("collate")

    t0 = time.perf_counter()
    table, dictionary = build(data_dir / "1_extract", cfg)
    stem = g.get("output", "postcode_lookup")
    io.write_table(table, out_dir / f"{stem}.parquet", meta={"config": cfg})
    # coverage: share of rows with a value, handy for spotting broken joins
    import pyarrow as pa
    cov = [round(1 - table[c].null_count / max(table.num_rows, 1), 4) for c in dictionary["column"].to_pylist()]
    from geodata.dictionary import LOOKUP, standard_table
    d = dictionary
    dictionary = standard_table(LOOKUP, d["column"].to_pylist(), d["note"].to_pylist(), d["source"].to_pylist(),
                                d["join"].to_pylist(), [str(table.schema.field(c).type) for c in d["column"].to_pylist()],
                                cov, d["key"].to_pylist())   # data dictionary structure (geodata/dictionary.py)
    pacsv.write_csv(dictionary, out_dir / f"{stem}_dictionary.csv")
    if a.csv or g.get("write_csv"):
        pacsv.write_csv(table, out_dir / f"{stem}.csv")
    log.info("postcode lookup: %s rows x %d columns in %.0fs", f"{table.num_rows:,}", table.num_columns,
             time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
