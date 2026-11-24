#!/usr/bin/env python
"""Section 3 - new data creation: per-postcode features.

Computes densities, rates, nearest distances, network travel times, land cover,
terrain, water, house-price and crime trends and LSOA neighbourhood averages as
configured in ``config/features.yaml``. Output keyed on ``postcode_ns``:
``<data_dir>/features/postcode_features.parquet`` + ``_dictionary.csv``.

    python 03_features.py
    python 03_features.py --bbox 320000 420000 380000 520000     # test area
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

from geodata import io
from geodata.config import REPO_ROOT
from geodata.features.build import build


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "features.yaml"))
    ap.add_argument("--data-dir")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="only postcodes in this BNG box (test runs)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    g = cfg.setdefault("global", {})
    if a.bbox:
        cfg.setdefault("spine", {})["bbox"] = a.bbox
    data_dir = Path(a.data_dir or os.environ.get("GEODATA_DIR") or g.get("data_dir", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    out_dir = data_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(data_dir / "features.log")])
    log = logging.getLogger("features")
    t0 = time.perf_counter()

    table, dictionary = build(data_dir, cfg)
    stem = g.get("output", "postcode_features")
    io.write_table(table, out_dir / f"{stem}.parquet", meta={"config": cfg})
    cols = dictionary["column"].to_pylist()
    cov = [round(1 - table[c].null_count / max(table.num_rows, 1), 4) for c in cols]
    dictionary = dictionary.append_column("coverage", pa.array(cov))
    pacsv.write_csv(dictionary, out_dir / f"{stem}_dictionary.csv")

    if g.get("merge_lookup"):
        lookup = data_dir / "collate" / "postcode_lookup.parquet"
        lt = pq.read_table(lookup)
        idx = pc.index_in(lt["postcode_ns"], value_set=table["postcode_ns"])
        for c in table.column_names[1:]:
            lt = lt.append_column(c, table[c].take(idx))
        io.write_table(lt, out_dir / "postcode_dataset.parquet")
    log.info("features: %s postcodes x %d columns in %.0fs", f"{table.num_rows:,}", table.num_columns - 1,
             time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
