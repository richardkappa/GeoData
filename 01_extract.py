#!/usr/bin/env python
"""Section 1 - data extraction.

Downloads every enabled source in ``config/1_extract.yaml`` and writes
standardised Parquet/GeoParquet files to ``<data_dir>/1_extract/``.

Examples
--------
    python 01_extract.py                       # all enabled sources
    python 01_extract.py --list                # show sources and status
    python 01_extract.py --only ons_nspl naptan
    python 01_extract.py --skip police --reprocess
    python 01_extract.py --config my.yaml --data-dir /path/to/GeoData
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from geodata.config import load_settings
from geodata.step1_extract import load_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config (default config/1_extract.yaml)")
    ap.add_argument("--data-dir", help="override global.data_dir")
    ap.add_argument("--only", nargs="+", help="run just these sources (ignores 'enabled')")
    ap.add_argument("--skip", nargs="+", default=[], help="skip these sources")
    ap.add_argument("--force", action="store_true", help="re-download cached files")
    ap.add_argument("--reprocess", action="store_true", help="rebuild outputs from cached downloads")
    ap.add_argument("--list", action="store_true", help="list sources and exit")
    ap.add_argument("--geocode-only", action="store_true",
                    help="only add / fill easting-northing in the tables already extracted (no downloads)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    settings = load_settings(a.config, data_dir=Path(a.data_dir) if a.data_dir else None,
                             force=a.force or None, reprocess=a.reprocess or None)
    log_file = settings.data_dir / "extract.log"
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    for noisy in ("urllib3", "pyogrio", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("extract")

    sources = load_all()
    if a.list:
        print(f"{'source':<18} {'on':<4} keys / description")
        for name, s in sources.items():
            on = settings.sources.get(name, {}).get("enabled", False)
            print(f"{name:<18} {'yes' if on else 'no':<4} [{', '.join(s.keys)}] {s.description}")
        return 0

    unknown = [n for n in (a.only or []) + a.skip if n not in sources]
    if unknown:
        ap.error(f"unknown source(s): {unknown}. Use --list.")

    if a.only:
        todo = [n for n in sources if n in a.only]
    else:
        todo = [n for n in sources if settings.sources.get(n, {}).get("enabled", False)]
    todo = [n for n in todo if n not in a.skip]
    log.info("data_dir=%s  sources=%s", settings.data_dir, ", ".join(todo))

    from geodata.geocode import geocode_files
    gcfg = settings.geocode or {}
    nspl = settings.out_dir / "ons_nspl" / "nspl.parquet"

    def geocode(name, paths):
        if gcfg.get("enabled", True) and name not in (gcfg.get("skip") or []):
            geocode_files([Path(p) for p in paths], nspl)

    if a.geocode_only:
        for name in todo:
            d = settings.out_dir / name
            if d.is_dir():
                geocode(name, sorted(d.rglob("*.parquet")))
        log.info("geocoding done")
        return 0

    manifest_path = settings.out_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failed = []
    for name in todo:
        params = {k: v for k, v in (settings.sources.get(name) or {}).items() if k != "enabled"}
        t0 = time.perf_counter()
        log.info("=== %s ===", name)
        try:
            outs = sources[name].func(settings, params) or []
            geocode(name, outs)   # easting/northing for point tables (own, lat/long, or postcode centre)
            dt = time.perf_counter() - t0
            log.info("=== %s done in %.0fs: %d file(s)", name, dt, len(outs))
            manifest[name] = {
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(dt, 1),
                "keys": list(sources[name].keys),
                "licence": sources[name].licence,
                "files": [str(Path(p).relative_to(settings.out_dir)) for p in outs],
            }
        except Exception as e:
            failed.append(name)
            log.error("=== %s FAILED: %s", name, e)
            log.debug(traceback.format_exc())
            manifest.setdefault(name, {})["last_error"] = str(e)
        settings.out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    from geodata.dictionary import write_extract_dictionaries
    write_extract_dictionaries(settings.data_dir)   # data dictionary sections 1 (sources) and 2b (plottable files)
    if failed:
        log.error("failed: %s (see %s)", ", ".join(failed), log_file)
        return 1
    log.info("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
