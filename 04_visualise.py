#!/usr/bin/env python
"""Section 4 - visualisations, written as self-contained HTML pages.

* mini maps: a 5 km square around each key location with roads, water,
  greenspace, heritage and stations (``viz/minimaps.html``)
* interactive: per area, every postcode on a street map coloured by a chosen
  feature from sections 2 and 3 (``viz/interactive_<area>.html``)
* cartograms: features averaged to constituencies / local authorities and
  drawn as equal hexagons or squares, MSOA hexes included (``viz/cartograms.html``)
* choropleths: the same averaged to real boundaries (constituencies, local
  authorities, MSOAs) and OS national grid squares (``viz/choropleths.html``)

Configured in ``config/visualise.yaml``. Open ``<data_dir>/viz/index.html``.

    python 04_visualise.py
    python 04_visualise.py --only minimaps
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from geodata.config import REPO_ROOT
from geodata.viz import PAGE_CSS, label, resolve_locations

STEPS = ("minimaps", "interactive", "cartograms", "choropleths", "heatmaps", "variables")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "visualise.yaml"))
    ap.add_argument("--data-dir")
    ap.add_argument("--only", nargs="+", choices=STEPS)
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    g = cfg.get("global", {})
    data_dir = Path(a.data_dir or os.environ.get("GEODATA_DIR") or g.get("data_dir", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    out_dir = data_dir / g.get("output_dir", "viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(data_dir / "visualise.log")])
    log = logging.getLogger("visualise")
    t0 = time.perf_counter()

    import geodata.viz as viz
    viz.BASEMAPS.update(g.get("basemaps") or {})
    locs = resolve_locations(data_dir, cfg.get("locations", {}))
    log.info("%d locations", len(locs))
    steps = a.only or STEPS
    if "minimaps" in steps and cfg.get("minimaps"):
        from geodata.viz.minimaps import build_minimaps
        build_minimaps(data_dir, out_dir, locs, cfg["minimaps"],
                       variables=(cfg.get("interactive") or {}).get("variables"),
                       area_variables=(cfg.get("cartograms") or {}).get("variables"))   # same list as the choropleths
    if "interactive" in steps and cfg.get("interactive"):
        from geodata.viz.interactive import build_interactive
        build_interactive(data_dir, out_dir, locs, cfg["interactive"])
    if "cartograms" in steps and cfg.get("cartograms"):
        from geodata.viz.cartograms import build_cartograms
        build_cartograms(data_dir, out_dir, cfg["cartograms"])
    if "choropleths" in steps and cfg.get("choropleths"):
        from geodata.viz.choropleths import build_choropleths
        build_choropleths(data_dir, out_dir, cfg["choropleths"])
    if "heatmaps" in steps and cfg.get("heatmaps"):
        from geodata.viz.heatmaps import build_heatmaps
        build_heatmaps(data_dir, out_dir, cfg["heatmaps"])
    if "variables" in steps:
        from geodata.viz.variables import build_variables
        build_variables(data_dir, out_dir)

    # index of whatever pages exist
    links = []
    if (out_dir / "minimaps.html").exists():
        links.append('<li><a href="minimaps.html">Mini maps</a>: 5 km squares around every key location</li>')
    if (out_dir / "minimap_explorer.html").exists():
        links.append('<li><a href="minimap_explorer.html">Mini map explorer</a>: pick a location, the map layers '
                     'and a postcode feature to plot over it</li>')
    if (out_dir / "cartograms.html").exists():
        links.append('<li><a href="cartograms.html">Cartograms</a>: constituencies and local authorities as equal '
                     'hexagons, plus the Ordnance Survey equal-area square cartogram</li>')
    if (out_dir / "choropleths.html").exists():
        links.append('<li><a href="choropleths.html">Choropleth maps</a>: constituencies, local authorities, MSOAs '
                     'and OS national grid squares (5, 10, 25, 50 km) on a street map</li>')
    if (out_dir / "heatmaps.html").exists():
        links.append('<li><a href="heatmaps.html">Heat maps</a>: where things are concentrated (crime, collisions, '
                     'listed buildings, food outlets, house sales...)</li>')
    if (out_dir / "variables.html").exists():
        links.append('<li><a href="variables.html">Variables</a>: every column in the postcode lookup and features, '
                     'with descriptions and coverage</li>')
    inter = sorted(out_dir.glob("interactive_*.html"))
    if inter:
        items = "".join(f'<li><a href="{p.name}">{label(p.stem.removeprefix("interactive_"))}</a></li>' for p in inter)
        links.append(f"<li>Interactive postcode features<ul>{items}</ul></li>")
    (out_dir / "index.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>GeoData visualisations</title>'
        f'<style>{PAGE_CSS} li{{margin:6px 0}}</style></head><body><header><h1>GeoData visualisations</h1>'
        f'<p>UK open data by postcode</p></header><main><ul>{"".join(links)}</ul></main></body></html>',
        encoding="utf-8")
    log.info("visualisations written to %s in %.0fs", out_dir, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
