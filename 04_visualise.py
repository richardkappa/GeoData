#!/usr/bin/env python
"""Section 4 - visualisations, written as self-contained HTML pages.

* mini maps: a 5 km square around each key location with roads, water,
  greenspace, heritage and stations (``4_visualise/minimaps.html``)
* cartograms: features averaged to constituencies / local authorities and
  drawn as equal hexagons or squares, MSOA hexes included (``4_visualise/cartograms.html``)
* choropleths: the same averaged to real boundaries (constituencies, local
  authorities, MSOAs) and OS national grid squares (``4_visualise/choropleths.html``)
* heat maps: where things are concentrated, rates per head and access values (``4_visualise/heatmaps.html``)
* networks: road and area networks around each key location (data for the map explorer)
* explorer: the map explorer - map layers, postcode features and road network views for each
  key location on one page (``4_visualise/explorer.html``)
* dynamic: morphing (Gastner-Newman) and Dorling cartograms (``4_visualise/dynamic_cartograms.html``)
* variables: the data dictionary - every column, by theme (``4_visualise/variables.html``)

Configured in ``config/4_visualise.yaml``. Open ``<data_dir>/4_visualise/index.html``.

    python 04_visualise.py
    python 04_visualise.py --only minimaps
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import yaml

from geodata.config import REPO_ROOT
from geodata.step4_visualise import PAGE_CSS, label, resolve_locations

STEPS = ("minimaps", "cartograms", "choropleths", "heatmaps", "networks", "explorer", "dynamic", "variables")


def steps_explorer(steps) -> bool:
    """The explorer page is rewritten whenever its data (mini maps, networks) is."""
    return bool({"explorer", "minimaps", "networks"} & set(steps))


NAV = re.compile(r'<a href="index\.html">(?:All visualisations|GeoData home)</a>(?: \| <a class="(?:site|repo)"[^>]*>[^<]*</a>)*')


def _add_repo_link(out_dir: Path, url: str | None, site: str | None = None) -> None:
    """Every page header links to the GeoData home page (index.html), the main website (`global.site_url`) and the
    code on GitHub (`global.repo_url`). Re-running replaces the links rather than adding more."""
    import html as _html
    parts = ['<a href="index.html">GeoData home</a>']
    if site:
        name = re.sub(r"^https?://(www\.)?", "", site).rstrip("/")
        parts.append(f'<a class="site" href="{_html.escape(site)}">{_html.escape(name)}</a>')
    if url:
        parts.append(f'<a class="repo" href="{_html.escape(url)}">Code on GitHub</a>')
    nav = " | ".join(parts)
    n = 0
    for p in out_dir.glob("*.html"):
        s = p.read_text(encoding="utf-8")
        new = NAV.sub(lambda m: nav, s, count=1)
        if new != s:
            p.write_text(new, encoding="utf-8")
            n += 1
    logging.getLogger("visualise").info("header links on %d pages", n)


def _postcode_variables(data_dir: Path, variables: dict) -> dict:
    """Only the variables that exist as postcode columns (step 2 lookup or step 3 features): area rates per head and
    derived variables are worked out for areas, so they are left out of the postcode overlays."""
    import pyarrow.parquet as pq
    have = set()
    for f in (data_dir / "2_collate" / "postcode_lookup.parquet", data_dir / "3_features" / "postcode_features.parquet"):
        if f.exists():
            have |= set(pq.read_schema(f).names)
    return {c: v for c, v in variables.items() if c in have}


def _merge(base: dict, over: dict) -> dict:
    """Profile overrides: nested dicts merge; anything else (lists included) replaces."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "4_visualise.yaml"))
    ap.add_argument("--data-dir")
    ap.add_argument("--only", nargs="+", choices=STEPS)
    ap.add_argument("--profile", help="apply a `profiles:` override from the config, e.g. web (copy for a public site)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.profile:
        prof = (cfg.get("profiles") or {}).get(a.profile)
        if prof is None:
            ap.error(f"no profile {a.profile!r} under `profiles:` in {a.config}")
        cfg = _merge(cfg, prof)
    g = cfg.get("global", {})
    data_dir = Path(a.data_dir or os.environ.get("GEODATA_DIR") or g.get("data_dir", "data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    out_dir = data_dir / g.get("output_dir", "4_visualise")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(data_dir / "visualise.log")])
    log = logging.getLogger("visualise")
    t0 = time.perf_counter()

    import geodata.step4_visualise as viz
    viz.BASEMAPS.update(g.get("basemaps") or {})
    viz.DERIVED.update(cfg.get("derived") or {})
    locs = resolve_locations(data_dir, cfg.get("locations", {}))
    log.info("%d locations", len(locs))
    steps = a.only or STEPS
    if "minimaps" in steps and cfg.get("minimaps"):
        from geodata.step4_visualise.minimaps import build_minimaps
        build_minimaps(data_dir, out_dir, locs, cfg["minimaps"],
                       # the explorer's postcode overlay: the same metrics as the area maps (the postcode-level ones -
                       # area rates and derived variables are skipped), plus explorer-only extras from minimaps.variables
                       variables=_postcode_variables(data_dir, {**((cfg.get("cartograms") or {}).get("variables") or {}),
                                                                **(cfg["minimaps"].get("variables") or {})}),
                       area_variables=(cfg.get("cartograms") or {}).get("variables"))   # same list as the choropleths
    if "cartograms" in steps and cfg.get("cartograms"):
        from geodata.step4_visualise.cartograms import build_cartograms
        build_cartograms(data_dir, out_dir, cfg["cartograms"])
    if "choropleths" in steps and cfg.get("choropleths"):
        from geodata.step4_visualise.choropleths import build_choropleths
        build_choropleths(data_dir, out_dir, cfg["choropleths"])
    if "heatmaps" in steps and cfg.get("heatmaps"):
        from geodata.step4_visualise.heatmaps import build_heatmaps
        build_heatmaps(data_dir, out_dir, cfg["heatmaps"])
    if "networks" in steps and cfg.get("networks"):
        from geodata.step4_visualise.networks import build_networks
        build_networks(data_dir, out_dir, locs, cfg["networks"])
    if steps_explorer(steps):   # one page for the mini map and network data
        from geodata.step4_visualise.explorer import build_explorer
        build_explorer(out_dir, float((cfg.get("minimaps") or {}).get("size_m", 5000)))
    if "dynamic" in steps and cfg.get("dynamic"):
        from geodata.step4_visualise.dynamic import build_dynamic
        build_dynamic(data_dir, out_dir, cfg["dynamic"])
    if "variables" in steps:
        from geodata.step4_visualise.variables import build_variables
        build_variables(data_dir, out_dir, cfg)   # the data dictionary (map-only variables from cfg)

    # index of whatever pages exist
    links = []
    if (out_dir / "minimaps.html").exists():
        links.append('<li><a href="minimaps.html">Mini maps</a>: 5 km squares around every key location, as '
                     'static maps (including the national hotspots for collisions within a 30-minute drive and for takeaways)</li>')
    if (out_dir / "explorer.html").exists():
        links.append('<li><a href="explorer.html">Map explorer</a>: the mini map squares, interactive. Tick map layers '
                     '(roads, water, greenspace, heritage, stations, takeaways, GP surgeries), plot a postcode feature as '
                     'dots, a smooth surface or LSOA areas, and add a road network view: through-routes, walking and '
                     'driving times, routes to the nearest station, collision hotspots, collisions within a 30-minute '
                     'drive, the junction graph and the LSOA neighbour network</li>')
    if (out_dir / "cartograms.html").exists():
        links.append('<li><a href="cartograms.html">Cartograms</a>: constituencies and local authorities as equal '
                     'hexagons, plus the Ordnance Survey equal-area square cartogram</li>')
    if (out_dir / "choropleths.html").exists():
        links.append('<li><a href="choropleths.html">Choropleth maps</a>: constituencies, local authorities, MSOAs '
                     'and OS national grid squares (5, 10, 25, 50 km) on a street map</li>')
    if (out_dir / "dynamic_cartograms.html").exists():
        links.append('<li><a href="dynamic_cartograms.html">Dynamic cartograms</a>: areas resized by the people who live '
                     'there - a shape-keeping cartogram that morphs from the real map, and Dorling circles</li>')
    if (out_dir / "small_multiples.html").exists():
        links.append('<li><a href="small_multiples.html">Small multiples</a>: nine variables side by side on the same '
                     'cartogram</li>')
    if (out_dir / "heatmaps.html").exists():
        links.append('<li><a href="heatmaps.html">Heat maps</a>: where things are concentrated (crime, collisions, '
                     'listed buildings, food outlets, house sales...)</li>')
    if (out_dir / "variables.html").exists():
        links.append('<li><a href="variables.html">Data dictionary</a>: every column in the postcode lookup and '
                     'features, by theme, with description, source, method, type and coverage</li>')
    for old in out_dir.glob("interactive_*.html"):   # retired: the map explorer's postcode overlay replaces them
        old.unlink()
    import html as _html
    links = [l for l in links if 'href="variables.html"' not in l]   # the data dictionary has its own block
    cards = "".join(
        f'<a class="card" href="{_html.escape(f["href"])}"><b>{_html.escape(f["title"])}</b>'
        f'<span>{_html.escape(f.get("text", ""))}</span></a>'
        for f in cfg.get("featured") or [] if (out_dir / f["href"].split("#")[0]).exists())
    dd = [("sec-1", "1", "Sources", "every dataset: publisher, licence, coverage, gaps, links to the data and its guide"),
          ("sec-2a", "2a", "Postcode lookup", "the columns joined to every UK postcode"),
          ("sec-2b", "2b", "Plottable data", "points, boundaries, lines, grids and the road network, ready for GIS"),
          ("sec-3a", "3a", "Calculated Postcode Features", "distances, densities, travel times, trends for each postcode"),
          ("sec-3b", "3b", "Calculated Plottable Features", "reach squares and the rates per head used on the maps")]
    dict_html = "".join(f'<a class="dd" href="variables.html#{a}"><b>{n}</b> {t}<span>{x}</span></a>'
                        for a, n, t, x in dd) if (out_dir / "variables.html").exists() else ""
    title = "Open source UK geographic data collation and visualisations"
    (out_dir / "index.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>'
        f'<style>{PAGE_CSS} main{{max-width:1200px}} li{{margin:6px 0}} h2{{font-size:18px;margin:22px 0 8px}}'
        f'p.lead{{font-size:15px;line-height:1.55;max-width:900px;margin:0 0 6px}}'
        f'.dds{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}'
        f'.dd{{background:#fff;border-radius:6px;padding:10px 12px;box-shadow:0 1px 3px #0002;text-decoration:none;color:#222;'
        f'border-top:3px solid #1d5f8a}} .dd b{{color:#1d5f8a;margin-right:4px}} .dd span{{display:block;font-size:12.5px;'
        f'color:#555;margin-top:4px}} .dd:hover,.card:hover{{box-shadow:0 2px 8px #0003}}'
        f'.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}'
        f'.card{{background:#fff;border-radius:6px;padding:12px;box-shadow:0 1px 3px #0002;text-decoration:none;color:#222}}'
        f'.card b{{display:block;margin-bottom:4px;color:#1d5f8a}} .card span{{font-size:13px;color:#555}}'
        f'.credits{{font-size:12px;color:#666;margin-top:26px}}'
        f'@media (max-width:1000px){{.dds{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body>'
        f'<header><h1>{title}</h1><p><a href="index.html">GeoData home</a></p></header><main>'
        f'<p class="lead">This project collates as many open UK datasets as possible into one place, starting with data '
        f'published by UK government bodies, and links every one of them to the postcode. It has two main outputs: '
        f'<b>(2a)</b> a postcode lookup - one row for every UK postcode with the values from each source, such as area '
        f'statistics, the boundaries and designations it falls in and the services near it - and <b>(2b)</b> the '
        f'plottable data itself (points, boundaries, lines, grids and the road network), ready to open in GIS '
        f'software. The maps below are examples that investigate it.</p>'
        f'<h2>Data dictionary</h2><div class="dds">{dict_html}</div>'
        f'<h2>Featured maps</h2><div class="cards">{cards}</div>'
        f'<h2>All map pages</h2><ul>{"".join(links)}</ul>'
        f'<p class="credits">Sources include ONS, OS, DfT, Defra, Forest Research, ORR, MHCLG, NHS, Land Registry and '
        f'others, mostly under the Open Government Licence (see the data dictionary for each). Sports facilities: '
        f'Contains Data © Sport England (CC BY 4.0).</p></main></body></html>',
        encoding="utf-8")
    _add_repo_link(out_dir, g.get("repo_url"), g.get("site_url"))
    from geodata.step4_visualise.mobile import add_mobile
    log.info("phone layout on %d pages", add_mobile(out_dir))
    log.info("visualisations written to %s in %.0fs", out_dir, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
