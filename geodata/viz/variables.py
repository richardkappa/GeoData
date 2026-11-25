"""Variables page: every column of the section 2 postcode lookup and the section 3
features, grouped by theme, with description, source, coverage and type, and a
search box. Built from the dictionary CSVs that 02_collate.py and 03_features.py write."""
from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path

import pyarrow.csv as pacsv

from . import PAGE_CSS

log = logging.getLogger(__name__)

# (theme, regex on the section 2 `source` value)
THEMES = [
    ("Postcode spine and geographies", r"^ons_nspl|^ons_boundaries|^ons_nsul"),
    ("Census 2021 (England & Wales, output area)", r"^census_2021"),
    ("Deprivation (IMD 2025, England, LSOA)", r"^imd_england"),
    ("Crime (police, LSOA)", r"^police"),
    ("House sales (Land Registry)", r"^land_registry"),
    ("Energy use (DESNZ)", r"^energy_postcode"),
    ("Flood risk (Environment Agency)", r"^environment_agency"),
    ("Broadband and mobile (Ofcom)", r"^ofcom"),
    ("Health and care services", r"^nhs_ods|^cqc"),
    ("Food businesses (FSA)", r"^fsa"),
    ("Designations, heritage and land", r"^natural_england|^historic_england|^os_open/builtup"),
    ("Ground, radon and geology (BGS / UKHSA)", r"^os_open/(radon|gb_hex)"),
    ("Climate 1991-2020 (Met Office)", r"^metoffice"),
]


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pacsv.read_csv(path).to_pylist()


def _fmt_cov(c) -> str:
    try:
        return f"{float(c) * 100:.0f}%"
    except (TypeError, ValueError):
        return ""


def build_variables(data_dir: Path, out_dir: Path) -> Path | None:
    lookup = _rows(data_dir / "collate" / "postcode_lookup_dictionary.csv")
    feats = _rows(data_dir / "features" / "postcode_features_dictionary.csv")
    if not lookup and not feats:
        log.warning("variables: no dictionaries found - run 02_collate.py / 03_features.py")
        return None
    groups: dict[str, list] = {}
    for r in lookup:
        src = str(r.get("source") or "").replace("\\", "/")
        theme = next((t for t, rx in THEMES if re.search(rx, src)), "Other")
        groups.setdefault(("Section 2: postcode lookup", theme), []).append(
            (r["column"], r.get("note") or "", src, r.get("join") or "", r.get("type") or "", _fmt_cov(r.get("coverage"))))
    for r in feats:
        groups.setdefault(("Section 3: postcode features", str(r.get("group") or "other").replace("_", " ")), []).append(
            (r["column"], r.get("note") or "", "03_features.py", r.get("group") or "", "", _fmt_cov(r.get("coverage"))))
    order = [t for t, _ in THEMES] + ["Other"]
    keys = sorted(groups, key=lambda k: (k[0], order.index(k[1]) if k[1] in order else 99, k[1]))
    toc, body = [], []
    for i, (section, theme) in enumerate(keys):
        rows = groups[(section, theme)]
        toc.append(f'<li><a href="#g{i}">{html.escape(section.split(":")[0])}: {html.escape(theme)}</a> ({len(rows)})</li>')
        trs = "".join(f"<tr><td><code>{html.escape(c)}</code></td><td>{html.escape(n)}</td><td>{html.escape(s)}</td>"
                      f"<td>{html.escape(j)}</td><td>{html.escape(ty)}</td><td class=n>{cv}</td></tr>"
                      for c, n, s, j, ty, cv in rows)
        body.append(f'<section id="g{i}"><h2>{html.escape(section)} - {html.escape(theme)} <span>{len(rows)}</span></h2>'
                    f'<table><thead><tr><th>column</th><th>description</th><th>source</th><th>join / group</th>'
                    f'<th>type</th><th>coverage</th></tr></thead><tbody>{trs}</tbody></table></section>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>GeoData variables</title>
<style>{PAGE_CSS}
#q{{font-size:15px;padding:6px 10px;width:340px;margin-bottom:12px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin-bottom:22px}}
th,td{{border-bottom:1px solid #e5e5e5;padding:4px 8px;text-align:left;vertical-align:top}} th{{background:#eef1f3}}
td.n{{text-align:right}} h2{{font-size:16px;margin:18px 0 6px}} h2 span{{color:#888;font-weight:400}}
ul.toc{{columns:2;font-size:13px}} code{{font-size:12px}}
</style></head><body><header><h1>Variables</h1><p>{len(lookup)} columns in the section 2 postcode lookup
(<code>collate/postcode_lookup.parquet</code>, all postcodes) and {len(feats)} in the section 3 features
(<code>features/postcode_features.parquet</code>, live postcodes). Coverage = share of postcodes with a value.
<a href="index.html">All visualisations</a></p></header><main>
<input id="q" placeholder="search columns and descriptions..." oninput="filt()">
<ul class="toc">{''.join(toc)}</ul>{''.join(body)}</main>
<script>function filt(){{const q=document.getElementById('q').value.toLowerCase();
document.querySelectorAll('section').forEach(s=>{{let n=0;s.querySelectorAll('tbody tr').forEach(tr=>{{
const on=!q||tr.textContent.toLowerCase().includes(q);tr.style.display=on?'':'none';n+=on;}});s.style.display=n?'':'none';}});}}</script>
</body></html>"""
    out = out_dir / "variables.html"
    out.write_text(page, encoding="utf-8")
    log.info("wrote   %s (%d lookup + %d feature columns)", out.name, len(lookup), len(feats))
    return out
