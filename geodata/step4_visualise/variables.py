"""The data dictionary page (``4_visualise/variables.html``), laid out like the pipeline, every part collapsible:

1. Sources - publisher, licence, coverage, imputation, merge key, output files, dataset and guide
2. Outputs - 2a the postcode lookup's columns (taken from the sources); 2b the plottable (geocoded) files
3. Calculated variables - 3a the postcode features (with how each group is made); 3b map-only variables

Each part is also written as a CSV with the same columns (see geodata/dictionary.py). A search box filters every
table as you type and opens the sections with matches.
"""
from __future__ import annotations

import html
import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv

from . import PAGE_CSS

log = logging.getLogger(__name__)

URL = re.compile(r"(https?://[^\s;,]+)")
LABELS = {"commercial_use": "commercial use", "postcodes_with_data": "postcodes with data", "licence_note": "licence note",
          "merge_key": "merge key", "output_files": "output files", "join_level": "joined at", "size_mb": "size",
          "how_to_use": "how to use it on a map", "area_covered": "covers", "last_run": "last run"}


def _short(u: str) -> str:
    t = re.sub(r"^https?://(www\.)?", "", u).rstrip("/")
    return t if len(t) <= 48 else t[:45] + "..."


def _cell(v, col: str) -> str:
    if v is None:
        return ""
    if col == "coverage":
        try:
            return f"{float(v) * 100:.0f}%"
        except (TypeError, ValueError):
            return ""
    if col == "gaps" and len(str(v)) > 70:   # one short line in tables; the full text on hover and in the CSV
        full = html.escape(str(v))
        short = html.escape(re.split(r"; |, so |\(", str(v))[0][:70])
        return f'<span title="{full}">{short} <b class="more-i">+</b></span>'
    if col == "size_mb":
        try:
            return f"{float(v):,.0f} MB" if float(v) >= 10 else f"{float(v):.1f} MB"
        except (TypeError, ValueError):
            return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int):
        return f"{v:,}"
    t = html.escape(str(v))
    if col in ("column", "file", "source", "merge_key", "output_files"):
        return "<code>" + t.replace("; ", "</code><br><code>") + "</code>" if t else ""
    return URL.sub(lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener">{_short(m.group(1))}</a>', t)


def _table(rows: list[dict], cols: list[str]) -> str:
    head = "".join(f"<th>{html.escape(LABELS.get(c, c.replace('_', ' ')))}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f'<td class="c-{c}">{_cell(r.get(c), c)}</td>' for c in cols) + "</tr>"
                   for r in rows)
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _cards(rows: list[dict], title_col: str, sub_col: str, fields: list[str], more: list[str] = ()) -> str:
    """One card per row: a title, a line of description and labelled fields; `more` fields fold away."""
    out = []
    for r in rows:
        dl = "".join(f"<dt>{html.escape(LABELS.get(c, c.replace('_', ' ')))}</dt><dd>{_cell(r.get(c), c)}</dd>"
                     for c in fields if r.get(c) not in (None, ""))
        extra = "".join(f"<dt>{html.escape(LABELS.get(c, c.replace('_', ' ')))}</dt><dd>{_cell(r.get(c), c)}</dd>"
                        for c in more if r.get(c) not in (None, ""))
        flag = ' <span class="flag">commercial use needs a licence</span>' if r.get("commercial_use") == "needs a licence" else ""
        out.append(f'<div class="card"><h4>{_cell(r.get(title_col), "column")}{flag}</h4>'
                   f'<p>{html.escape(str(r.get(sub_col) or ""))}</p><dl>{dl}</dl>'
                   + (f'<details class="more"><summary>more</summary><dl>{extra}</dl></details>' if extra else "")
                   + "</div>")
    return f'<div class="cards">{"".join(out)}</div>'


def _groups(rows: list[dict], render, sec: str, notes: dict | None = None, order=None, shared_gaps=False) -> tuple[str, list]:
    """One collapsible block per theme; returns (html, [(anchor, theme, n)]) for the contents list. With
    shared_gaps, a gaps text every row of a theme shares is shown once under its heading instead of on each row."""
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r.get("theme") or "Other", []).append(r)
    keys = sorted(by, key=lambda t: (order.index(t) if order and t in order else 999, t))
    out, toc = [], []
    for i, t in enumerate(keys):
        rs = by[t]
        note = (notes or {}).get(t) or ""
        drop = False
        if shared_gaps:
            g = {r.get("gaps") or "" for r in rs}
            if len(g) == 1 and next(iter(g)):
                note = (note + " " if note else "") + "Known gaps: " + next(iter(g))
                drop = True
        anchor = f"s{sec}-{i}"
        out.append(f'<details class="grp" id="{anchor}"><summary>{html.escape(t)} <span class="n">{len(rs)}</span></summary>'
                   + (f'<p class="how">{_cell(note, "note")}</p>' if note else "") + render(rs, drop) + "</details>")
        toc.append((anchor, t, len(rs)))
    return "".join(out), toc


def _section(num: str, title: str, intro: str, inner: str, count: int, cls: str = "sec") -> str:
    return (f'<details class="{cls}" id="sec-{num}" data-sec="{num}"><summary><b>{num}</b> {html.escape(title)} '
            f'<span class="n">{count:,}</span></summary><p class="intro">{intro}</p>{inner}</details>')


def build_variables(data_dir: Path, out_dir: Path, vcfg: dict | None = None) -> Path | None:
    from ..dictionary import (COLUMNS, FEATURE_METHODS, FEATURE_THEMES, FEATURES, LOOKUP, LOOKUP_THEMES, MAP_COLUMNS,
                              SOURCE_COLUMNS, SPATIAL_COLUMNS, from_legacy_csv, map_variables_table, sources_table,
                              spatial_table)
    read = lambda f, sec: from_legacy_csv(pacsv.read_csv(f), sec).to_pylist() if f.exists() else []
    lookup = read(data_dir / "2_collate" / "postcode_lookup_dictionary.csv", LOOKUP)
    feats = read(data_dir / "3_features" / "postcode_features_dictionary.csv", FEATURES)
    if not lookup and not feats:
        log.warning("dictionary: no column dictionaries found - run 02_collate.py / 03_features.py")
        return None
    # sections 1, 2b and 3b are rebuilt from the catalogue, the extract folder and the map config each time
    srcs, spatial = sources_table(data_dir), spatial_table(data_dir)
    maps = map_variables_table(vcfg or {})
    pacsv.write_csv(srcs, data_dir / "1_extract" / "sources_dictionary.csv")
    pacsv.write_csv(spatial, data_dir / "1_extract" / "spatial_dictionary.csv")
    pacsv.write_csv(maps, out_dir / "map_variables_dictionary.csv")
    calc = spatial_table(data_dir, "3_features")      # calculated plottable files (e.g. the reach squares)
    pacsv.write_csv(calc, data_dir / "3_features" / "plottable_dictionary.csv")
    calc = calc.to_pylist()
    srcs, spatial, maps = srcs.to_pylist(), spatial.to_pylist(), maps.to_pylist()

    feature_methods = {FEATURE_THEMES[g][0]: m for g, m in FEATURE_METHODS.items() if g in FEATURE_THEMES}
    colcols = [c for c in COLUMNS if c not in ("section", "theme")]
    coltable = lambda rs, drop: _table(rs, [c for c in colcols if not (drop and c == "gaps")])
    g1, t1 = _groups(srcs, lambda rs, d: _cards(rs, "source", "description",
                     ["website", "licence", "commercial_use", "licence_note", "area_covered", "postcodes_with_data",
                      "imputation", "merge_key", "dataset", "guide"], ["output_files", "last_run"]), "1")
    g2a, t2a = _groups(lookup, coltable, "2a", order=[t for t, _ in LOOKUP_THEMES], shared_gaps=True)
    g2b, t2b = _groups(spatial, lambda rs, d: _cards(rs, "file", "description",
                       ["kind", "geometry", "rows", "size_mb", "area_covered", "gaps", "source"],
                       ["how_to_use", "crs", "columns"]), "2b")
    g3a, t3a = _groups(feats, coltable, "3a", feature_methods, [t for t, _ in FEATURE_THEMES.values()], shared_gaps=True)
    g3b, t3b = _groups(maps, coltable, "3b", shared_gaps=True)
    g3c, t3c = _groups([{**r, "theme": "Calculated files: " + r["theme"].lower()} for r in calc],
                       lambda rs, d: _cards(rs, "file", "description", ["kind", "geometry", "rows", "size_mb", "source"],
                                            ["how_to_use", "crs", "columns"]), "3c")
    s1 = _section("1", "Sources", "Every source the pipeline downloads (step 1): publisher, licence, what it covers, how "
                  "much of the postcode lookup it fills, anything the pipeline fills in (imputation), the key it joins "
                  "on, and links to the exact dataset and its guide; <i>more</i> shows the files it writes. "
                  "File: <code>1_extract/sources_dictionary.csv</code>.", g1, len(srcs))
    s2a = _section("2a", "Postcode lookup: columns taken from the sources",
                   "One row per UK postcode (<code>2_collate/postcode_lookup.parquet</code>, step 2): each source's values "
                   "joined to the postcode at the level shown (its own postcode, an area it is in, or the polygon its "
                   "point falls in). Known gaps shared by a whole theme are shown once under its heading. File: "
                   "<code>2_collate/postcode_lookup_dictionary.csv</code>.", g2a, len(lookup), "sub")
    s2b = _section("2b", "Plottable data: points, polygons, lines, grids and the road network",
                   "The geocoded files in <code>&lt;data_dir&gt;/1_extract/</code> that can go on a map, all on the British "
                   "National Grid (EPSG:27700); <i>more</i> shows how to open each in GIS software or Python and its "
                   "columns. File: <code>1_extract/spatial_dictionary.csv</code>.", g2b, len(spatial), "sub")
    s3a = _section("3a", "Calculated Postcode Features", "Calculated for every live postcode (step 3, "
                   "<code>3_features/postcode_features.parquet</code>); how each group is made is under its heading. "
                   "File: <code>3_features/postcode_features_dictionary.csv</code>.", g3a, len(feats), "sub")
    s3b = _section("3b", "Calculated Plottable Features", "Calculated data that can be mapped: files step 3 writes "
                   "(the 500 m squares holding what can be reached within N minutes, "
                   "<code>3_features/plottable_dictionary.csv</code>), and variables worked out for each area when the area "
                   "maps are built (step 4): rates per head (points placed on their nearest postcode, totalled and "
                   "divided by the area's residents, households or addresses) and variables derived from other columns "
                   "(<code>4_visualise/map_variables_dictionary.csv</code>).", g3c + g3b, len(maps) + len(calc), "sub")
    body = (s1 + f'<details class="sec" id="sec-2" data-sec="2"><summary><b>2</b> Outputs <span class="n">'
            f'{len(lookup) + len(spatial):,}</span></summary>{s2a}{s2b}</details>'
            f'<details class="sec" id="sec-3" data-sec="3"><summary><b>3</b> Calculated variables <span class="n">'
            f'{len(feats) + len(maps) + len(calc):,}</span></summary>{s3a}{s3b}</details>')
    nav = ""
    for num, title, toc in (("1", "Sources", t1), ("2a", "Postcode lookup", t2a), ("2b", "Plottable data", t2b),
                            ("3a", "Calculated Postcode Features", t3a),
                            ("3b", "Calculated Plottable Features", t3c + t3b)):
        items = "".join(f'<li><a href="#{a}" data-go="{a}">{html.escape(t)}</a> <span class="n">{n}</span></li>'
                        for a, t, n in toc)
        nav += f'<h5><a href="#sec-{num}" data-go="sec-{num}"><b>{num}</b> {title}</a></h5><ul>{items}</ul>'
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>GeoData data dictionary</title>
<style>{PAGE_CSS}
#wrapd{{display:grid;grid-template-columns:250px 1fr;gap:18px;align-items:start}}
nav#toc{{position:sticky;top:64px;max-height:calc(100vh - 74px);overflow:auto;font-size:12.5px;background:#fff;border-radius:6px;
  box-shadow:0 1px 3px #0002;padding:8px 10px}}
nav#toc h5{{font-size:13px;margin:10px 0 3px}} nav#toc ul{{list-style:none;margin:0;padding:0 0 0 8px}} nav#toc li{{margin:2px 0}}
nav#toc a{{color:#1d5f8a;text-decoration:none}} nav#toc a:hover{{text-decoration:underline}}
#tools{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px;position:sticky;top:0;background:#f6f6f3;padding:8px 0;z-index:3}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px;margin:6px 0 12px}}
.card{{border:1px solid #e3e3e0;border-radius:6px;padding:8px 10px;background:#fcfcfb}}
.card h4{{margin:0 0 2px;font-size:14px}} .card p{{margin:0 0 6px;font-size:12.5px;color:#333}}
.card dl{{display:grid;grid-template-columns:130px 1fr;gap:2px 8px;margin:0;font-size:12px}}
.card dt{{color:#666}} .card dd{{margin:0}} .card details.more summary{{font-size:12px;color:#1d5f8a;cursor:pointer;margin-top:4px}}
.flag{{background:#fde2e1;color:#a4161a;font-size:11px;font-weight:600;padding:1px 6px;border-radius:8px;margin-left:6px}}
.tw{{overflow-x:auto}} td.c-description{{min-width:260px}} td.c-gaps{{font-size:11.5px;color:#7a4b00;min-width:200px;max-width:280px}} .more-i{{color:#1d5f8a;cursor:help}}
@media (max-width:900px){{#wrapd{{grid-template-columns:1fr}} nav#toc{{position:static;max-height:none}} .cards{{grid-template-columns:1fr}}}}
#q{{font-size:15px;padding:6px 10px;width:380px;max-width:100%}} #tools select,#tools button{{font-size:14px;padding:5px 8px}}
#hits{{color:#666;font-size:13px}}
details.sec{{background:#fff;border-radius:6px;box-shadow:0 1px 3px #0002;margin-bottom:12px;padding:4px 12px}}
details.sec>summary{{font-size:18px;padding:8px 0;cursor:pointer}} details.sub>summary{{font-size:16px;padding:6px 0;cursor:pointer}}
details.sub{{margin:4px 0 4px 14px}} details.grp{{margin:2px 0 2px 28px}} details.grp>summary{{cursor:pointer;padding:3px 0;font-size:14px}}
summary .n{{color:#888;font-weight:400;font-size:13px}} p.intro{{font-size:13px;color:#444;margin:2px 0 8px}}
p.how{{font-size:12.5px;color:#355;background:#f2f7f6;border-left:3px solid #6aa;padding:5px 8px;margin:4px 0 6px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:4px 0 12px}}
th,td{{border-bottom:1px solid #e8e8e8;padding:4px 7px;text-align:left;vertical-align:top}} th{{background:#eef1f3}}
td.c-coverage,td.c-rows,td.c-size_mb{{text-align:right;white-space:nowrap}} td.c-columns{{font-size:11px;color:#555;max-width:340px}}
td.c-how_to_use{{font-size:11.5px;max-width:360px}} code{{font-size:12px}} td a{{word-break:break-all}}
mark{{background:#ffe48a;padding:0}}
</style></head><body><header><h1>Data dictionary</h1><p>Every source, file and column the pipeline produces, in
pipeline order: <b>1</b> the sources; <b>2</b> the outputs - the postcode lookup's columns and the plottable files;
<b>3</b> calculated variables - postcode features and plottable features. Click a heading to open it, or search.
<a href="index.html">All visualisations</a></p></header><main>
<div id="tools"><input id="q" placeholder="search sources, files and columns (all words must match)..." oninput="filt()">
<select id="sec" onchange="filt()"><option value="">all sections</option><option value="1">1 Sources</option>
<option value="2">2 Outputs</option><option value="2a">2a Postcode lookup</option><option value="2b">2b Plottable data</option>
<option value="3">3 Calculated variables</option><option value="3a">3a Calculated Postcode Features</option>
<option value="3b">3b Calculated Plottable Features</option></select>
<button type="button" onclick="openAll(true)">Open all</button><button type="button" onclick="openAll(false)">Close all</button>
<span id="hits"></span></div>
<div id="wrapd"><nav id="toc"><b>Contents</b>{nav}</nav><div>{body}</div></div></main>
<script>
const secs = [...document.querySelectorAll('details[data-sec]')];
function openAll(on) {{ document.querySelectorAll('details').forEach(d => {{ if (d.style.display !== 'none') d.open = on; }}); }}
function filt() {{
  const words = document.getElementById('q').value.toLowerCase().split(/\\s+/).filter(Boolean);
  const pick = document.getElementById('sec').value;
  let hits = 0;
  secs.forEach(s => {{ const id = s.dataset.sec;
    s.style.display = !pick || id === pick || id.startsWith(pick) || pick.startsWith(id) ? '' : 'none'; }});
  document.querySelectorAll('details.grp').forEach(g => {{
    const inSec = g.closest('details[data-sec]');
    let n = 0;
    g.querySelectorAll('tbody tr, .card').forEach(tr => {{
      const t = tr.textContent.toLowerCase(), on = words.every(w => t.includes(w));
      tr.style.display = on ? '' : 'none'; n += on; }});
    const visible = n > 0 && inSec.style.display !== 'none';
    g.style.display = n || !words.length ? '' : 'none';
    if (words.length) g.open = visible;
    hits += visible ? n : 0; }});
  document.querySelectorAll('details[data-sec]').forEach(s => {{
    if (s.style.display === 'none') return;
    const any = [...s.querySelectorAll('details.grp')].some(g => g.style.display !== 'none');
    if (words.length) {{ s.open = any; if (!any && s.classList.contains('sub')) s.style.display = 'none'; }} }});
  if (pick) document.querySelectorAll('details[data-sec]').forEach(s => {{ if (s.style.display !== 'none') s.open = true; }});
  document.getElementById('hits').textContent = words.length ? hits.toLocaleString('en-GB') + ' matches' : '';
}}
// contents: open the target and every section around it, then scroll to it
document.querySelectorAll('[data-go]').forEach(a => a.addEventListener('click', e => {{
  const t = document.getElementById(a.dataset.go); if (!t) return;
  e.preventDefault();
  for (let d = t; d; d = d.parentElement) if (d.tagName === 'DETAILS') d.open = true;
  t.scrollIntoView({{behavior: 'smooth', block: 'start'}}); history.replaceState(null, '', '#' + a.dataset.go); }}));
const h = decodeURIComponent(location.hash.slice(1));
if (h && document.getElementById(h)) {{ const t = document.getElementById(h);
  for (let d = t; d; d = d.parentElement) if (d.tagName === 'DETAILS') d.open = true; t.scrollIntoView(); }}
else if (h) {{ document.getElementById('q').value = h; filt(); }}
</script></body></html>"""
    out = out_dir / "variables.html"
    out.write_text(page, encoding="utf-8")
    log.info("wrote   %s (%d sources, %d lookup columns, %d plottable files, %d features, %d map variables)",
             out.name, len(srcs), len(lookup), len(spatial), len(feats), len(maps))
    return out
