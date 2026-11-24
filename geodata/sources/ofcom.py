"""Ofcom Connected Nations: fixed broadband and mobile coverage by postcode.

Ofcom publishes, on its Connected Nations "data downloads" page, postcode-level

* **fixed broadband coverage** (premises able to get superfast / ultrafast /
  full fibre / gigabit, and those below the USO), and
* **mobile coverage** (premises / outdoor coverage by operator and technology,
  4G / 5G, indoor and outdoor),

each as a zip of CSVs per postcode area. The page link is configurable because
the URL changes each release.

Ofcom's website sits behind a Cloudflare browser check that can refuse
automated downloads from some networks (cloud servers especially). If that
happens the extractor says so; then either set ``url:`` / ``mobile_url:`` to
the zip's direct link, or download the zips in a browser and drop them into
``<data_dir>/raw/ofcom/``. A zip whose name matches ``mobile_file_regex``
(default ``mobile``) is treated as mobile coverage; any other zip as fixed.

Every CSV in a product's zip(s) that has a postcode column is read; columns are
snake_cased, numeric columns typed automatically and prefixed ``bb_`` (fixed)
or ``mob_`` (mobile). Each product is written to its own postcode-keyed file.
"""
from __future__ import annotations

import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import pyarrow as pa
import pyarrow.compute as pc

from .. import download as dl
from .. import io
from ..postcode import POSTCODE_KEY, add_postcode_ns
from . import source

log = logging.getLogger(__name__)

PAGE = "https://www.ofcom.org.uk/phones-and-broadband/coverage-and-speeds/connected-nations-2025/data-downloads-2025"


def _find_links(page: str, regex: str, exclude: str | None = None) -> list[str]:
    html = dl.get_text(page)
    if "Just a moment" in html[:2000]:
        raise PermissionError("Ofcom page returned a Cloudflare browser check")
    links = re.findall(r'href="([^"]+\.zip[^"]*)"', html, re.I)
    rx = re.compile(regex, re.I)
    ex = re.compile(exclude, re.I) if exclude else None
    return [urljoin(page, l.replace("&amp;", "&")) for l in links
            if rx.search(l) and not (ex and ex.search(l))]


def _postcode_csvs(zf: zipfile.ZipFile):
    """Yield the bytes of every CSV with a postcode column in a zip, recursing into nested
    zips (Ofcom ships the postcode files as a zip inside the release zip)."""
    for m in zf.namelist():
        low = m.lower()
        if low.endswith(".zip"):
            with zipfile.ZipFile(BytesIO(zf.read(m))) as inner:
                yield from _postcode_csvs(inner)
        elif low.endswith(".csv"):
            with zf.open(m) as f:
                header = f.readline().decode("utf-8-sig", "replace").lower()
            if re.search(r"postcode|\bpcds?\b", header):   # skip area-level tables without parsing them
                yield m.rsplit("/", 1)[-1], zf.read(m)


# Ofcom ships an all-premises and a residential-only file per postcode area
# (..._pc_coverage_r01_AB.csv / ..._pc_coverage_res_r01_AB.csv): same columns,
# different values. Residential columns get a "res_" prefix and are joined on.
RES_FILE = re.compile(r"(?i)[_-](res|residential)[_-]")


def _read_postcode_zips(zips) -> pa.Table | None:
    groups: dict[str, list] = {"": [], "res_": []}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            csvs = list(_postcode_csvs(zf))
        for fname, data in csvs:
            parts = groups["res_" if RES_FILE.search(fname) else ""]
            t = io.read_csv(BytesIO(data), strings=True, encoding="utf-8-sig")
            pcol = next((c for c in t.column_names if re.fullmatch(r"(?i)postcode(_space)?|pcds|pcd", c.strip())),
                        None) or next((c for c in t.column_names if "postcode" in c.lower()), None)
            if not pcol:
                continue
            t = add_postcode_ns(t, pcol, validate=True)
            t = t.drop_columns([c for c in t.column_names if c != POSTCODE_KEY and "postcode" in c.lower()])
            parts.append(t)
    tables = {}
    for pre, parts in groups.items():
        if parts:
            t = pa.concat_tables(parts, promote_options="permissive")
            t = t.filter(pc.is_valid(t[POSTCODE_KEY]))
            tables[pre] = t.rename_columns([c if c == POSTCODE_KEY else f"{pre}{c}" for c in t.column_names])
    if not tables:
        return None
    base = tables.get("") or tables.get("res_")
    if "" in tables and "res_" in tables:   # residential columns onto the all-premises rows
        res = tables["res_"]
        idx = pc.index_in(base[POSTCODE_KEY], value_set=res[POSTCODE_KEY])
        for c in res.column_names:
            if c != POSTCODE_KEY:
                base = base.append_column(c, res[c].take(idx))
    return base


def _product(settings, params, raw, name, zips, urls_found, prefix, out):
    """Download (if links found), read and write one product. Returns the output path or None."""
    if io.is_fresh(out, settings.reprocess or settings.force):
        return out
    for u in urls_found:
        try:
            dl.download(u, raw / re.sub(r"[?#].*$", "", u.rsplit("/", 1)[-1]), force=settings.force,
                        timeout=max(settings.timeout, 600))
        except Exception as e:
            log.warning("ofcom %s: download failed for %s (%s)", name, u, e)
    zs = zips()
    if not zs:
        log.warning("ofcom %s: no zip in %s - download it in a browser and place it there", name, raw)
        return None
    t = _read_postcode_zips(zs)
    if t is None:
        log.warning("ofcom %s: no postcode-level CSV in %s (area-level tables only?) - skipped",
                    name, [z.name for z in zs])
        return None
    if pc.count_distinct(t[POSTCODE_KEY]).as_py() != t.num_rows:
        raise ValueError(f"ofcom {name}: duplicate postcodes across {[z.name for z in zs]} - keep only one "
                         f"{name} postcode coverage zip (one release) in {raw}")
    t = io.snake_columns(t)
    t = io.auto_numeric(t, [c for c in t.column_names if c != POSTCODE_KEY])
    t = t.rename_columns([c if c == POSTCODE_KEY else f"{prefix}{c}" for c in t.column_names])
    t = t.sort_by(POSTCODE_KEY)
    io.write_table(t, out, meta={"files": [z.name for z in zs]})
    return out


# Local-authority mobile table: each measure is split into the % covered by 0..4
# operators (suffix _0.._4). These readable summaries are kept by default.
MOBILE_AREA_COLUMNS = {
    "mob_4g_outdoor_all_operators_pct": ("4G_prem_out_4", "premises with outdoor 4G from all operators (%)"),
    "mob_4g_indoor_all_operators_pct": ("4G_prem_in_4", "premises with indoor 4G from all operators (%)"),
    "mob_4g_indoor_none_pct": ("4G_prem_in_0", "premises with no indoor 4G from any operator (%)"),
    "mob_4g_landmass_all_operators_pct": ("4G_geo_out_4", "land area with outdoor 4G from all operators (%)"),
    "mob_4g_landmass_none_pct": ("4G_geo_out_0", "land area with no outdoor 4G (%)"),
    "mob_voice_indoor_all_operators_pct": ("Voice_prem_in_4", "premises with indoor voice calls from all operators (%)"),
    "mob_voice_indoor_none_pct": ("Voice_prem_in_0", "premises with no indoor voice signal (%)"),
    "mob_5g_outdoor_any_operator_pct": ("100-5G_high_confidence_prem_out_0",
                                        "premises with outdoor 5G (high confidence) from at least one operator (%)"),
    "mob_5g_outdoor_all_operators_pct": ("5G_high_confidence_prem_out_4",
                                         "premises with outdoor 5G (high confidence) from all operators (%)"),
    "mob_4g_a_b_roads_in_car_all_operators_pct": ("4G_abrd_in_4", "A and B roads with in-car 4G from all operators (%)"),
}


def _mobile_area(zs, out, columns="default", recode: dict | None = None) -> Path | None:
    """Ofcom's local-authority mobile table -> one row per local authority (key ``laua``)."""
    import pandas as pd
    frame = None
    for z in zs:
        with zipfile.ZipFile(z) as zf:
            m = next((n for n in zf.namelist() if re.search(r"(?i)mobile_coverage_laua.*\.csv$", n)), None)
            if m:
                frame = pd.read_csv(BytesIO(zf.read(m)), dtype={"laua": str, "laua_name": str})
                break
    if frame is None:
        return None
    vals = frame.drop(columns=["laua", "laua_name"]).apply(pd.to_numeric, errors="coerce")
    # blanks mean 0% where the other operator counts of the same measure are filled in
    for grp in {re.sub(r"_\d$", "_", c) for c in vals.columns if re.search(r"_\d$", c)}:
        cols = [c for c in vals.columns if c.startswith(grp) and re.fullmatch(r"\d", c[len(grp):])]
        filled = vals[cols].notna().any(axis=1)
        vals.loc[filled, cols] = vals.loc[filled, cols].fillna(0)
    # Ofcom may use older local authority codes than the NSPL (e.g. Barnsley/Sheffield recoded in 2026)
    out_df = pd.DataFrame({"laua": frame["laua"].replace(recode or {})})
    if columns == "all":
        for c in vals.columns:
            out_df["mob_" + io.snake(c)] = vals[c]
    else:
        for name, (src, _) in MOBILE_AREA_COLUMNS.items():
            if src.startswith("100-"):
                if src[4:] in vals:
                    out_df[name] = 100 - vals[src[4:]]
            elif src in vals:
                out_df[name] = vals[src]
    io.write_table(pa.Table.from_pandas(out_df, preserve_index=False), out,
                   meta={"files": [z.name for z in zs], "level": "local authority (laua)"})
    return out


@source("ofcom", order=55, keys=["postcode_ns"],
        description="Ofcom Connected Nations fixed broadband and mobile coverage by postcode")
def ofcom(settings, params):
    raw = settings.raw("ofcom")
    page = params.get("page", PAGE)
    mob_file_rx = re.compile(params.get("mobile_file_regex", r"mobile"), re.I)

    def is_mobile(z):
        return bool(mob_file_rx.search(z.name))

    products = [
        # name, url param, link regex, exclude, zip filter, prefix, output
        ("fixed", "url", params.get("link_regex", r"postcode"), params.get("mobile_link_regex_exclude", r"mobile"),
         lambda: sorted(z for z in raw.glob("*.zip") if not is_mobile(z)), "bb_", "broadband_postcode.parquet"),
    ]
    if params.get("mobile", True):
        products.append(
            ("mobile", "mobile_url", params.get("mobile_link_regex", r"mobile.*postcode|postcode.*mobile"), None,
             lambda: sorted(z for z in raw.glob("*.zip") if is_mobile(z)), "mob_", "mobile_postcode.parquet"))

    written = []
    for name, url_key, link_rx, exclude, zips, prefix, fname in products:
        out = settings.out("ofcom") / fname
        urls = []
        if not io.is_fresh(out, settings.reprocess or settings.force) and not zips():
            if params.get(url_key):
                urls = [params[url_key]]
            else:
                try:
                    urls = _find_links(page, link_rx, exclude)
                    if not urls:
                        log.warning("ofcom %s: no zip links matching %r on %s", name, link_rx, page)
                except Exception as e:
                    log.warning("ofcom %s: could not read the data downloads page (%s). Set `%s:` in the "
                                "config, or download the zip in a browser into %s", name, e, url_key, raw)
        p = _product(settings, params, raw, name, zips, urls, prefix, out)
        if p:
            written.append(p)
        elif name == "mobile" and zips():
            # no postcode-level mobile file in the release: fall back to local authority level
            area_out = settings.out("ofcom") / "mobile_laua.parquet"
            if io.is_fresh(area_out, settings.reprocess or settings.force):
                written.append(area_out)
            elif _mobile_area(zips(), area_out, params.get("mobile_area_columns", "default"),
                              params.get("laua_recode")):
                log.info("ofcom mobile: using local authority level coverage instead (mobile_laua.parquet)")
                written.append(area_out)
    if not written:
        raise FileNotFoundError(f"no Ofcom zip available - see the notes above, or place it in {raw}")
    return written
