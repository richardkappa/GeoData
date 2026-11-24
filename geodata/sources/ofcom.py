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
    if not written:
        raise FileNotFoundError(f"no Ofcom zip available - see the notes above, or place it in {raw}")
    return written
