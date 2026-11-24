"""Ofcom Connected Nations: broadband (and mobile) coverage by postcode.

Ofcom publishes postcode-level fixed broadband coverage (premises able to get
superfast / ultrafast / full fibre / gigabit, and those below the USO) as a
zip of CSVs per postcode area, on its Connected Nations "data downloads"
page. The page link is configurable because the URL changes each release.

Ofcom's website sits behind a Cloudflare browser check that can refuse
automated downloads from some networks (cloud servers especially). If that
happens the extractor says so; then either set ``url:`` to the zip's direct
link, or download the zip in a browser and drop it into
``<data_dir>/raw/ofcom/`` - any zip found there is processed.

Every CSV in the zip(s) that has a postcode column is read; columns are
snake_cased, numeric columns typed automatically and prefixed ``bb_``.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

import pyarrow as pa
import pyarrow.compute as pc

from .. import download as dl
from .. import io
from ..postcode import POSTCODE_KEY, add_postcode_ns
from . import source

log = logging.getLogger(__name__)

PAGE = "https://www.ofcom.org.uk/phones-and-broadband/coverage-and-speeds/connected-nations-2025/data-downloads-2025"


def _find_links(page: str, regex: str) -> list[str]:
    html = dl.get_text(page)
    if "Just a moment" in html[:2000]:
        raise PermissionError("Ofcom page returned a Cloudflare browser check")
    links = re.findall(r'href="([^"]+\.zip[^"]*)"', html, re.I)
    rx = re.compile(regex, re.I)
    return [urljoin(page, l.replace("&amp;", "&")) for l in links if rx.search(l)]


@source("ofcom", order=55, keys=["postcode_ns"],
        description="Ofcom Connected Nations fixed broadband coverage by postcode")
def ofcom(settings, params):
    out = settings.out("ofcom") / "broadband_postcode.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    raw = settings.raw("ofcom")
    urls = [params["url"]] if params.get("url") else []
    if not urls:
        try:
            urls = _find_links(params.get("page", PAGE), params.get("link_regex", r"postcode"))
            if not urls:
                log.warning("ofcom: no zip links matching %r on %s", params.get("link_regex", "postcode"),
                            params.get("page", PAGE))
        except Exception as e:
            log.warning("ofcom: could not read the data downloads page (%s). Set `url:` in the config, "
                        "or download the postcode zip in a browser into %s", e, raw)
    for u in urls:
        try:
            dl.download(u, raw / re.sub(r"[?#].*$", "", u.rsplit("/", 1)[-1]), force=settings.force,
                        timeout=max(settings.timeout, 600))
        except Exception as e:
            log.warning("ofcom: download failed for %s (%s)", u, e)
    zips = sorted(raw.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no Ofcom zip available - see the notes above, or place it in {raw}")

    parts = []
    for z in zips:
        for m in io.zip_members(z, r"\.csv$"):
            t = io.read_csv_from_zip(z, m, strings=True, encoding="utf-8-sig")
            pcol = next((c for c in t.column_names if re.fullmatch(r"(?i)postcode(_space)?|pcds|pcd", c.strip())),
                        None) or next((c for c in t.column_names if "postcode" in c.lower()), None)
            if not pcol:
                continue
            t = add_postcode_ns(t, pcol, validate=True)
            t = t.drop_columns([c for c in t.column_names if c != POSTCODE_KEY and "postcode" in c.lower()])
            parts.append(t)
    if not parts:
        raise ValueError("no CSV with a postcode column found in the Ofcom zip(s)")
    t = pa.concat_tables(parts, promote_options="permissive")
    t = t.filter(pc.is_valid(t[POSTCODE_KEY]))
    t = io.snake_columns(t)
    t = io.auto_numeric(t, [c for c in t.column_names if c != POSTCODE_KEY])
    t = t.rename_columns([c if c == POSTCODE_KEY else f"bb_{c}" for c in t.column_names])
    t = t.sort_by(POSTCODE_KEY)
    io.write_table(t, out, meta={"files": [z.name for z in zips]})
    return [out]
