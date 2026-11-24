"""Resolve file attachments on GOV.UK pages via the public Content API.

GOV.UK asset URLs contain an opaque media id that changes with each upload,
so instead of hard-coding them we ask ``https://www.gov.uk/api/content/<path>``
for the page and pick the attachment whose title matches a regex.
"""
from __future__ import annotations

import logging
import re

from . import download as dl

log = logging.getLogger(__name__)

API = "https://www.gov.uk/api/content"


def page(path: str) -> dict:
    return dl.get_json(f"{API}/{path.strip('/')}")


def attachments(path: str) -> list[dict]:
    return page(path).get("details", {}).get("attachments", [])


def find_attachment(path: str, title_regex: str) -> dict:
    rx = re.compile(title_regex, re.I)
    for a in attachments(path):
        if rx.search(a.get("title", "")) or rx.search(a.get("filename", "") or ""):
            log.info("gov.uk  %s -> %s", a.get("title"), a.get("url"))
            return a
    raise LookupError(f"No attachment matching {title_regex!r} on gov.uk/{path}")


def latest_in_series(path_template: str, years: range, title_regex: str) -> tuple[int, dict]:
    """Try ``path_template.format(year=y)`` newest-first and return the first hit."""
    for y in sorted(years, reverse=True):
        try:
            return y, find_attachment(path_template.format(year=y), title_regex)
        except Exception:  # page missing (404) or attachment not there yet
            continue
    raise LookupError(f"No page found for {path_template} in {years}")
