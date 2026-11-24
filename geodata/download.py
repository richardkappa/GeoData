"""HTTP helpers: cached, resumable, retrying downloads.

Files are streamed to ``<name>.part`` in 1 MiB chunks and renamed only when
complete, so an interrupted run never leaves a truncated file that looks
finished. Re-running skips files that already exist unless ``force``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

_SESSION: requests.Session | None = None
CHUNK = 1 << 20


def session(user_agent: str | None = None) -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers["User-Agent"] = user_agent or "GeoData-extract/0.1"
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _SESSION = s
    return _SESSION


def _retry(fn, *, tries: int = 5, what: str = ""):
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            # 4xx other than 429 will not fix themselves
            if status and 400 <= status < 500 and status != 429:
                raise
            # a proxy/firewall refusing the host will not change on retry
            if isinstance(e, requests.exceptions.ProxyError) and "403" in str(e):
                raise
            if attempt == tries:
                raise
            log.warning("%s failed (%s), retry %d/%d in %.0fs", what, e, attempt, tries - 1, delay)
            time.sleep(delay)
            delay *= 2


def get_json(url: str, params: dict | None = None, timeout: int = 120, **kw: Any) -> Any:
    def go():
        r = session().get(url, params=params, timeout=timeout, **kw)
        r.raise_for_status()
        return r.json()
    return _retry(go, what=url)


def get_text(url: str, params: dict | None = None, timeout: int = 120) -> str:
    def go():
        r = session().get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.text
    return _retry(go, what=url)


def get_bytes(url: str, params: dict | None = None, timeout: int = 120, data: dict | None = None) -> bytes:
    def go():
        if data is not None:
            r = session().post(url, data=data, timeout=timeout)
        else:
            r = session().get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.content
    return _retry(go, what=url)


def download(url: str, dest: Path, *, force: bool = False, timeout: int = 120,
             headers: dict | None = None, expected_size: int | None = None) -> Path:
    """Download ``url`` to ``dest`` (resuming a ``.part`` file if present)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if expected_size is None or dest.stat().st_size == expected_size:
            log.info("cached  %s", dest.name)
            return dest
    part = dest.with_name(dest.name + ".part")
    if force and part.exists():
        part.unlink()

    def go():
        have = part.stat().st_size if part.exists() else 0
        h = dict(headers or {})
        if have:
            h["Range"] = f"bytes={have}-"
        with session().get(url, stream=True, timeout=timeout, headers=h, allow_redirects=True) as r:
            if have and r.status_code == 200:      # server ignored Range: start again
                have = 0
            elif r.status_code == 416:            # already complete
                return
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            total = int(total) + have if total else None
            mode = "ab" if have else "wb"
            done, last = have, time.monotonic()
            with open(part, mode) as f:
                for chunk in r.iter_content(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if time.monotonic() - last > 15:
                        pct = f" {100 * done / total:.0f}%" if total else ""
                        log.info("  ... %s %.0f MB%s", dest.name, done / 1e6, pct)
                        last = time.monotonic()
            if total and done < total:
                raise requests.ConnectionError(f"incomplete download {done}/{total}")

    log.info("download %s -> %s", url, dest)
    _retry(go, what=url)
    part.replace(dest)
    return dest


def head_ok(url: str, timeout: int = 30) -> bool:
    """True when a GET for the first byte succeeds (some hosts reject HEAD)."""
    try:
        r = session().get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, stream=True)
        r.close()
        return r.status_code in (200, 206)
    except requests.RequestException:
        return False
