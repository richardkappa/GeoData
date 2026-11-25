"""Load the YAML run configuration and expose resolved paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "extract.yaml"


@dataclass
class Settings:
    data_dir: Path
    crs: str = "EPSG:27700"
    force: bool = False            # re-download even if a cached file exists
    reprocess: bool = False        # rebuild parquet outputs even if they exist
    threads: int = 4               # parallel HTTP requests (I/O bound)
    timeout: int = 120             # seconds per HTTP request
    user_agent: str = "GeoData-extract/0.1 (+open data research)"
    keep_raw: bool = True          # keep downloaded zips/csvs after processing
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    geocode: dict[str, Any] = field(default_factory=dict)   # coordinates for point tables (geodata/geocode.py)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def out_dir(self) -> Path:
        return self.data_dir / "extract"

    def raw(self, source: str) -> Path:
        p = self.raw_dir / source
        p.mkdir(parents=True, exist_ok=True)
        return p

    def out(self, source: str) -> Path:
        p = self.out_dir / source
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_settings(path: str | os.PathLike | None = None, **overrides: Any) -> Settings:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    g = cfg.get("global", {}) or {}
    # env var wins over the file so the same config works on several machines
    data_dir = os.environ.get("GEODATA_DIR") or g.get("data_dir") or "data"
    data_dir = Path(data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()

    s = Settings(
        data_dir=data_dir,
        crs=g.get("crs", "EPSG:27700"),
        force=bool(g.get("force", False)),
        reprocess=bool(g.get("reprocess", False)),
        threads=int(g.get("threads", 4)),
        timeout=int(g.get("timeout", 120)),
        user_agent=g.get("user_agent", Settings.user_agent),
        keep_raw=bool(g.get("keep_raw", True)),
        sources=cfg.get("sources", {}) or {},
        geocode=cfg.get("geocode", {"enabled": True}) or {},
    )
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, v)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
