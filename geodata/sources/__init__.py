"""Registry of extractors.

Each extractor is a function ``run(settings, params) -> list[Path]`` decorated
with :func:`source`. ``params`` is that source's block from
``config/extract.yaml`` (minus ``enabled``). Extractors must be idempotent:
cached downloads are reused and finished outputs are skipped unless
``force``/``reprocess`` is set.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable


@dataclass
class Source:
    name: str
    func: Callable
    description: str
    keys: tuple[str, ...]     # merge keys produced, e.g. ("postcode_ns",) or ("lsoa21cd",)
    licence: str
    order: int


REGISTRY: dict[str, Source] = {}


def source(name: str, *, description: str, keys: tuple[str, ...] | list[str],
           licence: str = "OGL v3", order: int = 50):
    def deco(fn):
        REGISTRY[name] = Source(name, fn, description, tuple(keys), licence, order)
        return fn
    return deco


def load_all() -> dict[str, Source]:
    for m in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{m.name}")
    return dict(sorted(REGISTRY.items(), key=lambda kv: (kv[1].order, kv[0])))
