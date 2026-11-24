"""ONS Open Geography boundaries (OA/LSOA/MSOA, LAD, wards, constituencies...).

Each entry under ``layers`` in the config becomes one GeoParquet file in
EPSG:27700. Use generalised (``BGC``/``BSC``) versions unless you need full
resolution: they are 10-50x smaller and plenty for postcode lookups (which
come from the NSPL anyway).
"""
from __future__ import annotations

import logging

from .. import arcgis, io
from . import source

log = logging.getLogger(__name__)


def _deg(metres):
    # queries return WGS84, so the server-side tolerance is in degrees (~111 km each)
    return None if not metres else float(metres) / 111_000


def fetch_arcgis_layers(settings, name: str, layers: dict, default_root: str | None = None,
                        simplify_m: float | None = None):
    """Download each configured layer. ``simplify_m`` (metres) asks the server
    to generalise geometries, which can shrink detailed polygons 5-10x."""
    outs = []
    for key, spec in (layers or {}).items():
        if spec is None:
            continue
        if isinstance(spec, str):
            spec = {"service": spec}
        if spec.get("enabled", True) is False:
            continue
        out = settings.out(name) / f"{key}.parquet"
        if io.is_fresh(out, settings.reprocess or settings.force):
            log.info("fresh   %s", out.name)
            outs.append(out)
            continue
        try:
            url = arcgis.resolve_service(spec, root=spec.get("root", default_root or arcgis.ONS_SERVICES))
            gdf = arcgis.fetch_layer(url, fields=spec.get("fields", "*"), where=spec.get("where", "1=1"),
                                     threads=settings.threads,
                                     max_allowable_offset=_deg(spec.get("simplify_m", simplify_m)))
        except Exception as e:  # keep going: one broken layer should not stop the run
            log.error("layer %s failed: %s", key, e)
            continue
        if gdf is None or len(gdf) == 0:
            log.warning("layer %s returned no features", key)
            continue
        if hasattr(gdf, "geometry"):
            drop = [c for c in gdf.columns if c.lower() in ("shape__area", "shape__length", "globalid")]
            gdf = gdf.drop(columns=drop)
            io.write_gdf(gdf, out, settings.crs)
        else:
            import pyarrow as pa
            io.write_table(io.snake_columns(pa.Table.from_pandas(gdf, preserve_index=False)), out)
        outs.append(out)
    return outs


@source("ons_boundaries", order=3, keys=["oa21cd", "lsoa21cd", "msoa21cd", "lad*cd", "..."],
        description="ONS Open Geography boundary polygons (configurable list of layers)")
def boundaries(settings, params):
    return fetch_arcgis_layers(settings, "ons_boundaries", params.get("layers", {}),
                               simplify_m=params.get("simplify_m"))
