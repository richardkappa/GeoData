"""Streaming vector conversion: any GDAL source -> GeoParquet.

Large OS products (e.g. Open Roads, ~4 million links) are converted in Arrow
record batches straight from GDAL to Parquet, so memory stays flat and no
Shapely objects are created. Geometry is stored as WKB with GeoParquet 1.0
metadata, readable by ``geopandas.read_parquet``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .io import PARQUET_KW, snake

log = logging.getLogger(__name__)


def list_layers(path: str | Path) -> list[str]:
    import pyogrio
    return [l[0] for l in pyogrio.list_layers(str(path))]


def to_geoparquet(src: str | Path, out: Path, *, layer: str | None = None,
                  columns: list[str] | None = None, target_crs: str = "EPSG:27700",
                  where: str | None = None, batch_size: int = 65536) -> Path:
    import pyogrio
    from pyproj import CRS

    info = pyogrio.read_info(str(src), layer=layer)
    src_crs = CRS.from_user_input(info["crs"]) if info.get("crs") else None
    tgt = CRS.from_user_input(target_crs)

    if src_crs is not None and not src_crs.equals(tgt):
        # rare for OS data; fall back to an in-memory reprojection
        import geopandas as gpd
        from .io import write_gdf
        gdf = gpd.read_file(src, layer=layer, columns=columns, where=where, engine="pyogrio", use_arrow=True)
        return write_gdf(gdf.to_crs(tgt), out, target_crs)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    writer = None
    rows = 0
    geom_types: set[str] = set()
    with pyogrio.open_arrow(str(src), layer=layer, columns=columns, where=where,
                            batch_size=batch_size, use_pyarrow=True) as (meta, reader):
        gcol = meta.get("geometry_name") or "wkb_geometry"
        gt = meta.get("geometry_type")
        if gt:
            geom_types.add(gt.replace(" Z", "").replace("3D ", ""))
        for batch in reader:
            t = pa.Table.from_batches([batch])
            names = ["geometry" if c == gcol else snake(c) for c in t.column_names]
            t = t.rename_columns(names)
            # GDAL may hand back an extension type; store plain binary WKB
            gi = t.schema.get_field_index("geometry")
            if gi >= 0 and not pa.types.is_binary(t.schema.field(gi).type):
                t = t.set_column(gi, "geometry", t["geometry"].cast(pa.binary()))
            if writer is None:
                geo = {"version": "1.0.0", "primary_column": "geometry", "columns": {"geometry": {
                    "encoding": "WKB", "geometry_types": sorted(geom_types),
                    "crs": json.loads(tgt.to_json())}}}
                schema = t.schema.with_metadata({b"geo": json.dumps(geo).encode()})
                writer = pq.ParquetWriter(tmp, schema, **PARQUET_KW)
            writer.write_table(t.cast(schema))
            rows += t.num_rows
    if writer is None:
        log.warning("no features in %s %s", src, layer or "")
        return out
    writer.close()
    tmp.replace(out)
    log.info("wrote   %s  (%s features)", out.name, f"{rows:,}")
    return out
