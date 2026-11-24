"""GeoData: download and standardise UK open geographic data.

Section 1 (extraction) lives in :mod:`geodata.sources`; run it with
``python 01_extract.py``. Every output is written as Parquet (tabular) or
GeoParquet (geometries, EPSG:27700) under ``<data_dir>/extract/<source>/``.
"""

__version__ = "0.1.0"
