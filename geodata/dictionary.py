"""The data dictionary: what every source, file and column is, laid out like the pipeline.

The dictionary page (``4_visualise/variables.html``) has a collapsible section per pipeline step, and each section has a CSV
with exactly the columns the page shows:

1. Sources                     ``1_extract/sources_dictionary.csv``    (``SOURCE_COLUMNS``)
2. Outputs
   2a. postcode lookup         ``2_collate/postcode_lookup_dictionary.csv``   (``COLUMNS``)
   2b. plottable data          ``1_extract/spatial_dictionary.csv``    (``SPATIAL_COLUMNS``)
3. Calculated variables
   3a. postcode features       ``3_features/postcode_features_dictionary.csv`` (``COLUMNS``)
   3b. map-only variables      ``4_visualise/map_variables_dictionary.csv``   (``COLUMNS``)

Source details (publisher, dataset, guide, licence, coverage, imputation) live in ``config/sources.yaml``; merge keys
come from each extractor's registration, output files from the last step 1 run (``1_extract/_manifest.json``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow as pa

# --- columns (2a, 3a, 3b) -----------------------------------------------------------------------------------------
COLUMNS = ["section", "theme", "column", "description", "source", "method", "join_level", "gaps", "type", "coverage"]
LOOKUP, FEATURES, MAPS = "Postcode lookup (step 2)", "Postcode features (step 3)", "Map variables (step 4)"
MAP_COLUMNS = COLUMNS   # 3b uses the same columns as 2a and 3a

# step 2: (theme, regex on the extracted file path the column comes from)
LOOKUP_THEMES = [
    ("Postcode spine and geographies", r"^ons_nspl|^ons_boundaries|^ons_nsul"),
    ("Census 2021 (England and Wales, output area)", r"^census_2021"),
    ("Deprivation (IMD 2025, England, LSOA)", r"^imd_england"),
    ("Household income (ONS, MSOA)", r"^income"),
    ("Crime (police.uk, LSOA)", r"^police"),
    ("Crime (Scotland, council areas)", r"^scotland_crime"),
    ("House sales (Land Registry)", r"^land_registry"),
    ("Energy use (DESNZ)", r"^energy_postcode"),
    ("Vehicles (DfT, LSOA)", r"^vehicles"),
    ("Flood risk (Environment Agency)", r"^environment_agency"),
    ("Road and rail noise (Defra)", r"^noise"),
    ("Broadband and mobile (Ofcom)", r"^ofcom"),
    ("Health and care services", r"^nhs_ods|^cqc"),
    ("Food businesses (FSA)", r"^fsa"),
    ("Designations, heritage and land", r"^natural_england|^historic_england|^os_open/builtup"),
    ("Ground, radon and geology (BGS / UKHSA)", r"^os_open/(radon|gb_hex)"),
    ("Climate 1991-2020 (Met Office)", r"^metoffice"),
]

# step 3: feature group -> (theme, sources it is built from)
FEATURE_THEMES = {
    "density": ("Counts within a distance", "point sources in config/3_features.yaml (points:)"),
    "rate": ("Rates per head within a distance", "point sources + census population"),
    "nearest": ("Distance to the nearest", "point and line sources in config/3_features.yaml"),
    "network": ("Travel times and reach over the road network", "os_open (Open Roads) + point sources"),
    "landcover": ("Land cover, trees and roads nearby", "os_open, natural_england, trees"),
    "terrain": ("Height, slope and terrain", "os_open (Terrain 50)"),
    "water": ("Rivers, lakes and coast", "os_open (Open Rivers, Zoomstack)"),
    "climate": ("Climate by year (HadUK-Grid)", "metoffice_haduk"),
    "air_quality": ("Air quality (Defra)", "air_quality"),
    "assign": ("Assigned areas", "nhs_ods"),
    "house_prices": ("House price trends", "land_registry"),
    "crime": ("Crime trends and rates", "police"),
    "neighbours": ("Neighbourhood averages (adjoining LSOAs)", "ons_boundaries + the measures averaged"),
}


# how each step 3 feature group is made (shown under its heading on the dictionary page)
FEATURE_METHODS = {
    "density": "Straight-line counts: every point of a dataset (crimes, collisions, schools, takeaways...) is binned onto "
               "a 100 m national grid, and the cells within 1 km and 5 km of each postcode are summed with a disc-shaped "
               "window. Residents, households and addresses are counted the same way from the census and address counts.",
    "rate": "A count within a distance divided by the residents (or households, addresses) within the same distance, "
            "per 1,000 or 10,000; blank where too few people live within it.",
    "nearest": "Straight-line distance from each postcode to the nearest (and second nearest) point or line of a "
               "dataset, using a KD-tree; lines (roads, rivers, coast) are sampled every 25 m.",
    "network": "Network analysis over OS Open Roads (3.98 million links): walking (4.8 km/h, no motorways) and driving "
               "(typical free-flow speeds by road class) times from each postcode's nearest junction to the nearest "
               "station, GP, A&E..., using Dijkstra's shortest paths; and what can be reached within N minutes (collisions, "
               "fires, takeaways, people...), searched per 500 m grid square in parallel tiles.",
    "landcover": "Shares of land within 500 m and 2 km covered by greenspace, woodland, built-up areas, water and trees "
                 "outside woodland (polygons rasterised to a 100 m grid, then summed with a disc window); km of road per km².",
    "terrain": "OS Terrain 50 (50 m elevation) sampled at each postcode: height, slope, height above the nearest river.",
    "water": "Distances to rivers, lakes and the coast from OS Open Rivers and Zoomstack.",
    "climate": "Met Office HadUK-Grid 1 km yearly grids: each postcode takes its nearest land square; averages over the "
               "years extracted and the latest year.",
    "air_quality": "Defra's modelled 1 km pollution grid: each postcode takes the value of its square.",
    "assign": "Each postcode assigned to the Primary Care Network of its nearest GP practice.",
    "house_prices": "Land Registry sales within 1 km of each postcode: median prices for the latest 12 months and "
                    "earlier windows, and their change.",
    "crime": "Police.uk crimes: latest 12 months vs the 12 before, and LSOA crime rates per 1,000 residents.",
    "neighbours": "LSOA neighbour network: areas that touch, plus areas joined by a road across a river or estuary "
                  "('bridges'); values pooled over the LSOAs within 1, 2 or 5 boundary steps, weighted by residents.",
}

# step 3 columns -> the catalogue sources whose gaps they inherit (first match wins), plus method gaps
FEATURE_GAP_RULES = [
    (r"scot_crime", ["scotland_crime"]),
    (r"crime|burglar|antisocial|vehicle_crime|violent|stop_search", ["police"]),
    (r"collision|casualt", ["road_safety"]),
    (r"fire", ["fire"]),
    (r"hp_|price|sales|property", ["land_registry"]),
    (r"tree", ["trees"]),
    (r"climate_|_avg$", ["metoffice_haduk"]),
    (r"air_", ["air_quality"]),
    (r"gp_|pharmac|dentist|pcn", ["nhs_ods"]),
    (r"school", ["schools"]),
    (r"swimming|gym|sports_hall|pitch", ["active_places"]),
    (r"takeaway|pub|restaurant|supermarket", ["fsa"]),
    (r"hospital|care_home", ["cqc"]),
    (r"rail_footfall", ["rail_usage"]),
    (r"listed|monument", ["historic_england"]),
    (r"per_\d+_(residents|households)|population|residents", ["census_2021"]),
]
GROUP_GAPS = {
    "network": "No Northern Ireland (OS Open Roads covers Great Britain); island postcodes only reach their own island",
    "terrain": "Great Britain only (OS Terrain 50)",
    "neighbours": "England and Wales LSOAs only",
}


def feature_sources(column: str, group: str) -> str:
    """The catalogue sources a step 3 column is built from (by its name), else the group's usual inputs."""
    for rx, keys in FEATURE_GAP_RULES:
        if re.search(rx, column):
            base = {"network": "os_open (Open Roads)", "terrain": "os_open (Terrain 50)"}.get(group)
            return ", ".join(([base] if base else []) + keys)
    return FEATURE_THEMES.get(group, ("", ""))[1]


def feature_gaps(column: str, group: str) -> str:
    out = []
    for rx, keys in FEATURE_GAP_RULES:
        if re.search(rx, column):
            out += [source_gaps(k) for k in keys]
            break
    if GROUP_GAPS.get(group):
        out.append(GROUP_GAPS[group])
    return "; ".join(x for x in out if x)


def lookup_theme(source: str) -> str:
    src = str(source or "").replace("\\", "/")
    return next((t for t, rx in LOOKUP_THEMES if re.search(rx, src)), "Other")


LEVELS = {"postcode_ns": "postcode", "oa21cd": "output area", "lsoa21cd": "LSOA / data zone", "msoa21cd": "MSOA",
          "lad26cd": "local authority", "lad19cd": "local authority (2019)", "rgn26cd": "region", "ctry26cd": "country",
          "pcon24cd": "Westminster constituency", "wd26cd": "ward", "easting/northing": "postcode point"}


def join_level(method: str, key: str, source: str) -> str:
    """How a lookup column reaches the postcode: its own postcode, an area the postcode is in, or the polygon its
    point falls in."""
    m, k = str(method or ""), str(key or "")
    if m.startswith("polygon"):     # the layer is in the source column
        return "point in polygon"
    if m == "spine":
        return "postcode (spine)"
    return LEVELS.get(k, k or m)


def source_gaps(source: str) -> str:
    """Coverage and known systematic gaps of the catalogue source a column comes from."""
    cat = _catalogue().get("sources") or {}
    src = str(source or "").replace("\\", "/")
    key = next((k for k in sorted(cat, key=len, reverse=True) if src.startswith(k)), None)
    if not key:
        return ""
    s = cat[key]
    # the specific gap first (it is what a reader most needs), then the coverage
    return "; ".join(x for x in (s.get("gaps", ""), f"covers {s['coverage']}" if s.get("coverage") else "") if x)


def standard_table(section: str, column, description, source, method, type_, coverage, key=None) -> pa.Table:
    """A column dictionary (2a / 3a / 3b) in the shared structure: themes, join level and gaps worked out here."""
    key = key if key is not None else [""] * len(column)
    if section == LOOKUP:
        theme = [lookup_theme(s) for s in source]
        level = [join_level(m, k, s) for m, k, s in zip(method, key, source)]
        gaps = [source_gaps(s) for s in source]
    else:
        level = ["postcode (calculated)" if section == FEATURES else "area (calculated for the maps)"] * len(column)
        gaps = [feature_gaps(str(c), str(m)) for c, m in zip(column, method)]
        theme = [FEATURE_THEMES.get(str(m), (str(m).replace("_", " ").capitalize(), ""))[0] for m in method]
        source = [s or feature_sources(str(c), str(m)) for s, c, m in zip(source, column, method)]
    n = len(column)
    return pa.table({"section": [section] * n, "theme": theme, "column": list(column),
                     "description": list(description), "source": [str(s).replace("\\", "/") for s in source],
                     "method": list(method), "join_level": level, "gaps": gaps, "type": list(type_),
                     "coverage": list(coverage)})


def from_legacy_csv(t: pa.Table, section: str) -> pa.Table:
    """Older dictionary CSVs (column/source/join/key/note or column/group/note) -> the shared structure."""
    names = t.column_names
    if names[:len(COLUMNS)] == COLUMNS:
        return t
    get = lambda c, default="": t[c].to_pylist() if c in names else [default] * t.num_rows
    if "description" in names:   # the shared structure from before join_level / gaps were added
        return standard_table(section, get("column"), get("description"), get("source"), get("method"),
                              get("type"), get("coverage", None))
    return standard_table(section, get("column"), get("note"), get("source"),
                          get("join") if "join" in names else get("group"), get("type"), get("coverage", None),
                          get("key") if "key" in names else None)


# --- 1. sources -----------------------------------------------------------------------------------------------------
SOURCE_COLUMNS = ["theme", "source", "description", "website", "licence", "commercial_use", "licence_note",
                  "area_covered", "postcodes_with_data", "imputation", "merge_key", "output_files", "dataset", "guide",
                  "last_run"]


def _catalogue() -> dict:
    import yaml
    from .config import REPO_ROOT
    f = REPO_ROOT / "config" / "sources.yaml"
    return yaml.safe_load(open(f, encoding="utf-8")) if f.exists() else {"sources": {}, "licences": {}}


def _postcode_share(data_dir) -> dict:
    """Source -> share of postcodes with a value from it in the step 2 lookup (its best-covered column)."""
    import pyarrow.csv as pacsv
    f = Path(data_dir) / "2_collate" / "postcode_lookup_dictionary.csv"
    if not f.exists():
        return {}
    t = from_legacy_csv(pacsv.read_csv(f), LOOKUP)
    keys = sorted(_catalogue().get("sources") or {}, key=len, reverse=True)
    out: dict[str, float] = {}
    for src, cov, method in zip(t["source"].to_pylist(), t["coverage"].to_pylist(), t["method"].to_pylist()):
        if cov is None:
            continue
        s = str(src).replace("\\", "/")
        key = next((k for k in keys if s.startswith(k)), s.split("/")[0])
        out[key] = max(out.get(key, 0.0), float(cov))
    return out


def sources_table(data_dir) -> pa.Table:
    """Section 1: every source - the README's details (publisher, licence), then where exactly it comes from, its
    guide, merge key, output files, coverage and what is imputed."""
    cat = _catalogue()
    lic = cat.get("licences") or {}
    mf = Path(data_dir) / "1_extract" / "_manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    try:
        from .step1_extract import load_all
        registry = load_all()
    except Exception:
        registry = {}
    share = _postcode_share(data_dir)
    rows = {c: [] for c in SOURCE_COLUMNS}
    for key, s in (cat.get("sources") or {}).items():
        m = man.get(key) or {}
        reg = registry.get(key)
        sh = share.get(key)
        vals = {"theme": s.get("theme", ""), "source": key, "description": s.get("description", ""),
                "website": s.get("website", ""),
                "licence": (lic.get(s.get("licence")) or {}).get("name", s.get("licence", "")),
                "commercial_use": "yes" if s.get("commercial", True) else "needs a licence",
                "licence_note": s.get("note", ""), "area_covered": s.get("coverage", ""),
                "postcodes_with_data": "" if sh is None else f"{sh:.0%} ({1 - sh:.0%} missing)",
                "imputation": s.get("imputation", ""),
                "merge_key": ", ".join(reg.keys) if reg else "",
                "output_files": "; ".join(m.get("files", [])).replace("\\", "/"),
                "dataset": s.get("dataset", ""), "guide": s.get("guide", ""), "last_run": m.get("finished", "")}
        for c in SOURCE_COLUMNS:
            rows[c].append(vals[c])
    return pa.table(rows)


# --- 2b. plottable (geocoded) data ----------------------------------------------------------------------------------
SPATIAL_COLUMNS = ["theme", "file", "source", "description", "kind", "geometry", "how_to_use", "rows", "size_mb",
                   "area_covered", "gaps", "columns", "crs"]
GRID_SOURCES = {"air_quality": "1 km", "metoffice_haduk": "1 km", "trees": "100 m"}
KIND_THEMES = {"points": "Points", "area centres": "Area centres (counts per small area, placed at its centre)",
               "grid": "Grid squares", "lines": "Lines", "network": "Road network (links and junctions)",
               "polygons": "Polygons", "mixed": "Mixed geometry"}


PRODUCTS = {"openroads": "OS Open Roads", "opengreenspace": "OS Open Greenspace", "openrivers": "OS Open Rivers",
            "builtupareas": "OS Open Built Up Areas", "openzoomstack": "OS Open Zoomstack", "opennames": "OS Open Names",
            "radon_indicative_atlas": "BGS / UKHSA radon indicative atlas (1 km)",
            "gb_hex_5km_geosure": "BGS GeoSure ground stability (5 km hexes)"}
GEOSURE = {"shrinkswell": "shrink-swell clays", "solublerocks": "soluble rocks", "compressibleground": "compressible ground",
           "collapsibledeposits": "collapsible deposits", "runningsand": "running sand", "landslides": "landslides"}
FILE_DESC = {
    "police/crimes.parquet": "Street-level crimes, one point each (police.uk's anonymised map points)",
    "police/stop_and_search.parquet": "Stop and search events",
    "fsa/food_hygiene.parquet": "Food businesses with their hygiene ratings",
    "land_registry/price_paid.parquet": "House sales, each at its postcode's centre",
    "road_safety/collision.parquet": "Road collisions (STATS19) with severity",
    "naptan/naptan_stops.parquet": "Bus stops, stations, ferry and air terminals",
    "schools/schools.parquet": "Schools and other education establishments",
    "cqc/cqc_locations.parquet": "Care homes, clinics, hospitals and other registered care locations",
    "fire/fire_points.parquet": "Fire and rescue incidents a year per LSOA (England) or data zone (Scotland), at the area's centre",
    "vehicles/lsoa_vehicles.parquet": "Licensed cars, plug-in and electric vehicles per LSOA / data zone, at the area's centre",
    "air_quality/pcm_2024.parquet": "Modelled NO2, PM2.5 and PM10 on 1 km squares (2024)",
    "metoffice_haduk/haduk_1km_ann.parquet": "Yearly climate (rainfall, temperature, sunshine...) on 1 km squares",
    "trees/tow_canopy_100m.parquet": "Tree canopy outside woodland per 100 m square, by type",
    "rail_usage/station_usage.parquet": "Railway stations with entries and exits a year",
    "ons_nspl/nspl.parquet": "Every UK postcode as a point, with its areas",
    "noise/road_lden.parquet": "Road noise bands (Lden, from 55 dB)",
    "noise/rail_lden.parquet": "Railway noise bands (Lden, from 55 dB)",
    "ons_boundaries/lad.parquet": "Local authority boundaries", "ons_boundaries/lad19.parquet": "Local authority boundaries (2019)",
    "ons_boundaries/lsoa21.parquet": "LSOA boundaries (2021, generalised)",
    "ons_boundaries/lsoa21_detail.parquet": "LSOA boundaries (2021, full resolution)",
    "ons_boundaries/msoa21.parquet": "MSOA boundaries (2021)", "ons_boundaries/westminster_2024.parquet": "Westminster constituencies (2024)",
    "os_open/codepoint_open.parquet": "OS Code-Point Open: postcode centres",
}


def file_description(rel: str, source: str) -> str:
    """A short description of one file (not the whole source)."""
    if rel in FILE_DESC:
        return FILE_DESC[rel]
    stem = rel.split("/")[-1].removesuffix(".parquet")
    if "__" in stem:
        product, layer = stem.split("__", 1)
        name = PRODUCTS.get(product, product.replace("_", " "))
        if product == "gb_hex_5km_geosure":
            key = re.sub(r"^gb_hex_5km_gs_|_v\d+$", "", layer)
            return f"{name}: {GEOSURE.get(key, key)}"
        if product == "radon_indicative_atlas":
            return name
        return f"{name}: {layer.replace('os_open_', '').replace('_', ' ')}"
    text = stem.replace("_", " ")
    lead = {"active_places": "Sport England: ", "metoffice_cdp": "Met Office 1991-2020 averages: ",
            "natural_england": "Natural England: ", "historic_england": "Historic England: ",
            "nhs_ods": "NHS: ", "ons_boundaries": "ONS boundaries: "}.get(source, "")
    return lead + text[:1].upper() + text[1:] if not lead else lead + text


def reach_description(name: str) -> str:
    """reach_cells_<mode>_<N>min.parquet -> what it holds."""
    m = re.match(r"reach_cells_(drive|walk)_(\d+)min", name)
    if not m:
        return name
    mode = "drive" if m.group(1) == "drive" else "walk"
    return (f"Calculated: for each 500 m square with postcodes, what can be reached within a {m.group(2)}-minute {mode} "
            f"over OS Open Roads (one column per measure in config/3_features.yaml network.reach); its postcodes share "
            f"these values")


def spatial_table(data_dir, folder: str = "1_extract") -> pa.Table:
    """Section 2b (and the calculated files of 3b with folder="3_features"): the geocoded files in <data_dir>/1_extract that can be mapped (points, polygons, lines, grids, the road
    network), from Parquet metadata only (fast, even for the largest files)."""
    import pyarrow.parquet as pq
    cat = _catalogue().get("sources") or {}
    root = Path(data_dir) / folder
    rows = {c: [] for c in SPATIAL_COLUMNS}
    for f in sorted(root.rglob("*.parquet")):
        try:
            md = pq.read_metadata(f)
            names = md.schema.to_arrow_schema().names
            geo = json.loads((md.metadata or {}).get(b"geo", b"{}") or b"{}")
        except Exception:
            continue
        source = f.relative_to(root).parts[0] if folder == "1_extract" else "step 3 (03_features.py)"
        if f.name.startswith("reach_cells_"):     # calculated: what can be reached from each 500 m square
            kind_override = "grid"
        else:
            kind_override = None
        if "geometry" in names:
            types = ((geo.get("columns") or {}).get("geometry") or {}).get("geometry_types") or []
            gtypes, crs = ", ".join(types), "EPSG:27700"
            low = [t.lower() for t in types]
            kind = ("network" if {"start_node", "end_node"} <= set(names) else
                    "points" if low and all("point" in t for t in low) else
                    "lines" if low and all("line" in t for t in low) else
                    "polygons" if not low or all("polygon" in t for t in low) else "mixed")
        elif {"easting", "northing"} <= set(names) or {"cell_x", "cell_y"} <= set(names):
            xy = ("cell_x", "cell_y") if "cell_x" in names else ("easting", "northing")
            crs = f"EPSG:27700 ({xy[0]} / {xy[1]} columns)"
            kind = kind_override or ("grid" if source in GRID_SOURCES else
                                     "area centres" if source in ("fire", "vehicles") else "points")
            gtypes = ("500 m squares (start point of each square's searches)" if kind_override else
                      f"{GRID_SOURCES[source]} squares" if kind == "grid" else "Point")
        else:
            continue
        rel = str(f.relative_to(root)).replace("\\", "/")
        if "geometry" in names:
            how = (f"GeoParquet in British National Grid (EPSG:27700). QGIS: Layer > Add Layer > Add Vector Layer, "
                   f"choose <data_dir>/{folder}/{rel} (needs GDAL 3.8+; or convert with "
                   f"ogr2ogr -f GPKG out.gpkg {f.name}). Python: geopandas.read_parquet")
        else:
            x, y = ("cell_x", "cell_y") if "cell_x" in names else ("easting", "northing")
            how = (f"Table with x = {x}, y = {y} in British National Grid (EPSG:27700). QGIS: open "
                   f"<data_dir>/{folder}/{rel} as a layer and set the geometry from the {x} / {y} fields (or "
                   f"export to CSV and use Add Delimited Text Layer). Python: pandas.read_parquet, then "
                   f"geopandas.points_from_xy({x}, {y}, crs=27700)")
        sc = cat.get(source) or {}
        vals = {"theme": KIND_THEMES[kind], "file": rel, "source": source,
                "description": (reach_description(f.name) if kind_override else file_description(rel, source)),
                "kind": kind, "geometry": gtypes, "how_to_use": how,
                "rows": md.num_rows, "size_mb": round(f.stat().st_size / 1e6, 1), "area_covered": sc.get("coverage", ""),
                "gaps": sc.get("gaps", ""), "columns": ", ".join(c for c in names if c != "geometry"), "crs": crs}
        for c in SPATIAL_COLUMNS:
            rows[c].append(vals[c])
    return pa.table(rows)


def write_extract_dictionaries(data_dir) -> None:
    """Sections 1 and 2b: 1_extract/sources_dictionary.csv and 1_extract/spatial_dictionary.csv (end of step 1)."""
    import pyarrow.csv as pacsv
    out = Path(data_dir) / "1_extract"
    out.mkdir(parents=True, exist_ok=True)
    pacsv.write_csv(sources_table(data_dir), out / "sources_dictionary.csv")
    pacsv.write_csv(spatial_table(data_dir), out / "spatial_dictionary.csv")


# --- 3b. map-only calculated variables ------------------------------------------------------------------------------
def map_variables_table(vcfg: dict) -> pa.Table:
    """Section 3b: variables calculated only for the maps (config/4_visualise.yaml): area rates per head and derived
    expressions. Same columns as 2a / 3a."""
    derived = vcfg.get("derived") or {}
    variables = (vcfg.get("cartograms") or {}).get("variables") or {}
    cols, desc, src, meth = [], [], [], []
    for name, spec in variables.items():
        spec = spec if isinstance(spec, dict) else {"label": spec}
        if spec.get("rate"):
            w = f"sum of {spec['weight']} of " if spec.get("weight") else "number of "
            how = (f"{w}{spec['rate']} (each placed on its nearest postcode) per {float(spec.get('scale', 1000)):,.0f} "
                   f"{spec.get('per', 'residents')}, totalled over the area")
            src.append(spec["rate"]), meth.append("rate")
        elif name in derived:
            d = derived[name]
            how = f"{d.get('expr')} (from {', '.join(d.get('uses', []))})"
            src.append(", ".join(d.get("uses", []))), meth.append("derived")
        else:
            continue
        cols.append(name)
        desc.append(f"{spec.get('label', name)}: {how}")
    theme = ["Area rates per head" if m == "rate" else "Derived from other columns" for m in meth]
    n = len(cols)
    return pa.table({"section": [MAPS] * n, "theme": theme, "column": cols, "description": desc, "source": src,
                     "method": meth, "join_level": ["area (calculated for the maps)"] * n,
                     "gaps": [feature_gaps(c, "") for c in cols],
                     "type": ["double"] * n, "coverage": pa.nulls(n, pa.float64())})
