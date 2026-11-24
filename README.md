# GeoData

Builds a UK geographic feature set from open government data. Everything is
downloaded by Python straight from source, so nothing needs fetching by hand.
Every output can be joined to a **postcode** (`postcode_ns`), a **UPRN** or an
ONS area code that the postcode lookup maps to postcodes.

The pipeline runs in four sections, in order:

| # | Section | Entry point | Status |
|---|---------|-------------|--------|
| 1 | **Data extraction**: download and standardise each source | `01_extract.py` + `config/extract.yaml` | ✅ built |
| 2 | Collation and merges: join everything onto the postcode spine | `02_collate.py` (planned) | to do |
| 3 | New data creation: distances, densities, neighbourhood averages | `03_features.py` (planned) | to do |
| 4 | Visualisations: maps and charts | notebooks (planned) | to do |

Sections 1–3 are plain `.py` scripts. They are long-running batch jobs, so they
are easier to rerun, schedule and log than notebooks. Section 4 will use notebooks.

---

## Quick start

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate GeoData

python 01_extract.py --list              # what's available / enabled
python 01_extract.py                     # run every enabled source
python 01_extract.py --only ons_nspl census_2021 imd_england
```

Data goes to `data/` inside the repo by default. To keep it elsewhere, e.g. on
the XPS's D: drive, use one of these (the environment variable wins over the
config file):

```bash
set GEODATA_DIR=D:/GeoData                # Windows cmd   (PowerShell: $env:GEODATA_DIR="D:/GeoData")
python 01_extract.py --data-dir D:/GeoData
# or edit global.data_dir in config/extract.yaml
```

Re-running is cheap. Downloads are cached in `<data_dir>/raw/<source>/`, and
outputs that already exist are skipped. Interrupted downloads resume from where
they stopped. Use `--reprocess` to rebuild the outputs from the cached files, or
`--force` to download everything again.

Some sources need a key, set as an environment variable:

| Variable | Needed by | How to get it |
|---|---|---|
| `CEDA_TOKEN` | `metoffice_haduk` (1 km HadUK-Grid yearly grids) | free account at https://services.ceda.ac.uk → access token |

---

## Configuration: `config/extract.yaml`

* **`global`**: `data_dir`, `crs` (EPSG:27700), `threads` (parallel HTTP
  requests), `force`, `reprocess`, `keep_raw` (set this to false to delete large
  zips once they've been processed).
* **`sources.<name>.enabled`**: turns a source on or off. `--only` ignores
  this setting.
* **Source parameters** choose what each source pulls. Examples: which census
  tables and at what level, how many months of police data, Land Registry years,
  which NaPTAN stop types, which OS products and GeoPackage layers, which
  boundary or designation layers, and how much to simplify polygons.

Boundary and designation layers can be given in three ways:
* by exact service name: `lsoa21: LSOA_2021_EW_BSC_V4_RUC`
* by regex, which always picks the newest vintage: `lad: {service_regex: '^Local_Authority_Districts_.*_Boundaries_UK_BGC$'}`
* by full URL: `{url: https://.../FeatureServer/0}`

---

## Output conventions

```
<data_dir>/
  raw/<source>/...                 cached downloads (zip/csv/gpkg)
  extract/<source>/*.parquet       standardised outputs
  extract/_manifest.json           what ran, when, files, keys, last errors
  extract.log                      full log of every run
```

* **Tables** are written as Parquet (zstd). **Geometries** are written as
  GeoParquet in **EPSG:27700** (British National Grid). Read them with
  `pandas.read_parquet` / `geopandas.read_parquet`, or query them in place with
  DuckDB.
* **`postcode_ns`** is the postcode in upper case with all whitespace removed
  (`"sw1a 1aa"` → `SW1A1AA`). Every postcode-level table has this column.
* **Point data** carries `easting`/`northing` (EPSG:27700). If the source gave
  only latitude/longitude, these are converted with PROJ. Northern Ireland
  postcodes are on the Irish Grid in the NSPL and are converted to the British
  grid as well. Points are kept as plain x/y columns rather than geometry
  objects, so section 2 can use fast numpy/KD-tree joins.
* **Area codes** use NSPL column names (`oa21cd`, `lsoa21cd`, `msoa21cd`...), so
  joining an area table to postcodes is a single merge on the NSPL.
* ONS "not applicable" pseudo codes (`E99999999`, `S99999999`…) become null.
* Column names are lower_snake_case.

> **UDPRN vs UPRN.** The UDPRN is Royal Mail's address key and is *not* open
> data. The open equivalent is the **UPRN**: `ons_nsul` maps UPRN → postcode and
> geographies, and `os_open: OpenUPRN` gives UPRN → coordinates.

### Performance (designed for a laptop like the XPS)
* CSVs are parsed by Arrow's multi-threaded C++ reader, which only materialises
  the columns asked for. The 2.7m-row NSPL processes in about 13 s.
* Postcode cleaning, filtering, type casting and grouping run as Arrow compute
  kernels, and coordinate conversion runs on numpy arrays through PROJ. There
  are no row-by-row Python loops.
* OS GeoPackages are streamed GDAL → Arrow → GeoParquet in batches, so even
  Open Roads (~4m links) uses little memory.
* Paged ArcGIS APIs are fetched with a few parallel threads, which is I/O work.
  Polygons can be simplified on the server (`simplify_m`): Natural England
  shrinks from 484 MB to 61 MB at 5 m.
* Low-cardinality text such as crime types is dictionary-encoded.

---

## Sources (section 1)

All of these are open data under the Open Government Licence v3 unless noted.
✅ means tested end to end; the timings are from a test run on a 4-core machine.

### Postcode / address spine
| Source key | Data | Merge key | Output | Notes |
|---|---|---|---|---|
| `ons_nspl` ✅ | [ONS National Statistics Postcode Lookup](https://geoportal.statistics.gov.uk/) (latest release found automatically) | `postcode_ns` | `nspl.parquet`: 2.73m postcodes, live and terminated, OA/LSOA/MSOA 2021, LAD, ward, constituency, region, national park, built-up area, rural-urban class, OAC, IMD rank, lat/long and BNG easting/northing | ~185 MB download, 13 s |
| `ons_nsul` ✅ | ONS National Statistics UPRN Lookup | `uprn`, `postcode_ns` | `nsul.parquet`: 41.6m UPRNs | ~560 MB download, ~2.5 min, off by default; can be limited to some regions |
| `os_open: CodePointOpen` ✅ | OS Code-Point Open | `postcode_ns` | `codepoint_open.parquet` | GB postcode centroids and positional quality |
| `os_open: OpenUPRN` | OS Open UPRN | `uprn` | `open_uprn.parquet` | 41m address points, off by default |

### Census, deprivation, energy
| Source key | Data | Merge key | Notes |
|---|---|---|---|
| `census_2021` ✅ | [Census 2021 topic summaries via Nomis bulk](https://www.nomisweb.co.uk/census/2021/bulk), England & Wales | `oa21cd` / `lsoa21cd` / `msoa21cd` | Pick tables (or `all`) and levels. One file per table, plus `census_2021_catalogue.parquet` mapping columns to their source descriptions |
| `imd_england` ✅ | English Indices of Deprivation **2025** (2019 optional), found through the GOV.UK content API | `lsoa21cd` | Every domain's score, rank and decile, plus population denominators |
| `energy_postcode` ✅ | DESNZ postcode-level domestic **electricity and gas** consumption (latest year) | `postcode_ns` | **New source**: meters, total, mean and median kWh per postcode |

### Crime, safety, transport
| Source key | Data | Merge key | Notes |
|---|---|---|---|
| `police` ✅ | [data.police.uk](https://data.police.uk/data/archive/) street crime and stop & search, all forces | `lsoa21cd`; points as easting/northing | Uses the latest monthly archive (~1.7 GB, 36 months) and keeps the last `months`. Also writes `crimes_lsoa_month.parquet` with counts per LSOA × month × crime type |
| `road_safety` ✅ | DfT STATS19 road collisions (vehicles and casualties optional) | easting/northing, `lsoa_of_collision` | Choose `last-5-years`, a list of years, or the full 1979+ file |
| `naptan` ✅ | DfT NaPTAN: every bus, rail, tram, ferry and air access node | `atcocode`, easting/northing | Adds a simple `mode` column. Can filter on `stop_types` |

### Property & services
| Source key | Data | Merge key | Notes |
|---|---|---|---|
| `land_registry` ✅ | HM Land Registry Price Paid (England & Wales sales) | `postcode_ns` | Choose years or the complete file (~5 GB). The current year's file is refreshed if it's older than 20 days |
| `schools` ✅ | DfE Get Information about Schools: all open establishments | `postcode_ns`, `urn`, easting/northing | Finds the latest daily extract automatically |

### Boundaries
| Source key | Data | Notes |
|---|---|---|
| `ons_boundaries` ✅ | ONS Open Geography boundaries: LSOA21, MSOA21, OA21, LAD (latest), countries, regions, national parks, Westminster 2024, wards, built-up areas, police force areas | Any ONS feature service can be added in the config |

### Ordnance Survey / British Geological Survey (via the OS Downloads API, no key needed)
| Product | Data | Notes |
|---|---|---|
| `OpenGreenspace` ✅ | Parks, playing fields, allotments and their access points | Polygons and points |
| `OpenRivers` ✅ | Watercourse network | Lines |
| `BuiltUpAreas` ✅ | OS Open Built Up Areas | Polygons |
| `Radon_Indicative_Atlas` ✅ | **New source**: BGS/UKHSA radon potential classes | 1 km polygons. NI available with `area: NI` |
| `GB-Hex-5km-GeoSure` ✅ | **New source**: BGS ground stability hazards (shrink-swell, landslide, soluble rocks, compressible ground, collapsible deposits, running sand) | 5 km hexes |
| `OpenRoads` ✅ | Road links (4m) and motorway junctions | ~590 MB download, streamed in ~1 min, off by default |
| `OpenNames` | Places, roads and postcode districts | Off by default |
| `Terrain50` | 50 m elevation model | Download only (raw). It will be sampled at postcodes in section 3 |

### Environment, heritage, climate
| Source key | Data | Notes |
|---|---|---|
| `natural_england` ✅ | SSSI, AONB (National Landscapes), ancient woodland, local and national nature reserves, SAC, SPA, Ramsar, country parks, heritage coasts (priority habitats optional) | Any of NE's ~900 ArcGIS layers can be added by name |
| `historic_england` ✅ | Listed buildings (380k points), scheduled monuments, registered parks & gardens, battlefields, World Heritage Sites, conservation areas | |
| `metoffice_cdp` ✅ | Met Office Climate Data Portal 1991–2020 averages: air frost, icing, summer and 10 mm rain days (fine grid), plus 12 km annual rainfall and mean, max and min temperature | Open, no login |
| `metoffice_haduk` | HadUK-Grid 1 km annual, seasonal or monthly grids for chosen years and variables (rainfall, temperature, sun, frost, snow, wind, humidity) | **Needs `CEDA_TOKEN`**: CEDA now requires a login for these files. Written as a land-only x/y grid |

### Candidates for later (not built yet)
These are open UK sources that also join at postcode or small-area level:
* **Environment Agency** flood risk: *Risk of Flooding from Rivers and Sea – postcodes in areas at risk* (postcode level), and the flood risk postcode search tool data.
* **Ofcom Connected Nations**: broadband speed and availability by postcode.
* **Food Standards Agency** food hygiene ratings: every food business with a postcode, via an API.
* **CQC** care homes and GP practices: locations with postcodes, via an API.
* **NHS ODS / NHS website datasets**: GP surgeries, pharmacies, hospitals (the old `media.nhs.uk` URLs are no longer reachable).
* **Defra air quality** background maps: 1 km grid of NO₂ and PM2.5.
* **VOA council tax** stock by band (LSOA), **ONS small area income** estimates (MSOA), **EPC** certificates (postcode/UPRN; free API key).
* **Scotland**: SIMD and the Scotland's Census 2022 bulk data. **Wales**: WIMD. **NI**: NIMDM.
* **OpenStreetMap** for amenities (deliberately left until the government sources are done).

---

## Repository layout

```
01_extract.py            section 1 runner (CLI)
config/extract.yaml      what to download, and the source parameters
geodata/
  config.py              loads YAML, resolves data_dir
  download.py            cached, resumable, retrying HTTP downloads
  io.py                  Arrow CSV reader, column cleaning, BNG conversion, Parquet writers
  postcode.py            vectorised postcode_ns standardisation
  arcgis.py              ArcGIS Online item search and parallel FeatureServer paging
  govuk.py               GOV.UK Content API attachment resolver (follows new releases)
  vector.py              streaming GDAL -> GeoParquet conversion
  sources/               one module per data family; each registers its extractors
legacy/                  the original notebooks and KeyFunctions.py (manual downloads, D:/ paths)
Investigations/          earlier experiments (they import ../KeyFunctions, now in legacy/)
```

### Adding a source
Create a function in `geodata/sources/<family>.py`:

```python
from . import source
from .. import download as dl, io
from ..postcode import add_postcode_ns

@source("my_source", order=60, keys=["postcode_ns"], description="What it is")
def my_source(settings, params):
    out = settings.out("my_source") / "my_source.parquet"
    if io.is_fresh(out, settings.reprocess or settings.force):
        return [out]
    f = dl.download(URL, settings.raw("my_source") / "file.csv", force=settings.force)
    t = add_postcode_ns(io.read_csv(f, strings=True), "Postcode")
    io.write_table(t, out)
    return [out]
```

Then add a `my_source:` block with `enabled: true` to `config/extract.yaml`.

---

## Legacy
The original workflow (`legacy/01_GetData.ipynb`, `legacy/02_CreatePostcodeFile.ipynb`,
`legacy/KeyFunctions.py`) read hand-downloaded files from `D:/GeoData` into one
large GeoDataFrame. It is kept for reference while sections 2–4 are rebuilt.
Its pygeos functions will move to shapely 2 / scipy KD-trees.
