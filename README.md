# GeoData

Builds a UK geographic feature set from open government data. Everything is
downloaded by Python straight from source, so nothing needs fetching by hand.
Every output can be joined to a **postcode** (`postcode_ns`), a **UPRN** or an
ONS area code that the postcode lookup maps to postcodes.

The pipeline runs in four sections, in order:

| # | Section | Entry point | Status |
|---|---------|-------------|--------|
| 1 | **Data extraction**: download and standardise each source | `01_extract.py` + `config/extract.yaml` | ✅ built |
| 2 | **Collation and merges**: one flat lookup row per postcode | `02_collate.py` + `config/collate.yaml` | ✅ flat lookup built |
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

python 02_collate.py                     # flat postcode lookup from whatever has been extracted
```

Data goes to `data/` inside the repo by default. To keep it elsewhere (e.g. a
larger data drive), use one of these (the environment variable wins over the
config file):

```bash
export GEODATA_DIR=/path/to/GeoData       # Windows cmd: set GEODATA_DIR=... | PowerShell: $env:GEODATA_DIR="..."
python 01_extract.py --data-dir /path/to/GeoData
# or edit global.data_dir in config/extract.yaml
```

Re-running is cheap. Downloads are cached in `<data_dir>/raw/<source>/`, and
outputs that already exist are skipped. Interrupted downloads resume from where
they stopped. Use `--reprocess` to rebuild the outputs from the cached files, or
`--force` to download everything again.

Some sources need a key, set as an environment variable:

| Variable | Needed by | How to get it |
|---|---|---|
| `CEDA_USERNAME` + `CEDA_PASSWORD` | `metoffice_haduk` (1 km HadUK-Grid yearly grids) | free account at https://services.ceda.ac.uk; a 3-day access token is then created automatically through CEDA's token API |
| `CEDA_TOKEN` (alternative) | `metoffice_haduk` | an access token made by hand on the CEDA site; used while it has more than an hour left |

---

## Configuration: `config/extract.yaml`

* **`global`**: `data_dir`, `crs` (EPSG:27700), `threads` (parallel HTTP
  requests), `force`, `reprocess`, `keep_raw` (set this to false to delete large
  zips once they've been processed).
* **`sources.<name>.enabled`**: turns a source on or off. `--only` ignores
  this setting.
* **Source parameters** choose what each source pulls. Examples: which census
  tables and at what level, the **start/end period** for police data and Land Registry sales,
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

### Performance (designed to run on a modest laptop or desktop)
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

## Section 2: flat postcode lookup (`02_collate.py`)

Builds `<data_dir>/collate/postcode_lookup.parquet`, whose **primary key is
`postcode_ns`**, and `postcode_lookup_dictionary.csv`. The dictionary lists every
column with its source, join type, key, data type and **coverage** (the share
of postcodes that have a value).

The primary key is enforced before the file is written. `postcode_ns` is the
first column, declared `not null` in the Parquet schema, unique, sorted, and
recorded in the file metadata (`primary_key: postcode_ns`). If any join would
produce a null or duplicate key, the run stops with an error rather than
writing a bad file. In pandas, use
`pd.read_parquet(...).set_index("postcode_ns")`.

Nothing here involves distances, radii or neighbourhoods; those belong to
section 3. Each column comes from one of four join types set in
`config/collate.yaml`:

| Join type | What it does | Used for |
|---|---|---|
| `postcode` | table already keyed on `postcode_ns` | DESNZ electricity and gas; EA flood risk; Ofcom broadband |
| `area` | table keyed on an area code the NSPL carries (`spine_col`, plus `key` if the file names it differently) | area names, Census 2021 (OA), IMD 2025 (LSOA) |
| `polygon` | attributes of the polygon the postcode centroid falls in. `how: first` (attributes), `flag` (inside any), `count`, `max`/`min` | AONB, SSSI, ancient woodland, national park, conservation area, WHS, built-up area, radon class, BGS GeoSure classes, Met Office 1991–2020 climate cell values |
| `summary` | simple counts or summaries of record tables inside the postcode or its area | addresses (UPRNs) per postcode; Land Registry sales count, median price, last sale date and property-type mix per postcode; police crimes by type in the postcode's LSOA (last N months); generic `count` summaries (with per-category counts, sums and means) for food businesses and their mean hygiene rating, CQC locations, care homes and beds, GP practices, pharmacies, dentists and NHS sites. Counts are 0 where a postcode has none |

A test run on the full UK (2.70m postcodes with grid references) produced 383
columns (without Ofcom) in about 90 s, peaking at about 6.7 GB of RAM. Coverage follows each
source's geography:

| Source | Coverage | Why |
|---|---|---|
| Census | 89% | England and Wales only |
| IMD | 84% | England only |
| Police | 86% | England, Wales and Northern Ireland, without Scotland |
| Energy | 41–48% | live domestic postcodes |
| Land Registry | 29% | postcodes with sales in the years extracted |
| EA flood risk | 54% (search tool), 6% (rivers & sea) | England; the rivers & sea file lists only postcodes in risk areas |

How it stays fast and light:
* Joins are **index lookups**. `pyarrow.compute.index_in` finds each postcode's
  row in the source table, and only the new columns are `take`n and appended.
  The existing columns are never copied, and row order never changes.
* Count columns are stored as int32.
* Polygon lookups put all postcode points in one GEOS STRtree, built once. Each
  polygon layer is then queried against it, with every polygon prepared a single
  time. A layer takes 0.1–5 s.
* Medians and type shares are computed with numpy sorts and `bincount` on
  integer group ids.

To add a join, append an entry to `joins:` in `config/collate.yaml`. A join whose
input file hasn't been extracted is skipped with a warning. Set
`spine.countries: [E, W]` or `include_terminated: false` to make the table
smaller. `shares: true` on the census join turns counts into proportions of each
table's total.

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
| `police` ✅ | [data.police.uk](https://data.police.uk/data/archive/) street crime and stop & search, all forces | `lsoa21cd`; points as easting/northing | Set `start`/`end` (`YYYY-MM`, `end: latest`). Each archive is ~1.7 GB and holds 36 months, so a period inside the last three years needs one download, and the fewest extra archives are fetched for longer histories (data goes back to Dec 2010). Also writes `crimes_lsoa_month.parquet` with counts per LSOA × month × crime type |
| `road_safety` ✅ | DfT STATS19 road collisions (vehicles and casualties optional) | easting/northing, `lsoa_of_collision` | Choose `last-5-years`, a list of years, or the full 1979+ file |
| `naptan` ✅ | DfT NaPTAN: every bus, rail, tram, ferry and air access node | `atcocode`, easting/northing | Adds a simple `mode` column. Can filter on `stop_types` |

### Property & services
| Source key | Data | Merge key | Notes |
|---|---|---|---|
| `land_registry` ✅ | HM Land Registry Price Paid (England & Wales sales) | `postcode_ns` | Set `start`/`end` as a year, `YYYY-MM` or a date (`end: latest`); sales are filtered to that range. Yearly files are fetched, or the complete file (~5 GB) for periods longer than `complete_file_years`. Files updated monthly are refreshed when older than 20 days |
| `schools` ✅ | DfE Get Information about Schools: all open establishments | `postcode_ns`, `urn`, easting/northing | Finds the latest daily extract automatically |

### Flood risk, broadband, health, food
| Source key | Data | Merge key | Notes |
|---|---|---|---|
| `environment_agency` ✅ | Environment Agency flood risk by postcode (England): **Risk of Flooding from Rivers and Sea – postcodes in areas at risk** (properties by likelihood band and type) and the **flood risk postcode search tool** data (properties at high/medium/low risk, groundwater risk class) | `postcode_ns` | Files found through the data.gov.uk catalogue, so new releases are picked up automatically |
| `ofcom` ⚠️ | Ofcom Connected Nations by postcode: fixed broadband coverage (superfast, ultrafast, full fibre, gigabit, below-USO) → `broadband_postcode.parquet` (`bb_*`), and mobile coverage (4G/5G by operator, indoor/outdoor) → `mobile_postcode.parquet` (`mob_*`) | `postcode_ns` | **Not tested.** Ofcom's site may block automated requests. If it does, set `url:` / `mobile_url:` to the zip links, or save the two postcode zips from a browser into `<data_dir>/raw/ofcom/` (a zip with "mobile" in its name is treated as mobile). Keep one release of each, as duplicate postcodes stop the run |
| `nhs_ods` ✅ | NHS Organisation Data Service (England & Wales): GP practices (6.6k, role RO76), pharmacies, dental practices, NHS trust sites (hospitals and clinics); optionally branch surgeries and independent healthcare | `postcode_ns`, `org_code` | ODS Data Search & Export API, no key needed. Active organisations only |
| `cqc` ✅ | Care Quality Commission active registered locations (England): care homes with bed numbers, home care, GP surgeries, dentists, hospitals and clinics, with overall rating, UPRN and coordinates | `postcode_ns`, `uprn`, easting/northing | Link found on CQC's "using CQC data" page. The ODS file is read with `python-calamine` in seconds |
| `fsa` ✅ | Food Standards Agency hygiene ratings: all ~615k rated food businesses in the UK, with business type, rating, scores and coordinates | `postcode_ns`, `fhrsid`, easting/northing | Uses the single all-establishments CSV, falling back to the 363 per-council XML files |

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
These are other open UK sources that join at postcode or small-area level:
* **Defra air quality** background maps: 1 km grid of NO₂ and PM2.5.
* **VOA council tax** stock by band (LSOA), **ONS small area income** estimates (MSOA), **EPC** certificates (postcode/UPRN; free API key).
* **Scotland**: SIMD and the Scotland's Census 2022 bulk data. **Wales**: WIMD. **NI**: NIMDM.
* **OpenStreetMap** for amenities (deliberately left until the government sources are done).

---

## Repository layout

```
01_extract.py            section 1 runner (CLI)
02_collate.py            section 2 runner: flat postcode lookup
config/extract.yaml      what to download, and the source parameters
config/collate.yaml      which columns and joins go into the postcode lookup
geodata/
  config.py              loads YAML, resolves data_dir
  download.py            cached, resumable, retrying HTTP downloads
  io.py                  Arrow CSV reader, column cleaning, BNG conversion, Parquet writers
  postcode.py            vectorised postcode_ns standardisation
  arcgis.py              ArcGIS Online item search and parallel FeatureServer paging
  govuk.py               GOV.UK Content API attachment resolver (follows new releases)
  vector.py              streaming GDAL -> GeoParquet conversion
  collate.py             section 2 join engine (postcode / area / polygon / summary)
  sources/               one module per data family; each registers its extractors
legacy/                  the original notebooks and KeyFunctions.py (manual downloads, hard-coded local paths)
  Investigations/        earlier experiments (they import ../KeyFunctions, i.e. legacy/KeyFunctions.py)
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
`legacy/KeyFunctions.py`) read hand-downloaded files from a fixed local folder into one
large GeoDataFrame. It is kept for reference while sections 2–4 are rebuilt.
Its pygeos functions will move to shapely 2 / scipy KD-trees.
