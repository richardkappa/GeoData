# GeoData

Builds a UK geographic feature set from open data. Every source is downloaded by Python straight from the
publisher, so nothing needs fetching by hand. Everything ends up joined to the **postcode** (`postcode_ns`),
and can also be linked through a UPRN or any ONS area code the postcode lookup carries.

The pipeline runs in four steps, in order. Each step reads the previous step's outputs from the data folder:

| # | Step | Script and config | What it does | Writes to `<data_dir>/` |
|---|---|---|---|---|
| 1 | **Extract** | `01_extract.py`, `config/1_extract.yaml` | Downloads each enabled source (cached, resumable), standardises it to Parquet / GeoParquet on the British National Grid, and gives every point table coordinates: its own, converted from latitude / longitude, or its postcode's centre | `1_raw/<source>/` downloads, `1_extract/<source>/*.parquet` |
| 2 | **Collate** | `02_collate.py`, `config/2_collate.yaml` | Starts from every UK postcode in the ONS postcode directory and adds columns: tables keyed on an area code (census, deprivation, income, vehicles, Scottish crime, names), tables keyed on the postcode (energy, flood risk, broadband), the polygon the postcode falls in (designations, radon, ground stability, noise, climate averages), and summaries of records (house sales, crimes, food businesses, health services) | `2_collate/postcode_lookup.parquet`: one row per postcode, plus a column dictionary |
| 3 | **Features** | `03_features.py`, `config/3_features.yaml` | For every live postcode: counts and rates of things within set distances, distances to the nearest, land cover and tree shares, height and slope, climate and air quality, walking and driving times over OS Open Roads, what can be reached within N minutes (collisions, fires, takeaways, people...), house price and crime trends, and neighbourhood averages over adjoining LSOAs | `3_features/postcode_features.parquet`, plus a column dictionary and per-square reach values |
| 4 | **Visualise** | `04_visualise.py`, `config/4_visualise.yaml` | Self-contained HTML maps from steps 2 and 3: mini maps, the map explorer, cartograms (fixed and dynamic), choropleths, heat maps, small multiples and the data dictionary (`variables.html`). `--profile web` writes a lighter copy for publishing | `4_visualise/` (`4_visualise_web/` for the web copy) |

All four are plain `.py` batch scripts, so they are easy to rerun, schedule and log.

---

## Quick start

**1. Folders.** Clone the repo, and choose a folder for the data. By default the data goes in `data/` inside
the repo, but you may want to keep it somewhere separate (for example a larger drive), as the full run needs
about **55 GB**, most of it cached downloads. Point the scripts at it with the `GEODATA_DIR` environment variable:

```bash
git clone https://github.com/richardkappa/GeoData.git
export GEODATA_DIR=/path/to/geodata-data      # PowerShell: $env:GEODATA_DIR = "D:\geodata-data"
```

**2. Environment.** One conda environment, from the repo (conda is easiest on Windows, where it supplies GDAL
and PROJ; `pip install -r requirements.txt` also works elsewhere):

```bash
conda env create -f environment.yml
conda activate GeoData
```

**3. Keys.** Only one source needs a login; set these as environment variables, never in a repo file:

| Variable | Needed for | How to get it |
|---|---|---|
| `CEDA_USERNAME`, `CEDA_PASSWORD` | Met Office HadUK-Grid 1 km climate grids (`metoffice_haduk`) | Free account at [services.ceda.ac.uk](https://services.ceda.ac.uk). A short-lived access token is then made automatically. (`CEDA_TOKEN` can be used instead, with a token made by hand on the CEDA site.) Without these, that one source is skipped |
| `CARTO_API_KEY` (optional) | Street-map tiles in step 4 | Free key at [carto.com/basemaps](https://carto.com/basemaps/apikey). Without it the pages use Esri tiles |

**4. Run** the four steps:

```bash
python 01_extract.py      # first run: about 45 GB of downloads, so an hour or more depending on your connection
python 02_collate.py      # about 2 minutes
python 03_features.py     # about 25 minutes
python 04_visualise.py    # about 10 minutes; open <data_dir>/4_visualise/index.html
```

On Windows, `scripts/setup_and_run_windows.ps1` does the clone, environment and all four steps in one go.

Reruns are cheap: downloads are cached in `<data_dir>/1_raw/`, finished outputs are skipped, and interrupted
downloads resume. `python 01_extract.py --list` shows the sources and which are enabled; `--only <source>`
runs just those; `--reprocess` rebuilds outputs from the cache; `--force` downloads again.

---

## Configuration: `config/1_extract.yaml`

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
  1_raw/<source>/...                 cached downloads (zip/csv/gpkg)
  1_extract/<source>/*.parquet       standardised outputs
  1_extract/_manifest.json           what ran, when, files, keys, last errors
  extract.log                      full log of every run
```

* **Tables** are written as Parquet (zstd). **Geometries** are written as
  GeoParquet in **EPSG:27700** (British National Grid). Read them with
  `pandas.read_parquet` / `geopandas.read_parquet`, or query them in place with
  DuckDB.
* **`postcode_ns`** is the postcode in upper case with all whitespace removed
  (`"sw1a 1aa"` → `SW1A1AA`). Every postcode-level table has this column.
* **Point data** carries `easting`/`northing` (EPSG:27700), checked as each
  table is extracted (`geodata/geocode.py`, settings under `geocode:` in
  `config/1_extract.yaml`). Every row gets the source's own coordinates if they
  are valid, else its latitude/longitude converted with PROJ, else the centre of
  its postcode from the NSPL. Tables that only have a postcode (GP practices,
  pharmacies, house sales...) get the columns added. When anything is filled,
  `xy_source` records where each row's coordinates came from (`source`,
  `latlong`, `postcode`, or empty if there was nothing to go on, e.g. food
  businesses run from private homes, whose addresses the FSA withholds).
  Tables with one row per postcode are postcode attributes, not points, and are
  left alone. `python 01_extract.py --geocode-only` re-runs this on what has
  already been extracted. Northern Ireland postcodes are on the Irish Grid in
  the NSPL and are converted to the British grid as well. Points are kept as
  plain x/y columns rather than geometry objects, so section 2 can use fast
  numpy/KD-tree joins.
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

Builds `<data_dir>/2_collate/postcode_lookup.parquet`, whose **primary key is
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
`config/2_collate.yaml`:

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

Column descriptions come from each join's `note`; columns without one (the NSPL
spine, energy, IMD, designations, climate ...) are described in
`config/column_descriptions.yaml` (exact names or patterns). The NSPL entries are
summarised from the ONS NSPL User Guide in the NSPL download. `variables.html`
(section 4) lists every column with its description and coverage.

To add a join, append an entry to `joins:` in `config/2_collate.yaml`. A join whose
input file hasn't been extracted is skipped with a warning. Set
`spine.countries: [E, W]` or `include_terminated: false` to make the table
smaller. `shares: true` on the census join turns counts into proportions of each
table's total.

---

## Section 3: postcode features (`03_features.py`)

Distances, densities, travel times, climate, trends and neighbourhood averages
for every live postcode, configured in `config/3_features.yaml`.

**What can be reached within a time limit.** `network.reach.measures` counts
points (any entry under `points:`) on the roads reachable within N minutes'
drive or walk of every postcode, for the whole country, or sums a weight
column instead of counting:

```yaml
network:
  reach:
    measures:
      collisions_within_30min_drive: {mode: drive, minutes: 30, points: collisions}
      bus_stops_within_10min_walk:   {mode: walk, minutes: 10, points: bus_stops}
      casualties_within_30min_drive: {mode: drive, minutes: 30, points: collisions, weight: number_of_casualties}
```

Drive times use typical free-flow speeds by road class; walking uses 4.8 km/h
and leaves out motorways. Each 500 m grid square with postcodes is searched from
the road junction nearest its postcodes, and its postcodes share the result.
The searches are exact but only use the roads that could be reached in time, so
the country is split into tiles that run in parallel. Measures with the same
mode and time limit share one set of searches, so extra ones are cheap (30
minutes' drive for all 268,000 squares takes about 11 minutes on 10 cores). The
per-square values are also saved as `3_features/reach_cells_<mode>_<N>min.parquet`.

### Planned: a vehicle value proxy for every postcode

No open dataset gives the value of the cars at a postcode, so the plan is a three-stage estimate, with the
inputs already extracted (`mot`, `vehicles`, `income`, house prices, deprivation):

1. **Value per car (national model).** The MOT results give, per car tested, make, model, fuel, engine size,
   year of first use, mileage and whether it failed. A car's relative value is modelled as a new-price level
   for its make / model / fuel / engine band, reduced for age (depreciation curve) and mileage, and slightly
   for condition (initial MOT failure). Without an open price list the default output is an index
   (100 = the average car); an editable table of typical new prices by make or segment would turn it into £.
2. **Average per postcode area.** Car-weighted mean of that value over the MOT tests in each of the ~120
   postcode areas (test station's area: close to, but not exactly, where cars are kept).
3. **Down to LSOAs and postcodes (small-area model).** The area means are regressed on area-level signals
   that exist for every LSOA - plug-in and company-car shares and cars per household (DfT), household income
   (ONS), house prices, deprivation, rurality - aggregated to postcode areas. The fitted model predicts each
   LSOA, and predictions are scaled so the car-weighted LSOA mean matches its postcode area's MOT mean
   (the same idea as ONS small-area income estimates). Postcodes take their LSOA's value.

Known gaps: cars under three years old have no MOT (newest, most valuable), partly covered by the LSOA
plug-in and company-car shares; the test station is not always near the keeper's home; Northern Ireland
MOTs are separate.

## Section 4: visualisations (`04_visualise.py`)

Writes self-contained HTML pages to `<data_dir>/4_visualise/`; open `index.html`.

* **Mini maps** (`minimaps.html`): a 5 km square around each key location
  (the original project's list, with Furness now centred on Barrow-in-Furness,
  plus tourist areas: Bath, York, Edinburgh, St Ives, Keswick, Llanberis),
  showing roads by class, railways, rivers and lakes, woodland, greenspace,
  built-up areas, conservation areas, scheduled monuments, listed buildings and
  stations. One PNG per map is also saved in `4_visualise/minimaps/`.
* **Map explorer** (`explorer.html`): the same squares, interactive, plus the
  three national hotspots for road collisions within a 30-minute drive. The
  controls come in three groups: the **location**; a **road network** view
  (road centrality, walking and driving time from the centre and to the
  nearest railway station, the shortest walk to a station, collision hotspots,
  collisions within a 30-minute drive, the major road junction graph, and the
  LSOA neighbour network with road "bridges" across rivers and the rings within
  1, 2 and 3 boundaries used for spatial smoothing); and a **postcode overlay**
  (postcode dots, a smooth surface, or an LSOA choropleth of a feature, with
  its own opacity; the postcode features are listed under `minimaps.variables`
  in `config/4_visualise.yaml`). Map layers (including takeaways and GP surgeries) and the
  background map are ticked in the side panel. A view can be opened directly,
  e.g. `explorer.html#loc=Bath&view=walk&mode=lsoa&var=2&layers=none&bg=light`.
  The old `minimap_explorer.html` address forwards here.
* **Cartograms** (`cartograms.html`): features averaged to each area, weighted
  by the people living in each postcode (each output area's Census 2021
  `ts001_total` shared over its postcodes by their addresses; estimated from
  addresses in Scotland and Northern Ireland), plus rates per head (crime,
  collisions, takeaways, GP practices... per 1,000 residents: points placed on
  their nearest postcode and totalled per area), drawn with every area the
  same size: Westminster
  constituencies and local authorities as hexes, NHS Primary Care Networks
  (1,261 hexes; each postcode takes the PCN of its nearest GP practice, an
  approximation because patients register where they choose) and MSOAs as 7,264
  hexes ([hexmap.uk](https://hexmap.uk), MIT), and the Ordnance Survey equal-area
  square cartogram of 2019 districts (OGL; postcodes are given their 2019
  district through the ONS December 2019 boundaries).
* **Choropleths** (`choropleths.html`): the same variables on real geography:
  constituencies, local authorities and MSOAs (ONS Open Geography boundaries,
  simplified for the web) and Ordnance Survey national grid squares of 5, 10, 25
  and 50 km, labelled with their grid reference. The background map can be
  switched off.
* **Dynamic cartograms** (`dynamic_cartograms.html`): local authorities and
  constituencies resized by their residents. A shape-keeping cartogram
  (Gastner-Newman diffusion, worked out at build time; sizes within 2-3% of
  population) that morphs from the real map with a slider, and Dorling circles
  pushed apart by a live force simulation, sized by residents, households,
  addresses or land area.
* **Heat maps** (where points are concentrated), **small multiples** and the searchable **data dictionary** (`variables.html`); the
  index has links straight to featured views.

Locations can be given as BNG coordinates or as a postcode. The layers, styles,
areas, layouts and variables are all set in `config/4_visualise.yaml`.

Section 3's LSOA neighbourhood averages use the same network: areas that touch,
plus areas joined by a road that crosses a gap between them (a bridge or tunnel
over a river or estuary), as in the "Building Bridges" approach.

**Background maps.** The pages are opened as local files, which OpenStreetMap's
own tile servers refuse. CARTO tiles are used when the environment variable
`CARTO_API_KEY` holds a (free) CARTO basemaps key; never put the key in a repo
file. Without it the pages use Esri's tiles, which need no key.

---

## Sources

Every source, where it comes from and the licence it is used under. **OGL** = [Open Government Licence v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/):
free for any use, commercial included, with attribution. ⚠️ marks anything that **cannot be used commercially
without a separate (paid) licence**. What each source contributes, column by column, is in the **data dictionary**: the variables page
(`4_visualise/variables.html`), with the same table in `2_collate/postcode_lookup_dictionary.csv` and
`3_features/postcode_features_dictionary.csv`.

### Postcodes, addresses and boundaries
| Source | What it is | Website | Licence |
|---|---|---|---|
| `ons_nspl` | ONS National Statistics Postcode Lookup: every UK postcode with its areas and grid reference | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk/) | OGL (contains OS, Royal Mail and GeoPlace data, free to use under the OGL) |
| `ons_nsul` | ONS National Statistics UPRN Lookup: every address (UPRN) with its postcode | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk/) | OGL |
| `ons_boundaries` | ONS boundaries: output areas, LSOAs, MSOAs, councils, wards, constituencies, regions, national parks | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk/) | OGL |
| `os_open` | Ordnance Survey OpenData: Code-Point Open, Open UPRN, Open Roads, Open Rivers, Open Greenspace, Built Up Areas, Open Names, Terrain 50, Open Zoomstack | [osdatahub.os.uk](https://osdatahub.os.uk/) | OGL |
| `os_open` (BGS) | British Geological Survey radon potential and GeoSure ground stability 5 km hexes (via the OS downloads service) | [bgs.ac.uk](https://www.bgs.ac.uk/) | OGL |

### People, deprivation, income, energy, vehicles
| Source | What it is | Website | Licence |
|---|---|---|---|
| `census_2021` | Census 2021 topic summaries (England and Wales) | [nomisweb.co.uk](https://www.nomisweb.co.uk/) | OGL |
| `imd_england` | English Indices of Deprivation 2025 | [gov.uk](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025) | OGL |
| `income` | ONS modelled household income by MSOA (England and Wales) | [ons.gov.uk](https://www.ons.gov.uk/) | OGL |
| `energy_postcode` | DESNZ domestic electricity and gas use by postcode | [gov.uk](https://www.gov.uk/government/collections/sub-national-electricity-consumption-data) | OGL |
| `vehicles` | DfT licensed vehicles by small area: cars, plug-in and electric vehicles | [gov.uk](https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-files) | OGL |
| `mot` | DVSA anonymised MOT test results: make, model, age, mileage, outcome | [open.data.dvsa.gov.uk](https://open.data.dvsa.gov.uk/) | OGL |

### Crime, safety, fire, transport
| Source | What it is | Website | Licence |
|---|---|---|---|
| `police` | Street-level crime and stop and search, England, Wales and Northern Ireland | [data.police.uk](https://data.police.uk/) | OGL |
| `scotland_crime` | Recorded crime by council area (Scotland) | [data.gov.scot](https://data.gov.scot/) | OGL |
| `fire` | Fire and rescue incidents: England by LSOA (MHCLG), Scotland by data zone (Scottish Fire and Rescue Service) | [gov.uk](https://www.gov.uk/government/statistical-data-sets/fire-statistics-incident-level-datasets), [data.gov.scot](https://data.gov.scot/) | OGL |
| `road_safety` | DfT STATS19 road collisions | [gov.uk](https://www.gov.uk/government/collections/road-accidents-and-safety-statistics) | OGL |
| `naptan` | DfT NaPTAN: bus stops, stations, ferry and air terminals | [gov.uk](https://www.gov.uk/government/publications/national-public-transport-access-node-schema) | OGL |
| `rail_usage` | ORR estimates of station usage | [dataportal.orr.gov.uk](https://dataportal.orr.gov.uk/) | OGL |

### Property, services, health, food, sport
| Source | What it is | Website | Licence |
|---|---|---|---|
| `land_registry` | HM Land Registry Price Paid: house sales in England and Wales | [gov.uk](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads) | OGL ⚠️ The **address data** in it (from Royal Mail's PAF) may only be used for personal or non-commercial purposes, or to show property price information; other commercial use needs a Royal Mail licence |
| `schools` | DfE Get Information about Schools | [get-information-schools.service.gov.uk](https://get-information-schools.service.gov.uk/) | OGL |
| `nhs_ods` | NHS Organisation Data Service: GP practices, pharmacies, dentists, NHS sites, Primary Care Networks | [digital.nhs.uk](https://digital.nhs.uk/services/organisation-data-service) | OGL |
| `cqc` | Care Quality Commission registered locations (care homes, clinics, hospitals) | [cqc.org.uk](https://www.cqc.org.uk/about-us/transparency/using-cqc-data) | OGL |
| `fsa` | Food Standards Agency food hygiene ratings | [ratings.food.gov.uk](https://ratings.food.gov.uk/) | OGL |
| `active_places` | Sport England Active Places: sports facilities (England) | [activeplacespower.com](https://www.activeplacespower.com/) | CC BY 4.0 (commercial use allowed); "Contains Data © Sport England" must go with any use |

### Environment, heritage, climate
| Source | What it is | Website | Licence |
|---|---|---|---|
| `environment_agency` | Flood risk by postcode (England) | [environment.data.gov.uk](https://environment.data.gov.uk/) | OGL |
| `ofcom` | Ofcom Connected Nations: broadband and mobile coverage | [ofcom.org.uk](https://www.ofcom.org.uk/) | OGL |
| `natural_england` | Protected sites: SSSI, National Landscapes, ancient woodland, nature reserves... | [naturalengland-defra.opendata.arcgis.com](https://naturalengland-defra.opendata.arcgis.com/) | OGL |
| `historic_england` | Listed buildings, scheduled monuments, parks and gardens, conservation areas... | [historicengland.org.uk](https://historicengland.org.uk/listing/the-list/data-downloads/) | OGL |
| `metoffice_cdp` | Met Office 1991-2020 climate averages | [climatedataportal.metoffice.gov.uk](https://climatedataportal.metoffice.gov.uk/) | OGL |
| `metoffice_haduk` | Met Office HadUK-Grid 1 km climate by year (needs a free CEDA account) | [catalogue.ceda.ac.uk](https://catalogue.ceda.ac.uk/) | OGL |
| `air_quality` | Defra modelled background NO₂, PM2.5 and PM10 on a 1 km grid | [uk-air.defra.gov.uk](https://uk-air.defra.gov.uk/data/pcm-data) | OGL |
| `noise` | Defra strategic noise maps, road and rail (England, 2017) | [environment.data.gov.uk](https://environment.data.gov.uk/) | OGL |
| `trees` | Forest Research trees outside woodland (England) | [forestresearch.gov.uk](https://www.forestresearch.gov.uk/tools-and-resources/trees-outside-woodland-map/) | OGL |

### Used by the maps (step 4)
| Source | What it is | Website | Licence |
|---|---|---|---|
| hexmap.uk | Hex cartogram layouts (constituencies, councils, MSOAs, Primary Care Networks) | [hexmap.uk](https://hexmap.uk) | MIT |
| OS equal-area cartogram | Square cartogram of 2019 local authorities | [ordnancesurvey.co.uk](https://www.ordnancesurvey.co.uk/) | OGL |
| Leaflet | Web map library | [leafletjs.com](https://leafletjs.com/) | BSD-2 |
| Background map tiles | Esri World Street Map / Light Grey Canvas, or CARTO when `CARTO_API_KEY` is set | [esri.com](https://www.esri.com/), [carto.com](https://carto.com/basemaps) | ⚠️ Not open data: both are free for low-volume, non-commercial use under their providers' terms; commercial or high-traffic use needs a paid plan |

### Candidates for a later build
Open sources considered but not built yet:
* **VOA council tax** stock by band (LSOA).
* **EPC** energy performance certificates (postcode / UPRN; free registration).
* **Ofsted** inspection ratings (join to schools).
* **Bus frequency** from the Bus Open Data Service timetables (free API key): buses per hour near each postcode.
* **planning.data.gov.uk** (England): green belt, tree preservation orders, brownfield land, Article 4 areas.
* **Renewable Energy Planning Database** (DESNZ): wind and solar sites.
* **Noise round 4** (2022): newer maps, currently only offered through an interactive download.
* **Deprivation for the other nations**: SIMD (Scotland), WIMD (Wales), NIMDM (Northern Ireland), and Scotland's Census 2022.
* **A vehicle value proxy** per postcode (the plan is under step 3 below; the MOT and vehicle inputs are already extracted).
* **OpenStreetMap** amenities (left until the government sources are done; ODbL licence).

---

## Repository layout

Files and folders are numbered by the pipeline step they belong to; the rest is shared.

```
01_extract.py                    step 1 runner: download and standardise every source
02_collate.py                    step 2 runner: the flat postcode lookup
03_features.py                   step 3 runner: calculated postcode features
04_visualise.py                  step 4 runner: the HTML maps and the data dictionary page
config/
  1_extract.yaml                 step 1: which sources, and their parameters
  2_collate.yaml                 step 2: the joins and columns of the postcode lookup
  3_features.yaml                step 3: feature groups and their settings
  4_visualise.yaml               step 4: pages, locations, map layers and variables
  sources.yaml                   source catalogue: publisher, dataset, guide, licence, coverage, gaps
  column_descriptions.yaml       column descriptions for the data dictionary
geodata/
  step1_extract/                 one module per data family; each registers its extractors
  step2_collate/                 the join engine (postcode / area / polygon / summary)
  step3_features/                feature groups: densities, distances, network travel and reach, land cover...
  step4_visualise/               mini maps, map explorer, area maps, dynamic cartograms, data dictionary page
  config.py                      loads the YAML, resolves the data folder
  download.py                    cached, resumable, retrying HTTP downloads
  io.py                          Arrow CSV reader, column cleaning, BNG conversion, Parquet writers
  postcode.py                    vectorised postcode_ns standardisation
  geocode.py                     coordinates for every point table (own, from lat/long, or postcode centre)
  dictionary.py                  the data dictionary tables (sources, columns, plottable files)
  descriptions.py                fills column descriptions from config/column_descriptions.yaml
  arcgis.py, govuk.py, vector.py ArcGIS paging, GOV.UK attachment resolver, streaming GDAL -> GeoParquet
scripts/                         Windows set-up and run script
```

### Adding a source
Create a function in `geodata/step1_extract/<family>.py`:

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

Then add a `my_source:` block with `enabled: true` to `config/1_extract.yaml`.
