"""Section 3 driver: compute the configured feature groups for every postcode."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .. import io
from ..postcode import POSTCODE_KEY
from . import adjacency, nearest, network, terrain, trends
from .data import Spine, load_geoms, load_points, load_spine
from .grid import Grid, disc_cells, disc_sums

log = logging.getLogger(__name__)


class Features:
    def __init__(self, spine: Spine):
        self.spine = spine
        self.cols: dict[str, np.ndarray | pa.Array] = {}
        self.dict_rows: list[tuple[str, str, str]] = []

    def add(self, name: str, values, group: str, note: str = ""):
        v = values
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            v = pa.array(v, mask=~np.isfinite(v))
        self.cols[name] = v
        self.dict_rows.append((name, group, note))

    def table(self) -> pa.Table:
        t = pa.table({POSTCODE_KEY: self.spine.postcode, **self.cols})
        from ..collate import enforce_primary_key
        return enforce_primary_key(t)

    def dictionary(self) -> pa.Table:
        from ..descriptions import fill
        cols = list(zip(*self.dict_rows)) if self.dict_rows else [[]] * 3
        if self.dict_rows:   # blank notes filled from config/column_descriptions.yaml
            cols[2] = fill(list(cols[0]), list(cols[2]))
        return pa.table(cols, names=["column", "group", "note"])


def _r(r) -> str:
    r = int(r)
    return f"{r // 1000}km" if r % 1000 == 0 else f"{r}m"


# ------------------------------------------------------------- denominators
def postcode_population(full: Spine) -> dict[str, np.ndarray]:
    """Residents / households per postcode: each output area's census total
    shared across its live postcodes in proportion to their addresses (UPRNs);
    equally when the OA has no address counts. Addresses = UPRN count."""
    t = full.table
    n = t.num_rows
    live = t["live"].to_numpy(zero_copy_only=False) if "live" in t.column_names else np.ones(n, bool)
    uprn = pc.fill_null(t["uprn_count"], 0).to_numpy().astype("f8") if "uprn_count" in t.column_names \
        else np.ones(n)
    uprn = np.where(live, uprn, 0)
    out = {"addresses": uprn}
    if "oa21cd" not in t.column_names:
        return out
    enc = pc.dictionary_encode(pc.fill_null(t["oa21cd"], "")).combine_chunks()
    g = enc.indices.to_numpy()
    sum_u = np.bincount(g, weights=uprn)
    n_live = np.bincount(g, weights=live.astype("f8"))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(sum_u[g] > 0, uprn / sum_u[g], np.where(live, 1.0 / n_live[g], 0.0))
    for name, col in (("residents", "ts001_total"), ("households", "ts017_total_all_household_spaces")):
        if col in t.column_names:
            tot = pc.cast(t[col], pa.float64()).to_numpy(zero_copy_only=False)
            out[name] = np.nan_to_num(tot * share)
    return out


def area_population(full: Spine, area_col: str, value: np.ndarray) -> pa.Table:
    t = full.table
    enc = pc.dictionary_encode(t[area_col]).combine_chunks()
    g = enc.indices.to_numpy(zero_copy_only=False)
    valid = pc.is_valid(t[area_col]).to_numpy(zero_copy_only=False)
    tot = np.bincount(g[valid], weights=value[valid], minlength=len(enc.dictionary))
    return pa.table({area_col: enc.dictionary, "pop": tot})


# ------------------------------------------------------------------ groups
def run_densities(F: Features, cfg, data_dir, full, grid, bbox_pad, pop):
    """Counts of each point dataset (and denominators) within each radius."""
    d = cfg.get("densities") or {}
    radii = d.get("radii", [1000, 5000])
    sp = F.spine
    counts = {}
    for name in d.get("points", []):
        spec = cfg["points"][name]
        try:
            P = load_points(data_dir, spec, full, bbox=bbox_pad)
        except FileNotFoundError as e:
            log.warning("density %s skipped: %s", name, e)
            continue
        res = disc_sums(grid.bin_points(P.x, P.y), grid, sp.x, sp.y, radii)
        for r, v in res.items():
            F.add(f"{name}_within_{_r(r)}", v, "density", f"{name} within {r} m")
            counts[(name, r)] = v
    for name in d.get("denominators", ["residents", "households", "addresses"]):
        if name not in pop:
            continue
        ok = np.isfinite(full.x)
        res = disc_sums(grid.bin_points(full.x[ok], full.y[ok], pop[name][ok]), grid, sp.x, sp.y, radii)
        for r, v in res.items():
            F.add(f"{name}_within_{_r(r)}", np.round(v, 1), "density", f"estimated {name} within {r} m")
            counts[(name, r)] = v
    return counts, radii


def run_rates(F: Features, cfg, counts, radii):
    for spec in cfg.get("rates", []) or []:
        num, den, per = spec["numerator"], spec["denominator"], spec.get("per", 1000)
        for r in radii:
            a, b = counts.get((num, r)), counts.get((den, r))
            if a is None or b is None:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(b >= spec.get("min_denominator", 50), per * a / b, np.nan)
            F.add(f"{spec['name']}_{_r(r)}", np.round(v, 3), "rate",
                  f"{num} per {per} {den} within {r} m (null if fewer than {spec.get('min_denominator', 50)} {den})")


def run_nearest(F: Features, cfg, data_dir, full, bbox_pad):
    nc = cfg.get("nearest") or {}
    sp = F.spine
    step = nc.get("line_step_m", 25)
    for name in nc.get("points", []):
        try:  # not clipped: the nearest target may be far away (airports, ferries)
            P = load_points(data_dir, cfg["points"][name], full)
        except FileNotFoundError as e:
            log.warning("nearest %s skipped: %s", name, e)
            continue
        d, _ = nearest.nearest_points(sp.x, sp.y, P.x, P.y)
        F.add(f"dist_{name}_m", np.round(d, 1), "nearest", f"straight-line metres to nearest {name}")
        if nc.get("second_nearest") and name in nc["second_nearest"] and len(P.x) > 1:
            d2, _ = nearest.nearest_points(sp.x, sp.y, P.x, P.y, k=2)
            F.add(f"dist_{name}_2nd_m", np.round(d2[:, 1], 1), "nearest",
                  f"metres to the second-nearest {name} (choice / resilience)")
    for name, spec in (nc.get("lines") or {}).items():
        try:
            geoms = _coast(data_dir, spec, bbox_pad) if spec.get("coast") else load_geoms(data_dir, spec, bbox_pad)
        except FileNotFoundError as e:
            log.warning("nearest %s skipped: %s", name, e)
            continue
        d, _, _ = nearest.nearest_lines(sp.x, sp.y, geoms, step)
        F.add(f"dist_{name}_m", np.round(d, 0), "nearest", f"metres to nearest {name} (lines sampled every {step} m)")
    for name, spec in (nc.get("polygons") or {}).items():
        try:
            geoms = load_geoms(data_dir, spec, bbox_pad)
        except FileNotFoundError as e:
            log.warning("nearest %s skipped: %s", name, e)
            continue
        d = nearest.nearest_polygons(sp.x, sp.y, geoms, step)
        F.add(f"dist_{name}_m", np.round(d, 0), "nearest", f"metres to nearest {name} (0 = inside)")


def _coast(data_dir, spec, bbox):
    """Coastline = boundary of the dissolved UK country polygons.
    Caveat: the Northern Ireland / Ireland land border is included."""
    import shapely
    geoms = load_geoms(data_dir, {"file": spec.get("file", "ons_boundaries/countries.parquet")})
    line = shapely.boundary(shapely.union_all(geoms))
    parts = shapely.get_parts(line)
    if bbox is not None:
        box = shapely.box(*bbox)
        parts = parts[shapely.intersects(parts, box)]
    return parts


def run_landcover(F: Features, cfg, data_dir, grid, bbox_pad):
    lc = cfg.get("landcover") or {}
    sp = F.spine
    radii = lc.get("radii", [500, 2000])
    for name, spec in (lc.get("layers") or {}).items():
        try:
            geoms = load_geoms(data_dir, spec, bbox_pad)
        except FileNotFoundError as e:
            log.warning("landcover %s skipped: %s", name, e)
            continue
        cov = grid.coverage(geoms)
        res = disc_sums(cov, grid, sp.x, sp.y, radii)
        for r, v in res.items():
            F.add(f"share_{name}_{_r(r)}", np.round(v / disc_cells(r, grid.res), 3), "landcover",
                  f"share of land within {r} m that is {name} ({grid.res:g} m grid)")
    for name, spec in (lc.get("road_density") or {}).items():
        try:
            geoms = load_geoms(data_dir, spec, bbox_pad)
        except FileNotFoundError as e:
            log.warning("road density %s skipped: %s", name, e)
            continue
        lengths = grid.line_length(geoms)
        res = disc_sums(lengths, grid, sp.x, sp.y, radii)
        for r, v in res.items():
            area_km2 = disc_cells(r, grid.res) * grid.res ** 2 / 1e6
            F.add(f"{name}_km_per_km2_{_r(r)}", np.round(v / 1000 / area_km2, 3), "landcover",
                  f"km of {name} per km² within {r} m")


def run_network(F: Features, cfg, data_dir, full, bbox_pad):
    nw = cfg.get("network") or {}
    if not nw:
        return
    path = data_dir / "extract" / nw.get("roads", "os_open/openroads__road_link.parquet")
    if not path.exists():
        log.warning("network skipped: %s missing (enable OpenRoads in extract.yaml)", path.name)
        return
    sp = F.spine
    g = network.load_roads(path, bbox_pad)
    walk = nw.get("walk") or {}
    if walk:
        speed = walk.get("speed_kmh", 4.8) / 3.6
        mask = ~np.isin(g.cls, walk.get("exclude_classes", ["Motorway"]))
        for name in walk.get("targets", []):
            try:
                P = load_points(data_dir, cfg["points"][name], full, bbox=bbox_pad)
            except FileNotFoundError as e:
                log.warning("walk %s skipped: %s", name, e)
                continue
            with io.Timer(f"walk to {name}"):
                d = network.nearest_via_network(g, g.length, mask, P.x, P.y, sp.x, sp.y,
                                                limit=walk.get("max_m", 20000))
            F.add(f"walk_{name}_m", np.round(d, 0), "network", f"walking distance (m) to nearest {name}")
            F.add(f"walk_{name}_min", np.round(d / speed / 60, 1), "network",
                  f"walking time (min at {walk.get('speed_kmh', 4.8)} km/h) to nearest {name}")
        for name, minutes in (walk.get("count_within_min") or {}).items():
            try:
                P = load_points(data_dir, cfg["points"][name], full, bbox=bbox_pad)
            except FileNotFoundError:
                continue
            if len(P.x) > walk.get("max_count_targets", 20000):
                log.warning("walk count %s skipped: %d targets is too many for per-target searches",
                            name, len(P.x))
                continue
            with io.Timer(f"count {name} within {minutes} min walk"):
                c = network.count_within_network(g, g.length, mask, P.x, P.y, sp.x, sp.y,
                                                 limit=minutes * 60 * speed)
            F.add(f"{name}_within_{minutes}min_walk", c, "network", f"{name} reachable within {minutes} min walk")
    drive = nw.get("drive") or {}
    if drive:
        speeds = {**network.DEFAULT_SPEEDS, **(drive.get("speeds_kmh") or {})}
        v = np.array([speeds.get(c, 25) for c in g.cls], dtype="f8") / 3.6
        secs = g.length / v
        hop = drive.get("hop_speed_kmh", 20) / 3.6
        for name in drive.get("targets", []):
            try:
                P = load_points(data_dir, cfg["points"][name], full, bbox=bbox_pad)
            except FileNotFoundError as e:
                log.warning("drive %s skipped: %s", name, e)
                continue
            with io.Timer(f"drive to {name}"):
                s = network.nearest_via_network(g, secs, None, P.x, P.y, sp.x, sp.y, hop_speed=hop,
                                                limit=drive.get("max_min", 180) * 60)
            F.add(f"drive_{name}_min", np.round(s / 60, 1), "network",
                  f"free-flow drive time (min) to nearest {name}")


def run_water(F: Features, cfg, data_dir, full, bbox_pad, tif):
    w = cfg.get("water") or {}
    rivers = w.get("height_above")
    if not rivers or tif is None:
        return
    sp = F.spine
    try:
        geoms = load_geoms(data_dir, rivers, bbox_pad)
    except FileNotFoundError as e:
        log.warning("height above river skipped: %s", e)
        return
    d, pts, i = nearest.nearest_lines(sp.x, sp.y, geoms, w.get("step_m", 25))
    ok = i >= 0
    river_h = np.full(sp.n, np.nan)
    river_h[ok] = terrain.heights_at(tif, pts[i[ok], 0], pts[i[ok], 1])
    elev = F.cols.get("elevation_m")
    elev = elev.to_numpy(zero_copy_only=False) if isinstance(elev, pa.Array) else elev
    if elev is None:
        elev = terrain.heights_at(tif, sp.x, sp.y)
    F.add("height_above_nearest_river_m", np.round(elev - river_h, 1), "water",
          "elevation minus elevation of the nearest watercourse point (low values near rivers = flood-prone)")
    low = w.get("low_lying", {"height_m": 3, "within_m": 500})
    F.add("low_lying_near_river", pa.array((elev - river_h < low["height_m"]) & (d < low["within_m"])),
          "water", f"within {low['within_m']} m of a watercourse and under {low['height_m']} m above it")


def run_terrain(F: Features, cfg, data_dir):
    tc = cfg.get("terrain") or {}
    if not tc:
        return None
    tif = data_dir / "extract" / "os_open" / "terrain50.tif"
    if not tif.exists():
        zs = sorted((data_dir / "raw" / "os_open").glob("terr50*.zip"))
        z = zs[0] if zs else None
        if z is None:
            log.warning("terrain skipped: download Terrain50 (os_open products) first")
            return None
        with io.Timer("terrain mosaic"):
            terrain.build_mosaic(z, tif)
    sp = F.spine
    with io.Timer("terrain sample"), np.errstate(all="ignore"):
        res = terrain.sample(tif, sp.x, sp.y, radii=tc.get("relative_height_radii", [500, 2000]))
    notes = {"elevation_m": "metres above sea level (OS Terrain 50)", "slope_deg": "ground slope (degrees)",
             "aspect_deg": "direction the slope faces (0 = north, 90 = east)"}
    for k, v in res.items():
        F.add(k, np.round(v, 2), "terrain", notes.get(k, "height minus mean height on a ring at that radius"))
    return tif


def run_assign(F: Features, cfg, data_dir, full):
    """Give each postcode an attribute of its nearest point, e.g. the Primary Care Network of the nearest
    GP practice. Points without the attribute (e.g. practices outside any PCN) are ignored."""
    for name, spec in (cfg.get("assign") or {}).items():
        try:
            P = load_points(data_dir, cfg["points"][spec["points"]], full, columns=[spec["point_key"]])
        except FileNotFoundError as e:
            log.warning("assign %s skipped: %s", name, e)
            continue
        lk = spec["lookup"]
        f = data_dir / "extract" / lk["file"]
        if not f.exists():
            log.warning("assign %s skipped: %s missing - run 01_extract.py", name, lk["file"])
            continue
        lt = pq.read_table(f, columns=[lk["key"], lk["value"]])
        idx = pc.index_in(P.table[spec["point_key"]], value_set=lt[lk["key"]])
        vals = lt[lk["value"]].take(pc.fill_null(idx, 0))
        ok = pc.and_(pc.is_valid(idx), pc.is_valid(vals)).to_numpy(zero_copy_only=False) & np.isfinite(P.x)
        if not ok.any():
            log.warning("assign %s skipped: no %s matched %s", name, spec["points"], lk["file"])
            continue
        d, i = nearest.nearest_points(F.spine.x, F.spine.y, P.x[ok], P.y[ok])
        values = vals.filter(pa.array(ok)).take(pa.array(i))
        F.add(name, values, "assign", spec.get("note", f"{lk['value']} of the nearest {spec['points']}"))
        log.info("assign %s: %d of %d %s have a value", name, int(ok.sum()), len(ok), spec["points"])


CLIMATE_VARS = {   # HadUK-Grid annual variable -> (column stem, description)
    "rainfall": ("rainfall_mm", "annual rainfall (mm)"),
    "tas": ("temp_mean_c", "mean air temperature (°C)"),
    "tasmax": ("temp_max_c", "mean daily maximum temperature (°C)"),
    "tasmin": ("temp_min_c", "mean daily minimum temperature (°C)"),
    "sun": ("sunshine_hours", "annual sunshine (hours)"),
    "groundfrost": ("ground_frost_days", "days with ground frost"),
    "airfrost": ("air_frost_days", "days with air frost"),
    "snowLying": ("snow_lying_days", "days with snow lying"),
    "sfcWind": ("wind_ms", "mean 10 m wind speed (m/s)"),
    "hurs": ("humidity_pct", "mean relative humidity (%)"),
}


def run_climate(F: Features, cfg, data_dir):
    """Met Office HadUK-Grid 1 km annual grids sampled at each postcode (nearest land cell)."""
    cc = cfg.get("climate") or {}
    if not cc:
        return
    f = data_dir / "extract" / cc.get("file", "metoffice_haduk/haduk_1km_ann.parquet")
    if not f.exists():
        log.warning("climate skipped: %s missing - run 01_extract.py --only metoffice_haduk (needs CEDA login)",
                    f.relative_to(data_dir))
        return
    t = pq.read_table(f)
    cols = t.column_names
    years = sorted({int(m.group(1)) for c in cols if (m := re.search(r"_(\d{4})$", c))})
    if not years:
        log.warning("climate skipped: no <variable>_<year> columns in %s", f.name)
        return
    sp = F.spine
    d, i = nearest.nearest_points(sp.x, sp.y, t["easting"].to_numpy(), t["northing"].to_numpy())
    far = d > float(cc.get("max_distance_m", 2000))
    span = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])

    def at_postcodes(c):
        v = t[c].to_numpy(zero_copy_only=False).astype("f8")[i]
        v[far] = np.nan
        return v

    import warnings

    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)

    warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)

    per_var = {}
    for var in cc.get("variables", list(CLIMATE_VARS)):
        stem, desc = CLIMATE_VARS.get(var, (io.snake(var), var))
        ycols = [f"{var}_{y}" for y in years if f"{var}_{y}" in cols]
        if not ycols:
            continue
        stack = np.column_stack([at_postcodes(c) for c in ycols])
        per_var[var] = stack
        with np.errstate(all="ignore"):
            F.add(f"climate_{stem}_avg", np.round(np.nanmean(stack, axis=1), 2), "climate",
                  f"{desc}, average of {span} (HadUK-Grid 1 km)")
        if var in cc.get("latest", []):
            F.add(f"climate_{stem}_{years[-1]}", np.round(stack[:, -1], 2), "climate",
                  f"{desc} in {years[-1]} (HadUK-Grid 1 km)")
    with np.errstate(all="ignore"):
        if "tasmax" in per_var and "tasmin" in per_var:
            F.add("climate_temp_range_c", np.round(np.nanmean(per_var["tasmax"] - per_var["tasmin"], axis=1), 2),
                  "climate", f"mean daily max minus min temperature, {span} (°C): high inland, low on coasts")
        if "rainfall" in per_var and per_var["rainfall"].shape[1] > 1:
            r = per_var["rainfall"]
            F.add("climate_rainfall_cv", np.round(np.nanstd(r, axis=1) / np.nanmean(r, axis=1), 3), "climate",
                  f"year-to-year variability of annual rainfall, {span} (std / mean)")
        if "tas" in per_var and per_var["tas"].shape[1] > 1:
            tas = per_var["tas"]
            F.add(f"climate_temp_change_{years[-1]}_c", np.round(tas[:, -1] - np.nanmean(tas[:, :-1], axis=1), 2),
                  "climate", f"mean temperature in {years[-1]} minus the average of {years[0]}-{years[-2]} (°C)")
    log.info("climate: %s postcodes had no land cell within %s m", f"{int(far.sum()):,}",
             cc.get("max_distance_m", 2000))


def run_house_prices(F: Features, cfg, data_dir, full, grid, bbox_pad, counts):
    hp = cfg.get("house_prices") or {}
    if not hp:
        return
    spec = {"file": hp.get("file", "land_registry/price_paid.parquet"),
            "filter": hp.get("filter", {"ppd_category_type": "A"})}
    try:
        P = load_points(data_dir, spec, full, columns=["price", "date_of_transfer", "old_new"], bbox=bbox_pad)
    except FileNotFoundError as e:
        log.warning("house prices skipped: %s", e)
        return
    sp = F.spine
    r = hp.get("radius", 1000)
    addr = counts.get(("addresses", r))
    if addr is None:
        ok = np.isfinite(full.x)
        pop = postcode_population(full)
        addr = disc_sums(grid.bin_points(full.x[ok], full.y[ok], pop["addresses"][ok]), grid, sp.x, sp.y, [r])[r]
    res = trends.house_prices(P.x, P.y, pc.cast(P.table["price"], pa.float64()).to_numpy(zero_copy_only=False),
                              trends.days(P.table["date_of_transfer"]),
                              pc.equal(P.table["old_new"], "Y").to_numpy(zero_copy_only=False),
                              grid, sp.x, sp.y, radius=r, windows=tuple(hp.get("windows", [0, 1, 3, 5])),
                              addresses_in_radius=addr, min_sales=hp.get("min_sales", 5))
    last = int(np.nanmax(trends.days(P.table["date_of_transfer"])))
    for k, v in res.items():
        F.add(k, v, "house_prices", _house_price_note(k, r, last, hp.get("min_sales", 5)))


def _house_price_note(col: str, radius, last_day: int, min_sales: int) -> str:
    """Plain-English description of a house price column, with the dates each window covers."""
    import datetime as dt

    def dates(w: int) -> str:
        end = dt.date(1970, 1, 1) + dt.timedelta(days=last_day - round(w * 365.25))
        start = dt.date(1970, 1, 1) + dt.timedelta(days=last_day - round((w + 1) * 365.25) + 1)
        return f"{start:%b %Y} to {end:%b %Y}"

    def period(w: int) -> str:
        if w == 0:
            return f"the latest 12 months of sales ({dates(0)})"
        return f"a 12-month period {w} year{'s' if w > 1 else ''} before the latest sales ({dates(w)})"

    km = f"{int(radius) / 1000:g} km"
    where = f"within {km} of the postcode"
    std = "standard residential sales only (excludes repossessions, buy-to-let portfolios and other non-market sales)"
    if m := re.fullmatch(r"hp_change_\d+m_w0_vs_w(\d+)", col):
        w = int(m.group(1))
        return (f"How much the typical sale price {where} has changed over {w} year{'s' if w > 1 else ''}: "
                f"sales in {dates(0)} compared with sales in {dates(w)}, as a fraction "
                f"(0.10 = prices 10% higher now, -0.05 = 5% lower). "
                f"Blank where either period has fewer than {min_sales} sales")
    w = int(re.search(r"_w(\d+)$", col).group(1))
    if col.startswith("hp_price_"):
        return (f"Typical sale price (£, geometric mean, rounded to £100) of homes sold {where} in {period(w)}; "
                f"{std}. The geometric mean is less pulled up by a few very expensive sales than a simple average. "
                f"Blank where there were fewer than {min_sales} sales")
    if col.startswith("hp_sales_per_100_addresses_"):
        return (f"Homes sold {where} in {period(w)} per 100 addresses in the same area: "
                f"how often property changes hands (turnover). Standard sales only")
    if col.startswith("hp_sales_"):
        return f"Number of homes sold {where} in {period(w)}; {std}"
    if col.startswith("hp_newbuild_share_"):
        return (f"Share of homes sold {where} in {period(w)} that were new builds (0 to 1: 0.25 = a quarter). "
                f"Standard sales only")
    return f"house price measure {where}, {period(w)}"


def _lookup_by_area(F: Features, area_tab: pa.Table, key: str, group: str, notes: dict | None = None):
    sp_key = F.spine.table[key]
    idx = pc.index_in(sp_key, value_set=area_tab[key])
    for c in area_tab.column_names:
        if c == key:
            continue
        F.add(c, area_tab[c].take(idx), group, (notes or {}).get(c, ""))


def run_crime(F: Features, cfg, data_dir, full, pop):
    cc = cfg.get("crime") or {}
    if not cc:
        return None
    path = data_dir / "extract" / "police" / "crimes_lsoa_month.parquet"
    if not path.exists():
        log.warning("crime features skipped: run the police extract")
        return None
    lm = pq.read_table(path)
    lm = lm.set_column(lm.schema.get_field_index("crime_type"), "crime_type", pc.cast(lm["crime_type"], pa.string()))
    tab = None
    if cc.get("trends", True):
        tab = trends.crime_trends(lm)
    if cc.get("lsoa_rates", True) and "residents" in pop:
        months = sorted(set(lm["month"].to_pylist()))[-12:]
        last = lm.filter(pc.is_in(lm["month"], pa.array(months)))
        lp = area_population(full, "lsoa21cd", pop["residents"])
        import pandas as pd
        df = last.to_pandas()
        wide = df.pivot_table(index="lsoa21cd", columns="crime_type", values="crimes", aggfunc="sum", fill_value=0)
        wide.columns = [f"crime_rate_lsoa_{re.sub(r'[^0-9a-z]+', '_', c.lower()).strip('_')}" for c in wide.columns]
        wide["crime_rate_lsoa_total"] = wide.sum(axis=1)
        popser = pd.Series(lp["pop"].to_numpy(), index=lp["lsoa21cd"].to_pylist())
        p = popser.reindex(wide.index).to_numpy()
        rates = wide.div(np.where(p >= 100, p, np.nan), axis=0) * 1000
        rates = rates.round(2).reset_index()
        rt = pa.Table.from_pandas(rates, preserve_index=False)
        tab = rt if tab is None else tab.join(rt, keys="lsoa21cd", join_type="full outer", coalesce_keys=True)
    if tab is not None:
        _lookup_by_area(F, tab, "lsoa21cd", "crime",
                        {c: "per 1,000 LSOA residents, last 12 months" for c in tab.column_names if "rate" in c})
    return tab


def run_neighbours(F: Features, cfg, data_dir, full, pop, crime_tab):
    nb = cfg.get("neighbours") or {}
    if not nb:
        return
    key = nb.get("key", "lsoa21cd")
    try:
        import geopandas as gpd
        gdf = gpd.read_parquet(data_dir / "extract" / nb.get("file", "ons_boundaries/lsoa21.parquet"),
                               columns=[key, "geometry"])
    except Exception as e:
        log.warning("neighbours skipped: %s", e)
        return
    gdf = gdf.dropna(subset=[key]).drop_duplicates(key).reset_index(drop=True)
    codes = pa.array(gdf[key].astype(str).to_numpy())
    A = adjacency.contiguity(gdf.geometry.values, nb.get("tolerance_m", 1.0))
    br = nb.get("bridges") or {}
    if br.get("roads"):   # areas across rivers/estuaries joined by a road count as neighbours too
        rp = data_dir / "extract" / br["roads"]
        if rp.exists():
            import geopandas as gpd
            roads = gpd.read_parquet(rp, columns=["geometry"]).geometry.values
            B, _ = adjacency.road_bridges(gdf.geometry.values, roads, A)
            A = (A + B).tocsr()
            A.data[:] = 1.0
        else:
            log.warning("neighbours: bridges skipped - %s missing", br["roads"])
    lp = area_population(full, key, pop.get("residents", np.ones(len(full.x))))
    w = pc.fill_null(lp["pop"].take(pc.index_in(codes, value_set=lp[key])), 0).to_numpy()
    # variables at area level
    vars_: dict[str, np.ndarray] = {}
    lookup_path = data_dir / "collate" / "postcode_lookup.parquet"
    wanted = nb.get("lookup_columns", [])
    if wanted and lookup_path.exists():
        have = [c for c in wanted if c in pq.read_schema(lookup_path).names]
        lt = pq.read_table(lookup_path, columns=[key, *have]).filter(pc.is_valid(pc.field(key)))
        lt = lt.group_by(key, use_threads=False).aggregate([(c, "first") for c in have])
        for c in have:
            vars_[c] = pc.cast(lt[f"{c}_first"].take(pc.index_in(codes, value_set=lt[key])),
                               pa.float64()).to_numpy(zero_copy_only=False)
    if crime_tab is not None and "crime_rate_lsoa_total" in crime_tab.column_names:
        vars_["crime_rate_lsoa_total"] = pc.cast(
            crime_tab["crime_rate_lsoa_total"].take(pc.index_in(codes, value_set=crime_tab["lsoa21cd"])),
            pa.float64()).to_numpy(zero_copy_only=False)
    out = {key: codes}
    notes = {}
    # collisions per LSOA (STATS19 points located in the LSOA polygons), as annual rates per resident
    coll = _collision_counts(data_dir, nb.get("collisions") or {}, gdf.geometry.values)
    min_pop = float((nb.get("collisions") or {}).get("min_population", 100))
    if coll:
        years = coll.pop("_years")
        for m, cnt in coll.items():
            with np.errstate(invalid="ignore", divide="ignore"):
                own = np.where(w >= min_pop, 1000 * cnt / years / w, np.nan)
            out[f"{m}_per_1000_res_yr_lsoa"] = pa.array(np.round(own, 3))
            notes[f"{m}_per_1000_res_yr_lsoa"] = COLLISION_NOTES[m] + f" per 1,000 residents per year in the LSOA ({years} years of STATS19)"
    for k in nb.get("steps", [1, 2, 5]):
        M = adjacency.k_step(A, int(k))
        out[f"lsoa_neighbours_{k}step_n"] = pa.array(np.diff(M.indptr))
        out[f"lsoa_neighbours_{k}step_population"] = pa.array(np.round(adjacency.neighbourhood_sum(M, w), 0))
        for name, v in vars_.items():
            out[f"{name}_nbhd_{k}step"] = pa.array(np.round(adjacency.neighbourhood_mean(M, v, w), 3))
        if coll:   # pooled: all collisions in the neighbourhood / all its residents
            pw = adjacency.neighbourhood_sum(M, w)
            for m, cnt in coll.items():
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(pw >= min_pop, 1000 * adjacency.neighbourhood_sum(M, cnt) / years / pw, np.nan)
                c = f"{m}_per_1000_res_yr_nbhd_{k}step"
                out[c] = pa.array(np.round(r, 3))
                notes[c] = (COLLISION_NOTES[m] + f" per 1,000 residents per year across the LSOAs within {k} "
                            f"boundar{'y' if k == 1 else 'ies'} (own included, road bridges count): all of them "
                            f"divided by all their residents")
    _lookup_by_area(F, pa.table(out), key, "neighbours",
                    {c: notes.get(c, "population-weighted mean over LSOAs within k boundaries (incl. own)") for c in out})


COLLISION_NOTES = {"collisions": "Road collisions", "collisions_ksi": "Fatal or serious road collisions",
                   "casualties": "Road casualties"}


def _collision_counts(data_dir, ccfg: dict, polys) -> dict | None:
    """Collisions, fatal/serious collisions and casualties per area (point in polygon)."""
    if not ccfg.get("file"):
        return None
    f = data_dir / "extract" / ccfg["file"]
    if not f.exists():
        log.warning("neighbours: collisions skipped - %s missing", ccfg["file"])
        return None
    import shapely
    t = pq.read_table(f, columns=["easting", "northing", "collision_severity", "number_of_casualties", "collision_year"])
    x = pc.cast(t["easting"], pa.float64()).to_numpy(zero_copy_only=False)
    y = pc.cast(t["northing"], pa.float64()).to_numpy(zero_copy_only=False)
    ok = np.isfinite(x) & np.isfinite(y)
    p_idx, a_idx = shapely.STRtree(polys).query(shapely.points(x[ok], y[ok]), predicate="within")
    area = np.full(ok.sum(), -1)
    area[p_idx] = a_idx
    sev = pc.fill_null(t["collision_severity"], 3).to_numpy()[ok]
    cas = pc.fill_null(t["number_of_casualties"], 0).to_numpy().astype("f8")[ok]
    inn = area >= 0
    n = len(polys)
    years = len(set(t["collision_year"].to_pylist()) - {None}) or 1
    log.info("neighbours: %s collisions over %d years located in LSOAs", f"{int(inn.sum()):,}", years)
    return {"collisions": np.bincount(area[inn], minlength=n).astype("f8"),
            "collisions_ksi": np.bincount(area[inn & (sev <= 2)], minlength=n).astype("f8"),
            "casualties": np.bincount(area[inn], weights=cas[inn], minlength=n),
            "_years": years}


# ------------------------------------------------------------------ driver
def build(data_dir: Path, cfg: dict[str, Any]) -> tuple[pa.Table, pa.Table]:
    spine, full = load_spine(data_dir, cfg)
    F = Features(spine)
    res = float((cfg.get("global") or {}).get("grid_res", 100))
    radii_all = [*(cfg.get("densities") or {}).get("radii", [1000, 5000]),
                 *(cfg.get("landcover") or {}).get("radii", [500, 2000]),
                 (cfg.get("house_prices") or {}).get("radius", 1000)]
    pad = max(radii_all) + 2 * res
    netpad = float((cfg.get("network") or {}).get("pad_m", 15000))
    grid = Grid.around(spine.x, spine.y, res, pad)
    bbox = (float(np.min(spine.x)), float(np.min(spine.y)), float(np.max(spine.x)), float(np.max(spine.y)))
    bbox_pad = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    bbox_net = (bbox[0] - netpad, bbox[1] - netpad, bbox[2] + netpad, bbox[3] + netpad)
    pop = postcode_population(full)

    with io.Timer("densities"):
        counts, radii = run_densities(F, cfg, data_dir, full, grid, bbox_pad, pop)
    with io.Timer("rates"):
        run_rates(F, cfg, counts, radii)
    npad = float((cfg.get("nearest") or {}).get("pad_m", 50000))
    with io.Timer("nearest"):
        run_nearest(F, cfg, data_dir, full, (bbox[0] - npad, bbox[1] - npad, bbox[2] + npad, bbox[3] + npad))
    with io.Timer("landcover"):
        run_landcover(F, cfg, data_dir, grid, bbox_pad)
    with io.Timer("terrain"):
        tif = run_terrain(F, cfg, data_dir)
    with io.Timer("water"):
        run_water(F, cfg, data_dir, full, bbox_pad, tif)
    with io.Timer("climate"):
        run_climate(F, cfg, data_dir)
    with io.Timer("assign"):
        run_assign(F, cfg, data_dir, full)
    with io.Timer("network"):
        run_network(F, cfg, data_dir, full, bbox_net)
    with io.Timer("house prices"):
        run_house_prices(F, cfg, data_dir, full, grid, bbox_pad, counts)
    with io.Timer("crime"):
        crime_tab = run_crime(F, cfg, data_dir, full, pop)
    with io.Timer("neighbours"):
        run_neighbours(F, cfg, data_dir, full, pop, crime_tab)
    return F.table(), F.dictionary()
