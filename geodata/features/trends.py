"""House-price and crime trends.

House prices (per postcode, within a radius, on the grid):
  windows are 12-month periods counted back from the latest sale in the data
  (``w0`` = last 12 months, ``w1`` = 12-24 months ago, ``w3`` = 36-48 ...).
  For each window: number of sales and the geometric mean price (exp of the
  mean log price - a median cannot be summed on a grid, the geometric mean
  can, and it is similarly robust to a few very expensive sales). Changes are
  ``price_w0 / price_wk - 1``. Also new-build share and sales per 100 addresses
  in the latest window. Windows older than the extracted period come out empty:
  set land_registry ``start`` far enough back (e.g. 6 years for a 5-year change).

Crime (per LSOA, joined to postcodes):
  last 12 months vs the 12 before, by crime type, as a ratio; and seasonality
  (share of the year's crime in the peak 3 consecutive months, 0.25 = flat).
  Needs 24 months of police data for the change.
"""
from __future__ import annotations

import logging

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .grid import Grid, disc_sums

log = logging.getLogger(__name__)


def house_prices(sales_x, sales_y, price, date_days, new_build, grid: Grid, px, py, *,
                 radius: float, windows=(0, 1, 3, 5), addresses_in_radius=None, min_sales: int = 5):
    out = {}
    last = int(np.nanmax(date_days))
    logp = np.log(np.asarray(price, dtype="f8"))
    age = (last - np.asarray(date_days)) // 365.25    # whole years before the latest sale
    r = int(radius)
    base = None
    for w in windows:
        m = age == w
        if not m.any():
            log.warning("house prices: no sales %d-%d months before the latest sale - extend land_registry start",
                        12 * w, 12 * (w + 1))
            continue
        n = disc_sums(grid.bin_points(sales_x[m], sales_y[m]), grid, px, py, [radius])[radius]
        s = disc_sums(grid.bin_points(sales_x[m], sales_y[m], logp[m]), grid, px, py, [radius])[radius]
        with np.errstate(invalid="ignore", divide="ignore"):
            gm = np.where(n >= min_sales, np.exp(s / n), np.nan)
        out[f"hp_sales_{r}m_w{w}"] = n
        out[f"hp_price_{r}m_w{w}"] = np.round(gm, -2)
        if w == windows[0]:
            base = gm
            nb = disc_sums(grid.bin_points(sales_x[m & new_build], sales_y[m & new_build]), grid, px, py,
                           [radius])[radius]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"hp_newbuild_share_{r}m_w{w}"] = np.where(n > 0, np.round(nb / n, 3), np.nan)
                if addresses_in_radius is not None:
                    out[f"hp_sales_per_100_addresses_{r}m_w{w}"] = np.where(
                        addresses_in_radius > 0, np.round(100 * n / addresses_in_radius, 2), np.nan)
        elif base is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"hp_change_{r}m_w0_vs_w{w}"] = np.round(base / gm - 1, 4)
    return out


def crime_trends(lsoa_month: pa.Table, types=None):
    """Per-LSOA trend table keyed on lsoa21cd."""
    import pandas as pd
    df = lsoa_month.to_pandas()
    df["crime_type"] = df["crime_type"].astype(str)
    months = sorted(df["month"].unique())
    out = pd.DataFrame(index=pd.Index(sorted(df["lsoa21cd"].unique()), name="lsoa21cd"))
    if types:
        df = df[df["crime_type"].isin(types)]
    groups = {"total": df}
    for t_ in sorted(df["crime_type"].unique()):
        groups[t_] = df[df["crime_type"] == t_]
    if len(months) >= 24:
        recent, prior = set(months[-12:]), set(months[-24:-12])
        for name, g in groups.items():
            a = g[g["month"].isin(recent)].groupby("lsoa21cd")["crimes"].sum()
            b = g[g["month"].isin(prior)].groupby("lsoa21cd")["crimes"].sum()
            a, b = a.reindex(out.index, fill_value=0), b.reindex(out.index, fill_value=0)
            # +1 smoothing so small counts do not explode
            out[f"crime_change_12m_{_snake(name)}"] = ((a + 1) / (b + 1) - 1).round(4)
    else:
        log.warning("crime trends: %d months of police data; 24 needed for the 12-month change "
                    "(set police start earlier)", len(months))
    if len(months) >= 12:
        last12 = df[df["month"].isin(months[-12:])]
        tot = last12.groupby(["lsoa21cd", "month"])["crimes"].sum().unstack(fill_value=0)
        tot = tot.reindex(columns=months[-12:], fill_value=0)
        arr = tot.to_numpy(dtype="f8")
        roll = arr + np.roll(arr, -1, axis=1) + np.roll(arr, -2, axis=1)   # circular 3-month windows
        with np.errstate(invalid="ignore", divide="ignore"):
            peak = roll.max(axis=1) / arr.sum(axis=1)
        out["crime_peak_3m_share"] = pd.Series(np.round(peak, 3), index=tot.index).reindex(out.index)
    return pa.Table.from_pandas(out.reset_index(), preserve_index=False)


def _snake(s: str) -> str:
    import re
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def days(col) -> np.ndarray:
    """date32 column -> int days since epoch."""
    return pc.cast(col, pa.int32()).to_numpy(zero_copy_only=False).astype("f8")
