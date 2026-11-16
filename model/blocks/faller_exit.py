"""Faller exit: how well a horse was going when it fell or pulled up, from its in-running low (jumps).

About 16% of jumps runs end with no finishing position; the form book shows F or P and the served windows read the
run as last (fv_nfp0) or as nothing. The in-running market watched the horse travel: one that fell at the last when
trading 1.5 ran a different race from one pulled up at 200 (reports/discovery_1111.md 4.6; ir_ holds the in-running
low of every run without regard to how the run ended, model/blocks/inrunning.py).

For each earlier jumps run whose comment says the horse fell, unseated or was brought down (a fall), or was pulled
up, refused, ran out or was carried out (a pull-up), with a price-file record (BSP > 1, in-running low > 1, in-play
volume at least IP_VOL_FLOOR):

    x = max(0, logit(1 / low) - logit(1 / BSP))      how far the race shortened it before it left

    fx_fall_x  fx_pu_x            x at the horse's last fall / last pull-up
    fx_fall_mean  fx_pu_mean      the mean of x over its earlier falls / pull-ups
    fx_fall_half  fx_pu_half      1 when the low at that last exit was half its BSP or shorter
    fx_fall_since  fx_pu_since    racing days since that exit (1 = its last run)
    fx_nfp_m3  fx_nfp_m5  fx_nfp_car   normalised finishing position with a jumps non-finisher placed by its rank on
                                       the in-running low among the race's runners with a record (a non-finisher
                                       with no record reads as last, as fv_nfp0)

Earlier racing days only. No price record: NaN. The price files lag a day, so yesterday's run has none at 06:00
and none here either when the file is not loaded.
"""

from __future__ import annotations

import os
import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd

from model.blocks import inrunning as _ir
from model.blocks.form_variants import _ladder, career_totals
from model.blocks.jumps_form import FALL_RX, PU_RX
from model.freshness_features import _Days, _col
from model.race_shape import _codes, day_index, race_code, race_key

FEATURES = ["fx_fall_x", "fx_pu_x", "fx_fall_mean", "fx_pu_mean", "fx_fall_half", "fx_pu_half", "fx_fall_since",
            "fx_pu_since", "fx_nfp_m3", "fx_nfp_m5", "fx_nfp_car"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "race_type", "surface_type", "comment",
         "placing_numerical", "number_of_runners", "bfsp"]
IP_VOL_FLOOR = 100.0          # GBP matched in running on the runner, fixed before the first screen (12 Nov 2026)
SOURCE = "db"                 # "synthetic": the block tests' made-up readings (mechanics only, never scored)
FALL_PATTERN, PU_PATTERN = FALL_RX, PU_RX
# the test history has no fall in its comments: a phrase it does carry stands in for one (mechanics only)
TEST_REACH = {"SOURCE": "synthetic", "FALL_PATTERN": r"\bnever dangerous\b"}


@lru_cache(maxsize=2)
def _vol_table(path: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["day", "name_norm", "pf_ipvol"])
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return empty
    try:
        with sqlite3.connect(path) as conn:
            t = pd.read_sql_query("SELECT race_date, horse_norm AS name_norm, ip_vol AS pf_ipvol FROM betfair_prices "
                                  "WHERE market_type = 'win' AND race_date IS NOT NULL AND horse_norm IS NOT NULL", conn)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return empty
    t["day"] = pd.to_datetime(t["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return t.drop(columns="race_date").dropna(subset=["day"]).drop_duplicates(["day", "name_norm"], keep=False)


def _readings(df: pd.DataFrame) -> pd.DataFrame:
    """Each row's price-file BSP, in-running low and in-play volume (NaN where none)."""
    if SOURCE == "synthetic":
        return _ir._synthetic(df).assign(pf_ipvol=1000.0)
    r = _ir.readings(df)
    q = pd.DataFrame({"day": pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy(),
                      "name_norm": df["horse_name"].map(_ir.norm_name).to_numpy()})
    v = q.merge(_vol_table(_ir.DB_PATH), on=["day", "name_norm"], how="left")["pf_ipvol"]
    r["pf_ipvol"] = pd.to_numeric(v, errors="coerce").to_numpy()
    return r


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy()
    H = _Days(np.where(bad, -1, _codes(name)), day)
    r = _readings(df)
    bsp, lo = r["pf_bsp"].to_numpy(float), r["pf_ipmin"].to_numpy(float)
    vol = r["pf_ipvol"].to_numpy(float)
    jumps = race_code(df).isin(["hurdle", "chase", "nhflat"]).to_numpy()
    text = _col(df, "comment").fillna("").astype(str).str.lower()
    fall = jumps & text.str.contains(FALL_PATTERN, regex=True).to_numpy()
    pu = jumps & ~fall & text.str.contains(PU_PATTERN, regex=True).to_numpy()
    held = np.isfinite(bsp) & (bsp > 1.0) & np.isfinite(lo) & (lo > 1.0) & (lo < 1000.0)
    ok = held & np.isfinite(vol) & (vol >= IP_VOL_FLOOR)
    with np.errstate(invalid="ignore", divide="ignore"):
        logit = lambda p: np.log(p / (1.0 - p))  # noqa: E731
        x = np.where(ok, np.maximum(0.0, logit(1.0 / np.where(ok, lo, 2.0)) - logit(1.0 / np.where(ok, bsp, 2.0))),
                     np.nan)
    half = np.where(ok, (lo <= 0.5 * bsp).astype(float), np.nan)

    cols: dict[str, np.ndarray] = {}
    blocks = np.arange(len(H.u))
    for tag, flag in (("fall", fall), ("pu", pu)):
        xb = H.per_block(np.where(flag, x, np.nan))
        hb = H.per_block(np.where(flag, half, np.nan))
        last = H.last_before(np.isfinite(xb))
        has = last >= 0
        at = np.clip(last, 0, None)
        cols[f"fx_{tag}_x"] = H.to_rows(np.where(has, xb[at], np.nan))
        cols[f"fx_{tag}_half"] = H.to_rows(np.where(has, hb[at], np.nan))
        cols[f"fx_{tag}_since"] = H.to_rows(np.where(has, blocks - at, np.nan))
        s, c = career_totals(H, xb)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"fx_{tag}_mean"] = H.to_rows(np.where(c > 0, s / c, np.nan))

    # the finishing position, a jumps non-finisher placed by its in-running low
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(float)
    n = pd.to_numeric(_col(df, "number_of_runners"), errors="coerce").to_numpy(float)
    race = pd.Series(pd.factorize(race_key(df), sort=True)[0])
    rk = pd.Series(np.where(held, lo, np.nan)).groupby(race).rank(method="average").to_numpy()
    nrk = pd.Series(held.astype(float)).groupby(race).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        fin = np.where(np.isfinite(pos) & (pos >= 1) & (n > 1), np.clip((n - pos) / (n - 1), 0.0, 1.0), np.nan)
        imp = np.where(held & (nrk > 1), (nrk - rk) / (nrk - 1), 0.0)
    nonfin = ~np.isfinite(pos) | (pos < 1)
    nfp = np.where(nonfin, np.where(jumps & (fall | pu), imp, 0.0), fin)
    lad = _ladder(H, H.per_block(nfp))
    for w in ("m3", "m5", "car"):
        cols[f"fx_nfp_{w}"] = H.to_rows(lad[w])

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
