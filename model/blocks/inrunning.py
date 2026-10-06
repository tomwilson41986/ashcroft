"""In running: how short, and how long, each of a horse's past runs traded in running (Betfair's price files).

The owner's list of 6 Oct (horseracebase's system builder) offers "BF In Play (Min)" and "(Max)" for a horse's runs.
Betfair's price files hold them for every GB/IE run from 2018 (betfair_prices: IPMIN, IPMAX and BSP of the win
market; the place files' in-play columns are placeholders on most rows, ledger place-fields-check-1002). A beaten
horse that traded at 1.3 in running ran a race its finishing position hides; one that drifted to 900 was never
involved. RESEARCH_FRAMEWORK.md (6.2) named this the hidden in-running form the market may lack, and
model/market_features.py built it for the opt-in research path; it has never stood in front of the served model.

    ir_low_l1  ir_low_m3  ir_low_w5  ir_low_car        log of the in-running low (a winner's is about 0): the last
                                                       run, the mean of the last 3, the last 5 weighted 5..1, career
    ir_lowr_l1  ir_lowr_m3  ir_lowr_w5  ir_lowr_car    log(in-running low / BSP): how much shorter than its BSP the
                                                       race made it at its best
    ir_short_m5  ir_short_car                          the share of its beaten runs that traded at 2.0 or shorter
                                                       (the last 5 beaten runs held, career)
    ir_highr_l1  ir_highr_m3                           log(in-running high / BSP): how far it drifted at its worst
    ir_runs                                            earlier runs with an in-running record

Every window reads the horse's earlier racing days only. A run joins its price-file record by day and the
normalised horse name (betfair_prices.normalise_horse); a name two records share on one day joins neither.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from functools import lru_cache

import numpy as np
import pandas as pd

from model.blocks.form_variants import _ladder
from model.freshness_features import _Days, _col
from model.race_shape import _codes, day_index

PREFIX = "ir_"
FEATURES = (["ir_low_l1", "ir_low_m3", "ir_low_w5", "ir_low_car", "ir_lowr_l1", "ir_lowr_m3", "ir_lowr_w5",
             "ir_lowr_car", "ir_short_m5", "ir_short_car", "ir_highr_l1", "ir_highr_m3", "ir_runs"])
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "placing_numerical", "bfsp"]
SHORT = 2.0                                  # a beaten run that traded at this or shorter "should have won"

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DB_PATH = os.environ.get("ASHCROFT_DB") or os.path.join(_ROOT, "horse_racing.db")
# "db": the price files' table in DB_PATH; "synthetic": readings made up from the frame's own results, for the
# block tests' synthetic history, which no price file holds (mechanics only, never scored)
SOURCE = "db"
TEST_REACH = {"SOURCE": "synthetic"}


def norm_name(name) -> str:
    """'1. Casts Tasha (IRE)' -> 'caststasha': betfair_prices.normalise_horse, kept here so the block and the
    price-file loader cannot drift apart."""
    s = str(name or "").lower()
    s = re.sub(r"^\d+\.\s*", "", s)
    s = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


@lru_cache(maxsize=2)
def _price_table(path: str) -> pd.DataFrame:
    """The win markets' BSP and in-running range, one row a (day, horse); empty when there is no table."""
    empty = pd.DataFrame(columns=["day", "name_norm", "pf_bsp", "pf_ipmin", "pf_ipmax"])
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return empty
    try:
        with sqlite3.connect(path) as conn:
            t = pd.read_sql_query(
                "SELECT race_date, horse_norm, bsp, ipmin, ipmax FROM betfair_prices "
                "WHERE market_type = 'win' AND race_date IS NOT NULL AND horse_norm IS NOT NULL", conn)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return empty
    t = t.rename(columns={"horse_norm": "name_norm", "bsp": "pf_bsp", "ipmin": "pf_ipmin", "ipmax": "pf_ipmax"})
    t["day"] = pd.to_datetime(t["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    t = t.drop(columns="race_date").dropna(subset=["day"])
    return t.drop_duplicates(["day", "name_norm"], keep=False)        # a name twice on a day: neither


def _synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Made-up readings from the frame's results (block tests only): a winner traded at 1.01, a beaten horse at a
    hashed share of its BSP, and drifted to a hashed multiple of it."""
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(dtype=float)
    won = (pd.to_numeric(_col(df, "placing_numerical"), errors="coerce") == 1).to_numpy()
    keys = (df["horse_name"].astype(str) + "|" + pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")).to_numpy()
    u = np.array([int(hashlib.md5(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for k in keys])
    ipmin = np.where(won, 1.01, np.maximum(1.02, bsp * (0.15 + 0.85 * u)))
    ipmax = np.where(won, bsp * 1.5, bsp * (2.0 + 20.0 * u))
    return pd.DataFrame({"pf_bsp": bsp, "pf_ipmin": ipmin, "pf_ipmax": ipmax}, index=df.index)


def readings(df: pd.DataFrame) -> pd.DataFrame:
    """Each row's price-file BSP, in-running low and high (NaN where none)."""
    if SOURCE == "synthetic":
        return _synthetic(df)
    t = _price_table(DB_PATH)
    q = pd.DataFrame({"day": pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy(),
                      "name_norm": df["horse_name"].map(norm_name).to_numpy()})
    out = q.merge(t, on=["day", "name_norm"], how="left")[["pf_bsp", "pf_ipmin", "pf_ipmax"]]
    out.index = df.index
    return out.apply(pd.to_numeric, errors="coerce")


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy()
    horse = np.where(bad, -1, _codes(name))
    H = _Days(horse, day)

    r = readings(df)
    bsp = r["pf_bsp"].to_numpy(dtype=float)
    lo = r["pf_ipmin"].to_numpy(dtype=float)
    hi = r["pf_ipmax"].to_numpy(dtype=float)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    ok_lo = np.isfinite(lo) & (lo > 1.0) & (lo < 1000.0)
    ok_bsp = np.isfinite(bsp) & (bsp > 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_low = np.where(ok_lo, np.log(lo), np.nan)
        v_lowr = np.where(ok_lo & ok_bsp, np.log(lo / bsp), np.nan)
        v_highr = np.where(np.isfinite(hi) & (hi > 1.0) & ok_bsp, np.log(np.minimum(hi, 1000.0) / bsp), np.nan)
    beaten = np.isfinite(pos) & (pos > 1)
    v_short = np.where(ok_lo & beaten, (lo <= SHORT).astype(float), np.nan)

    cols: dict[str, np.ndarray] = {}
    for tag, v, windows in (("low", v_low, ("l1", "m3", "w5", "car")), ("lowr", v_lowr, ("l1", "m3", "w5", "car")),
                            ("highr", v_highr, ("l1", "m3"))):
        lad = _ladder(H, H.per_block(v))
        for w in windows:
            cols[f"{PREFIX}{tag}_{w}"] = H.to_rows(lad[w])
        if tag == "low":
            cols["ir_runs"] = H.to_rows(lad["_cnt"])          # earlier runs with an in-running record
    lad = _ladder(H, H.per_block(v_short))
    cols["ir_short_m5"], cols["ir_short_car"] = H.to_rows(lad["m5"]), H.to_rows(lad["car"])

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
