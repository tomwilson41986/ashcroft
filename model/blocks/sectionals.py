"""Sectionals: how each of a horse's past Flat and all-weather runs was run, from the Blandford site's files.

The external-sectionals screen (reports/external_data_review.md, 5 Oct 2026) found that a horse's past finishing
speed and the shape of its runs cut the within-race error of the BSP forecast by 1.2-1.4% in Flat and all-weather
races (t 10.8 by month parity, 6.6 forward), against an older model, while adding nothing beyond the BSP on the
winner: the market prices them and the model could not see them. The table (data/external/sectionals.csv.gz, built by
scripts/build_sectionals.py from Abiqb93/horses-website-deployed public/data/rtv) holds the racing2 export of 30 Jun
2024 - 29 Jun 2025 and Racing TV / Timeform's files from 18 Apr 2026, every reading against the field the run met:

    sc_fsp_l1  sc_fsp_m3  sc_fsp_w5  sc_fsp_car      finishing speed less the field's (fsp_rel): the last run, the
                                                     mean of the last 3, the last 5 weighted 5..1, career
    sc_fvp_l1  sc_fvp_m3                             finished faster than it placed (fsp rank less finishing rank)
    sc_early_l1  sc_early_m3                         its place early as a share of the field (0 in front)
    sc_late_l1  sc_late_m3                           its last two furlongs less the field's mean, seconds
    sc_sl_l1  sc_sl_car                              stride length less the field's, metres
    sc_runs                                          earlier runs with a sectional record

Every window reads the horse's earlier racing days only; a jumps run has no record and leaves its window empty. A
run joins its record by day and the normalised horse name; a name two records share on one day joins neither.
Coverage: Flat and all-weather, 30 Jun 2024 - 29 Jun 2025 and from 18 Apr 2026; nothing in between.
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

PREFIX = "sc_"
FEATURES = ["sc_fsp_l1", "sc_fsp_m3", "sc_fsp_w5", "sc_fsp_car", "sc_fvp_l1", "sc_fvp_m3", "sc_early_l1",
            "sc_early_m3", "sc_late_l1", "sc_late_m3", "sc_sl_l1", "sc_sl_car", "sc_runs"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "placing_numerical"]
READINGS = ["fsp_rel", "fsp_vs_pos", "early_pct", "late_rel", "sl_rel"]

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TABLE_PATH = os.path.join(_ROOT, "data", "external", "sectionals.csv.gz")
# the backend's runs (blandford_sectionals.py --build), in horse_racing.db's sectional_runs table
DB_PATH = os.environ.get("ASHCROFT_DB") or os.path.join(_ROOT, "horse_racing.db")
# "table": the site's runs in TABLE_PATH; "synthetic": readings made up from the frame's own results, for the
# block tests' synthetic history, which the site does not hold (mechanics only, never scored)
SOURCE = "table"
TEST_REACH = {"SOURCE": "synthetic"}


def norm_name(name) -> str:
    """'1. Casts Tasha (IRE)' -> 'caststasha': betfair_prices.normalise_horse, kept here so the block and the
    table builder cannot drift apart."""
    s = str(name or "").lower()
    s = re.sub(r"^\d+\.\s*", "", s)
    s = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


@lru_cache(maxsize=2)
def _table(path: str) -> pd.DataFrame:
    """One row a (day, horse); empty when there is no table. A name twice on a day joins neither."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["day", "name_norm", *READINGS])
    t = pd.read_csv(path, usecols=["day", "name_norm", *READINGS], dtype={"name_norm": str})
    t["day"] = pd.to_datetime(t["day"], errors="coerce").dt.strftime("%Y-%m-%d")
    t = t.dropna(subset=["day", "name_norm"])
    return t.drop_duplicates(["day", "name_norm"], keep=False)


def _synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Made-up readings from the frame's results (block tests only): a winner finished fastest and led late, the
    rest at hashed values."""
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    keys = (df["horse_name"].astype(str) + "|" + pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")).to_numpy()
    u = np.array([int(hashlib.md5(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for k in keys])
    won = pos == 1
    return pd.DataFrame({"fsp_rel": np.where(won, 2.0, 4.0 * u - 2.0), "fsp_vs_pos": u - 0.5,
                         "early_pct": u, "late_rel": np.where(won, -0.5, u - 0.2), "sl_rel": 0.4 * u - 0.2},
                        index=df.index)


@lru_cache(maxsize=2)
def _db_runs(path: str) -> pd.DataFrame:
    """The backend's runs (blandford_sectionals.py --build: At The Races and Racing TV, in sectional_runs), one row
    a (day, horse), Racing TV's readings before At The Races' where both hold the run; empty when there is no
    table."""
    empty = pd.DataFrame(columns=["day", "name_norm", *READINGS])
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return empty
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            t = pd.read_sql_query(f"SELECT day, name_norm, src, {', '.join(READINGS)} FROM sectional_runs", conn)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return empty
    t["_rank"] = t["src"].map({"rtv": 0, "atr": 1}).fillna(2)
    t = t.sort_values(["day", "name_norm", "_rank"]).drop_duplicates(["day", "name_norm"], keep="first")
    return t[["day", "name_norm", *READINGS]]


def readings(df: pd.DataFrame) -> pd.DataFrame:
    """Each row's sectional readings for that day's run (NaN where none): the site's table first, the backend's
    runs where the site has none."""
    if SOURCE == "synthetic":
        return _synthetic(df)
    q = pd.DataFrame({"day": pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy(),
                      "name_norm": df["horse_name"].map(norm_name).to_numpy()})
    site = q.merge(_table(TABLE_PATH), on=["day", "name_norm"], how="left")[READINGS].apply(pd.to_numeric, errors="coerce")
    db = q.merge(_db_runs(DB_PATH), on=["day", "name_norm"], how="left")[READINGS].apply(pd.to_numeric, errors="coerce")
    has_site = site["fsp_rel"].notna().to_numpy()
    vals = np.where(has_site[:, None], site.to_numpy(dtype=float), db.to_numpy(dtype=float))
    return pd.DataFrame(vals, index=df.index, columns=READINGS)


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy()
    horse = np.where(bad, -1, _codes(name))
    H = _Days(horse, day)
    r = readings(df)

    cols: dict[str, np.ndarray] = {}
    for tag, src, windows in (("fsp", "fsp_rel", ("l1", "m3", "w5", "car")), ("fvp", "fsp_vs_pos", ("l1", "m3")),
                              ("early", "early_pct", ("l1", "m3")), ("late", "late_rel", ("l1", "m3")),
                              ("sl", "sl_rel", ("l1", "car"))):
        lad = _ladder(H, H.per_block(r[src].to_numpy(dtype=float)))
        for w in windows:
            cols[f"{PREFIX}{tag}_{w}"] = H.to_rows(lad[w])
        if tag == "fsp":
            cols["sc_runs"] = H.to_rows(lad["_cnt"])          # earlier runs with a sectional record

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
