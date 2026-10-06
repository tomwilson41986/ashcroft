"""Horseracebase extras: the system builder's readings the model had not taken (the owner's list of 6 Oct).

Horseracebase's system builder lists some 800 criteria. Nearly all are already the model's in some form (a horse's,
rider's, yard's, sire's and dam's records by condition and window, the last four runs, weights against the mark,
penalties, class and trip moves, draw, pace, prize money, travel; reports/hrb_system_builder_review.md maps them).
These are the few the records can give that the model did not read:

    hx_same_surname     the rider and the trainer share a surname (a family yard)          "Same Surname"
    hx_second_car       the share of its earlier finishes that were seconds, career          "H-% (Second)"
    hx_second_m10       ... over its last 10 runs
    hx_lr_winner_lbsp   log BSP of the winner of its last race (a race won by an outsider      "(LR) Winners Odds"
                        is a weaker form line)
    hx_lr_nonfin        the share of its last race's runners that did not finish             "(LR) Non Completers"
    hx_maxfield_won     the biggest field it has won in                                      "Max Field Size Won"
    hx_lastyear_nfp     its normalised finishing position (0 for a non-finisher) when it ran   "Placing In Race Last
                        at this course and trip about a year ago (335-395 days)               Year"

Earlier days only: every reading but the surnames (today's card) comes from the horse's earlier racing days.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.blocks.form_variants import _ladder
from model.freshness_features import _Days, _col
from model.race_shape import _codes, day_index, field_size, race_key

PREFIX = "hx_"
FEATURES = ["hx_same_surname", "hx_second_car", "hx_second_m10", "hx_lr_winner_lbsp", "hx_lr_nonfin",
            "hx_maxfield_won", "hx_lastyear_nfp"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "jockey_name", "trainer", "number_of_runners",
         "placing_numerical", "bfsp", "dist_furlongs"]
LAST_YEAR_MIN, LAST_YEAR_MAX = 335, 395          # days back that count as "this race last year"
TEST_REACH = {"LAST_YEAR_MIN": 20, "LAST_YEAR_MAX": 40}


def surname(name) -> str:
    """The last word of a name, letters only, lower case ('Mrs J Harrington' -> 'harrington'; claims and
    apostrophes dropped: "R P O'Brien (3)" -> 'obrien')."""
    s = re.sub(r"\(.*?\)", " ", str(name or "")).strip().lower()
    words = [w for w in (re.sub(r"[^a-z]", "", w) for w in s.split()) if w]
    return words[-1] if words else ""


def build(df: pd.DataFrame) -> pd.DataFrame:
    day = day_index(df)
    name = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
    bad = name.isin(["", "nan", "none"]).to_numpy()
    horse = np.where(bad, -1, _codes(name))
    H = _Days(horse, day)
    race = pd.factorize(race_key(df))[0].astype(np.int64)
    n = field_size(df).to_numpy(dtype=float)
    pos = pd.to_numeric(_col(df, "placing_numerical"), errors="coerce").to_numpy(dtype=float)
    pos = np.where(pos > 0, pos, np.nan)
    fin = np.isfinite(pos)
    has_result = pd.Series(fin).groupby(race).transform("any").to_numpy()
    nonfin = ~fin & has_result
    bsp = pd.to_numeric(_col(df, "bfsp"), errors="coerce").to_numpy(dtype=float)

    cols: dict[str, np.ndarray] = {}
    js = _col(df, "jockey_name").map(surname)
    ts = _col(df, "trainer").map(surname)
    cols["hx_same_surname"] = np.where((js != "") & (ts != ""), (js == ts).astype(float), np.nan)

    v_second = np.where(fin, (pos == 2).astype(float), np.nan)
    lad = _ladder(H, H.per_block(v_second))
    cols["hx_second_car"], cols["hx_second_m10"] = H.to_rows(lad["car"]), H.to_rows(lad["m10"])

    with np.errstate(invalid="ignore", divide="ignore"):
        win_bsp = pd.Series(np.where(pos == 1, bsp, np.nan)).groupby(race).transform("max").to_numpy()
        v_wlb = np.where(has_result & np.isfinite(win_bsp) & (win_bsp > 1), np.log(win_bsp), np.nan)
        share_nf = pd.Series(nonfin.astype(float)).groupby(race).transform("sum").to_numpy() / np.where(n > 0, n, np.nan)
    cols["hx_lr_winner_lbsp"] = H.to_rows(_ladder(H, H.per_block(v_wlb))["l1"])
    cols["hx_lr_nonfin"] = H.to_rows(_ladder(H, H.per_block(np.where(has_result, share_nf, np.nan)))["l1"])

    # the biggest field won in, over earlier days
    won_n = H.per_block(np.where(pos == 1, n, np.nan), how="max")
    run_max = pd.Series(np.where(np.isfinite(won_n), won_n, -np.inf)).groupby(H.key).cummax().to_numpy()
    prev = np.where(H.valid & (H.pos >= 1), np.r_[np.nan, run_max[:-1]], np.nan)
    cols["hx_maxfield_won"] = H.to_rows(np.where(np.isfinite(prev), prev, np.nan))

    # the same course and trip about a year ago
    with np.errstate(invalid="ignore", divide="ignore"):
        nfp0 = np.where(fin, np.clip((n - pos) / np.where(n > 1, n - 1, np.nan), 0.0, 1.0),
                        np.where(nonfin, 0.0, np.nan))
    dist = pd.to_numeric(_col(df, "dist_furlongs"), errors="coerce").to_numpy(dtype=float)
    runs = pd.DataFrame({"row": np.arange(len(df)), "horse": horse, "track": _codes(_col(df, "track").astype(str)),
                         "dist": np.round(dist * 2), "day": day, "nfp0": nfp0})
    runs = runs[(runs["horse"] >= 0) & (runs["day"] >= 0) & np.isfinite(runs["dist"])]
    past = runs[np.isfinite(runs["nfp0"])][["horse", "track", "dist", "day", "nfp0"]]
    pairs = runs[["row", "horse", "track", "dist", "day"]].merge(past, on=["horse", "track", "dist"],
                                                                suffixes=("", "_p"))
    gap = pairs["day"] - pairs["day_p"]
    pairs = pairs[(gap >= LAST_YEAR_MIN) & (gap <= LAST_YEAR_MAX)].assign(off=(gap - 365).abs())
    best = pairs.sort_values(["row", "off", "day_p"]).drop_duplicates("row")
    last_year = np.full(len(df), np.nan)
    last_year[best["row"].to_numpy()] = best["nfp0"].to_numpy()
    cols["hx_lastyear_nfp"] = last_year

    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
