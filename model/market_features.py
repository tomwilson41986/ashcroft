"""
Market-movement features from the Betfair historic price files.

Data: the ``betfair_prices`` table loaded by betfair_prices.py (morning
WAP, pre-off WAP, BSP, pre-off and in-play highs/lows, traded volumes,
win and place markets), linked to race_results by race_results_id.

Two kinds of feature:

1. Lag-safe form features (default) — built only from a horse's, trainer's
   or jockey's *previous* runs, so they are legitimate at any decision
   time:
      LR_mkt_ip_low_ratio   how short the horse traded in-running last time
                            relative to its BSP (< 0.3 = looked the winner
                            and got beaten: hidden form)
      horse_ip_low_3r       same, mean of last three
      LR_mkt_steam          log(morning WAP / BSP) last time (> 0 = backed)
      horse_steam_rate_5r   share of last five runs that were steamers
      LR_mkt_vol_share      share of the race's pre-off volume last time
      LR_mkt_outperform     finishing position beat the market rank last time
      horse_outperform_3r   mean over last three
      LR_mkt_place_ratio    win BSP / place BSP last time (consistency read)
      LR_mkt_drift_rate     last time's drift per hour of the pre-off window
      LR_mkt_pp_vol_gk      how much the price churned before the off
      trainer_steam_rate    lag-safe expanding steam rate for the yard
                            ("stable money"), jockey likewise
      trainer_mkt_p_mean    the yard's mean BSP-implied probability, and
      trainer_mkt_ae        its actual strike rate over that expectation
                            (A/E at the market's own price), jockey likewise
      horse_mkt_runs        number of previous runs with market data

2. Same-day morning features (``include_same_day_morning=True``) — the
   horse's *own* morning WAP today. Legitimate only if bets are placed
   after the morning market forms (which is when daily_predictions runs),
   and never for the BSP-regression target without that caveat:
      today_morning_implied_p, today_morning_rank, today_morning_vol_share

The same-run descriptors (``mkt_*`` without a lag) are outcome-adjacent
and are computed for diagnostics and the lag features only.

Part 4.4 notes on two of the descriptors. ``mkt_drift_rate`` is the spec's
``drift / time elapsed``: the price files carry no timestamp for the morning
WAP, so elapsed time is taken from a fixed morning reference clock
(``MORNING_CLOCK_HOURS``) to the race's own off time, which still varies by
several hours across a card and so is not merely a rescaling of the drift.
``mkt_pp_vol`` is the pre-off volatility: the files carry the window's high,
low, average and close but not the path, so the SD of ln π is estimated from
the range (Parkinson 1980) and, in ``mkt_pp_vol_gk``, from the range together
with the net move (Garman-Klass 1980). A genuine path SD needs tick data the
``betfair_prices`` table does not hold.

Everything in this module is **Stage C** (Part 4): it is all a function of the
market's price and must never reach the fundamental model.

The entity aggregates (trainer / jockey) go through
``model.lagsafe.race_lagged_expanding_mean``, not ``shift(1).expanding()``.
A trainer regularly saddles two runners in one race, so in a frame sorted by
trainer the previous row is a *stablemate in the same race* and its result
leaks; worse, the feature is otherwise constant across the yard's runners, so
the leaked result is the only thing varying within the race.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from model.lagsafe import race_lagged_expanding_mean, race_minutes
from model.perf_figures import ensure_raceid

#: Reference clock for "the morning price", in hours after midnight. The
#: morning WAP has no timestamp in the price files; the off time does.
MORNING_CLOCK_HOURS = 9.0
MIN_DRIFT_HOURS = 1.0
#: Parkinson (1980): SD of a log price over a window from its range alone.
PARKINSON_K = 1.0 / (2.0 * np.sqrt(np.log(2.0)))
_GK_C = 2.0 * np.log(2.0) - 1.0

RUN_DESCRIPTORS = ["mkt_bsp", "mkt_implied_p", "mkt_rank", "mkt_steam", "mkt_pp_drift", "mkt_pp_range",
                   "mkt_drift_rate", "mkt_pp_vol", "mkt_pp_vol_gk", "mkt_race_vol_ln",
                   "mkt_ip_low_ratio", "mkt_ip_hit_low", "mkt_vol_ln", "mkt_vol_share", "mkt_ip_vol_share",
                   "mkt_place_ratio", "mkt_outperform"]

MARKET_FEATURES = [
    "LR_mkt_ip_low_ratio", "horse_ip_low_3r", "horse_ip_hit_low_5r",
    "LR_mkt_steam", "horse_steam_3r", "horse_steam_rate_5r",
    "LR_mkt_pp_range", "LR_mkt_vol_share", "horse_vol_share_3r",
    "LR_mkt_outperform", "horse_outperform_3r", "horse_outperform_career",
    "LR_mkt_place_ratio", "LR_mkt_implied_p",
    "LR_mkt_drift_rate", "horse_drift_rate_3r", "LR_mkt_pp_vol_gk", "horse_pp_vol_gk_3r",
    "trainer_steam_rate", "jockey_steam_rate", "trainer_outperform", "horse_mkt_runs",
    "trainer_mkt_p_mean", "trainer_mkt_ae", "jockey_mkt_p_mean", "jockey_mkt_ae",
]
#: §4.5's own warning, kept next to the features it describes: a trainer's mean
#: BSP probability explains most of the variance in that trainer's PRB, so these
#: are close to a restatement of the market. They are Stage C, they earn their
#: place only if they add calibration information the price does not, and they
#: must be validated on *future* PRB out of sample before being trusted.
CONNECTION_MARKET_FEATURES = ["trainer_mkt_p_mean", "trainer_mkt_ae", "jockey_mkt_p_mean", "jockey_mkt_ae"]
SAME_DAY_MORNING_FEATURES = ["today_morning_implied_p", "today_morning_rank", "today_morning_vol_share",
                             "today_morning_ln_price"]


def load_market_frame(db_path: str, date_from: str | None = None) -> pd.DataFrame:
    """race_results_id -> win + place market columns (one row per runner)."""
    conn = sqlite3.connect(db_path)
    where = "WHERE race_results_id IS NOT NULL" + (" AND race_date >= ?" if date_from else "")
    params = (date_from,) if date_from else ()
    win = pd.read_sql_query(f"""
        SELECT race_results_id, bsp, ppwap, morningwap, ppmax, ppmin, ipmax, ipmin, morning_vol, pp_vol, ip_vol
        FROM betfair_prices {where} AND market_type = 'win'""", conn, params=params)
    place = pd.read_sql_query(f"""
        SELECT race_results_id, bsp AS place_bsp FROM betfair_prices {where} AND market_type = 'place'""",
                              conn, params=params)
    conn.close()
    win = win.drop_duplicates("race_results_id").rename(columns={c: f"bfp_{c}" for c in win.columns if c != "race_results_id"})
    place = place.drop_duplicates("race_results_id").rename(columns={"place_bsp": "bfp_place_bsp"})
    return win.merge(place, on="race_results_id", how="left")


def off_hours(race_time) -> pd.Series:
    """Race time as hours after midnight. One parser for the repo: see
    `model.lagsafe.race_minutes` for the convention and why it is not a string."""
    return race_minutes(race_time) / 60.0


def _time_24h(race_time) -> pd.Series:
    """'2.30' -> '14:30': a sortable off time, so that a frame ordered by
    (entity, race_date, race_time) really is in running order within the day."""
    hrs = off_hours(race_time)
    hh = hrs.fillna(0.0).astype(int)
    mm = ((hrs.fillna(0.0) % 1) * 60).round().astype(int)
    return hh.astype(str).str.zfill(2) + ":" + mm.astype(str).str.zfill(2)


def compute_run_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Same-run market descriptors (outcome-adjacent; feed the lag features)."""
    d = ensure_raceid(df).copy()
    bsp = pd.to_numeric(d["bfp_bsp"], errors="coerce").where(lambda s: s > 1.0)
    d["mkt_bsp"] = bsp
    inv = 1.0 / bsp
    d["mkt_implied_p"] = inv / inv.groupby(d["raceid"]).transform("sum")
    d["mkt_rank"] = bsp.groupby(d["raceid"]).rank(method="min")
    mwap = pd.to_numeric(d["bfp_morningwap"], errors="coerce").where(lambda s: s > 1.0)
    ppwap = pd.to_numeric(d["bfp_ppwap"], errors="coerce").where(lambda s: s > 1.0)
    d["mkt_steam"] = np.log(mwap / bsp)          # > 0: shortened from the morning
    d["mkt_pp_drift"] = np.log(ppwap / bsp)      # > 0: BSP shorter than the pre-off average
    ppmax = pd.to_numeric(d["bfp_ppmax"], errors="coerce"); ppmin = pd.to_numeric(d["bfp_ppmin"], errors="coerce")
    d["mkt_pp_range"] = np.log(ppmax / ppmin).where((ppmax > 1) & (ppmin > 1))
    ipmin = pd.to_numeric(d["bfp_ipmin"], errors="coerce")
    d["mkt_ip_low_ratio"] = (ipmin / bsp).where(ipmin > 0)
    d["mkt_ip_hit_low"] = ((ipmin > 0) & (ipmin <= 2.0)).astype(float).where(ipmin.notna())
    vol = pd.to_numeric(d["bfp_pp_vol"], errors="coerce").fillna(0) + pd.to_numeric(d["bfp_morning_vol"], errors="coerce").fillna(0)
    d["mkt_vol_ln"] = np.log1p(vol)
    d["mkt_vol_share"] = vol / vol.groupby(d["raceid"]).transform("sum").replace(0, np.nan)
    ipvol = pd.to_numeric(d["bfp_ip_vol"], errors="coerce").fillna(0)
    d["mkt_ip_vol_share"] = ipvol / ipvol.groupby(d["raceid"]).transform("sum").replace(0, np.nan)
    pbsp = pd.to_numeric(d.get("bfp_place_bsp"), errors="coerce") if "bfp_place_bsp" in d.columns else pd.Series(np.nan, index=d.index)
    d["mkt_place_ratio"] = (bsp / pbsp).where(pbsp > 1.0)
    pos = pd.to_numeric(d["placing_numerical"], errors="coerce")
    n = pd.to_numeric(d["number_of_runners"], errors="coerce")
    # positive = finished better than the market's rank implied, scaled by field size
    d["mkt_outperform"] = ((d["mkt_rank"] - pos) / n.replace(0, np.nan)).where(pos.notna())

    # §4.4 drift rate: the move divided by the hours it had to happen in. The
    # elapsed window is the race's own off time measured from a fixed morning
    # clock, because the price files date the off but not the morning WAP.
    hrs = off_hours(d["race_time"]) if "race_time" in d.columns else pd.Series(np.nan, index=d.index)
    elapsed = (hrs - MORNING_CLOCK_HOURS).clip(lower=MIN_DRIFT_HOURS)
    d["mkt_drift_rate"] = d["mkt_steam"] / elapsed
    # §4.4 pre-off volatility. ln π differs from −ln(price) by a race constant,
    # so the SD of one is the SD of the other. Parkinson from the range alone;
    # Garman-Klass adds the net move, so it separates a price that churned from
    # one that walked steadily to the same place.
    d["mkt_pp_vol"] = d["mkt_pp_range"] * PARKINSON_K
    open_to_close = np.log(bsp / mwap)
    gk = 0.5 * d["mkt_pp_range"] ** 2 - _GK_C * open_to_close ** 2
    d["mkt_pp_vol_gk"] = np.sqrt(gk.clip(lower=0.0))
    # race liquidity: a selection constraint, not a signal (§4.4)
    d["mkt_race_vol_ln"] = np.log1p(vol.groupby(d["raceid"]).transform("sum"))
    # post-race, never exported: the numerator of the A/E ratios below
    d["_mkt_won"] = (pos == 1).astype(float).where(pos.notna())
    return d


def _lag_group(df: pd.DataFrame, group: str, col: str):
    return df.groupby(group, group_keys=False)[col]


def add_market_features(df: pd.DataFrame, market: pd.DataFrame | None = None, db_path: str | None = None,
                        include_same_day_morning: bool = False) -> pd.DataFrame:
    """Attach lag-safe market features to a race_results-shaped frame.

    ``market`` is the frame from load_market_frame (or pass ``db_path`` to
    load it). Rows without market data get NaN features.
    """
    if market is None:
        if db_path is None:
            raise ValueError("pass market= or db_path=")
        market = load_market_frame(db_path)
    d = df.copy()
    if "id" not in d.columns:
        raise ValueError("race_results frame needs the `id` column to join market data")
    d = d.merge(market, left_on="id", right_on="race_results_id", how="left").drop(columns=["race_results_id"])
    d = compute_run_descriptors(d)
    d = d.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
    g = lambda col: _lag_group(d, "horse_name", col)

    d["LR_mkt_ip_low_ratio"] = g("mkt_ip_low_ratio").shift(1)
    d["horse_ip_low_3r"] = g("mkt_ip_low_ratio").apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    d["horse_ip_hit_low_5r"] = g("mkt_ip_hit_low").apply(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    d["LR_mkt_steam"] = g("mkt_steam").shift(1)
    d["horse_steam_3r"] = g("mkt_steam").apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    d["_steamer"] = (d["mkt_steam"] > 0.1).astype(float).where(d["mkt_steam"].notna())
    d["horse_steam_rate_5r"] = g("_steamer").apply(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    d["LR_mkt_pp_range"] = g("mkt_pp_range").shift(1)
    d["LR_mkt_vol_share"] = g("mkt_vol_share").shift(1)
    d["horse_vol_share_3r"] = g("mkt_vol_share").apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    d["LR_mkt_outperform"] = g("mkt_outperform").shift(1)
    d["horse_outperform_3r"] = g("mkt_outperform").apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    d["horse_outperform_career"] = g("mkt_outperform").apply(lambda x: x.shift(1).expanding().mean())
    d["LR_mkt_place_ratio"] = g("mkt_place_ratio").shift(1)
    d["LR_mkt_implied_p"] = g("mkt_implied_p").shift(1)
    d["LR_mkt_drift_rate"] = g("mkt_drift_rate").shift(1)
    d["horse_drift_rate_3r"] = g("mkt_drift_rate").apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    d["LR_mkt_pp_vol_gk"] = g("mkt_pp_vol_gk").shift(1)
    d["horse_pp_vol_gk_3r"] = g("mkt_pp_vol_gk").apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    d["horse_mkt_runs"] = g("mkt_bsp").apply(lambda x: x.shift(1).notna().cumsum())

    # Trainer and jockey aggregates. These keys repeat inside a race — a yard
    # runs two, and sorting by trainer parks the stablemates next to each other
    # — so `shift(1).expanding()` would read a rival's result out of the race
    # being predicted. race_lagged_expanding_mean aggregates to races first and
    # then lags, so a runner sees every earlier race and no part of its own.
    # `lag_src` carries a sortable 24h off time so that within a day the
    # ordering is the real running order.
    lag_src = d.assign(race_time=_time_24h(d["race_time"])) if "race_time" in d.columns else d
    for ent, prefix in (("trainer", "trainer"), ("jockey_name", "jockey")):
        if ent not in d.columns:
            continue
        d[f"{prefix}_steam_rate"] = race_lagged_expanding_mean(lag_src, ent, "_steamer")
        d[f"{prefix}_mkt_p_mean"] = race_lagged_expanding_mean(lag_src, ent, "mkt_implied_p")
        # §4.5 A/E: strike rate actually achieved over the strike rate the
        # market's own prices expected of this yard's / rider's book.
        d[f"{prefix}_mkt_ae"] = race_lagged_expanding_mean(lag_src, ent, "_mkt_won") / d[f"{prefix}_mkt_p_mean"].where(lambda x: x > 1e-6)
        if prefix == "trainer":
            d["trainer_outperform"] = race_lagged_expanding_mean(lag_src, ent, "mkt_outperform")
    if "sire" in d.columns:
        d["sire_mkt_p_mean"] = race_lagged_expanding_mean(lag_src, "sire", "mkt_implied_p")
        d["sire_mkt_ae"] = race_lagged_expanding_mean(lag_src, "sire", "_mkt_won") / d["sire_mkt_p_mean"].where(lambda x: x > 1e-6)

    if include_same_day_morning:
        mwap = pd.to_numeric(d["bfp_morningwap"], errors="coerce").where(lambda s: s > 1.0)
        inv = 1.0 / mwap
        d["today_morning_implied_p"] = inv / inv.groupby(d["raceid"]).transform("sum")
        d["today_morning_rank"] = mwap.groupby(d["raceid"]).rank(method="min")
        mv = pd.to_numeric(d["bfp_morning_vol"], errors="coerce").fillna(0)
        d["today_morning_vol_share"] = mv / mv.groupby(d["raceid"]).transform("sum").replace(0, np.nan)
        d["today_morning_ln_price"] = np.log(mwap)
    d = d.drop(columns=["_steamer", "_mkt_won"], errors="ignore")
    return d.sort_values(["race_date", "race_time"]).reset_index(drop=True)


def market_movement_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: win rate and ROI at BSP by steam/drift decile (is the
    move informative beyond the price?). Uses same-run descriptors."""
    d = compute_run_descriptors(df)
    d = d.dropna(subset=["mkt_steam", "mkt_bsp"])
    d["decile"] = pd.qcut(d["mkt_steam"], 10, labels=False, duplicates="drop")
    won = pd.to_numeric(d["placing_numerical"], errors="coerce") == 1
    d["ret"] = np.where(won, d["mkt_bsp"] - 1, -1.0)
    return d.groupby("decile").agg(n=("ret", "size"), mean_steam=("mkt_steam", "mean"), win_rate=("ret", lambda r: (r > 0).mean()),
                                   roi_at_bsp=("ret", "mean"), mean_implied_p=("mkt_implied_p", "mean")).reset_index()
