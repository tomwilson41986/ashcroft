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
      trainer_steam_rate    lag-safe expanding steam rate for the yard
                            ("stable money"), jockey likewise
      horse_mkt_runs        number of previous runs with market data

2. Same-day morning features (``include_same_day_morning=True``) — the
   horse's *own* morning WAP today. Legitimate only if bets are placed
   after the morning market forms (which is when daily_predictions runs),
   and never for the BSP-regression target without that caveat:
      today_morning_implied_p, today_morning_rank, today_morning_vol_share

The same-run descriptors (``mkt_*`` without a lag) are outcome-adjacent
and are computed for diagnostics and the lag features only.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from model.perf_figures import ensure_raceid

RUN_DESCRIPTORS = ["mkt_bsp", "mkt_implied_p", "mkt_rank", "mkt_steam", "mkt_pp_drift", "mkt_pp_range",
                   "mkt_ip_low_ratio", "mkt_ip_hit_low", "mkt_vol_ln", "mkt_vol_share", "mkt_ip_vol_share",
                   "mkt_place_ratio", "mkt_outperform"]

MARKET_FEATURES = [
    "LR_mkt_ip_low_ratio", "horse_ip_low_3r", "horse_ip_hit_low_5r",
    "LR_mkt_steam", "horse_steam_3r", "horse_steam_rate_5r",
    "LR_mkt_pp_range", "LR_mkt_vol_share", "horse_vol_share_3r",
    "LR_mkt_outperform", "horse_outperform_3r", "horse_outperform_career",
    "LR_mkt_place_ratio", "LR_mkt_implied_p",
    "trainer_steam_rate", "jockey_steam_rate", "trainer_outperform", "horse_mkt_runs",
]
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
    d["horse_mkt_runs"] = g("mkt_bsp").apply(lambda x: x.shift(1).notna().cumsum())

    for ent, prefix in (("trainer", "trainer"), ("jockey_name", "jockey")):
        d = d.sort_values([ent, "race_date", "race_time"]).reset_index(drop=True)
        d[f"{prefix}_steam_rate"] = _lag_group(d, ent, "_steamer").apply(lambda x: x.shift(1).expanding().mean())
        if prefix == "trainer":
            d["trainer_outperform"] = _lag_group(d, ent, "mkt_outperform").apply(lambda x: x.shift(1).expanding().mean())

    if include_same_day_morning:
        mwap = pd.to_numeric(d["bfp_morningwap"], errors="coerce").where(lambda s: s > 1.0)
        inv = 1.0 / mwap
        d["today_morning_implied_p"] = inv / inv.groupby(d["raceid"]).transform("sum")
        d["today_morning_rank"] = mwap.groupby(d["raceid"]).rank(method="min")
        mv = pd.to_numeric(d["bfp_morning_vol"], errors="coerce").fillna(0)
        d["today_morning_vol_share"] = mv / mv.groupby(d["raceid"]).transform("sum").replace(0, np.nan)
        d["today_morning_ln_price"] = np.log(mwap)
    d = d.drop(columns=["_steamer"])
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
