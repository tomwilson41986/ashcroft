"""
Features from the Blandford / Timeform results feed (blandford_results).

Same-day (legitimately pre-race) features:
    tf_master_pre        Timeform-style master rating before this race
    tf_adjusted_pre      pre-race adjusted rating (weight-adjusted)
    tf_master_rank       within-race rank of tf_master_pre (1 = top)
    tf_master_gap_top    top-rated minus this horse (lbs)
    tf_master_vs_or      master rating minus official rating

Lag-safe form features (previous runs only):
    LR_tf_perf, tf_perf_ewm, tf_perf_best3, tf_perf_sd     performance ratings
    LR_tf_timefig, tf_timefig_best3, tf_timefig_ewm         timefigures
    LR_tf_perf_vs_master                                    last run's perf minus its pre-race rating (improver?)
    LR_race_fsp_pct                                         finishing-speed % of the last race (race shape:
                                                            last-N-furlong speed / whole-race speed, winner)
    LR_tf_ipmin_ratio, tf_ipmin_3r                          in-running low / BSP on previous runs
    LR_tf_bsp_adv                                           Betfair vs ISP advantage last time
    tf_runs                                                 number of prior rows with Timeform data

All lag features use shift(1) per horse ordered by date/time.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

BLANDFORD_FEATURES = [
    "tf_master_pre", "tf_adjusted_pre", "tf_master_rank", "tf_master_gap_top", "tf_master_vs_or",
    "LR_tf_perf", "tf_perf_ewm", "tf_perf_best3", "tf_perf_sd", "LR_tf_timefig", "tf_timefig_best3", "tf_timefig_ewm",
    "LR_tf_perf_vs_master", "LR_race_fsp_pct", "LR_tf_ipmin_ratio", "tf_ipmin_3r", "LR_tf_bsp_adv", "tf_runs",
]


def load_blandford_frame(db_path: str, date_from: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    where = "WHERE race_results_id IS NOT NULL" + (" AND meeting_date >= ?" if date_from else "")
    q = f"""SELECT race_results_id, pre_race_master_rating, pre_race_adjusted_rating, performance_rating, timefigure,
                   finishing_time, winner_sectional, distance_sectional, distance_f, betfair_win_sp, ip_min, bsp_advantage
            FROM blandford_results {where}"""
    df = pd.read_sql_query(q, conn, params=(date_from,) if date_from else ())
    conn.close()
    return df.drop_duplicates("race_results_id")


def _ewm_by_horse(d: pd.DataFrame, col: str, half_life_days: float = 270.0) -> pd.Series:
    """Recency-weighted mean of prior values per horse (shift-1)."""
    out = np.full(len(d), np.nan)
    dates = pd.to_datetime(d["race_date"]).values.astype("datetime64[D]").astype(np.int64)
    horses = d["horse_name"].astype(str).values
    vals = d[col].values.astype(float)
    cur = None; sw = sx = 0.0; last = 0
    for i in range(len(d)):
        if horses[i] != cur:
            cur = horses[i]; sw = sx = 0.0; last = dates[i]
        if sw > 0:
            k = 0.5 ** ((dates[i] - last) / half_life_days)
            out[i] = (sx * k) / (sw * k)
        if not np.isnan(vals[i]):
            k = 0.5 ** ((dates[i] - last) / half_life_days) if sw > 0 else 1.0
            sw = sw * k + 1.0; sx = sx * k + vals[i]; last = dates[i]
    return pd.Series(out, index=d.index)


def add_blandford_features(df: pd.DataFrame, feed: pd.DataFrame | None = None, db_path: str | None = None) -> pd.DataFrame:
    """Attach Timeform-feed features to a race_results-shaped frame (needs `id`)."""
    if feed is None:
        if db_path is None:
            raise ValueError("pass feed= or db_path=")
        feed = load_blandford_frame(db_path)
    if "id" not in df.columns:
        raise ValueError("race_results frame needs the `id` column")
    d = df.merge(feed, left_on="id", right_on="race_results_id", how="left").drop(columns=["race_results_id"])
    if "raceid" not in d.columns:
        from model.perf_figures import ensure_raceid
        d = ensure_raceid(d)

    # same-day pre-race ratings
    d["tf_master_pre"] = d["pre_race_master_rating"]
    d["tf_adjusted_pre"] = d["pre_race_adjusted_rating"]
    d["tf_master_rank"] = d.groupby("raceid")["tf_master_pre"].rank(ascending=False, method="min")
    d["tf_master_gap_top"] = d.groupby("raceid")["tf_master_pre"].transform("max") - d["tf_master_pre"]
    orr = pd.to_numeric(d.get("official_rating"), errors="coerce").replace(0, np.nan) if "official_rating" in d.columns else np.nan
    d["tf_master_vs_or"] = d["tf_master_pre"] - orr

    # same-run descriptors that feed the lag features (never use un-lagged)
    fin = pd.to_numeric(d["finishing_time"], errors="coerce"); ws = pd.to_numeric(d["winner_sectional"], errors="coerce")
    ds = pd.to_numeric(d["distance_sectional"], errors="coerce"); dist = pd.to_numeric(d["distance_f"], errors="coerce")
    d["_race_fsp_pct"] = 100.0 * (ds / ws) / (dist / fin)
    d.loc[~((fin > 0) & (ws > 0) & (ds > 0) & (dist > 0)), "_race_fsp_pct"] = np.nan
    bsp = pd.to_numeric(d["betfair_win_sp"], errors="coerce"); ipm = pd.to_numeric(d["ip_min"], errors="coerce")
    d["_ipmin_ratio"] = (ipm / bsp).where((bsp > 1) & (ipm > 0))
    d["_perf_vs_master"] = d["performance_rating"] - d["pre_race_master_rating"]

    d = d.sort_values(["horse_name", "race_date", "race_time"]).reset_index(drop=True)
    g = lambda c: d.groupby("horse_name", group_keys=False)[c]
    d["LR_tf_perf"] = g("performance_rating").shift(1)
    d["tf_perf_ewm"] = _ewm_by_horse(d, "performance_rating")
    d["tf_perf_best3"] = g("performance_rating").apply(lambda x: x.shift(1).rolling(3, min_periods=1).max())
    d["tf_perf_sd"] = g("performance_rating").apply(lambda x: x.shift(1).expanding(min_periods=2).std())
    d["LR_tf_timefig"] = g("timefigure").shift(1)
    d["tf_timefig_best3"] = g("timefigure").apply(lambda x: x.shift(1).rolling(3, min_periods=1).max())
    d["tf_timefig_ewm"] = _ewm_by_horse(d, "timefigure")
    d["LR_tf_perf_vs_master"] = g("_perf_vs_master").shift(1)
    d["LR_race_fsp_pct"] = g("_race_fsp_pct").shift(1)
    d["LR_tf_ipmin_ratio"] = g("_ipmin_ratio").shift(1)
    d["tf_ipmin_3r"] = g("_ipmin_ratio").apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    d["LR_tf_bsp_adv"] = g("bsp_advantage").shift(1)
    d["tf_runs"] = g("performance_rating").apply(lambda x: x.shift(1).notna().cumsum())
    d = d.drop(columns=["_race_fsp_pct", "_ipmin_ratio", "_perf_vs_master"])
    return d.sort_values(["race_date", "race_time"]).reset_index(drop=True)
