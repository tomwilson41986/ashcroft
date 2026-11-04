"""Venue and competition scoring rates as of each match (build spec 6.2, v1).

- venue: runs and wickets per legal ball at the venue over the three years before the match (the venue's three-year
  bucket), shrunk to the format's population with n / (n + k), k = 50 overs' worth of balls;
- competition: runs per legal ball in the competition over the 365 days before the match, shrunk the same way. It
  carries the scoring trend (the IPL Impact Player rule from 2023) without giving the model a calendar year to
  extrapolate.

Only matches strictly before the match's start date count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cricket.ratings.shrinkage import n_over_n_plus_k

K_BALLS = 300.0


def _rolling_before(agg: pd.DataFrame, key: str, window: str) -> pd.DataFrame:
    """Sums of runs, wkts, balls over each key's matches in the ``window`` before the match's date (date-exclusive)."""
    a = agg.sort_values([key, "date"]).copy()
    out = []
    for _, g in a.groupby(key, sort=False):
        day = g.groupby("date")[["runs", "wkts", "balls"]].sum()
        roll = day.rolling(window, closed="left").sum().fillna(0)
        out.append(g[["match_id", "date"]].merge(roll, left_on="date", right_index=True, how="left"))
    return pd.concat(out, ignore_index=True)


def rates(d: pd.DataFrame, matches: pd.DataFrame, pop_before: str | None = None) -> pd.DataFrame:
    """Per match: venue_rpb, venue_wpb, comp_rpb (as of the match's start date). The population rates shrunk
    towards come from matches before ``pop_before`` (the first test season), so no later match sets them."""
    lg = d[d["legal"].astype(bool)]
    per = lg.groupby("match_id").agg(runs=("runs_total", "sum"), wkts=("team_wickets", "sum"),
                                     balls=("runs_total", "size")).reset_index()
    mm = matches[["match_id", "start_date", "venue_id", "event_name"]].copy()
    mm["date"] = pd.to_datetime(mm.start_date)
    per = per.merge(mm, on="match_id", how="right").fillna({"runs": 0, "wkts": 0, "balls": 0})
    base = per if pop_before is None else per[per.date < pd.Timestamp(pop_before)]
    pop_r = base.runs.sum() / base.balls.sum()
    pop_w = base.wkts.sum() / base.balls.sum()
    v = _rolling_before(per, "venue_id", "1095D")
    c = _rolling_before(per.assign(event_name=per.event_name.fillna("none")), "event_name", "365D")
    out = mm[["match_id"]].merge(v, on="match_id", how="left").merge(
        c[["match_id", "runs", "balls"]].rename(columns={"runs": "c_runs", "balls": "c_balls"}), on="match_id",
        how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["venue_rpb"] = n_over_n_plus_k(out.runs / out.balls, out.balls, pop_r, K_BALLS)
        out["venue_wpb"] = n_over_n_plus_k(out.wkts / out.balls, out.balls, pop_w, K_BALLS)
        out["comp_rpb"] = n_over_n_plus_k(out.c_runs / out.c_balls, out.c_balls, pop_r, K_BALLS)
    return out[["match_id", "venue_rpb", "venue_wpb", "comp_rpb"]]
