"""Handicap angles: where the weights and the mark disagree with the form.

A handicap weights each runner by its official rating, so a horse's chance
turns on whether its mark is right. Two situations are classically where it is
not, and both are visible before the off:

  Under a penalty. A horse that wins again before the handicapper reassesses it
  runs off its old mark plus a fixed penalty (typically 5-7 lb). When its win was
  worth more than the penalty -- a wide margin, measured in pounds at the trip --
  it is "well in": the market has to guess the rise the handicapper has not yet
  made.

  Wrong at the weights. Within a race, weight carried should follow the mark
  (top weight minus the rating gap). Carrying more than the mark implies means a
  penalty or running from out of the handicap; carrying less means a claim.

Every feature uses only the horse's earlier runs and today's declared weights
and marks; the race being predicted contributes nothing but its card.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.lagsafe import race_minutes
from model.perf_figures import lbs_per_length, parse_beaten_lengths

HANDICAP_FEATURES = ["hc_is_handicap", "hc_wt_vs_mark", "hc_lr_won", "hc_lr_win_margin_lbs", "hc_days_since_lr",
                     "hc_mark_unchanged_after_win", "hc_penalty_edge_lbs", "hc_mark_vs_last_win",
                     "hc_wins_off_higher_mark", "hc_lr_handicap"]
ASSUMED_PENALTY_LBS = 6.0


def _race_key(df: pd.DataFrame) -> pd.Series:
    if "raceid" in df.columns:
        return df["raceid"].astype(str)
    return (pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d") + "|" + df["track"].astype(str)
            + "|" + df["race_time"].astype(str))


def add_handicap_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    race = _race_key(df)
    d = pd.DataFrame(index=df.index)
    d["_race"] = race
    d["_h"] = df["horse_name"].astype(str)
    d["_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    d["_t"] = race_minutes(df["race_time"]) if "race_time" in df.columns else 0.0
    rt = df.get("race_type", pd.Series("", index=df.index)).astype(str).str.lower()
    rn = df.get("race_name", pd.Series("", index=df.index)).astype(str).str.lower()
    d["_hcap"] = (rt.str.contains("handicap") | rn.str.contains("handicap")).astype(float)
    orr = pd.to_numeric(df.get("official_rating"), errors="coerce").replace(0, np.nan)
    wt = pd.to_numeric(df.get("pounds"), errors="coerce").replace(0, np.nan)
    d["_or"], d["_wt"] = orr, wt
    pos = pd.to_numeric(df.get("placing_numerical"), errors="coerce")
    d["_won"] = (pos == 1).astype(float).where(pos.notna())
    dist = pd.to_numeric(df.get("dist_furlongs"), errors="coerce")

    # winning margin: the runner-up's beaten distance, in lengths and then pounds at the trip
    lb = df.get("total_dst_bt", pd.Series(np.nan, index=df.index)).map(parse_beaten_lengths).astype(float)
    second = lb.where(pos == 2)
    margin = second.groupby(race).transform("max")
    d["_margin_lbs"] = (margin * lbs_per_length(dist.fillna(8.0).to_numpy())).where(pos == 1)

    # today's card: weight against mark, within handicaps
    top_wt = wt.groupby(race).transform("max")
    top_or = orr.groupby(race).transform("max")
    d["hc_is_handicap"] = d["_hcap"]
    d["hc_wt_vs_mark"] = (wt - (top_wt - (top_or - orr))).where(d["_hcap"] > 0)

    # the horse's own earlier runs
    d = d.sort_values(["_h", "_date", "_t"], kind="stable")
    g = d.groupby("_h", sort=False)
    d["hc_lr_won"] = g["_won"].shift(1)
    d["hc_lr_win_margin_lbs"] = g["_margin_lbs"].shift(1)
    d["hc_days_since_lr"] = (d["_date"] - g["_date"].shift(1)).dt.days
    d["hc_lr_handicap"] = g["_hcap"].shift(1)
    lr_or = g["_or"].shift(1)
    d["hc_mark_unchanged_after_win"] = ((d["hc_lr_won"] == 1) & (d["_or"] == lr_or)).astype(float).where(
        d["hc_lr_won"].notna() & d["_or"].notna() & lr_or.notna())
    # a win worth more than the penalty, while the mark has not moved: "well in"
    d["hc_penalty_edge_lbs"] = (d["hc_lr_win_margin_lbs"] - ASSUMED_PENALTY_LBS).where(
        (d["hc_mark_unchanged_after_win"] == 1) & (d["_hcap"] > 0))
    # today's mark against the mark of the horse's last WIN (lower = better treated)
    win_or = d["_or"].where(d["_won"] == 1)
    last_win_or = win_or.groupby(d["_h"], sort=False).ffill()
    prev_win_or = last_win_or.groupby(d["_h"], sort=False).shift(1)
    d["hc_mark_vs_last_win"] = d["_or"] - prev_win_or
    # has it ever won off a higher mark than today's (the highest winning mark so far)
    # (cummax leaves NaN on the non-winning rows, so carry the running maximum forward before lagging it)
    max_win_or = (win_or.groupby(d["_h"], sort=False).cummax().groupby(d["_h"], sort=False).ffill()
                  .groupby(d["_h"], sort=False).shift(1))
    d["hc_wins_off_higher_mark"] = (max_win_or > d["_or"]).astype(float).where(max_win_or.notna() & d["_or"].notna())

    res = df.copy()
    for c in HANDICAP_FEATURES:
        res[c] = d[c].reindex(df.index)
    return res, list(HANDICAP_FEATURES)
