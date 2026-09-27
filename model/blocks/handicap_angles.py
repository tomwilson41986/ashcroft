"""Handicap angles: where the weights and the mark disagree with the form.

A handicap weights each runner by its official rating, so a horse's chance turns on
whether its mark is right. The classic cases where it is not are visible before the
off (model/handicap_features.py, which was only ever tested against the result; the
screen against the best model's errors, research/queries/travel_residuals.py on 27
Sep, found them worth -2.3 x 10^-4 jointly, resolved):

    hca_wt_vs_mark              weight carried less what the mark implies within the
                                handicap (top weight less the rating gap): + for a
                                penalty or running from out of the handicap, - for a claim
    hca_lr_won                  won its last run
    hca_lr_win_margin_lbs       that win's margin in pounds at the trip
    hca_days_since_lr           days since the last run
    hca_lr_handicap             the last run was a handicap
    hca_mark_unchanged_after_win  won last time and runs off the same mark today
    hca_penalty_edge_lbs        in a handicap off an unchanged mark after a win: the
                                win's margin less a typical 6 lb penalty ("well in")
    hca_mark_vs_last_win        today's mark less the mark of its last win
    hca_wins_off_higher_mark    has won off a higher mark than today's

"Last run" is the horse's last run on an earlier day, so a day's own results reach
none of its features; the weights and marks are the card's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.race_shape import day_index, race_key

FEATURES = ["hca_wt_vs_mark", "hca_lr_won", "hca_lr_win_margin_lbs", "hca_days_since_lr", "hca_lr_handicap",
            "hca_mark_unchanged_after_win", "hca_penalty_edge_lbs", "hca_mark_vs_last_win",
            "hca_wins_off_higher_mark"]
POST_RACE: set[str] = set()
READS = ["raceid", "race_date", "race_time", "track", "horse_name", "race_type", "race_name", "official_rating",
         "pounds", "placing_numerical", "total_dst_bt", "dist_furlongs"]
ASSUMED_PENALTY_LBS = 6.0


def _name_codes(s: pd.Series) -> np.ndarray:
    t = s.fillna("").astype(str).str.strip().str.lower()
    bad = t.isin(["", "nan", "none"]).to_numpy()
    return np.where(bad, -1, pd.factorize(t, sort=True)[0]).astype(np.int64)


def build(df: pd.DataFrame) -> pd.DataFrame:
    from model.perf_figures import lbs_per_length, winning_margin
    n = len(df)
    day = day_index(df)
    rk = race_key(df)
    race = pd.factorize(rk, sort=True)[0]
    h = _name_codes(df["horse_name"])
    text = (df["race_type"].fillna("").astype(str) + " " + df["race_name"].fillna("").astype(str)).str.lower()
    hcap = text.str.contains("handicap").to_numpy().astype(float)
    orr = pd.to_numeric(df["official_rating"], errors="coerce").where(lambda s: s > 0).to_numpy(float)
    wt = pd.to_numeric(df["pounds"], errors="coerce").where(lambda s: s > 0).to_numpy(float)
    pos = pd.to_numeric(df["placing_numerical"], errors="coerce").to_numpy(float)
    won = np.where(np.isfinite(pos), (pos == 1).astype(float), np.nan)
    dist = pd.to_numeric(df["dist_furlongs"], errors="coerce").fillna(8.0).to_numpy(float)
    src = df[["race_date", "race_time", "track", "placing_numerical", "total_dst_bt"]].copy()
    src["raceid"] = rk.to_numpy()
    margin_lbs = np.where(pos == 1, winning_margin(src).to_numpy(float) * lbs_per_length(dist), np.nan)

    # today's card: weight against mark, within handicaps
    top_wt = pd.Series(wt).groupby(race).transform("max").to_numpy(float)
    top_or = pd.Series(orr).groupby(race).transform("max").to_numpy(float)
    wt_vs_mark = np.where(hcap > 0, wt - (top_wt - (top_or - orr)), np.nan)

    # each row's last run on an earlier day (sorted position), and the horse's wins up to it
    o = np.lexsort((race, day, h))
    hs, ds = h[o], day[o]
    idx = np.arange(n)
    new_h = np.r_[True, hs[1:] != hs[:-1]]
    new_day = new_h | np.r_[True, ds[1:] != ds[:-1]]
    day_start = np.maximum.accumulate(np.where(new_day, idx, 0))
    prev = day_start - 1
    valid = (prev >= 0) & (hs >= 0)
    valid &= np.where(prev >= 0, hs[np.maximum(prev, 0)] == hs, False)
    p = np.maximum(prev, 0)
    won_s, or_s = won[o], orr[o]
    win_or = pd.Series(np.where(won_s == 1, or_s, np.nan))
    grp = pd.Series(hs)
    last_win_or = win_or.groupby(grp).ffill().to_numpy(float)              # up to and including each row
    max_win_or = win_or.groupby(grp).cummax().groupby(grp).ffill().to_numpy(float)

    def at_prev(arr_sorted):
        return np.where(valid, arr_sorted[p], np.nan)

    lr_won = at_prev(won_s)
    lr_margin = at_prev(margin_lbs[o])
    lr_days = np.where(valid, (ds - ds[p]).astype(float), np.nan)
    lr_hcap = at_prev(hcap[o])
    lr_or = at_prev(or_s)
    unchanged = np.where(np.isfinite(lr_won) & np.isfinite(or_s) & np.isfinite(lr_or),
                         ((lr_won == 1) & (or_s == lr_or)).astype(float), np.nan)
    edge = np.where((unchanged == 1) & (hcap[o] > 0), lr_margin - ASSUMED_PENALTY_LBS, np.nan)
    vs_last_win = or_s - at_prev(last_win_or)
    mx = at_prev(max_win_or)
    higher = np.where(np.isfinite(mx) & np.isfinite(or_s), (mx > or_s).astype(float), np.nan)

    cols = {"hca_wt_vs_mark": wt_vs_mark}
    for name, arr in (("hca_lr_won", lr_won), ("hca_lr_win_margin_lbs", lr_margin), ("hca_days_since_lr", lr_days),
                      ("hca_lr_handicap", lr_hcap), ("hca_mark_unchanged_after_win", unchanged),
                      ("hca_penalty_edge_lbs", edge), ("hca_mark_vs_last_win", vs_last_win),
                      ("hca_wins_off_higher_mark", higher)):
        full = np.full(n, np.nan)
        full[o] = arr
        cols[name] = full
    new = pd.DataFrame(cols, index=df.index)[FEATURES]
    return pd.concat([df.drop(columns=[c for c in FEATURES if c in df.columns]), new], axis=1)
