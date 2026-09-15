"""
Performance figures on the lbs (official-rating) scale.

Shared by the ABM agent builder (model/abm/agents.py) and the
cross-classified effects model (model/effects.py) so that every module
that reasons about ability uses the same scale.

Conventions follow scripts/simulate_h2h.py:
    * lbs-per-length curve: ~3 lbs/length at 5f down to 0.75 at 20f
    * winner credit = min(winning margin in lbs, 5)  (+2 if margin unknown)
    * beaten runs floored at base - 20 lbs (eased / tailed-off runs are
      bad days, not absurd ones)

Speed conversion used by the ABM: a horse `d` lbs better than par is
`d / lbs_per_length` lengths better over the trip, i.e. it is faster by
`lengths * 2.4 m / race_length_m` as a fraction of speed.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

LBS_PER_LENGTH_POINTS = [(5.0, 3.0), (8.0, 2.0), (12.0, 1.5), (16.0, 1.0), (20.0, 0.75)]
LENGTH_METRES = 2.4
FURLONG_METRES = 201.168

#: Beaten-margin words in lengths. One table for the whole repo: two modules
#: disagreeing on what "nk" means silently puts two different performance
#: figures on the same race.
MARGIN_WORDS = {
    "nse": 0.05, "nose": 0.05, "sh": 0.1, "shd": 0.1, "sht-hd": 0.1,
    "hd": 0.2, "snk": 0.25, "nk": 0.3, "dh": 0.0, "dht": 0.0, "dist": 30.0,
}
_MARGIN_WORDS = MARGIN_WORDS        # retained for existing callers
_FRACTIONS = {"¼": 0.25, "½": 0.5, "¾": 0.75, "1/4": 0.25, "1/2": 0.5, "3/4": 0.75}


def lbs_per_length(dist_furlongs) -> np.ndarray | float:
    """Weight-for-distance scale (lbs per length), linearly interpolated."""
    xs = np.array([p[0] for p in LBS_PER_LENGTH_POINTS])
    ys = np.array([p[1] for p in LBS_PER_LENGTH_POINTS])
    d = np.asarray(dist_furlongs, dtype=float)
    out = np.interp(np.clip(d, xs[0], xs[-1]), xs, ys)
    return float(out) if np.ndim(dist_furlongs) == 0 else out


def race_length_metres(dist_furlongs) -> np.ndarray | float:
    return np.asarray(dist_furlongs, dtype=float) * FURLONG_METRES


def lbs_to_lengths(lbs, dist_furlongs):
    return np.asarray(lbs, dtype=float) / lbs_per_length(dist_furlongs)


def lbs_to_speed_factor(lbs, dist_furlongs):
    """Multiplicative speed factor equivalent to an ability edge in lbs.

    +d lbs -> the horse finishes d/lpl lengths ahead of a par horse over
    the trip, which is a speed ratio of 1 + lengths*2.4/L.
    """
    lengths = lbs_to_lengths(lbs, dist_furlongs)
    return 1.0 + lengths * LENGTH_METRES / race_length_metres(dist_furlongs)


def parse_beaten_lengths(value) -> float:
    """Parse a beaten-distance string ('2.5', '1 1/2', 'nk', 'dist') to lengths."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().lower().rstrip("l").strip()
    if s in ("", "nan", "none"):
        return np.nan
    if s in _MARGIN_WORDS:
        return _MARGIN_WORDS[s]
    for frac, val in _FRACTIONS.items():
        if s.endswith(frac):
            head = s[: -len(frac)].strip()
            base = float(head) if head else 0.0
            return base + val
    m = re.match(r"^([\d.]+)$", s)
    if m:
        return float(m.group(1))
    m = re.match(r"^(\d+)\s+(\d)/(\d)$", s)
    if m:
        return float(m.group(1)) + float(m.group(2)) / float(m.group(3))
    return np.nan


def ensure_raceid(df: pd.DataFrame) -> pd.DataFrame:
    """Add the repo-standard `raceid` (date_time_track) column if missing."""
    if "raceid" not in df.columns:
        df = df.copy()
        df["raceid"] = (
            pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")
            + "_" + df["race_time"].astype(str)
            + "_" + df["track"].astype(str)
        )
    return df


def performance_figure_lbs(
    df: pd.DataFrame,
    or_col: str = "official_rating",
    beaten_col: str = "total_dst_bt",
    place_col: str = "placing_numerical",
    dist_col: str = "dist_furlongs",
    median_or_col: str = "median_or",
    winner_credit_cap: float = 5.0,
    unknown_margin_credit: float = 2.0,
    floor_lbs: float = 20.0,
) -> pd.Series:
    """Per-run performance figure in lbs (same-day, NOT lag-safe by design).

    This is an *outcome* measure: use it as a target (effects model) or as
    a lagged input (ABM ability), never as a same-race feature.
    """
    df = ensure_raceid(df)
    base = pd.to_numeric(df[or_col], errors="coerce")
    if median_or_col in df.columns:
        base = base.where(base > 0, pd.to_numeric(df[median_or_col], errors="coerce"))
    race_med = df.groupby("raceid")[or_col].transform(
        lambda s: pd.to_numeric(s, errors="coerce").replace(0, np.nan).median()
    )
    base = base.where(base > 0, race_med)

    lpl = pd.Series(lbs_per_length(df[dist_col].fillna(8.0).values), index=df.index)
    beaten = df[beaten_col].map(parse_beaten_lengths) if beaten_col in df.columns else pd.Series(np.nan, index=df.index)
    place = pd.to_numeric(df[place_col], errors="coerce")

    # Winning margin = beaten distance of the runner-up in the same race
    second = beaten.where(place == 2)
    win_margin = df.assign(_second=second).groupby("raceid")["_second"].transform("max")

    is_winner = place == 1
    credit = np.minimum(win_margin * lpl, winner_credit_cap).fillna(unknown_margin_credit)
    loss = (beaten * lpl).fillna(0.0)
    perf = np.where(is_winner, base + credit, np.maximum(base - loss, base - floor_lbs))
    perf = pd.Series(perf, index=df.index, dtype=float)
    perf[base.isna()] = np.nan
    return perf


PERF_FIGURE_FEATURES = [
    "horse_perf_lbs_ewm", "horse_perf_lbs_last", "horse_perf_lbs_best3", "horse_perf_lbs_sd",
    "horse_perf_n", "horse_perf_vs_or", "horse_perf_ewm_vs_or",
]


def add_perf_figure_features(df: pd.DataFrame, half_life_days: float = 270.0,
                             date_col: str = "race_date", time_col: str = "race_time") -> pd.DataFrame:
    """Lag-safe summaries of a horse's *previous* performance figures (lbs).

    Adds the same-run outcome ``perf_lbs`` (never use as a feature) and:
        horse_perf_lbs_ewm    recency-weighted mean of prior figures
                              (half-life ``half_life_days``), the H2H
                              script's estimate of race-day ability
        horse_perf_lbs_last   last run's figure
        horse_perf_lbs_best3  mean of the best three prior figures
        horse_perf_lbs_sd     weighted sd of prior figures (consistency)
        horse_perf_n          number of prior rated runs
        horse_perf_vs_or      last figure minus today's official rating
                              ("well in" if positive)
        horse_perf_ewm_vs_or  ewm figure minus today's official rating
    """
    out = df.copy()
    out["perf_lbs"] = performance_figure_lbs(out)
    tmp = out[["horse_name", date_col, time_col]].reset_index(drop=True)
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    order = tmp.sort_values(["horse_name", date_col, time_col]).index.values  # positions
    dates = tmp.loc[order, date_col].values.astype("datetime64[D]").astype(np.int64)
    horses = tmp.loc[order, "horse_name"].astype(str).values
    perf = out["perf_lbs"].values.astype(float)[order]
    n = len(order)
    ewm = np.full(n, np.nan); last = np.full(n, np.nan); best3 = np.full(n, np.nan)
    sd = np.full(n, np.nan); cnt = np.zeros(n)
    state_h = None; s_w = s_wx = s_wxx = 0.0; last_day = 0; last_v = np.nan; hist = []
    for i in range(n):
        h = horses[i]
        if h != state_h:
            state_h = h; s_w = s_wx = s_wxx = 0.0; last_v = np.nan; hist = []; last_day = dates[i]
        if s_w > 0:
            decay = 0.5 ** ((dates[i] - last_day) / half_life_days)
            w_, wx_, wxx_ = s_w * decay, s_wx * decay, s_wxx * decay
            m = wx_ / w_
            ewm[i] = m; sd[i] = np.sqrt(max(wxx_ / w_ - m * m, 0.0))
            last[i] = last_v; cnt[i] = len(hist)
            best3[i] = np.mean(sorted(hist, reverse=True)[:3])
        v = perf[i]
        if not np.isnan(v):
            decay = 0.5 ** ((dates[i] - last_day) / half_life_days) if s_w > 0 else 1.0
            s_w = s_w * decay + 1.0; s_wx = s_wx * decay + v; s_wxx = s_wxx * decay + v * v
            last_day = dates[i]; last_v = v; hist.append(v)
    for name, arr in (("horse_perf_lbs_ewm", ewm), ("horse_perf_lbs_last", last), ("horse_perf_lbs_best3", best3),
                      ("horse_perf_lbs_sd", sd), ("horse_perf_n", cnt)):
        full = np.full(n, np.nan); full[order] = arr
        out[name] = full
    orr = pd.to_numeric(out["official_rating"], errors="coerce").replace(0, np.nan)
    out["horse_perf_vs_or"] = out["horse_perf_lbs_last"] - orr
    out["horse_perf_ewm_vs_or"] = out["horse_perf_lbs_ewm"] - orr
    return out
