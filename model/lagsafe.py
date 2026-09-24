"""
Expanding means that lag by *race*, not by row.

`grp[col].apply(lambda x: x.shift(1).expanding().mean())` is the repo's usual
idiom for "the average over this horse's previous runs". For a group keyed on
the horse it is correct, because a horse runs once in a race, so the previous
row is the previous race.

It is wrong for every other kind of key. Group on track-and-distance, or on
trainer, and the runners of one race sit next to each other in the sorted
frame, so `shift(1)` steps back to a rival in the *same* race and the expanding
mean picks up that rival's result. The damage is worse than it first looks:
these features are otherwise constant across a race, so the leaked same-race
outcome is the only thing that varies within the race, which is exactly the
variation a race-grouped or race-demeaned model keys on.

`race_lagged_expanding_mean` aggregates to races first and then lags, so a
runner sees every earlier race in its group and no part of its own.

"Earlier" means earlier DAYS (`LAG_UNIT = "day"`). Lagging by race let the 4.10
see the 2.00's result on the same card, which a 06:00 forecast never can: the
trainer's and jockey's strike rates, the course-and-draw cells and the pace
indices all trained on information the forecast does not have. Worse, when a
live card is priced its runners have no result yet, so a race-lagged prior
counted the day's earlier runners as losers -- a skew between training and
serving on top of the leak. A day lag removes both: nothing from the day being
predicted enters any prior, in training or in serving. A horse runs at most once
a day, so horse-level features are unchanged. `LAG_UNIT = "race"` restores the
old rule for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The unit a prior steps back by: "day" (earlier days only) or "race".
LAG_UNIT = "day"


def race_minutes(race_time) -> pd.Series:
    """Off time as minutes after midnight.

    Two reasons this is not `race_time.astype(str)`.

    It is correct. The database writes afternoon cards as '2.30' (sometimes
    '2.30.'), so a lexicographic sort puts the 1.45 before the 12.40 and the
    "previous race" is then the wrong race. Anything before 11 is read as pm,
    the same convention `betfair_prices` uses to match price files to results.

    It is also about fifty times faster where it matters. The per-race
    aggregation below takes a `min` over this column, and a `min` over a string
    dtype runs per group in Python: it was 5.4 seconds of every 5.7-second call,
    which on the full history is hours across every caller in the repo.
    """
    s = pd.Series(race_time).astype(str).str.strip().str.rstrip(".").str.replace(".", ":", regex=False)
    parts = s.str.extract(r"^(\d{1,2})(?::(\d{1,2}))?")
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce").fillna(0.0)
    h = h.where((h >= 11) | h.isna(), h + 12)
    return (h * 60.0 + m).fillna(0.0).set_axis(pd.Series(race_time).index)


def ensure_race_key(df: pd.DataFrame, race_col: str = "raceid") -> pd.Series:
    """The frame's race identifier, rebuilt from date/time/track if absent."""
    if race_col in df.columns:
        return df[race_col].astype(str)
    parts = [pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")]
    for c in ("race_time", "track"):
        if c in df.columns:
            parts.append(df[c].astype(str))
    out = parts[0]
    for p in parts[1:]:
        out = out + "_" + p
    return out


def _per_race_priors(df: pd.DataFrame, keys: list[str], value_col: str,
                     race_col: str) -> pd.DataFrame:
    """Per-row totals over the group's earlier races -- on earlier days, under
    the default `LAG_UNIT` -- as sum, count and number of races.

    The subtraction form, `group_cumsum - own`, is the obvious way to write
    this and it is not quite right. It reaches the correct answer through the
    current race's own value, so the last bits of the result depend on a number
    the row is not allowed to see. Where the prior is then ranked within a race
    the dust is the whole signal: `dist_from_preferred` is exactly zero for a
    field that has all run this trip before, so `rDistApt` was ranking rounding
    error that moved when the race's own result moved.

    Taking the cumulative total as of the PREVIOUS race never adds the current
    value in the first place, so the prior is bit-identical however today
    finishes. It also drops a pass, since the sum no longer has to be
    reconstructed from the mean."""
    d = pd.DataFrame(index=df.index)
    for k in keys:
        d[k] = df[k]
    d["_race"] = ensure_race_key(df, race_col)
    d["_v"] = pd.to_numeric(df[value_col], errors="coerce")
    d["_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.normalize()
    d["_time"] = race_minutes(df["race_time"]) if "race_time" in df.columns else 0.0

    # One cell per group per unit (a day, or a race), then each cell steps back
    # a whole cell. `_prior_races` still counts earlier RACES either way, so a
    # `min_races` threshold means the same thing under both rules.
    unit = "_date" if LAG_UNIT == "day" else "_race"
    order = keys + ["_d", "_t"] + (["_race"] if unit == "_race" else [])
    per = (d.groupby(keys + [unit], dropna=False, observed=True)
             .agg(_s=("_v", "sum"), _n=("_v", "count"), _r=("_race", "nunique"),
                  _d=("_date", "min"), _t=("_time", "min"))
             .reset_index()
             .sort_values(order, kind="stable"))
    g = per.groupby(keys, dropna=False, observed=True)
    per["_cs"] = g["_s"].cumsum()
    per["_cn"] = g["_n"].cumsum()
    per["_cr"] = g["_r"].cumsum()
    gg = per.groupby(keys, dropna=False, observed=True)
    per["_prior_sum"] = gg["_cs"].shift(1).fillna(0.0)
    per["_prior_n"] = gg["_cn"].shift(1).fillna(0.0)
    per["_prior_races"] = gg["_cr"].shift(1).fillna(0).astype(int)

    cols = ["_prior_sum", "_prior_n", "_prior_races"]
    merged = d.reset_index().merge(per[keys + [unit] + cols], on=keys + [unit], how="left")
    out = merged[cols].set_index(merged["index"].values)
    return out.reindex(df.index)


def race_lagged_expanding_mean(df: pd.DataFrame, group_col: str | list[str], value_col: str,
                               race_col: str = "raceid", min_races: int = 1) -> pd.Series:
    """Mean of `value_col` over all *earlier races* in the group.

    Returns a Series aligned to `df.index`. Rows whose group has fewer than
    `min_races` earlier races get NaN rather than a number resting on one
    observation."""
    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    p = _per_race_priors(df, keys, value_col, race_col)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(p["_prior_n"] > 0, p["_prior_sum"] / p["_prior_n"], np.nan)
    return pd.Series(np.where(p["_prior_races"] >= min_races, mean, np.nan), index=df.index)


def race_lagged_expanding_sum(df: pd.DataFrame, group_col: str | list[str], value_col: str,
                              race_col: str = "raceid") -> pd.Series:
    """Total of `value_col` over earlier races in the group (same lagging rule).

    A group with no earlier race gets NaN, not zero. LightGBM reads a NaN as
    "no history" and learns its own default direction for it, which is not the
    same thing as a horse that has run five times and won nothing."""
    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    p = _per_race_priors(df, keys, value_col, race_col)
    out = np.where(p["_prior_races"] > 0, p["_prior_sum"], np.nan)
    return pd.Series(out, index=df.index)


def race_lagged_expanding_count(df: pd.DataFrame, group_col: str | list[str], value_col: str,
                                race_col: str = "raceid") -> pd.Series:
    """How many non-null `value_col` observations the group had in earlier races."""
    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    p = _per_race_priors(df, keys, value_col, race_col)
    return pd.Series(p["_prior_n"].values, index=df.index)
