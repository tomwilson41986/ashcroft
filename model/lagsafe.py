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
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def race_lagged_expanding_mean(df: pd.DataFrame, group_col: str | list[str], value_col: str,
                               race_col: str = "raceid", min_races: int = 1) -> pd.Series:
    """Mean of `value_col` over all *earlier races* in the group.

    Returns a Series aligned to `df.index`. Rows whose group has fewer than
    `min_races` earlier races get NaN rather than a number resting on one
    observation."""
    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    d = pd.DataFrame(index=df.index)
    for k in keys:
        d[k] = df[k]
    d["_race"] = ensure_race_key(df, race_col)
    d["_v"] = pd.to_numeric(df[value_col], errors="coerce")
    d["_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    d["_time"] = df["race_time"].astype(str) if "race_time" in df.columns else ""

    per_race = (d.groupby(keys + ["_race"], dropna=False, observed=True)
                 .agg(_s=("_v", "sum"), _n=("_v", "count"), _d=("_date", "min"), _t=("_time", "min"))
                 .reset_index()
                 .sort_values(keys + ["_d", "_t", "_race"], kind="stable"))
    g = per_race.groupby(keys, dropna=False, observed=True)
    prior_sum = g["_s"].cumsum() - per_race["_s"]
    prior_n = g["_n"].cumsum() - per_race["_n"]
    prior_races = g.cumcount()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(prior_n > 0, prior_sum / prior_n, np.nan)
    per_race["_mean"] = np.where(prior_races >= min_races, mean, np.nan)

    merged = d.reset_index().merge(per_race[keys + ["_race", "_mean"]], on=keys + ["_race"], how="left")
    return pd.Series(merged["_mean"].values, index=merged["index"].values).reindex(df.index)


def race_lagged_expanding_sum(df: pd.DataFrame, group_col: str | list[str], value_col: str,
                              race_col: str = "raceid") -> pd.Series:
    """Count-weighted sum over earlier races in the group (same lagging rule)."""
    m = race_lagged_expanding_mean(df, group_col, value_col, race_col)
    n = race_lagged_expanding_count(df, group_col, value_col, race_col)
    return m * n


def race_lagged_expanding_count(df: pd.DataFrame, group_col: str | list[str], value_col: str,
                                race_col: str = "raceid") -> pd.Series:
    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    d = pd.DataFrame(index=df.index)
    for k in keys:
        d[k] = df[k]
    d["_race"] = ensure_race_key(df, race_col)
    d["_v"] = pd.to_numeric(df[value_col], errors="coerce")
    d["_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    d["_time"] = df["race_time"].astype(str) if "race_time" in df.columns else ""
    per_race = (d.groupby(keys + ["_race"], dropna=False, observed=True)
                 .agg(_n=("_v", "count"), _d=("_date", "min"), _t=("_time", "min"))
                 .reset_index()
                 .sort_values(keys + ["_d", "_t", "_race"], kind="stable"))
    per_race["_cnt"] = per_race.groupby(keys, dropna=False, observed=True)["_n"].cumsum() - per_race["_n"]
    merged = d.reset_index().merge(per_race[keys + ["_race", "_cnt"]], on=keys + ["_race"], how="left")
    return pd.Series(merged["_cnt"].values, index=merged["index"].values).reindex(df.index)
