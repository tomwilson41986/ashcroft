"""Track-and-distance-normalised times with a meeting variant from a mixed model (reports/greyhound_roadmap.md F3).

The engine's GSR (greyhound/metrics.py) reads each run's calculated time against the track and distance's mean winning
time over every earlier day, less the meeting's variant: the median of the night's winners against that standard.
Two refinements here, both lag-safe:

- **Rolling standards.** The winners' **median** calculated time at the track and distance over the 28, 91 and 365
  days before the run's day (the day itself excluded), so a track whose surface or timing changed is read against its
  recent self, and one outlier winner does not move the standard.
- **A meeting variant from a mixed model.** A night's winners each read the going with noise; their mean deviation
  from the standard is a meeting (track x day) random effect plus that noise. The empirical-Bayes estimate shrinks the
  night's mean toward 0 by the number of winners: ``variant = mean_dev * n / (n + k)``, ``k = s2_within / s2_between``,
  the two variances estimated by moments from the meetings of **earlier days** only (a night with one winner is
  mostly noise; a night of twelve is mostly going).

Each run's normalised figure is in lengths (0.08 s a length), positive when faster than the standard:
``(standard - (calculated time - variant)) / 0.08``. A run's own figure enters only the features of later days (the
dog's windows are lagged a whole day), so the night's own winners are known when it is read.

Per dog, from its earlier days: the last figure, the mean of the last 3 and 6, and the best of 6, for the shrunk-variant
GSR and each rolling standard; the night's variant the dog last ran on and its reliability (n / (n + k)).

    tn_gsr_{last,mean3,mean6,best6}, tn_t28_*, tn_t91_*, tn_t365_*, tn_var_last, tn_var_rel_last, tn_t91_mean3_z,
    tn_t91_mean3_gap
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREFIX = "tn_"
WINDOWS = (28, 91, 365)
OUTLIER = 40.0                      # lengths: a figure further than this from the standard is a bad time, not a run


def rolling_standard(df: pd.DataFrame, days: int) -> pd.Series:
    """Each run's track-and-distance standard: the median calculated time of the winners there over the ``days``
    before the run's day (its own day excluded)."""
    key = df.track.astype(str) + "|" + pd.to_numeric(df.distance_m, errors="coerce").round().astype("Int64").astype(str)
    day = pd.to_datetime(df.race_date).dt.normalize()
    win = df.won.to_numpy(float) == 1
    t = pd.to_numeric(df.adjusted_time, errors="coerce").where(win)
    # the winners' times and, for every (key, day) a run is on, a placeholder: the window ends just before the day
    pts = pd.DataFrame({"key": key, "day": day, "t": t})
    pts = pd.concat([pts[pts.t.notna()], pts[["key", "day"]].drop_duplicates().assign(t=np.nan)], ignore_index=True)
    pts = pts.sort_values(["key", "day"], kind="stable")
    out = []
    for k, g in pts.groupby("key", sort=False):
        s = g.set_index("day").t
        med = s.rolling(f"{days}D", closed="left", min_periods=1).median()
        out.append(pd.DataFrame({"key": k, "day": s.index, "std": med.to_numpy()}))
    tab = pd.concat(out, ignore_index=True).drop_duplicates(["key", "day"], keep="last")
    m = pd.DataFrame({"key": key, "day": day}).merge(tab, on=["key", "day"], how="left")
    return pd.Series(m["std"].to_numpy(), index=df.index)


def meeting_variant(df: pd.DataFrame, standard: pd.Series) -> pd.DataFrame:
    """Each run's meeting variant (seconds, positive when the night ran slow) by empirical Bayes: the night's mean
    winner deviation from ``standard`` shrunk toward 0 by n / (n + k), k the within / between variance ratio from the
    meetings of earlier days. Also the reliability n / (n + k)."""
    win = df.won.to_numpy(float) == 1
    dev = (pd.to_numeric(df.adjusted_time, errors="coerce") - standard).where(win)
    dev = dev.where(dev.abs() <= OUTLIER * 0.08)
    day = pd.to_datetime(df.race_date).dt.normalize()
    mt = pd.DataFrame({"meeting": df.meeting_id.to_numpy(), "day": day, "dev": dev})
    g = mt.dropna(subset=["dev"]).groupby("meeting")
    stats = pd.DataFrame({"day": g.day.first(), "n": g.dev.size(), "mean": g.dev.mean(), "ss": g.dev.var(ddof=1) *
                          (g.dev.size() - 1)})
    stats["ss"] = stats.ss.fillna(0.0)
    # by day: cumulative moments of the meetings of earlier days only
    by_day = stats.groupby("day").agg(m=("mean", "size"), s1=("mean", "sum"), s2=("mean", lambda x: (x ** 2).sum()),
                                      ss=("ss", "sum"), dfw=("n", lambda x: (x - 1).clip(lower=0).sum()),
                                      inv_n=("n", lambda x: (1.0 / x).sum())).sort_index()
    cum = by_day.cumsum().shift(1)                                 # strictly before the day
    s2_w = cum.ss / cum.dfw.replace(0, np.nan)                     # pooled within-night variance of a winner's reading
    mean_of_means = cum.s1 / cum.m
    var_means = cum.s2 / cum.m - mean_of_means ** 2               # variance of the nights' means
    s2_b = (var_means - s2_w * cum.inv_n / cum.m).clip(lower=1e-6)  # less the noise the means carry
    k = (s2_w / s2_b).rename("k")
    stats = stats.join(k, on="day")
    rel = stats.n / (stats.n + stats.k)
    stats["variant"] = (stats["mean"] * rel).where(rel.notna(), 0.0)
    stats["rel"] = rel
    m = mt[["meeting"]].join(stats[["variant", "rel"]], on="meeting")
    return pd.DataFrame({"variant": m.variant.fillna(0.0).to_numpy(), "rel": m.rel.to_numpy()}, index=df.index)


def _race_z(v: pd.Series, race: pd.Series) -> tuple[pd.Series, pd.Series]:
    g = v.groupby(race)
    return (v - g.transform("mean")) / g.transform("std").replace(0, np.nan), v - g.transform("max")


def calculate(df: pd.DataFrame, dog_days) -> tuple[pd.DataFrame, list[str]]:
    """The block for the engine's frame (after ``_per_run``: ``std_time``, ``adjusted_time``, ``meeting_id``)."""
    from greyhound.metrics import LENGTH_S
    t = pd.to_numeric(df.adjusted_time, errors="coerce")
    bad = df.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False) if "comment" in df else \
        pd.Series(False, index=df.index)
    v = meeting_variant(df, df.std_time)
    figs = {"gsr": (df.std_time - (t - v.variant)) / LENGTH_S}
    for w in WINDOWS:
        figs[f"t{w}"] = (rolling_standard(df, w) - (t - v.variant)) / LENGTH_S
    tmp = df[["dog_id", "race_date"]].copy()
    for name, f in figs.items():
        tmp[f"_{name}"] = f.where((f.abs() <= OUTLIER) & ~bad).to_numpy()
    tmp["_var"] = v.variant.to_numpy() / LENGTH_S
    tmp["_rel"] = v.rel.to_numpy()
    dd = dog_days(tmp, [f"_{n}" for n in figs] + ["_var", "_rel"], [])
    cols = {}
    for name in figs:
        c = f"_{name}"
        cols[f"{PREFIX}{name}_last"] = dd.back(dd.window(c, 1))
        cols[f"{PREFIX}{name}_mean3"] = dd.back(dd.window(c, 3))
        cols[f"{PREFIX}{name}_mean6"] = dd.back(dd.window(c, 6))
        cols[f"{PREFIX}{name}_best6"] = dd.back(dd.window(c, 6, "max"))
    cols[f"{PREFIX}var_last"] = dd.back(dd.window("_var", 1))
    cols[f"{PREFIX}var_rel_last"] = dd.back(dd.window("_rel", 1))
    out = pd.DataFrame(cols, index=df.index)
    z, gap = _race_z(out[f"{PREFIX}t91_mean3"], df.race_id)
    out[f"{PREFIX}t91_mean3_z"], out[f"{PREFIX}t91_mean3_gap"] = z, gap
    out = out.astype(np.float32)
    return out, list(out.columns)
