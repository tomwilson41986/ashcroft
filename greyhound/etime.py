"""Expected final time (the owner, 3 Nov 2026: "should we develop features like 'expected final time'. And many
metrics, models and variants around that"). Each dog's time today at today's track and distance, from its earlier
runs, with its uncertainty, against the field, and the chance it is the fastest.

**Normalising a run.** Each earlier run's calculated time, less its night's meeting variant (the mixed-model variant,
``greyhound.times.meeting_variant``), as a fraction of its track and distance's standard (the winners' mean calculated
time there on earlier days, ``std_time``): ``rel = (time - variant) / standard - 1``, negative when faster. A fraction
carries across trips (a run 1% slow at 480 m is read as 1% slow at 500 m), so it is projected onto today's standard:
``expected = standard_today * (1 + rel_estimate)``. The sectional and the run-in the same way, against their own
standards (``std_sec``, ``std_runin``), and their sum a variant of the total.

**Estimates of rel**, from the dog's earlier days only: the last run, the mean of the last 3, a recency-weighted mean
of the last 6 (weights 6, 5, .. 1), the best of 6, and an empirical-Bayes estimate: the dog's mean over all earlier
runs shrunk toward the mean rel of today's grade at today's track (earlier days) by ``K`` runs.

**Uncertainty.** The dog's spread of rel over its last 6, shrunk toward the population's within-dog spread (each run
against the dog's mean before it, earlier days) by ``K`` runs; and the runs behind it.

**Against the field** (the shrunk estimate): expected time minus the field's best and minus the second best, in
lengths (0.08 s), the rank (1 fastest), the z-score in the race; and the chance the dog's time is the fastest, its time
read as normal(expected, sd) independently of the others (``greyhound.ability.p_best``, fixed seed).

Not built: a learned variant (a walk-forward LightGBM predicting the normalised time from the dog's lagged features),
which would fit several models over every run inside the cache build; it belongs in the fast loop as an experiment.

    et_<est>_s, et_<est>_rel for est in last, mean3, wmean6, best6, eb; et_split_s, et_sec_eb_s, et_runin_eb_s,
    et_sd_rel, et_sd_s, et_n, et_gap_best_len, et_gap_second_len, et_rank, et_z, et_p_fastest
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.lagsafe import race_lagged_expanding_count, race_lagged_expanding_mean

PREFIX = "et_"
K = 3.0                     # runs of prior: the dog's mean toward its grade at the track, and its spread
OUTLIER = 0.1               # a run more than 10% off its standard is a bad time or a wrong trip, not a run
SIMS = 400
SEED = 20261103


def _rel(df: pd.DataFrame, variant: pd.Series) -> dict[str, pd.Series]:
    bad = df.comment.fillna("").str.contains(r"\(Handslip\)|Fell|Fll|BrkLine", case=False) if "comment" in df else \
        pd.Series(False, index=df.index)
    t = pd.to_numeric(df.adjusted_time, errors="coerce")
    sec = pd.to_numeric(df.sectional, errors="coerce")
    out = {"tot": (t - variant) / df.std_time - 1.0, "sec": sec / df.std_sec - 1.0,
           "runin": ((t - sec) - variant) / df.std_runin - 1.0}
    return {k: v.where((v.abs() <= OUTLIER) & ~bad & ~df.is_trial.astype(bool)) for k, v in out.items()}


def _wmean6(dd, col: str) -> pd.Series:
    num = den = 0
    for k, w in zip(range(1, 7), (6, 5, 4, 3, 2, 1)):
        x = dd.prev(col, k)
        num = num + x.fillna(0.0) * w
        den = den + x.notna() * w
    return (num / den.replace(0, np.nan)) if not isinstance(den, int) else pd.Series(np.nan, index=dd.dd.index)


def calculate(df: pd.DataFrame, dog_days) -> tuple[pd.DataFrame, list[str]]:
    from greyhound.ability import p_best
    from greyhound.metrics import LENGTH_S
    from greyhound.times import meeting_variant
    v = meeting_variant(df, df.std_time).variant
    rel = _rel(df, v)
    tmp = df[["dog_id", "race_date"]].copy()
    for k, s in rel.items():
        tmp[f"_{k}"] = s.to_numpy()
    dd = dog_days(tmp, ["_tot", "_sec", "_runin"], [])
    est = {"last": dd.window("_tot", 1), "mean3": dd.window("_tot", 3), "wmean6": _wmean6(dd, "_tot"),
           "best6": dd.window("_tot", 6, "min")}
    est = {k: pd.Series(dd.back(s), index=df.index) for k, s in est.items()}
    sd6 = pd.Series(dd.back(dd.window("_tot", 6, "std")), index=df.index)
    n6 = pd.Series(dd.back(dd.prev("_tot").notna().groupby(dd.dd.dog_id).transform(
        lambda x: x.rolling(6, min_periods=1).sum())), index=df.index)
    # the empirical-Bayes estimate: the dog's mean over earlier days, shrunk toward today's grade at today's track
    d = df.assign(_tot=rel["tot"], _sec=rel["sec"], _runin=rel["runin"])
    cols = {}
    eb = {}
    for part in ("tot", "sec", "runin"):
        m = race_lagged_expanding_mean(d, "dog_id", f"_{part}")
        n = race_lagged_expanding_count(d, "dog_id", f"_{part}")
        prior = race_lagged_expanding_mean(d, ["track", "race_class"], f"_{part}")
        pop = race_lagged_expanding_mean(d.assign(_p=0), "_p", f"_{part}")
        prior = prior.fillna(pop).fillna(0.0)
        eb[part] = (m.fillna(prior) * n + prior * K) / (n + K)
        if part == "tot":
            n_tot = n
            m_tot = m
    est["eb"] = eb["tot"]
    std_today = df.std_time
    for k, r in est.items():
        cols[f"{PREFIX}{k}_rel"] = r
        cols[f"{PREFIX}{k}_s"] = std_today * (1.0 + r)
    cols[f"{PREFIX}sec_eb_s"] = df.std_sec * (1.0 + eb["sec"])
    cols[f"{PREFIX}runin_eb_s"] = df.std_runin * (1.0 + eb["runin"])
    cols[f"{PREFIX}split_s"] = cols[f"{PREFIX}sec_eb_s"] + cols[f"{PREFIX}runin_eb_s"]
    # uncertainty: the dog's spread over its last 6, shrunk toward the population's within-dog spread
    resid2 = (rel["tot"] - m_tot) ** 2
    pop_w = race_lagged_expanding_mean(d.assign(_r2=resid2, _p=0), "_p", "_r2")
    pop_w = pop_w.fillna(resid2.mean() if resid2.notna().any() else 0.0)
    s2 = ((n6 - 1).clip(lower=0) * sd6.fillna(0.0) ** 2 + K * pop_w) / ((n6 - 1).clip(lower=0) + K)
    sd_rel = np.sqrt(s2)
    cols[f"{PREFIX}sd_rel"] = sd_rel
    cols[f"{PREFIX}sd_s"] = std_today * sd_rel
    cols[f"{PREFIX}n"] = n_tot
    # against the field, on the shrunk estimate
    exp = cols[f"{PREFIX}eb_s"]
    g = exp.groupby(df.race_id)
    best = g.transform("min")
    second = g.transform(lambda x: x.nsmallest(2).iloc[-1] if x.notna().sum() >= 2 else np.nan)
    cols[f"{PREFIX}gap_best_len"] = (exp - best) / LENGTH_S
    cols[f"{PREFIX}gap_second_len"] = (exp - second) / LENGTH_S
    cols[f"{PREFIX}rank"] = g.rank(method="average")
    cols[f"{PREFIX}z"] = (exp - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    cols[f"{PREFIX}p_fastest"] = p_best(-exp.to_numpy(float), cols[f"{PREFIX}sd_s"].to_numpy(float),
                                        df.race_id.to_numpy(), sims=SIMS, seed=SEED)
    out = pd.DataFrame(cols, index=df.index).astype(np.float32)
    return out, list(out.columns)
