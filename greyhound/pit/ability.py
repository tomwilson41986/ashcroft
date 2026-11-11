"""Ability with uncertainty: a Kalman filter (a local level) on each dog's figures, so a dog's rating carries how sure
we are of it. Rolling windows cannot tell a dog with two runs from one with fifty; the filter's spread can.

For each figure (the engine's gsr, the speed rating; esr, early pace; fsr, finishing speed; all in lengths against the
track and distance's standard) the dog's ability is a level that drifts between runs: the variance grows by ``q`` a day
between runs, and each run is the level plus noise of variance ``r`` (1.5 r for a trial). The debut prior is the
population's first-run mean and variance. ``q``, ``r`` and the prior are estimated on the runs before ``FIT_BEFORE``
only (by the one-step-ahead predictive likelihood over a grid), then frozen: the meta records
them. The state a run reads is the state after the dog's last EARLIER day.

Columns (``kf_``): for each figure f in (gsr, esr, fsr): ``kf_<f>_mean``, ``kf_<f>_sd``; ``kf_n`` (figures seen),
``kf_days`` (days since the last), and against today's field: ``kf_gsr_gap`` (lengths behind the field's best mean),
``kf_gsr_rank``, ``kf_gsr_pbest`` (the chance this dog's level is the field's highest, from the means and spreads, by
simulation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.pit import core

PREFIX = "kf_"
FIGURES = ("gsr", "esr", "fsr")
FIT_BEFORE = "2018-01-01"
SOURCE = "GBGB results (sources/gbgb.py) through the metrics engine's per-run figures (greyhound/metrics.py)"


def estimate(df: pd.DataFrame, fig: str, before: str = FIT_BEFORE, max_dogs: int = 30_000, seed: int = 0) -> dict:
    """q (a day), r and the debut prior for ``fig``, from the runs before ``before`` (or the first fifth of the days when
    there are too few): the debut prior from the dogs' first figures; q and r by the one-step-ahead predictive
    likelihood (each figure scored against the filter's state before it) over a grid, on up to ``max_dogs`` dogs.
    (A moment fit of consecutive differences put most of the spread into drift and chased the last run: on a 2026
    sample its q of 0.63 a day scored 1.674 on the winner's log loss, q 0.1 1.665, uniform 1.689.)"""
    d = df[["dog_id", "t", "race_date", fig] + (["is_trial"] if "is_trial" in df else [])].dropna(subset=[fig])
    early = d[d.race_date < before]
    if len(early) < 50_000:
        cut = d.race_date.sort_values().iloc[int(len(d) * 0.2)] if len(d) else before
        early = d[d.race_date < cut]
    dogs = early.dog_id.unique()
    if len(dogs) > max_dogs:
        dogs = np.random.default_rng(seed).choice(dogs, max_dogs, replace=False)
        early = early[early.dog_id.isin(dogs)]
    early = early.sort_values(["dog_id", "t"])
    first = early[early.groupby("dog_id").cumcount() == 0][fig]
    m0 = float(first.mean()) if len(first) else 0.0
    p0 = float(first.var()) if len(first) > 1 else 25.0
    tot = float(early[fig].var()) if len(early) > 1 else 25.0
    if len(early) < 1000:
        return {"q": 0.05, "r": max(tot * 0.6, 1.0), "m0": m0, "p0": p0, "rows": int(len(early)), "fit": "default"}
    dog = early.dog_id.to_numpy()
    dayn = (core.day(early) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    obs = early[fig].to_numpy(float)
    trial = early.is_trial.fillna(False).to_numpy(bool) if "is_trial" in early else np.zeros(len(early), bool)
    best, grid = None, []
    for q in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
        for share in (0.4, 0.55, 0.7, 0.85):
            prm = {"q": q, "r": max(tot * share, 0.5), "m0": m0, "p0": p0}
            m, v, _, _ = filter_dog_states(dog, dayn, obs, trial, prm)
            s2 = v + prm["r"] * np.where(trial, 1.5, 1.0)
            ll = float(np.mean(-0.5 * (np.log(2 * np.pi * s2) + (obs - m) ** 2 / s2)))
            grid.append((q, round(prm["r"], 3), round(ll, 5)))
            if best is None or ll > best[0]:
                best = (ll, prm)
    return {**best[1], "rows": int(len(early)), "fit": "predictive likelihood", "loglik": round(best[0], 5),
            "grid": grid}


def filter_dog_states(dog: np.ndarray, dayn: np.ndarray, obs: np.ndarray, trial: np.ndarray, prm: dict):
    """Pre-run state (mean, variance, figures seen, days since the last figure) for rows sorted by dog then time.
    Every run on a day reads the state after the dog's EARLIER days; the variance grows by q a day since the last
    figure; a day without a figure changes nothing."""
    n = len(dog)
    mean, var, cnt, since = np.empty(n), np.empty(n), np.zeros(n), np.full(n, np.nan)
    q, r, m0, p0 = prm["q"], prm["r"], prm["m0"], prm["p0"]
    j = 0
    while j < n:
        d0 = dog[j]
        m, p, c, last = m0, p0, 0, None              # the state after the dog's last figure, and that figure's day
        while j < n and dog[j] == d0:
            k = j
            while k < n and dog[k] == d0 and dayn[k] == dayn[j]:
                k += 1
            gap = (dayn[j] - last) if last is not None else None
            p_now = p + q * gap if gap is not None else p
            mean[j:k], var[j:k], cnt[j:k] = m, p_now, c
            since[j:k] = gap if gap is not None else np.nan
            seen = False
            for t in range(j, k):                    # the day's figures, for the days after
                if np.isfinite(obs[t]):
                    pp = p_now if not seen else p
                    rr = r * (1.5 if trial[t] else 1.0)
                    kg = pp / (pp + rr)
                    m = m + kg * (obs[t] - m)
                    p = (1 - kg) * pp
                    c += 1
                    seen = True
            if seen:
                last = dayn[j]
            j = k
    return mean, var, cnt, since


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    df = df.sort_values(["dog_id", "t"], kind="stable")
    dayn = (core.day(df) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    dog = df.dog_id.to_numpy()
    trial = df.is_trial.fillna(False).to_numpy(bool) if "is_trial" in df else np.zeros(len(df), bool)
    out = pd.DataFrame({"race_id": df.race_id.to_numpy(), "dog_id": dog}, index=df.index)
    params = {}
    for f in FIGURES:
        prm = estimate(df, f)
        params[f] = prm
        m, v, c, since = filter_dog_states(dog, dayn, df[f].to_numpy(float), trial, prm)
        out[f"{PREFIX}{f}_mean"] = m
        out[f"{PREFIX}{f}_sd"] = np.sqrt(v)
        if f == "gsr":
            out[f"{PREFIX}n"] = c
            out[f"{PREFIX}days"] = since
    out = out.sort_index()
    rid = runs.loc[out.index, "race_id"] if "race_id" in runs else out.race_id
    g = out.groupby(rid)[f"{PREFIX}gsr_mean"]
    out[f"{PREFIX}gsr_gap"] = g.transform("max") - out[f"{PREFIX}gsr_mean"]
    out[f"{PREFIX}gsr_rank"] = g.rank(ascending=False, method="average")
    k, kgrid = race_noise(out, runs, params["gsr"]["r"])
    out[f"{PREFIX}gsr_pbest"] = p_best(out.race_id.to_numpy(), out[f"{PREFIX}gsr_mean"].to_numpy(),
                                       out[f"{PREFIX}gsr_sd"].to_numpy(), params["gsr"]["r"] * k)
    return out.reset_index(drop=True), {"params": params, "fit_before": FIT_BEFORE, "race_noise": k,
                                        "race_noise_grid": kgrid}


def race_noise(tab: pd.DataFrame, runs: pd.DataFrame, r: float, before: str = FIT_BEFORE, max_races: int = 20_000,
               seed: int = 0) -> tuple[float, dict]:
    """How many times the figure's run noise ``r`` a race's outcome carries (crowding, traps, checks: a race is
    noisier than one dog's figure), chosen on the races before ``before`` (or the first fifth) by the winner's log loss
    of ``p_best``. On a 2026 sample, ``p_best`` at k = 1 scored 1.751 against a uniform guess's 1.689."""
    d = tab.assign(won=runs.loc[tab.index, "won"].to_numpy(), race_date=runs.loc[tab.index, "race_date"].to_numpy(),
                   trial=runs.loc[tab.index, "is_trial"].fillna(False).to_numpy(bool) if "is_trial" in runs else False)
    d = d[~d.trial & d.won.notna()]
    early = d[d.race_date < before]
    if early.race_id.nunique() < 2000:
        cut = d.race_date.sort_values().iloc[int(len(d) * 0.2)] if len(d) else before
        early = d[d.race_date < cut]
    races = early.race_id.unique()
    if len(races) > max_races:
        races = np.random.default_rng(seed).choice(races, max_races, replace=False)
    e = early[early.race_id.isin(races)]
    if not len(e):
        return 1.0, {}
    grid = {}
    for k in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 27.0, 40.0):
        p = p_best(e.race_id.to_numpy(), e[f"{PREFIX}gsr_mean"].to_numpy(), e[f"{PREFIX}gsr_sd"].to_numpy(), r * k,
                   sims=500)
        w = e.won.to_numpy() == 1
        grid[k] = round(float(-np.log(p[w & np.isfinite(p)]).mean()), 5)
    return min(grid, key=grid.get), grid


def p_best(race: np.ndarray, mean: np.ndarray, sd: np.ndarray, r: float, sims: int = 2000, seed: int = 0) -> np.ndarray:
    """Each dog's chance of the highest figure in its race: level ~ N(mean, sd^2 + r) (the day's run noise too); the
    count smoothed by half a win a dog, so no dog's chance is 0."""
    order = np.argsort(race, kind="stable")
    rs = race[order]
    starts = np.r_[0, np.flatnonzero(rs[1:] != rs[:-1]) + 1, len(rs)]
    out = np.full(len(race), np.nan)
    # the same draws for every race (a dog's are those of its place in the race's rows), so a race's chances are its
    # own inputs' and nothing else's: one stream down the frame gave a card race other numbers than its result row
    # (the audit of 11 Nov 2026, findings 17 and 42)
    width = int(np.diff(starts).max()) if len(starts) > 1 else 1
    Z = np.random.default_rng(seed).standard_normal((width, sims)).T
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        mu, s = mean[idx], np.sqrt(sd[idx] ** 2 + r)
        if not np.all(np.isfinite(mu)):
            continue
        draw = mu[None, :] + s[None, :] * Z[:, :len(idx)]
        win = np.bincount(draw.argmax(1), minlength=len(idx)) + 0.5
        out[idx] = win / win.sum()
    return out


COLUMNS = {
    **{f"{PREFIX}{f}_mean": f"the dog's {f} level before the run (Kalman local level, lengths)" for f in FIGURES},
    **{f"{PREFIX}{f}_sd": f"the spread of that level (how sure: falls with runs, grows with days off)" for f in FIGURES},
    f"{PREFIX}n": "figures the filter has seen (trials included)",
    f"{PREFIX}days": "days since the dog's last figure",
    f"{PREFIX}gsr_gap": "lengths behind the field's best level",
    f"{PREFIX}gsr_rank": "the level's rank in the field (1 the best)",
    f"{PREFIX}gsr_pbest": "chance this dog runs the field's best figure, from the levels, their spreads and the race's "
                          "noise (fitted on early races)",
}
