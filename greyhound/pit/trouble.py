"""What trouble cost: each GBGB comment code priced in lengths, learnt from the data, so a troubled run is read at
what the dog would have run clear (the engine credits a flat 1.5 lengths a bump: greyhound/metrics.py gsr_trouble_adj).

Each run's surprise is its speed rating (gsr) less the dog's level before the run (the ability table's Kalman mean).
A ridge regression of the surprise on the run's comment codes (each comma-separated code, lower case: "crd1", "bmp",
"ldnrin", ...; the codes seen 200 times or more in the fit window), refitted for each year on the years before it
only, gives each code's lengths. A year with fewer than ``MIN_FIT`` earlier runs has no prices (NaN): nothing is
priced on its own year.

The run's cost (``cost``) is minus the sum of its codes' lengths (lengths lost); its clean figure is gsr + cost. Both
are the run's own result, so the table carries only what the dog's EARLIER days give (``tr_``): the last run's cost,
the mean cost over the last 5, the share of the last 10 with a cost over 1 length, and a Kalman level on the clean
figures (``tr_clean_mean``, ``tr_clean_sd``), as the ability table does on the raw ones.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from greyhound.pit import ability, core

PREFIX = "tr_"
MIN_COUNT = 200
MIN_FIT = 100_000
LAMBDA = 50.0
SOURCE = "GBGB race comments and times (sources/gbgb.py); the ability table's levels"


def codes(comment) -> list[str]:
    return [re.sub(r"\s+", "", t.lower()) for t in str(comment or "").split(",") if t.strip()]


def fit_costs(tokens: list[list[str]], y: np.ndarray, min_count: int = MIN_COUNT, lam: float = LAMBDA) -> dict:
    """Ridge of ``y`` on the code indicators (codes seen ``min_count`` times or more): {code: lengths}."""
    from collections import Counter
    cnt = Counter(c for ts in tokens for c in set(ts))
    vocab = sorted(c for c, n in cnt.items() if n >= min_count)
    if not vocab:
        return {}
    from scipy import sparse
    ix = {c: i for i, c in enumerate(vocab)}
    k = len(vocab)
    rows, cols = [], []
    for r_, ts in enumerate(tokens):
        for c in set(ts):
            j = ix.get(c)
            if j is not None:
                rows.append(r_)
                cols.append(1 + j)
    n = len(tokens)
    rows += list(range(n))
    cols += [0] * n
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, k + 1))
    xtx = (X.T @ X).toarray()
    xty = X.T @ y
    reg = lam * np.eye(k + 1)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xtx + reg, xty)
    return {c: float(beta[1 + ix[c]]) for c in vocab}


def build(runs: pd.DataFrame, figs: pd.DataFrame | None = None, ctx: dict | None = None) -> tuple[pd.DataFrame, dict]:
    df = figs if figs is not None else core.per_run_figures(runs)
    kf = (ctx or {}).get("ability")
    if kf is None:
        kf, _ = ability.build(runs, df)
    df = df.merge(kf[["race_id", "dog_id", "kf_gsr_mean"]], on=["race_id", "dog_id"], how="left")
    toks = [codes(c) for c in df.comment.fillna("").to_numpy()]
    surprise = (df.gsr - df.kf_gsr_mean).to_numpy(float)
    year = pd.to_datetime(df.race_date).dt.year.to_numpy()
    cost = np.full(len(df), np.nan)
    fits = {}
    for y in sorted(set(year)):
        prior = np.flatnonzero((year < y) & np.isfinite(surprise))
        if len(prior) < MIN_FIT:
            fits[int(y)] = {"note": f"{len(prior)} earlier runs: no prices"}
            continue
        costs = fit_costs([toks[i] for i in prior], np.clip(surprise[prior], -20, 20))
        fits[int(y)] = {"codes": len(costs), "fit_runs": int(len(prior)),
                        "top_costs": dict(sorted(costs.items(), key=lambda kv: kv[1])[:10])}
        rows = np.flatnonzero(year == y)
        cost[rows] = [-sum(costs.get(c, 0.0) for c in set(toks[i])) for i in rows]
    df["_cost"] = cost
    df["_clean"] = df.gsr + df._cost
    df["_costly"] = (df._cost > 1.0).astype(float).where(df._cost.notna())
    out = pd.DataFrame({"race_id": df.race_id, "dog_id": df.dog_id})
    out[f"{PREFIX}last_cost"] = core.dog_lagged(df, "_cost", 1)
    out[f"{PREFIX}cost5"] = core.dog_lagged(df, "_cost", 5)
    out[f"{PREFIX}costly10"] = core.dog_lagged(df, "_costly", 10)
    s = df.sort_values(["dog_id", "t"], kind="stable")
    dayn = (core.day(s) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    trial = s.is_trial.fillna(False).to_numpy(bool) if "is_trial" in s else np.zeros(len(s), bool)
    prm = ability.estimate(s.assign(_c=s._clean), "_c")
    m, v, _, _ = ability.filter_dog_states(s.dog_id.to_numpy(), dayn, s._clean.to_numpy(float), trial, prm)
    out.loc[s.index, f"{PREFIX}clean_mean"] = m
    out.loc[s.index, f"{PREFIX}clean_sd"] = np.sqrt(v)
    return out.reset_index(drop=True), {"fits": fits, "clean_params": prm}


COLUMNS = {
    f"{PREFIX}last_cost": "lengths the last run's trouble cost (learnt code prices)",
    f"{PREFIX}cost5": "mean trouble cost over the last 5 runs",
    f"{PREFIX}costly10": "share of the last 10 runs that lost over a length to trouble",
    f"{PREFIX}clean_mean": "the dog's level on clean figures (gsr + trouble cost), Kalman",
    f"{PREFIX}clean_sd": "the spread of that level",
}
