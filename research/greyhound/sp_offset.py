"""Does anything on the card add to the Betfair SP? A residual model on the BSP as an offset (the owner, 20 Nov 2026:
the discovery round, idea sp-01; reports/greyhound_discovery_1120.md; ledger greyhound-sp-offset-1120).

Every earlier reading put OUR price beside the market's (a two-input Benter stage: weight on the model 0.196, the
blend worse than the BSP). This is the many-input stage: a race softmax on the winner whose starting score is the
BSP's own log chance (LightGBM ``init_score``), so the trees can only learn what the card-safe features say the BSP
gets wrong. No price of the race is a feature. If nothing is learnt the model IS the BSP and the gain is zero.

Stage 1, here: the winner's log-loss a race, the BSP's against the BSP plus the residual, on three test years the
model never saw (fitted on everything before each, early-stopped on the last tenth of its days), with the day as the
cluster. A placebo beside it: the first fold refitted with the winner shuffled within each race, whose gain must be
zero. Stage 2 (how a bet at the SP would be chosen without knowing the SP) is only registered if stage 1 passes.

    python -m research.greyhound.sp_offset --out out/study
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-sp-offset-1120"
FOLDS = (("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-05-31"))
PARAMS = {"num_leaves": 15, "min_child_samples": 2000, "learning_rate": 0.05, "feature_fraction": 0.7,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 63, "verbose": -1, "seed": 0,
          "num_threads": 4, "deterministic": True, "force_col_wise": True}
ROUNDS, EARLY = 1500, 100


def race_ll(z: np.ndarray, won: np.ndarray, race: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The winner's negative log chance a race under the softmax of ``z`` (rows grouped by race): (race code, loss)."""
    from greyhound.model import _race_softmax, _starts
    starts = _starts(race)
    p = _race_softmax(np.asarray(z, float), starts)
    lw = np.where(won > 0, -np.log(np.clip(p, 1e-12, 1.0)), 0.0)
    return race[starts], np.add.reduceat(lw, starts)


def fit(X_tr, won_tr, race_tr, base_tr, X_es, won_es, race_es, base_es, params=None, rounds=ROUNDS, early=EARLY):
    """The residual softmax: trees on top of the BSP's log chance, early-stopped on the later days."""
    import lightgbm as lgb
    from greyhound.model import race_objective
    fobj, _ = race_objective(race_tr, won_tr)
    _, feval = race_objective(race_es, won_es)
    dtr = lgb.Dataset(X_tr, won_tr, init_score=base_tr, free_raw_data=False)
    des = lgb.Dataset(X_es, won_es, init_score=base_es, reference=dtr)
    return lgb.train(dict(params or PARAMS, objective=fobj, metric="None"), dtr, rounds, valid_sets=[des], feval=feval,
                     callbacks=[lgb.early_stopping(early, verbose=False)])


def read_gain(day: np.ndarray, ll_bsp: np.ndarray, ll_model: np.ndarray) -> dict:
    """The gain a race (the BSP's loss less the model's: above 0, the residual adds), its 90% interval by day."""
    from research.greyhound.sp_harness import ratio_t
    r = ratio_t(day, ll_bsp - ll_model, np.ones(len(day)))
    f = lambda v: None if v is None else round(float(v), 6)
    return {"races": int(len(day)), "days": r["days"], "ll_bsp": round(float(ll_bsp.mean()), 5),
            "ll_model": round(float(ll_model.mean()), 5), "gain": f(r["ratio"]), "gain_90%": [f(r["lo"]), f(r["hi"])],
            "t": None if r["t"] is None else round(r["t"], 2)}


def run_fold(d: pd.DataFrame, feats: list[str], test_from: str, test_to: str, shuffle_seed: int | None = None) -> dict:
    """One fold on a frame of wholly priced races grouped by race in time order: fit before ``test_from`` (the last
    tenth of the fit days for early stopping), read ``test_from``..``test_to``. ``shuffle_seed``: the placebo."""
    day = d.day.to_numpy()
    fit_i = np.flatnonzero(day < test_from)
    te_i = np.flatnonzero((day >= test_from) & (day <= test_to))
    if not len(fit_i) or not len(te_i):
        return {"note": "no data found: no fit or test rows", "test": [test_from, test_to]}
    days = np.unique(day[fit_i])
    cut = days[int(len(days) * 0.9)]
    tr_i, es_i = fit_i[day[fit_i] < cut], fit_i[day[fit_i] >= cut]
    won = d.won.to_numpy(float).copy()
    race = pd.factorize(d.race_id)[0]
    if shuffle_seed is not None:                       # the placebo: the winner moved at random within its race
        won[fit_i] = _shuffle_within(won, race, np.random.default_rng(shuffle_seed))[fit_i]
    base = np.log(np.clip(d.bsp_p_norm.to_numpy(float), 1e-9, 1.0))
    X = lambda rows: d[feats].to_numpy(np.float32, na_value=np.nan)[rows]
    m = fit(X(tr_i), won[tr_i], race[tr_i], base[tr_i], X(es_i), won[es_i], race[es_i], base[es_i])
    true_won = d.won.to_numpy(float)
    z = base[te_i] + m.predict(X(te_i), num_iteration=m.best_iteration, raw_score=True)
    rc, ll_m = race_ll(z, true_won[te_i], race[te_i])
    _, ll_b = race_ll(base[te_i], true_won[te_i], race[te_i])
    first = pd.Series(day[te_i]).groupby(race[te_i]).first()
    out = read_gain(first.reindex(rc).to_numpy(), ll_b, ll_m)
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    out.update({"test": [test_from, test_to], "fit_rows": int(len(fit_i)), "fit_from": str(day[fit_i].min()),
                "rounds": int(m.best_iteration or 0), "placebo": shuffle_seed is not None,
                "top_features": [(k, round(float(v), 1)) for k, v in imp.head(12).items() if v > 0]})
    out["_per_race"] = pd.DataFrame({"day": first.reindex(rc).to_numpy(), "ll_bsp": ll_b, "ll_model": ll_m})
    return out


def _shuffle_within(won: np.ndarray, race: np.ndarray, rng) -> np.ndarray:
    """``won`` permuted at random within each race (rows grouped by race)."""
    key = rng.random(len(won))
    order = np.lexsort((key, race))                    # within a race, a random order; races stay in place
    out = np.empty_like(won)
    out[np.lexsort((np.arange(len(won)), race))] = won[order]
    return out


def run(d: pd.DataFrame, feats: list[str], folds=FOLDS) -> dict:
    res = {"tag": TAG, "features": len(feats), "folds": [], "rows": int(len(d)), "races": int(d.race_id.nunique())}
    parts = []
    for a, b in folds:
        f = run_fold(d, feats, a, b)
        log.info("fold %s..%s: %s", a, b, {k: v for k, v in f.items() if k not in ("_per_race", "top_features")})
        if "_per_race" in f:
            parts.append(f.pop("_per_race"))
        res["folds"].append(f)
    if parts:
        x = pd.concat(parts, ignore_index=True)
        res["pooled"] = read_gain(x.day.to_numpy(), x.ll_bsp.to_numpy(), x.ll_model.to_numpy())
    p = run_fold(d, feats, *folds[0], shuffle_seed=7)
    p.pop("_per_race", None)
    res["placebo_fold_1"] = p
    pooled = res.get("pooled", {})
    lo = (pooled.get("gain_90%") or [None])[0]
    each = [f.get("gain") for f in res["folds"]]
    res["reading"] = {"pooled_interval_above_zero": bool(lo is not None and lo > 0),
                      "every_fold_positive": bool(each and all(g is not None and g > 0 for g in each)),
                      "placebo_interval_spans_zero": bool(p.get("gain_90%") and p["gain_90%"][0] is not None
                                                          and p["gain_90%"][0] <= 0)}
    res["reading"]["pass"] = all(res["reading"].values())
    return res


def markdown(res: dict) -> str:
    md = [f"## A residual model on the BSP ({TAG})", "",
          f"{res['rows']:,} runners of {res['races']:,} wholly priced races; {res['features']} card-safe features; the "
          "winner's log-loss a race, the BSP alone against the BSP plus the residual. Gain above 0: the card adds.", "",
          "| Test | Races | Days | Fit rows | Rounds | BSP | BSP + residual | Gain a race | 90% | t |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    row = lambda name, f: (f"| {name} | {f.get('races')} | {f.get('days')} | {f.get('fit_rows', '')} | "
                           f"{f.get('rounds', '')} | {f.get('ll_bsp')} | {f.get('ll_model')} | {f.get('gain')} | "
                           f"{f.get('gain_90%')} | {f.get('t')} |")
    for f in res["folds"]:
        md.append(row(" to ".join(f["test"]), f) if "gain" in f else f"| {' to '.join(f['test'])} | {f.get('note')} |")
    if res.get("pooled"):
        md.append(row("Pooled", res["pooled"]))
    md.append(row("Placebo (fold 1, winner shuffled in the fit)", res["placebo_fold_1"]))
    md += ["", f"Reading: {json.dumps(res['reading'])}", "",
           "Top features by gain, last fold: " + ", ".join(f"{k} ({v})" for k, v in res["folds"][-1].get("top_features", []))]
    return "\n".join(md)


def main(argv=None) -> int:
    from greyhound import fastloop as fl
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/sp_offset")
    a = ap.parse_args(argv)
    df, meta = fl.load_cache(Store())
    feats = [f for f in fl.feature_set(meta, "card_safe") if f in df.columns and f not in fl.LABELS]
    d = df[df.bsp_p_norm.notna()]
    d = d.assign(day=pd.to_datetime(d.t).dt.strftime("%Y-%m-%d"))
    log.info("cache: %d rows, %d wholly priced; %d features; %s to %s", len(df), len(d), len(feats), d.day.min(), d.day.max())
    del df
    res = run(d, feats)
    res["cache"] = {"hash": meta.get("hash"), "last_day": meta.get("last_day"), "first_day": str(d.day.min())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
