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


JOIN_KEYS = ("race_date", "race_time", "track", "trap", "dog_id", "field")
STAGE2 = (("back_top10", "back", 0.90), ("back_top5", "back", 0.95), ("back_top2", "back", 0.98),
          ("lay_bottom10", "lay", 0.10), ("lay_bottom5", "lay", 0.05), ("lay_bottom2", "lay", 0.02))


def run_fold(d: pd.DataFrame, feats: list[str], test_from: str, test_to: str, shuffle_seed: int | None = None,
             feature_shuffle_seed: int | None = None, seeds: tuple[int, ...] | None = None) -> dict:
    """One fold on a frame of wholly priced races grouped by race in time order: fit before ``test_from`` (the last
    tenth of the fit days for early stopping), read ``test_from``..``test_to``. ``shuffle_seed``: the first placebo
    (the winner shuffled within the race in the fit). ``feature_shuffle_seed``: the second (20 Nov 2026: each fit
    race's feature rows dealt at random among its runners, so nothing a dog's own card says can be learnt and the
    gain must be nothing; the first placebo teaches the trees to undo the BSP and proves nothing about a leak).
    ``seeds``: one fit a seed (its own bagging and feature draws, its own early stop), the residual their mean (20
    Nov 2026: with 34 of 252,670 races removed one fit stopped at 2 rounds where it had stopped at 60, and the pooled
    gain fell by a third: a single fit is too unstable to read or to freeze)."""
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
    A = d[feats].to_numpy(np.float32, na_value=np.nan)                 # the matrix once a fold
    X = lambda rows: A[rows]
    if feature_shuffle_seed is not None:
        deal = _deal_within(race, np.random.default_rng(feature_shuffle_seed))
        Xf = lambda rows: A[deal[rows]]
    else:
        Xf = X
    Xtr, Xes, Xte = Xf(tr_i), Xf(es_i), X(te_i)
    models = [fit(Xtr, won[tr_i], race[tr_i], base[tr_i], Xes, won[es_i], race[es_i], base[es_i],
                  params=None if sd is None else dict(PARAMS, seed=int(sd))) for sd in (seeds or (None,))]
    del Xtr
    true_won = d.won.to_numpy(float)
    each_te = [m.predict(Xte, num_iteration=m.best_iteration, raw_score=True) for m in models]
    f_te = np.mean(each_te, axis=0)
    z = base[te_i] + f_te
    rc, ll_m = race_ll(z, true_won[te_i], race[te_i])
    _, ll_b = race_ll(base[te_i], true_won[te_i], race[te_i])
    first = pd.Series(day[te_i]).groupby(race[te_i]).first()
    out = read_gain(first.reindex(rc).to_numpy(), ll_b, ll_m)
    imp = pd.Series(np.sum([m.feature_importance("gain") for m in models], axis=0) / len(models),
                    index=feats).sort_values(ascending=False)
    rounds = [int(m.best_iteration or 0) for m in models]
    out.update({"test": [test_from, test_to], "fit_rows": int(len(fit_i)), "fit_from": str(day[fit_i].min()),
                "rounds": int(np.median(rounds)),
                "placebo": shuffle_seed is not None or feature_shuffle_seed is not None,
                "top_features": [(k, round(float(v), 1)) for k, v in imp.head(12).items() if v > 0]})
    if seeds:                                          # each single fit's own gain: the spread a single fit hides
        one = [float((ll_b - race_ll(base[te_i] + f, true_won[te_i], race[te_i])[1]).mean()) for f in each_te]
        out.update({"seeds": [int(x) for x in seeds], "rounds_by_seed": rounds,
                    "gain_by_seed": [round(g, 6) for g in one]})
    out["_per_race"] = pd.DataFrame({"day": first.reindex(rc).to_numpy(), "ll_bsp": ll_b, "ll_model": ll_m})
    if "bsp" in d and shuffle_seed is None and feature_shuffle_seed is None:
        # stage 2: the residual centred on its race, a card-only signal (no price of the race in it); its cuts are
        # read off the early-stopping days of the fit, never off the test
        centre = lambda v, r: v - pd.Series(v).groupby(r).transform("mean").to_numpy()
        s_es = centre(np.mean([m.predict(Xes, num_iteration=m.best_iteration, raw_score=True) for m in models],
                              axis=0), race[es_i])
        out["cuts"] = {name: float(np.quantile(s_es, q)) for name, _, q in STAGE2}
        out["_runners"] = pd.DataFrame({"day": day[te_i], "race_id": d.race_id.to_numpy()[te_i],
                                        "won": true_won[te_i], "bsp": d.bsp.to_numpy(float)[te_i],
                                        "s": centre(f_te, race[te_i])})
        for k in JOIN_KEYS:                            # the price files' join (research/greyhound/residual_strategies.py)
            if k in d:
                out["_runners"][k] = d[k].to_numpy()[te_i]
    return out


def _deal_within(race: np.ndarray, rng) -> np.ndarray:
    """A row index for every row: another row of the same race, the race's rows dealt at random (grouped by race)."""
    order = np.lexsort((rng.random(len(race)), race))
    out = np.empty(len(race), dtype=np.int64)
    out[np.lexsort((np.arange(len(race)), race))] = order
    return out


def stage2(folds: list[dict], comm: float, n_trials: int) -> dict:
    """Bets at the SP chosen by the card-only residual: a back of 1 on a runner whose centred residual is in the top
    10, 5 or 2% (cuts from the fit's early-stopping days), a lay risking 1 on the bottom 10, 5 or 2%; netted by race,
    commission off a gain (research/greyhound/sp_harness.py). Each rule by fold and pooled."""
    from research.greyhound.sp_harness import read, settle
    out = {"rules": []}
    for name, side, _ in STAGE2:
        per_fold, parts = [], []
        for f in folds:
            r = f.get("_runners")
            if r is None:
                continue
            cut = f["cuts"][name]
            b = r[(r.s >= cut) if side == "back" else (r.s <= cut)]
            st = settle(b.assign(side=side), comm) if len(b) else pd.DataFrame(columns=["day", "race_id", "risk", "gross", "net"])
            parts.append(st)
            one = read(st, n_trials) if len(st) else {"bets": 0}
            per_fold.append({"test": f["test"], "bets": one.get("bets"), "roi_%": one.get("roi_%"),
                             "roi_90%": one.get("roi_90%"), "mean_bsp": round(float(b.bsp.mean()), 2) if len(b) else None,
                             "cut": round(cut, 4)})
        allp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        pooled = read(allp, n_trials) if len(allp) else {"bets": 0}
        pos = sum(1 for x in per_fold if (x.get("roi_%") or 0) > 0)
        lo = (pooled.get("roi_90%") or [None])[0]
        out["rules"].append({"rule": name, "pooled": pooled, "folds": per_fold, "folds_positive": pos,
                             "pass": bool(lo is not None and lo > 0 and pos >= 2)})
    out["passed"] = [r["rule"] for r in out["rules"] if r["pass"]]
    return out


def picks(runners: pd.DataFrame, cuts: dict) -> pd.DataFrame:
    """Every stage 2 bet of a frame of runners with their centred residual ``s``: its rows once a rule that picks
    them, with ``rule`` and ``side`` (the one place a cut is applied: the study and the forward test share it)."""
    out = []
    for name, side, _ in STAGE2:
        cut = cuts[name]
        b = runners[(runners.s >= cut) if side == "back" else (runners.s <= cut)]
        out.append(b.assign(rule=name, side=side))
    return pd.concat(out, ignore_index=True) if out else runners.iloc[0:0].assign(rule="", side="")


def stage2_daily(folds: list[dict], comm: float) -> pd.DataFrame:
    """The stage 2 rules' net and money at risk a day over the test years (rule, day, net, risk): what the forward
    test's days are pooled with (research/greyhound/sp_offset_forward.py)."""
    from research.greyhound.sp_harness import settle
    out = []
    for f in folds:
        r = f.get("_runners")
        if r is None:
            continue
        p = picks(r, f["cuts"])
        for name, b in p.groupby("rule"):
            st = settle(b, comm)
            out.append(st.groupby("day", as_index=False)[["net", "risk"]].sum().assign(rule=name))
    return pd.concat(out, ignore_index=True)[["rule", "day", "net", "risk"]] if out else \
        pd.DataFrame(columns=["rule", "day", "net", "risk"])


def _shuffle_within(won: np.ndarray, race: np.ndarray, rng) -> np.ndarray:
    """``won`` permuted at random within each race (rows grouped by race)."""
    key = rng.random(len(won))
    order = np.lexsort((key, race))                    # within a race, a random order; races stay in place
    out = np.empty_like(won)
    out[np.lexsort((np.arange(len(won)), race))] = won[order]
    return out


def run(d: pd.DataFrame, feats: list[str], folds=FOLDS, stage2_args: tuple | None = None,
        daily_out: list | None = None, seeds: tuple[int, ...] | None = None) -> dict:
    """``daily_out``: a list that is given the stage 2 rules' day-by-day net and risk (``stage2_daily``). ``seeds``:
    the residual is the mean of one fit a seed; the reading's placebo is then the second (the feature rows dealt at
    random, on the first three seeds), and the first, which proves nothing, is not run."""
    res = {"tag": TAG, "features": len(feats), "folds": [], "rows": int(len(d)), "races": int(d.race_id.nunique())}
    parts, kept = [], []
    for a, b in folds:
        f = run_fold(d, feats, a, b, seeds=seeds)
        log.info("fold %s..%s: %s", a, b, {k: v for k, v in f.items() if k not in ("_per_race", "_runners", "top_features")})
        if "_per_race" in f:
            parts.append(f.pop("_per_race"))
        kept.append(dict(f))
        f.pop("_runners", None)
        res["folds"].append(f)
    if stage2_args is not None and any("_runners" in k for k in kept):
        res["stage2"] = stage2(kept, *stage2_args)
        if daily_out is not None:
            daily_out.append(stage2_daily(kept, stage2_args[0]))
    p2 = run_fold(d, feats, *folds[0], feature_shuffle_seed=11, seeds=seeds[:3] if seeds else None)
    p2.pop("_per_race", None)
    res["placebo_features_fold_1"] = p2
    if parts:
        x = pd.concat(parts, ignore_index=True)
        res["pooled"] = read_gain(x.day.to_numpy(), x.ll_bsp.to_numpy(), x.ll_model.to_numpy())
    if seeds:
        res["seeds"] = [int(x) for x in seeds]
        p = p2
    else:
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


def study_rows(df: pd.DataFrame, drop_twins: bool) -> tuple[np.ndarray, int]:
    """The rows the study reads: the wholly priced races, less (``drop_twins``) every race a twin row or a short gap
    touches (research/greyhound/twin_check.py; the flags are read on every row: a twin may be an unpriced copy).
    Returns the mask and the number of priced races left out."""
    keep = df.bsp_p_norm.notna().to_numpy()
    if not drop_twins:
        return keep, 0
    from research.greyhound.twin_check import flags
    touched = flags(df).touched.to_numpy()
    return keep & ~touched, int(df.race_id[keep & touched].nunique())


def both_halves(a: dict, b: dict) -> dict:
    """What holds on two disjoint sets of seeds: stage 1's reading, and each stage 2 rule's."""
    ra, rb = set((a.get("stage2") or {}).get("passed") or []), set((b.get("stage2") or {}).get("passed") or [])
    return {"stage1": bool(a["reading"]["pass"] and b["reading"]["pass"]),
            "rules": [name for name, _, _ in STAGE2 if name in ra and name in rb]}


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
    if res.get("placebo_fold_1"):
        md.append(row("Placebo 1 (fold 1, winner shuffled in the fit: teaches the trees to undo the BSP)", res["placebo_fold_1"]))
    if res.get("placebo_features_fold_1"):
        md.append(row("Placebo 2 (fold 1, each fit race's feature rows dealt at random)", res["placebo_features_fold_1"]))
    if res.get("stage2"):
        md += ["", "### Stage 2: bets at the SP chosen by the card-only residual", "",
               "A back of 1 (top of the centred residual) or a lay risking 1 (bottom), cuts from each fit's early-"
               "stopping days, netted by race, 2% off a gain. Return on the money at risk, 90% by day.", "",
               "| Rule | Bets | Return % | 90% | t | 2024 | 2025 | Jan-May 2026 | Mean BSP | Folds positive | Pass |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["stage2"]["rules"]:
            p, fs = r["pooled"], r["folds"]
            cell = lambda x: f"{x.get('roi_%')} ({x.get('bets')})"
            mb = [x.get("mean_bsp") for x in fs if x.get("mean_bsp") is not None]
            md.append(f"| {r['rule']} | {p.get('bets')} | {p.get('roi_%')} | {p.get('roi_90%')} | {p.get('t')} | "
                      + " | ".join(cell(x) for x in fs) + f" | {round(float(np.mean(mb)), 2) if mb else ''} | "
                      f"{r['folds_positive']} | {r['pass']} |")
        md += ["", f"Passed stage 2: {res['stage2']['passed'] or 'none'}."]
    if res.get("seeds"):
        md += ["", f"The residual is the mean of {len(res['seeds'])} fits (seeds {res['seeds'][0]}-{res['seeds'][-1]}). "
               "Each single fit's own gain a race and its early stop:", "",
               "| Test | Least | Median | Most | Rounds, least to most |", "|---|---|---|---|---|"]
        for f in res["folds"]:
            g, r = sorted(f.get("gain_by_seed") or [0]), sorted(f.get("rounds_by_seed") or [0])
            md.append(f"| {' to '.join(f['test'])} | {g[0]} | {float(np.median(g)):.6f} | {g[-1]} | {r[0]} to {r[-1]} |")
    md += ["", f"Reading: {json.dumps(res['reading'])}", "",
           "Top features by gain, last fold: " + ", ".join(f"{k} ({v})" for k, v in res["folds"][-1].get("top_features", []))]
    return "\n".join(md)


def main(argv=None) -> int:
    from greyhound import fastloop as fl
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/sp_offset")
    ap.add_argument("--seed-halves", type=int, default=0,
                    help="N: the study twice, on the mean of seeds 1..N and of seeds N+1..2N (greyhound-sp-offset-seeds-1120)")
    ap.add_argument("--drop-twins", action="store_true",
                    help="leave out every race a twin row or a short gap touches (research/greyhound/twin_check.py)")
    ap.add_argument("--drop-features", default="",
                    help="features the residual may not read, comma-separated (the cut check's: greyhound-cut-check-1120)")
    a = ap.parse_args(argv)
    df, meta = fl.load_cache(Store())
    barred = {x.strip() for x in a.drop_features.split(",") if x.strip()}
    feats = [f for f in fl.feature_set(meta, "card_safe") if f in df.columns and f not in fl.LABELS and f not in barred]
    missing = sorted(barred - set(df.columns))
    if missing:
        raise SystemExit(f"--drop-features names features the cache does not hold: {missing}")
    df = df.assign(day=pd.to_datetime(df.t).dt.strftime("%Y-%m-%d"))
    keep, dropped = study_rows(df, a.drop_twins)
    d = df[keep]
    log.info("cache: %d rows, %d wholly priced (%d touched races left out); %d features; %s to %s", len(df), len(d),
             dropped, len(feats), d.day.min(), d.day.max())
    del df
    from greyhound.costs import commission
    from research.greyhound.trial_count import greyhound_trials
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args2 = (commission(), greyhound_trials())
    cache = {"hash": meta.get("hash"), "last_day": meta.get("last_day"), "first_day": str(d.day.min()),
             "touched_races_left_out": dropped}
    if a.seed_halves:
        n = a.seed_halves
        halves, mds = {}, []
        for name, seeds in (("A", tuple(range(1, n + 1))), ("B", tuple(range(n + 1, 2 * n + 1)))):
            daily: list = []
            r = run(d, feats, stage2_args=args2, daily_out=daily, seeds=seeds)
            r["cache"] = cache
            halves[name] = r
            if daily and len(daily[0]):
                daily[0].to_csv(out / f"stage2_daily_{name}.csv", index=False)
            mds += [f"# Half {name}: seeds {seeds[0]}-{seeds[-1]}", "", markdown(r), ""]
        both = both_halves(halves["A"], halves["B"])
        mds += ["# Both halves", "", f"Stage 1 passes in both: {both['stage1']}. Stage 2 rules that pass in both: "
                f"{both['rules'] or 'none'}."]
        (out / "report.json").write_text(json.dumps({"tag": "greyhound-sp-offset-seeds-1120", **halves, "both": both},
                                                    indent=1, default=str))
        (out / "report.md").write_text("\n".join(mds))
        print("\n".join(mds))
        return 0
    daily = []
    res = run(d, feats, stage2_args=args2, daily_out=daily)
    res["cache"] = cache
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text(markdown(res))
    if daily and len(daily[0]):
        daily[0].to_csv(out / "stage2_daily.csv", index=False)
    print(markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
