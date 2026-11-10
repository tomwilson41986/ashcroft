"""The greyhound fast experiment loop (the owner, 3 Nov 2026: "define a quicker way to test models; we can't have a
2-3 hour run each time"). The horse model's fast loop (docs/FAST_LOOP.md) carried to greyhounds.

Where the hours went: every run rebuilt the metrics on every GBGB run from 2018 (about 15 minutes, the parity, HRB
and library blocks) and fitted on the whole history at a slow learning rate (the log-BSP model's 20,000 rounds took
two hours: reports/greyhound_model.md section 15). Here:

1. **The feature matrix is built once and cached** in S3 (``greyhound/feature_cache/<hash>/``), every feature of every
   block (the weigh-in features too, so a test with and without them costs no rebuild), with the BSP from the price
   files, from ``KEEP_FROM`` on. The hash is of the feature code: a change to the engine rebuilds it; nothing else does.
2. **An experiment is a line of JSON** (``research/greyhound/experiments/current.json``): a feature set, blocks to add
   (``greyhound/blocks/``: computed on the cached matrix, no rebuild), prefixes to drop, an objective (``softmax``,
   ``binary``, ``pl2`` or ``logbsp``, the horse model's target), and a recipe.
3. **The quick recipe** fits on 2024 on with a learning rate of 0.08 and coarse bins (minutes a fit; 0.15 until batch 3); ``confirm`` fits
   on 2022 on at 0.08 for the winners, within a time budget (``max_seconds``). Both test on every race from ``TEST_FROM`` (out of sample) and report the
   log-loss against the result, against the Betfair SP on the same races, and Benter's test: whether the model adds to
   the SP in a blend fitted on the first half of the test and scored on the second.
4. **Experiments run side by side** (``greyhound-fastloop.yml``: one job each on the same cache, then a table); an
   experiment with ``"study": true`` also runs the trade-out study on its prices (``research.greyhound.tradeout_study``,
   the re-timed books). Two lanes (``lane``): the quick experiments' table comes as soon as they finish; the slow lane
   (the confirm recipe, the study) has a table of its own after, so a slow job never holds up the quick answer.

    python -m greyhound.fastloop --build-cache                     # once a code change; skipped when the cache is current
    python -m greyhound.fastloop --spec research/greyhound/experiments/current.json --out out/fastloop
    python -m greyhound.fastloop --spec ... --only with_weighin --out out/fastloop
    python -m greyhound.fastloop --collect out/fastloop
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_PREFIX = "greyhound/feature_cache"
#: the files whose code makes the features: a change to any of them is a new cache
CODE_FILES = ("greyhound/metrics.py", "greyhound/parity.py", "greyhound/hrb.py", "greyhound/lib.py",
              "greyhound/data.py", "greyhound/track.py", "greyhound/model.py", "sources/gbgb.py",
              "greyhound/times.py", "greyhound/intent.py", "greyhound/ability.py", "greyhound/etime.py")
KEEP_FROM = "2020-06-01"
BLOCKS = ("parity", "hrb", "lib", "times", "intent", "ability", "etime")   # F3, F5, F6 and the expected time
PRICE_YEARS = "2018-2026"
EXTRA_COLS = ("bsp", "bsp_p_norm", "weight_kg")     # the raw weight: the predicted weigh-in block (wpred)
TEST_FROM = "2026-06-01"
STUDY_FROM = "2026-10-06"
RECIPES = {
    # batch 3 (section 20): 0.08 on the quick window gave 72% of the slow recipe's gain at a quarter more time than 0.15
    "quick": {"train_from": "2024-01-01", "learning_rate": 0.08, "early_stop": 60, "max_bin": 63, "rounds": 6000},
    # batch 2 (section 19): 2021 on at 0.05 ran into its 8,000-round cap in 51 minutes; now 2022 on at 0.08, stopped
    # by early stopping, the cap or a time budget, whichever comes first
    "confirm": {"train_from": "2022-01-01", "learning_rate": 0.08, "early_stop": 100, "max_bin": 255, "rounds": 6000,
                "max_seconds": 1500},
}
BASE_PARAMS = {"num_leaves": 63, "min_child_samples": 300, "feature_fraction": 0.7, "bagging_fraction": 0.8,
               "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 0,
               # the same recipe on the same cache scored 0.3593 (batches 12 and 14) and 0.3579 (batch 13), stopping at
               # 3,163 and 4,372 rounds: GitHub's runners differ, and so did the fits. From 10 Nov 2026 every fit is
               # deterministic on a fixed thread count, so a difference between experiments is the experiment's
               "num_threads": 4, "deterministic": True, "force_col_wise": True}
OBJECTIVES = ("softmax", "binary", "pl2", "logbsp", "share", "mix")
#: the objectives whose forecast is a price (pred_bsp), scored by the RMSE of log(BSP)
PRICE_OBJECTIVES = ("logbsp", "share", "mix")
NAME = re.compile(r"^[a-z0-9_]{1,40}$")


def _log(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


# --------------------------------------------------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------------------------------------------------

def code_hash(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for f in CODE_FILES:
        p = root / f
        h.update(f.encode())
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()[:12]


def build_cache(store, last: date, keep_from: str = KEEP_FROM) -> dict:
    """Every feature of every block on every GBGB run to ``last``, the modelling rows from ``keep_from``, to S3."""
    from greyhound.live import CARD_UNSAFE
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices, modelling_rows, slim
    from greyhound.track import runs_through
    t0 = time.monotonic()
    runs = runs_through(store, last)
    eng = GreyhoundMetricsEngine(blocks=BLOCKS)
    df = eng.calculate_all(runs, prices=load_prices(store, PRICE_YEARS))
    del runs
    gc.collect()
    df = slim(df, eng.features + list(EXTRA_COLS))
    df = modelling_rows(df)
    df = df[df.t >= keep_from].reset_index(drop=True)
    h = code_hash()
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    store.put(f"{CACHE_PREFIX}/{h}/features.parquet", buf.getvalue())
    meta = {"hash": h, "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "last_day": str(last),
            "keep_from": keep_from, "rows": int(len(df)), "races": int(df.race_id.nunique()),
            "features": {b: f for b, f in eng.block_features.items()}, "card_unsafe": sorted(CARD_UNSAFE),
            "timings": eng.timings, "seconds": round(time.monotonic() - t0), "bytes": len(buf.getvalue())}
    store.put(f"{CACHE_PREFIX}/{h}/meta.json", json.dumps(meta, indent=1).encode())
    return meta


def cache_meta(store) -> dict | None:
    raw = store.get(f"{CACHE_PREFIX}/{code_hash()}/meta.json")
    return json.loads(raw) if raw else None


def load_cache(store) -> tuple[pd.DataFrame, dict]:
    meta = cache_meta(store)
    if meta is None:
        raise SystemExit(f"no feature cache for code {code_hash()}: run --build-cache")
    t0 = time.monotonic()
    df = pd.read_parquet(io.BytesIO(store.get(f"{CACHE_PREFIX}/{meta['hash']}/features.parquet")))
    _log(f"cache {meta['hash']}: {len(df):,} rows, last day {meta['last_day']} ({time.monotonic() - t0:.0f} s to load)")
    return df, meta


# --------------------------------------------------------------------------------------------------------------------
# An experiment
# --------------------------------------------------------------------------------------------------------------------

def feature_set(meta: dict, name: str) -> list[str]:
    """``all`` (every block, the weigh-in too), ``card_safe`` (what the morning card knows: the live model's view),
    ``base``, ``base+parity``, ``base+parity+hrb``, ``core`` (every block but times, intent, ability, etime),
    ``core+<block>[+<block>]`` (core and those blocks); each ``-card`` for card-safe."""
    blocks = meta["features"]
    card = name.endswith("-card") or name == "card_safe"
    key = name.removesuffix("-card")
    order = {"base": ["base"], "base+parity": ["base", "parity"], "base+parity+hrb": ["base", "parity", "hrb"],
             "all": list(blocks), "card_safe": list(blocks),
             # the blocks before the roadmap's F3/F5/F6 (3 Nov 2026), to read what those add
             "core": [b for b in blocks if b not in ("times", "intent", "ability", "etime")]}
    if key.startswith("core+"):                      # core and named blocks: what each new block adds on its own
        extra = key.split("+")[1:]
        unknown = [b for b in extra if b not in blocks]
        if unknown:
            raise ValueError(f"unknown blocks {unknown} in feature set {name!r}")
        order[key] = order["core"] + extra
    if key not in order:
        raise ValueError(f"unknown feature set {name!r}")
    feats = list(dict.fromkeys(f for b in order[key] for f in blocks.get(b, [])))
    if card:
        feats = [f for f in feats if f not in set(meta["card_unsafe"])]
    return feats


def lane(exp: dict, spec: dict) -> str:
    """``slow`` for an experiment off the quick recipe or with the study, ``quick`` otherwise: the CI runs the quick
    lane's table as soon as it is done, without waiting on the slow lane."""
    slow = (exp.get("recipe", spec.get("recipe", "quick")) != "quick" or exp.get("study") or exp.get("limit")
            or exp.get("bfsp") or exp.get("publish_pred"))
    return "slow" if slow else "quick"


def time_budget(seconds: float | None, stopper):
    """A LightGBM callback ending the fit once ``seconds`` have passed, at the early stopper's best round so far
    (``stopper``: ``lgb.early_stopping``'s callback, which runs before this one)."""
    import lightgbm as lgb
    t0 = time.monotonic()

    def cb(env) -> None:
        if seconds and time.monotonic() - t0 > seconds:
            if getattr(stopper, "best_iter", None):
                raise lgb.callback.EarlyStopException(stopper.best_iter[0], stopper.best_score_list[0])
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    cb.order = 40
    return cb


def check_spec(spec: dict) -> list[dict]:
    exps = spec.get("experiments") or []
    names = set()
    for e in exps:
        if not NAME.match(str(e.get("name", ""))):
            raise ValueError(f"experiment name must be [a-z0-9_]{{1,40}}: {e.get('name')!r}")
        if e["name"] in names:
            raise ValueError(f"duplicate experiment {e['name']}")
        names.add(e["name"])
        if e.get("objective", "softmax") not in OBJECTIVES:
            raise ValueError(f"{e['name']}: objective must be one of {OBJECTIVES}")
        if e.get("recipe", spec.get("recipe", "quick")) not in RECIPES:
            raise ValueError(f"{e['name']}: recipe must be one of {tuple(RECIPES)}")
        if e.get("seeds") is not None and not (1 <= int(e["seeds"]) <= 10):
            raise ValueError(f"{e['name']}: seeds must be 1 to 10")
        if e.get("halflife_days") is not None and float(e["halflife_days"]) <= 0:
            raise ValueError(f"{e['name']}: halflife_days must be above 0")
        walk = e.get("walk", spec.get("walk"))
        if walk and sorted(walk) != list(walk):
            raise ValueError(f"{e['name']}: walk dates must be in order")
        bad = [k for k in e.get("calibrate", []) if k not in CALIBRATIONS]
        if bad or (e.get("calibrate") and e.get("objective", "softmax") not in PRICE_OBJECTIVES):
            raise ValueError(f"{e['name']}: calibrate takes {CALIBRATIONS} on the price objectives only")
    return exps


def _starts(race: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(race[1:] != race[:-1]) + 1]


# --------------------------------------------------------------------------------------------------------------------
# The plan of 10 Nov 2026 (the owner: "Fundamental is to improve our model to predict BFSP / Closing Prices"; the plan
# in reports/greyhound_bfsp_plan.md). Step 1: fit the race, not the runner. Step 0: measure (importance by family,
# where the error sits, naive baselines, refits through the test window).
# --------------------------------------------------------------------------------------------------------------------

def race_soft_objective(race: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None):
    """A conditional logit fitted to soft labels: the race's runners compete in a softmax and ``target`` is each
    runner's share of the race (summing to 1 in each race): its share of the Betfair SP book (``share``), or that
    share mixed with the result (``mix``). The loss is the race's cross-entropy, -sum(target x log p), so the scores
    are the race's log-odds of the shares and the forecast is a coherent book by construction, not a runner-by-runner
    regression patched afterwards (the compression of batch 8). A race whose target does not sum to 1 is left out.
    ``weight``: a weight a row (the same for a race's runners), e.g. by recency. Rows grouped by race."""
    from greyhound.model import _race_softmax
    starts = _starts(race)
    sizes = np.diff(np.r_[starts, len(race)])
    tgt = np.asarray(target, float)
    ok_race = np.abs(np.add.reduceat(np.nan_to_num(tgt, nan=0.0), starts) - 1.0) < 1e-6
    ok = np.repeat(ok_race, sizes) & np.isfinite(tgt)
    y = np.where(ok, tgt, 0.0)
    w = np.ones(len(race)) if weight is None else np.asarray(weight, float)

    def fobj(score, data):
        p = _race_softmax(score, starts)
        return np.where(ok, w * (p - y), 0.0), np.maximum(np.where(ok, w * p * (1 - p), 0.0), 1e-6)

    def feval(score, data):
        p = _race_softmax(score, starts)
        return "race_xent", float(-(y * np.log(np.clip(p, 1e-12, 1))).sum() / max(ok_race.sum(), 1)), False
    return fobj, feval


def soft_target(df: pd.DataFrame, obj: str, mix_win: float = 0.3) -> np.ndarray:
    """``share``: each runner's share of its race's Betfair SP book (NaN where the race is not wholly priced).
    ``mix``: (1 - mix_win) x that share + mix_win x the result (the winner's share of the race's wins)."""
    share = df.bsp_p_norm.to_numpy(float)
    if obj == "share":
        return share
    won = df.won.to_numpy(float)
    wins = pd.Series(won).groupby(df.race_id.to_numpy()).transform("sum").to_numpy()
    res = np.where(wins > 0, won / np.where(wins > 0, wins, 1), np.nan)
    return (1 - mix_win) * share + mix_win * res


#: families for the importance table: each block, the metrics database's tables, and one theme across blocks
MARKET_HISTORY = re.compile(r"^(sp_p_mean3|dog_ae6|trainer_ae|gp_(mkt|ae|nres|bsp_ae|bsp_mkt)_|gp_trainer30_ae|"
                            r"ti_.*_(ae|lift)$|mk_)")


def families(feats: list[str], meta: dict, added: dict) -> dict[str, list[str]]:
    """Each feature's family: its engine block (``meta['features']``), the block it was added by, or ``other``."""
    of = {}
    for b, names in meta["features"].items():
        for f in names:
            of.setdefault(f, b)
    for b, names in added.items():
        for f in names:
            of[f] = b
    fam: dict[str, list[str]] = {}
    for f in feats:
        fam.setdefault(of.get(f, "other"), []).append(f)
    return fam


def _within_race_perm(race: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Two orderings of the rows, both grouped by race: ``base`` in row order, ``shuffled`` at random within each
    race. Rows ``base[i]`` take the values of rows ``shuffled[i]``: a permutation within each race."""
    n = len(race)
    return np.lexsort((np.arange(n), race)), np.lexsort((rng.random(n), race))


def importance(predict, X: np.ndarray, feats: list[str], fams: dict, race: np.ndarray, log_bsp: np.ndarray,
               won: np.ndarray, contrib=None, seed: int = 0) -> dict:
    """The model's reliance on each family, on test rows only:
    - ``perm``: each family's columns shuffled together WITHIN each race (the race's level kept, what tells its
      runners apart destroyed), the rise in RMSE of log(BSP) and in log-loss (positive: the model needs the family);
      the market-history theme the same way;
    - ``shap``: the mean absolute SHAP value (log-price scale) of each feature and each family, where ``contrib``
      gives LightGBM's ``pred_contrib`` for these rows.
    ``predict(X) -> (pred_bsp, p_model)``."""
    from greyhound.model import logloss
    rng = np.random.default_rng(seed)
    ok = np.isfinite(log_bsp)

    def score(Xa):
        pb, pm = predict(Xa)
        e = np.log(pb[ok]) - log_bsp[ok]
        return float(np.sqrt(np.mean(e ** 2))), logloss(pm, won)

    r0, l0 = score(X)
    out = {"rows": int(len(X)), "races": int(len(np.unique(race))), "rmse": round(r0, 5), "logloss": round(l0, 5),
           "perm": {}}
    base, shuf = _within_race_perm(race, rng)
    groups = dict(fams)
    groups["theme: market history"] = [f for f in feats if MARKET_HISTORY.match(f)]
    idx = {f: i for i, f in enumerate(feats)}
    for name, cols in groups.items():
        js = [idx[c] for c in cols if c in idx]
        if not js:
            continue
        Xp = X.copy()
        Xp[np.ix_(base, js)] = X[np.ix_(shuf, js)]
        r, ll = score(Xp)
        out["perm"][name] = {"features": len(js), "rmse_rise": round(r - r0, 5), "logloss_rise": round(ll - l0, 6)}
    if contrib is not None:
        a = np.abs(contrib[:, :-1]).mean(0)
        by_f = pd.Series(a, index=feats).sort_values(ascending=False)
        out["shap_top"] = {k: round(float(v), 5) for k, v in by_f.head(40).items()}
        out["shap_family"] = {name: round(float(by_f.reindex(cols).sum()), 5) for name, cols in groups.items()}
    return out


BSP_BANDS = (1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 1001.0)


def breakdown(pred: pd.DataFrame) -> dict:
    """Where the error in log(BSP) sits: overall and within the race (the share of each race's spread of log(BSP)
    the forecast explains: R2 within), and the RMSE and the bias (mean of log forecast less log BSP: above 0, priced
    too long) by BSP band, by runs before, by field and by track."""
    d = pred[(pred.bsp > 1) & (pred.pred_bsp > 1)].copy()
    if not len(d):
        return {}
    d["e"] = np.log(d.pred_bsp) - np.log(d.bsp)
    y = np.log(d.bsp)
    within = y - y.groupby(d.race_id).transform("mean")
    ew = d.e - d.e.groupby(d.race_id).transform("mean")
    out = {"runners": int(len(d)), "rmse": round(float(np.sqrt((d.e ** 2).mean())), 4),
           "bias": round(float(d.e.mean()), 4),
           # knowing only each race's mean log(BSP): the "race average" baseline of section 14 (0.754)
           "rmse_race_mean_only": round(float(np.sqrt((within ** 2).mean())), 4),
           "r2_within_race": round(float(1 - (ew ** 2).sum() / (within ** 2).sum()), 4),
           "rmse_of_race_level": round(float(np.sqrt((d.e.groupby(d.race_id).transform("mean") ** 2).mean())), 4)}

    def by(col, bins=None, labels=None, top=None):
        k = pd.cut(d[col], bins, right=False, labels=labels) if bins is not None else d[col]
        g = d.groupby(k, observed=True).e
        t = pd.DataFrame({"n": g.size(), "rmse": np.sqrt(g.apply(lambda v: (v ** 2).mean())), "bias": g.mean()})
        if top:
            t = t.sort_values("n", ascending=False).head(top)
        return {str(i): {"n": int(r.n), "rmse": round(float(r.rmse), 4), "bias": round(float(r.bias), 4)}
                for i, r in t.iterrows()}

    out["by_bsp"] = by("bsp", BSP_BANDS)
    if "runs_before" in d:
        out["by_runs_before"] = by("runs_before", (-0.5, 0.5, 2.5, 5.5, 19.5, 1e6), ("0", "1-2", "3-5", "6-19", "20+"))
    out["by_field"] = by("field")
    out["by_track"] = by("track", top=15)
    return out


def _block_prefix(name: str) -> str:
    """The column prefix of block ``name`` (``greyhound/blocks``, or a ``pit_`` table)."""
    import importlib
    if name.startswith("pit_"):
        from greyhound import pit
        return pit.module(name[4:]).PREFIX
    return importlib.import_module(f"greyhound.blocks.{name}").PREFIX


def run_experiment(df: pd.DataFrame, meta: dict, exp: dict, recipe: dict, test_from: str = TEST_FROM,
                   test_to: str | None = None) -> tuple[dict, pd.DataFrame]:
    """One experiment: its features (the set, the blocks added, the prefixes dropped), one fit on the recipe's window
    before ``test_from`` (early-stopped on its last tenth), every race from ``test_from`` priced; the scores."""
    import lightgbm as lgb
    from greyhound import blocks as gblocks
    from greyhound.model import blend_test, logloss, race_norm, race_objective, _race_softmax
    t0 = time.monotonic()
    feats = feature_set(meta, exp.get("features", "card_safe"))
    for b in exp.get("add_blocks", []):
        feats += [f for f in gblocks.attach(df, b) if f not in feats]
    for f in exp.get("add_features", []):             # named cached columns (e.g. the day's weigh-in features)
        if f not in feats:
            feats.append(f)
    added = {}
    for b in exp.get("add_blocks", []):
        added[b] = [c for c in df.columns if c in feats and c.startswith(_block_prefix(b))]
    for pre in exp.get("drop", []):
        feats = [f for f in feats if not f.startswith(pre)]
    if exp.get("keep"):                                # a baseline on named features only (exact names or prefixes)
        feats = [f for f in feats if any(f == k or f.startswith(k) for k in exp["keep"])]
    feats = [f for f in feats if f in df.columns]
    if not feats:
        raise ValueError(f"{exp['name']}: no features left")
    obj = exp.get("objective", "softmax")
    t = df.t.to_numpy().astype("datetime64[ns]")
    lo = np.datetime64(pd.Timestamp(recipe["train_from"]), "ns")
    edge = np.datetime64(pd.Timestamp(test_from), "ns")
    hi = np.datetime64(pd.Timestamp(test_to) + pd.Timedelta(days=1), "ns") if test_to else None
    fit_i = np.flatnonzero((t >= lo) & (t < edge))
    test_i = np.flatnonzero((t >= edge) & ((t < hi) if hi is not None else True))
    if obj == "logbsp":
        fit_i = fit_i[df.bsp.to_numpy(float)[fit_i] > 1]
    elif obj in ("share", "mix"):                     # races wholly priced by the Betfair SP: the book's shares exist
        fit_i = fit_i[df.bsp_p_norm.notna().to_numpy()[fit_i]]
    cut = np.quantile(t[fit_i].astype("int64"), 0.9).astype("datetime64[ns]")
    tr_i, es_i = fit_i[t[fit_i] < cut], fit_i[t[fit_i] >= cut]

    def matrix(rows: np.ndarray) -> np.ndarray:                 # only the rows a fit or a test reads
        X = np.empty((len(rows), len(feats)), dtype=np.float32)
        for j, f in enumerate(feats):
            X[:, j] = df[f].to_numpy(np.float32, na_value=np.nan)[rows]
        return X

    won = df.won.to_numpy(float)
    race = pd.factorize(df.race_id)[0]
    y = np.log(np.clip(df.bsp.to_numpy(float), 1.01, 1000.0)) if obj == "logbsp" else won
    soft = soft_target(df, obj, exp.get("mix_win", 0.3)) if obj in ("share", "mix") else None
    params = dict(BASE_PARAMS, learning_rate=recipe["learning_rate"], max_bin=recipe["max_bin"])
    params.update(exp.get("params", {}))
    # recency weights (step 1.4): a fit row weighs 0.5 ** (its age at the test's start / the half-life)
    wts = None
    if exp.get("halflife_days"):
        age = (edge - t[tr_i]).astype("timedelta64[s]").astype(float) / 86400.0
        wts = 0.5 ** (age / float(exp["halflife_days"]))
    dtr = lgb.Dataset(matrix(tr_i), y[tr_i], weight=wts if obj in ("logbsp", "binary") else None, feature_name=feats,
                      free_raw_data=True, params={"max_bin": recipe["max_bin"]})
    des = lgb.Dataset(matrix(es_i), y[es_i], reference=dtr, free_raw_data=True)
    models = []
    for k in range(int(exp.get("seeds", 1))):           # step 1.5: several seeds, their raw scores averaged
        p_k = dict(params, seed=int(params.get("seed", 0)) + k)
        es = lgb.early_stopping(recipe["early_stop"], verbose=False)
        stop = [es, time_budget(recipe.get("max_seconds"), es)]
        if obj == "binary":
            m = lgb.train(dict(p_k, objective="binary"), dtr, recipe["rounds"], valid_sets=[des], callbacks=stop)
        elif obj == "logbsp":
            m = lgb.train(dict(p_k, objective="huber", alpha=float(exp["huber"])) if exp.get("huber") else
                          dict(p_k, objective="regression"), dtr, recipe["rounds"], valid_sets=[des], callbacks=stop)
        elif obj in ("share", "mix"):
            fobj, _ = race_soft_objective(race[tr_i], soft[tr_i], wts)
            _, feval = race_soft_objective(race[es_i], soft[es_i])
            m = lgb.train(dict(p_k, objective=fobj, metric="None"), dtr, recipe["rounds"], valid_sets=[des],
                          feval=feval, callbacks=stop)
        else:
            second = (df.position.to_numpy(float) == 2).astype(float) if obj == "pl2" else None
            fobj, _ = race_objective(race[tr_i], won[tr_i], None if second is None else second[tr_i])
            _, feval = race_objective(race[es_i], won[es_i])
            m = lgb.train(dict(p_k, objective=fobj, metric="None"), dtr, recipe["rounds"], valid_sets=[des],
                          feval=feval, callbacks=stop)
        models.append(m)
    del dtr, des
    m = models[0]
    fit_s = time.monotonic() - t0
    budget = recipe.get("max_seconds")
    stopped = ("cap" if m.current_iteration() >= recipe["rounds"] else
               "time" if budget and fit_s > budget else "early_stop")
    gc.collect()

    def raw_of(X):
        return np.mean([mm.predict(X, num_iteration=mm.best_iteration) for mm in models], axis=0)

    X_test = matrix(test_i)
    raw = raw_of(X_test)
    keep = ["race_id", "t", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "field", "won",
            "sp_decimal", "sp_p_norm", "bsp", "bsp_p_norm", "runs_before"]
    pred = df.iloc[test_i][[c for c in keep if c in df.columns]].reset_index(drop=True)
    if obj == "binary":
        pred["p_model"] = race_norm(pd.Series(raw), pred.race_id).to_numpy()
    elif obj == "logbsp":
        pred["pred_bsp"] = np.exp(raw)
        pred["p_model"] = race_norm(1.0 / pred.pred_bsp, pred.race_id).to_numpy()
        if exp.get("rescale_book"):
            # the SP book's usual total from the fit's races: each race's chances normalised and scaled to it, so the
            # forecast's level is the SP's as well as its order (the closing study: the mid 0.635 -> 0.492 so)
            fr = pd.DataFrame({"r": df.race_id.to_numpy()[fit_i], "b": df.bsp.to_numpy(float)[fit_i]})
            book = float(np.median((1.0 / fr.b).groupby(fr.r).sum()))
            pred["pred_bsp"] = 1.0 / (book * pred.p_model)
    elif obj in ("share", "mix"):
        # the race's softmax is the forecast book's shares; the price is 1 / (the SP book's usual total x share)
        pred["p_model"] = _race_softmax(raw.astype(float), _starts(race[test_i]))
        fr = pd.DataFrame({"r": df.race_id.to_numpy()[fit_i], "b": df.bsp.to_numpy(float)[fit_i]})
        book = float(np.median((1.0 / fr.b).groupby(fr.r).sum()))
        pred["pred_bsp"] = 1.0 / (book * pred.p_model)
    else:
        pred["p_model"] = _race_softmax(raw.astype(float), _starts(race[test_i]))
    calibrated = {}
    if obj in PRICE_OBJECTIVES and exp.get("calibrate"):
        # fitted on the early-stopping rows (the trees never trained on them), scored on the test, paired by day. A race
        # objective's log price is minus its score plus a race constant, which the stretch centres away
        raw_es = raw_of(matrix(es_i))
        lbsp = np.log(np.clip(df.bsp.to_numpy(float), 1.01, 1000.0))
        sign = 1.0 if obj == "logbsp" else -1.0
        book = (float(np.median((1.0 / pd.Series(df.bsp.to_numpy(float)[fit_i])).groupby(race[fit_i]).sum()))
                if exp.get("rescale_book") or obj != "logbsp" else None)
        for kind in exp["calibrate"]:
            fit = fit_stretch(sign * raw_es, lbsp[es_i], won[es_i], race[es_i], df.track.to_numpy()[es_i], kind)
            cal = apply_stretch(pred, sign * raw, fit, book)
            calibrated[kind] = {"fit": fit, **score_pair(pred, cal)}
    imp = pd.Series(m.feature_importance("gain"), index=feats)
    whole = pred.bsp_p_norm.notna().groupby(pred.race_id).transform("all") if "bsp_p_norm" in pred else None
    res = {"name": exp["name"], "spec": exp, "recipe": recipe, "features": len(feats), "rounds": m.best_iteration,
           "stopped_by": stopped,
           "fit_rows": int(len(fit_i)), "test_runners": int(len(pred)), "test_races": int(pred.race_id.nunique()),
           "test_from": test_from, "test_to": str(pred.race_date.max()),
           "logloss": round(logloss(pred.p_model, pred.won), 5),
           "seconds_fit": round(fit_s), "top_features": {k: round(float(v), 4) for k, v in
                                                         (imp / imp.sum()).sort_values(ascending=False).head(15).items()}}
    if whole is not None and whole.any():
        b = pred[whole]
        res["on_bsp_races"] = {"runners": int(len(b)), "model": round(logloss(b.p_model, b.won), 5),
                               "bsp": round(logloss(b.bsp_p_norm, b.won), 5)}
        res["benter_vs_bsp"] = blend_test(b, "bsp_p_norm")
    if obj in PRICE_OBJECTIVES:
        ok = pred.bsp > 1
        e = np.log(pred.pred_bsp[ok]) - np.log(pred.bsp[ok])
        res["log_bsp_rmse"] = round(float(np.sqrt(np.mean(e ** 2))), 4)
        res["breakdown"] = breakdown(pred)
    res["seeds"] = len(models)
    if exp.get("importance") and obj in PRICE_OBJECTIVES:
        # on a sample of the test's races (the same races for every experiment: chosen by race id)
        rt = race[test_i]
        races = np.unique(rt)
        pick = np.isin(rt, np.random.default_rng(0).choice(races, min(len(races), int(exp.get("importance_races",
                                                                                            4000))), replace=False))
        Xs, rs = X_test[pick], rt[pick]
        sub = pred[pick].reset_index(drop=True)
        rescale = exp.get("rescale_book") or obj in ("share", "mix")
        fr = pd.DataFrame({"r": df.race_id.to_numpy()[fit_i], "b": df.bsp.to_numpy(float)[fit_i]})
        book_i = float(np.median((1.0 / fr.b).groupby(fr.r).sum()))
        st = _starts(rs)

        def predict(Xa):
            r = raw_of(Xa)
            if obj == "logbsp":
                p = race_norm(pd.Series(1.0 / np.exp(r)), pd.Series(rs)).to_numpy()
                pb = 1.0 / (book_i * p) if rescale else np.exp(r)
            else:
                p = _race_softmax(r.astype(float), st)
                pb = 1.0 / (book_i * p)
            return pb, p

        n_shap = min(len(Xs), int(exp.get("shap_rows", 8000)))
        contrib = m.predict(Xs[:n_shap], num_iteration=m.best_iteration, pred_contrib=True)
        lb = np.log(sub.bsp.where(sub.bsp > 1).to_numpy(float))
        res["importance"] = importance(predict, Xs, feats, families(feats, meta, added), rs, lb,
                                       sub.won.to_numpy(float), contrib)
        _log(f"importance: {json.dumps(res['importance']['perm'])[:2000]}")
    if calibrated:
        res["calibrated"] = calibrated
    if exp.get("groups"):
        res["by_group"] = {g: group_scores(pred[in_group(pred, tracks)], obj) for g, tracks in
                           split_groups(pred, exp["groups"]).items()}
    res["seconds"] = round(time.monotonic() - t0)
    return res, pred


def run_walk(df: pd.DataFrame, meta: dict, exp: dict, recipe: dict, walk: list[str],
             test_to: str | None = None) -> tuple[dict, pd.DataFrame]:
    """The test window priced in segments, the model refitted at the start of each on everything before it (step 0.3
    of the plan of 10 Nov 2026: one fit at 1 June priced five months on a model that never saw them); the segments'
    prices pooled and scored as one. The importance table, when asked, is read on the last segment only."""
    from greyhound.model import logloss
    preds, parts, res = [], [], {}
    for i, start in enumerate(walk):
        end = (pd.Timestamp(walk[i + 1]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") if i + 1 < len(walk) else test_to
        ex = exp if i == len(walk) - 1 else {k: v for k, v in exp.items() if k not in ("importance", "calibrate")}
        _log(f"{exp['name']}: segment {start} to {end or 'the last day'}")
        res, pred = run_experiment(df, meta, ex, recipe, start, end)
        parts.append({k: res.get(k) for k in ("test_from", "test_to", "rounds", "stopped_by", "test_runners",
                                              "logloss", "log_bsp_rmse")})
        preds.append(pred)
    pred = pd.concat(preds, ignore_index=True)
    res = dict(res, walk=parts, test_from=walk[0], test_to=str(pred.race_date.max()),
               test_runners=int(len(pred)), test_races=int(pred.race_id.nunique()),
               logloss=round(logloss(pred.p_model, pred.won), 5))
    res.pop("calibrated", None)
    if "pred_bsp" in pred:
        ok = pred.bsp > 1
        res["log_bsp_rmse"] = round(float(np.sqrt(np.mean((np.log(pred.pred_bsp[ok]) - np.log(pred.bsp[ok])) ** 2))), 4)
        res["breakdown"] = breakdown(pred)
    whole = pred.bsp_p_norm.notna().groupby(pred.race_id).transform("all")
    if whole.any():
        b = pred[whole]
        res["on_bsp_races"] = {"runners": int(len(b)), "model": round(logloss(b.p_model, b.won), 5),
                               "bsp": round(logloss(b.bsp_p_norm, b.won), 5)}
    return res, pred


def details(results: list[dict]) -> str:
    """Under the table: the importance tables and where the error sits, for each experiment that has them."""
    md = []
    for r in results:
        imp = r.get("importance")
        if imp:
            md += ["", f"### What {r['name']} relies on ({imp['races']} test races, {imp['rows']} runners; RMSE "
                   f"{imp['rmse']}, log-loss {imp['logloss']})", "",
                   "Each family's columns shuffled together within each race: the rise in RMSE of log(BSP) and in "
                   "log-loss (higher: the model needs it more); and the mean absolute SHAP value (log-price scale).", "",
                   "| Family | Features | RMSE rise | Log-loss rise | SHAP (sum) |", "|---|---|---|---|---|"]
            for name, v in sorted(imp["perm"].items(), key=lambda kv: -kv[1]["rmse_rise"]):
                md.append(f"| {name} | {v['features']} | {v['rmse_rise']:+.5f} | {v['logloss_rise']:+.6f} | "
                          f"{(imp.get('shap_family') or {}).get(name, '')} |")
            if imp.get("shap_top"):
                md += ["", "Top features by mean absolute SHAP value: " + ", ".join(
                    f"{k} {v}" for k, v in list(imp["shap_top"].items())[:25]) + "."]
        bd = r.get("breakdown")
        if bd:
            md += ["", f"### Where {r['name']}'s error sits ({bd['runners']} runners)", "",
                   f"RMSE {bd['rmse']}, bias {bd['bias']:+.4f} (above 0: priced too long). Knowing only each race's "
                   f"average log(BSP): {bd['rmse_race_mean_only']}. Share of the within-race spread explained: "
                   f"{bd['r2_within_race']:.1%}. Error in the race's level alone: {bd['rmse_of_race_level']}.", "",
                   "| By | Group | Runners | RMSE | Bias |", "|---|---|---|---|---|"]
            for key, lab in (("by_bsp", "BSP from"), ("by_runs_before", "Runs before"), ("by_field", "Field"),
                             ("by_track", "Track")):
                for g, v in (bd.get(key) or {}).items():
                    md.append(f"| {lab} | {g} | {v['n']} | {v['rmse']} | {v['bias']:+.4f} |")
        if r.get("walk"):
            md += ["", f"{r['name']}, segment by segment: " + "; ".join(
                f"{w['test_from']} to {w['test_to']}: RMSE {w.get('log_bsp_rmse')}, log-loss {w['logloss']}, "
                f"{w['rounds']} rounds" for w in r["walk"]) + "."]
    return "\n".join(md)


# --------------------------------------------------------------------------------------------------------------------
# Track groups (the owner, 4 Nov 2026: "use the weigh-in for the tracks we have it and then evaluate if we can cover
# other tracks with our model or just leave them alone")
# --------------------------------------------------------------------------------------------------------------------

def _norm_track(x) -> str:
    return "".join(c for c in str(x).lower() if c.isalnum())


def in_group(pred: pd.DataFrame, tracks) -> pd.Series:
    """The runners at these tracks (names matched without case, spaces or punctuation)."""
    names = {_norm_track(t) for t in tracks}
    return pred.track.map(_norm_track).isin(names)


def split_groups(pred: pd.DataFrame, groups: dict) -> dict:
    """The spec's named groups of tracks and ``rest``: every other track in the test. ``"each_track"``: a group per
    track (no rest)."""
    if groups == "each_track":
        return {str(t): [t] for t in sorted(pred.track.astype(str).unique())}
    named = {g: list(t) for g, t in groups.items()}
    taken = {_norm_track(t) for ts in named.values() for t in ts}
    named["rest"] = sorted({t for t in pred.track.astype(str).unique() if _norm_track(t) not in taken})
    return named


def group_scores(p: pd.DataFrame, obj: str) -> dict:
    """A group's runners and races, the model's log-loss and the BSP's on the same races (the gap: how far behind the
    market's final price), and the RMSE of log(BSP) for a log-BSP model."""
    from greyhound.model import logloss
    if not len(p):
        return {"runners": 0}
    out = {"runners": int(len(p)), "races": int(p.race_id.nunique()), "tracks": int(p.track.nunique()),
           "logloss": round(logloss(p.p_model, p.won), 5)}
    if "bsp_p_norm" in p:
        whole = p.bsp_p_norm.notna().groupby(p.race_id).transform("all")
        b = p[whole]
        if len(b):
            out["on_bsp_races"] = {"runners": int(len(b)), "model": round(logloss(b.p_model, b.won), 5),
                                   "bsp": round(logloss(b.bsp_p_norm, b.won), 5),
                                   "gap": round(logloss(b.p_model, b.won) - logloss(b.bsp_p_norm, b.won), 5)}
    if obj == "logbsp" and "pred_bsp" in p:
        ok = p.bsp > 1
        if ok.any():
            e = np.log(p.pred_bsp[ok]) - np.log(p.bsp[ok])
            out["log_bsp_rmse"] = round(float(np.sqrt(np.mean(e ** 2))), 4)
    return out


# --------------------------------------------------------------------------------------------------------------------
# Calibration of the log-BSP price (the owner, 4 Nov 2026: "we need to make our prices more accurate"; roadmap 4.3)
# --------------------------------------------------------------------------------------------------------------------

CALIBRATIONS = ("stretch", "stretch_track", "temper", "temper_track")


def _centre(x: np.ndarray, race: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, float))
    return (s - s.groupby(np.asarray(race)).transform("mean")).to_numpy()


def _race_nll(z: np.ndarray, won: np.ndarray, race: np.ndarray) -> float:
    """The mean over races of minus the log of the winner's softmax chance, the scores ``z`` within each race."""
    s = pd.Series(z)
    mx = s.groupby(race).transform("max")
    lse = np.log(np.exp(s - mx).groupby(race).transform("sum")) + mx
    races = len(np.unique(race[won > 0]))
    return float(-(won * (s - lse)).sum() / max(races, 1))


def fit_stretch(log_pred: np.ndarray, log_bsp: np.ndarray, won: np.ndarray, race: np.ndarray, track: np.ndarray,
                kind: str, shrink: float = 2000.0) -> dict:
    """How far to stretch the model's prices within each race: the chance becomes proportional to pred_bsp^-b (b above
    1 spreads a compressed price: favourites shorter, outsiders longer). ``stretch``: b by least squares of the
    race-centred log(BSP) on the race-centred log price; ``temper``: b that maximises the winners' likelihood. The
    ``_track`` forms fit a b per track, shrunk to the overall b by ``shrink`` runners."""
    if kind not in CALIBRATIONS:
        raise ValueError(f"calibrate must be among {CALIBRATIONS}: {kind!r}")
    c = _centre(log_pred, race)
    by_runner = kind.startswith("stretch")
    if by_runner:
        yc = _centre(log_bsp, race)

        def b_of(mask):
            return float((c[mask] * yc[mask]).sum() / max((c[mask] ** 2).sum(), 1e-12))
    else:
        from scipy.optimize import minimize_scalar

        def b_of(mask):
            r = minimize_scalar(lambda b: _race_nll(-b * c[mask], won[mask], race[mask]), bounds=(0.2, 4.0),
                                method="bounded")
            return float(r.x)
    every = np.ones(len(c), bool)
    out = {"kind": kind, "b": round(b_of(every), 4), "rows": int(len(c)), "races": int(len(np.unique(race)))}
    if kind.endswith("_track"):
        tr = pd.Series(track).astype(str).to_numpy()
        out["by_track"] = {}
        for t in sorted(set(tr)):
            mask = tr == t
            n = int(mask.sum())
            if n < 50 or len(np.unique(race[mask])) < 10:
                continue
            w = n / (n + shrink)
            out["by_track"][t] = round(w * b_of(mask) + (1 - w) * out["b"], 4)
    return out


def apply_stretch(pred: pd.DataFrame, log_pred: np.ndarray, fit: dict, book: float | None) -> pd.DataFrame:
    """The test prices stretched by ``fit`` (``fit_stretch``): the chance normalised in each race, the price at the
    SP book's usual total when ``book`` is given (``rescale_book``), else around the race's mean log price."""
    from greyhound.model import race_norm
    b = pred.track.astype(str).map(fit.get("by_track") or {}).fillna(fit["b"]).to_numpy(float)
    c = _centre(log_pred, pred.race_id.to_numpy())
    out = pred[["race_id", "race_date", "dog_id", "track", "won"] + [c_ for c_ in ("bsp",) if c_ in pred]].copy()
    out["p_model"] = race_norm(pd.Series(np.exp(-b * c)), out.race_id).to_numpy()
    if book:
        out["pred_bsp"] = 1.0 / (book * out.p_model)
    else:
        mean = pd.Series(np.asarray(log_pred, float)).groupby(pred.race_id.to_numpy()).transform("mean").to_numpy()
        out["pred_bsp"] = np.exp(mean + b * c)
    return out


def _boot_days(day: pd.DataFrame, n_boot: int = 2000, seed: int = 0, root: bool = False) -> list[float]:
    """The 90% interval of (sum e - sum b) / n over days resampled (the square roots' difference when ``root``)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(day), (n_boot, len(day)))
    e, b, n = (day[k].to_numpy()[idx].sum(1) for k in ("e", "b", "n"))
    diff = np.sqrt(e / n) - np.sqrt(b / n) if root else (e - b) / n
    return [round(float(np.percentile(diff, 5)), 5), round(float(np.percentile(diff, 95)), 5)]


def score_pair(base: pd.DataFrame, cal: pd.DataFrame) -> dict:
    """The calibrated prices against the uncalibrated on the same runners: RMSE of log(BSP) and log-loss, each with the
    difference and its 90% interval from a bootstrap over days (negative: the calibration is better)."""
    out = {}
    ok = (base.bsp > 1).to_numpy() if "bsp" in base else np.zeros(len(base), bool)
    if ok.any():
        se_b = (np.log(base.pred_bsp.to_numpy(float)) - np.log(base.bsp.to_numpy(float))) ** 2
        se_e = (np.log(cal.pred_bsp.to_numpy(float)) - np.log(cal.bsp.to_numpy(float))) ** 2
        day = pd.DataFrame({"d": base.race_date.astype(str), "e": se_e, "b": se_b})[ok].groupby("d").agg(
            e=("e", "sum"), b=("b", "sum"), n=("e", "size"))
        rmse_e, rmse_b = np.sqrt(day.e.sum() / day.n.sum()), np.sqrt(day.b.sum() / day.n.sum())
        out["log_bsp_rmse"] = round(float(rmse_e), 4)
        out["rmse_vs_uncalibrated"] = round(float(rmse_e - rmse_b), 4)
        out["rmse_ci90"] = _boot_days(day, root=True) if len(day) > 1 else None

    def ll(p, w):
        p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
        return -(w * np.log(p) + (1 - w) * np.log(1 - p))

    w = base.won.to_numpy(float)
    day = pd.DataFrame({"d": base.race_date.astype(str), "e": ll(cal.p_model, w), "b": ll(base.p_model, w)}).groupby(
        "d").agg(e=("e", "sum"), b=("b", "sum"), n=("e", "size"))
    out["logloss"] = round(float(day.e.sum() / day.n.sum()), 5)
    out["logloss_vs_uncalibrated"] = round(float((day.e.sum() - day.b.sum()) / day.n.sum()), 5)
    out["logloss_ci90"] = _boot_days(day) if len(day) > 1 else None
    by = pd.DataFrame({"t": base.track.astype(str), "e": ll(cal.p_model, w), "b": ll(base.p_model, w)}).groupby("t")
    out["logloss_vs_uncalibrated_by_track"] = {t: round(float((g.e.sum() - g.b.sum()) / len(g)), 5)
                                               for t, g in by if len(g) >= 500}
    return out


# --------------------------------------------------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------------------------------------------------

def _vs(r: dict, base: dict) -> str:
    pr = r.get("paired")
    if pr and r["name"] != base["name"]:
        return f"{pr['diff']} {pr['ci90']}" + (f" z {pr['z']:+.1f}" if pr.get("z") is not None else "")
    return str(round(r["logloss"] - base["logloss"], 5))


def paired_vs_baseline(folder: Path, baseline: str | None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Each experiment's log-loss less the baseline's on the same runners, and its 90% interval from a bootstrap over
    days (a day's races share their conditions, so the day is the unit). Negative: better than the baseline."""
    files = {p.stem.removeprefix("pred_"): p for p in folder.rglob("pred_*.parquet")}
    if not baseline or baseline not in files:
        return {}
    key = ["race_id", "dog_id"]

    def ll(d):
        p = d.p_model.clip(1e-12, 1 - 1e-12)
        return -(d.won * np.log(p) + (1 - d.won) * np.log(1 - p))

    base = pd.read_parquet(files[baseline]).assign(ll_b=lambda d: ll(d))[key + ["race_date", "ll_b"]]
    rng = np.random.default_rng(seed)
    out = {}
    for name, f in files.items():
        d = pd.read_parquet(f).assign(ll_e=lambda x: ll(x))[key + ["ll_e"]].merge(base, on=key, how="inner")
        day = d.groupby("race_date").agg(e=("ll_e", "sum"), b=("ll_b", "sum"), n=("ll_e", "size"))
        if len(day) < 2:
            continue
        idx = rng.integers(0, len(day), (n_boot, len(day)))
        diff = (day.e.to_numpy()[idx].sum(1) - day.b.to_numpy()[idx].sum(1)) / day.n.to_numpy()[idx].sum(1)
        d0 = float((day.e.sum() - day.b.sum()) / day.n.sum())
        out[name] = {"diff": round(d0, 5),
                     "ci90": [round(float(np.percentile(diff, 5)), 5), round(float(np.percentile(diff, 95)), 5)],
                     "z": round(d0 / float(diff.std()), 2) if diff.std() > 0 else None,
                     "runners": int(day.n.sum()), "days": int(len(day))}
    return out


def paired_bsp_rmse(folder: Path, baseline: str | None, n_boot: int = 2000, seed: int = 0) -> dict:
    """The accuracy of the BFSP forecast (the owner, 3 Nov 2026: "the final price / log of final price is key"): each
    log-BSP experiment's RMSE of log(BSP) less the baseline's on the same runners, with its 90% interval from a
    bootstrap over days. Negative: a more accurate SP forecast."""
    files = {p.stem.removeprefix("pred_"): p for p in folder.rglob("pred_*.parquet")}
    if not baseline or baseline not in files:
        return {}
    key = ["race_id", "dog_id"]

    def se(d, col):
        ok = (d.bsp > 1) & (d.pred_bsp > 1)
        d = d[ok]
        return d.assign(**{col: (np.log(d.pred_bsp) - np.log(d.bsp)) ** 2})

    b0 = pd.read_parquet(files[baseline])
    if "pred_bsp" not in b0:
        return {}
    base = se(b0, "se_b")[key + ["race_date", "se_b"]]
    rng = np.random.default_rng(seed)
    out = {}
    for name, f in files.items():
        e = pd.read_parquet(f)
        if "pred_bsp" not in e:
            continue
        d = se(e, "se_e")[key + ["se_e"]].merge(base, on=key, how="inner")
        day = d.groupby("race_date").agg(e=("se_e", "sum"), b=("se_b", "sum"), n=("se_e", "size"))
        if len(day) < 2:
            continue
        idx = rng.integers(0, len(day), (n_boot, len(day)))
        nn = day.n.to_numpy()[idx].sum(1)
        diff = np.sqrt(day.e.to_numpy()[idx].sum(1) / nn) - np.sqrt(day.b.to_numpy()[idx].sum(1) / nn)
        n = day.n.sum()
        d0 = float(np.sqrt(day.e.sum() / n) - np.sqrt(day.b.sum() / n))
        out[name] = {"rmse": round(float(np.sqrt(day.e.sum() / n)), 4),
                     "diff": round(d0, 4), "z": round(d0 / float(diff.std()), 2) if diff.std() > 0 else None,
                     "ci90": [round(float(np.percentile(diff, 5)), 4), round(float(np.percentile(diff, 95)), 4)],
                     "runners": int(n), "days": int(len(day))}
    return out


def _bsp_cells(r: dict) -> tuple[str, str]:
    p = r.get("paired_bsp")
    if not p:
        return (str(r.get("log_bsp_rmse", "")), "")
    z = f", z {p['z']:+.1f}" if p.get("z") is not None and p["diff"] != 0 else ""
    return (str(p["rmse"]), f"{p['diff']:+.4f} ({p['ci90'][0]:+.4f} to {p['ci90'][1]:+.4f}{z})")


def _short_cell(r: dict) -> str:
    sh = r.get("shortening") or {}
    if not sh or "skipped" in sh:
        return sh.get("skipped", "-") if sh else "-"
    return " / ".join(str((sh.get(m) or {}).get("model_adds", "-")) for m in SHORT_MARKS[:3])


def _short_vs(r: dict, base: dict) -> str:
    pr = r.get("paired_short")
    if not pr or r["name"] == base["name"]:
        return "-" if not pr else "0.0"
    return f"{pr['diff']} {pr['ci90']}"


def table(results: list[dict], baseline: str | None = None) -> str:
    """The trade-out's measure first (the owner, 3 Nov 2026: get on at prices expected to shorten at the SP): how much
    each experiment's price adds to the forecast of the SP from the book at first traded, 5 and 3 minutes out
    (higher is better) and the paired difference from the baseline (the SP book's cross-entropy a race, negative is
    better); then the log-loss against the result. Whether a model beats the SP itself is not the question: the SP is
    the market's best price; the results JSON keeps Benter's test for reference."""
    if not results:
        return "(no results)"
    base = next((r for r in results if r["name"] == baseline), results[0])
    lines = [f"Baseline: {base['name']}. Test {base['test_from']} to {base['test_to']} (log-loss); the SP forecast on "
             f"the recorded books from {STUDY_FROM}, walk-forward by day. z: the paired difference over its bootstrap "
             "standard error (by day); the bar from 10 Nov 2026 is 3 (reports/greyhound_bfsp_plan.md).", "",
             "| Experiment | Features | Objective | Rounds | RMSE log(BSP) | vs baseline (90%, by day) | Adds to the SP "
             "forecast a race: first traded / 5 min / 3 min | vs baseline (90%, by day) | Log-loss | vs baseline (90%, "
             "by day) | Seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        rmse, rmse_vs = _bsp_cells(r)
        lines.append(f"| {r['name']} | {r['features']} | {r['spec'].get('objective', 'softmax')} | {r['rounds']} | "
                     f"{rmse} | {rmse_vs} | {_short_cell(r)} | {_short_vs(r, base)} | {r['logloss']} | "
                     f"{_vs(r, base)} | {r['seconds']} |")
        for g, gs in (r.get("by_group") or {}).items():
            if not gs.get("runners"):
                continue
            ob = gs.get("on_bsp_races") or {}
            st = ((r.get("groups_study") or {}).get(g) or {}).get("shortening")
            short = _short_cell({"shortening": st}) if st else ""
            lines.append(f"| {r['name']} [{g}: {gs['tracks']} tracks, {gs['races']} races] | | | | "
                         f"{gs.get('log_bsp_rmse', '')} | | {short} | | {gs['logloss']} | "
                         f"vs the BSP {ob.get('gap', '')} | |")
        for kind, c in (r.get("calibrated") or {}).items():
            ci = c.get("rmse_ci90") or ["", ""]
            lci = c.get("logloss_ci90") or ["", ""]
            lines.append(f"| {r['name']} + {kind} (b {c['fit']['b']}) | | | | {c.get('log_bsp_rmse', '')} | "
                         f"{c.get('rmse_vs_uncalibrated', '')} ({ci[0]} to {ci[1]}) vs uncalibrated | | | "
                         f"{c['logloss']} | {c['logloss_vs_uncalibrated']} ({lci[0]} to {lci[1]}) vs uncalibrated | |")
    return "\n".join(lines)


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--build-cache", action="store_true", help="build the cache unless one for this code exists")
    ap.add_argument("--refresh", action="store_true", help="with --build-cache: build it again even so")
    ap.add_argument("--spec", default=None, help="the experiments (JSON)")
    ap.add_argument("--list", action="store_true", help="print the spec's experiment names as JSON (the CI matrix)")
    ap.add_argument("--lane", choices=("quick", "slow"), default=None,
                    help="with --list or --collect: the quick lane (the quick recipe, no study) or the slow one only")
    ap.add_argument("--only", default=None, help="run this experiment of the spec only")
    ap.add_argument("--collect", default=None, help="the table of every result JSON under this folder")
    ap.add_argument("--out", default="out/fastloop")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    if a.collect:
        rs = [json.loads(p.read_text()) for p in sorted(Path(a.collect).rglob("result_*.json"))]
        spec = json.loads(Path(a.spec).read_text()) if a.spec else {}
        if a.lane:
            rs = [r for r in rs if lane(r.get("spec", {}), spec) == a.lane or r["name"] == spec.get("baseline")]
        order = [e["name"] for e in spec.get("experiments", [])]
        rs.sort(key=lambda r: order.index(r["name"]) if r["name"] in order else len(order))
        paired = paired_vs_baseline(Path(a.collect), spec.get("baseline") or (rs[0]["name"] if rs else None))
        short = paired_shortening(Path(a.collect), spec.get("baseline") or (rs[0]["name"] if rs else None))
        bsp = paired_bsp_rmse(Path(a.collect), spec.get("baseline") or (rs[0]["name"] if rs else None))
        for r in rs:
            r["paired"] = paired.get(r["name"])
            r["paired_bsp"] = bsp.get(r["name"])
            r["paired_short"] = short.get(r["name"])
        md = table(rs, spec.get("baseline")) + "\n" + details(rs)
        Path(a.collect, "table.md").write_text(md + "\n")
        print(md)
        return 0
    if a.build_cache:
        meta = cache_meta(store)
        if meta and not a.refresh:
            print(json.dumps({"cache": "current", "hash": meta["hash"], "last_day": meta["last_day"]}))
        else:
            meta = build_cache(store, uk_today() - timedelta(days=1))
            print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=1))
    if not a.spec:
        return 0
    spec = json.loads(Path(a.spec).read_text())
    exps = check_spec(spec)
    if a.list:
        print(json.dumps([e["name"] for e in exps if a.lane is None or lane(e, spec) == a.lane]))
        return 0
    if a.only:
        exps = [e for e in exps if e["name"] == a.only]
        if not exps:
            ap.error(f"no experiment {a.only!r} in {a.spec}")
    df, meta = load_cache(store)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for e in exps:
        recipe = dict(RECIPES[e.get("recipe", spec.get("recipe", "quick"))], **e.get("recipe_overrides", {}))
        if spec.get("groups") and "groups" not in e:
            e = dict(e, groups=spec["groups"])
        _log(f"experiment {e['name']}")
        walk = e.get("walk", spec.get("walk"))
        if walk:
            res, pred = run_walk(df, meta, e, recipe, walk, spec.get("test_to"))
        else:
            res, pred = run_experiment(df, meta, e, recipe, spec.get("test_from", TEST_FROM), spec.get("test_to"))
        res["cache"] = {k: meta[k] for k in ("hash", "last_day", "rows")}
        if e.get("shortening", spec.get("shortening")):
            _log(f"shortening {e['name']}")
            res["shortening"] = shortening(store, pred, e["name"], out)
        if e.get("study"):
            res["study"] = run_study(store, pred, e["name"], out)
        if e.get("limit"):
            res["limit"] = run_limit(store, pred, e["name"], out)
        if e.get("publish_pred"):
            res["published"] = publish_pred(store, pred, e["name"], out, e["publish_pred"])
            _log(f"published {e['name']}: {res['published']}")
        if e.get("bfsp"):
            res["bfsp"] = run_bfsp(store, pred, e["name"], out)
        if e.get("groups") and (e.get("shortening", spec.get("shortening")) or e.get("limit")):
            # each group's races on their own: does our price add to the market there, and do the rules pay there
            res["groups_study"] = {}
            for g, tracks in split_groups(pred, e["groups"]).items():
                sub = pred[in_group(pred, tracks)]
                gname = f"{e['name']}__{g}"
                _log(f"group {g}: {len(sub)} runners")
                gs = {}
                if e.get("shortening", spec.get("shortening")):
                    gs["shortening"] = shortening(store, sub, gname, out)
                if e.get("limit"):
                    gs["limit"] = run_limit(store, sub, gname, out)
                res["groups_study"][g] = gs
        (out / f"result_{e['name']}.json").write_text(json.dumps(res, indent=1, default=str))
        # each runner's log-loss contribution, for the paired comparison against the baseline in --collect
        keep = ["race_id", "race_date", "dog_id", "won", "p_model"] + [c for c in ("bsp", "pred_bsp") if c in pred]
        # the race's identity too, for strategies scored on these prices (place markets joined by day, time, track, trap)
        keep += [c for c in ("race_time", "track", "trap", "dog_name", "field", "runs_before") if c in pred]
        pred[keep].to_parquet(out / f"pred_{e['name']}.parquet", index=False)      # bsp, pred_bsp: the paired RMSE
        results.append(res)
        _log(json.dumps({k: res[k] for k in ("name", "logloss", "rounds", "seconds")}))
    print(table(results, spec.get("baseline")))
    return 0


SHORT_MARKS = ("first_traded", "off_5", "off_3", "off_1")


def shortening_summary(acc: pd.DataFrame) -> dict:
    """Per mark, over every scored day: how much our price adds to the forecast of the SP from the book at that mark
    (the SP book's cross-entropy a race, market_micro less micro: the same conditional logit without and with our
    price), with micro's R2 of log(SP / mid) against no change and how often it calls the move the right way."""
    out = {}
    for mark, g in acc.groupby("mark"):
        mm, mi = g[g.model == "market_micro"], g[g.model == "micro"]
        if not len(mm) or not len(mi):
            continue
        races = mi.races.sum()
        out[mark] = {"days": int(mi.day.nunique()), "races": int(races),
                     "model_adds": round(float((mm.xent * mm.races).sum() - (mi.xent * mi.races).sum()) / races, 5),
                     "gain_over_market": round(float((mi.xent_market * mi.races).sum() - (mi.xent * mi.races).sum())
                                               / races, 5),
                     "r2": round(1 - float(mi.sse.sum() / mi.sse_nochange.sum()), 4),
                     "right_way%": round(100 * float((mi.right_way * mi.rows).sum() / mi.rows.sum()), 1)}
    return out


def shortening(store, pred: pd.DataFrame, name: str, out: Path) -> dict:
    """The trade-out's question (the owner, 3 Nov 2026: "we are just trying to get money on at higher prices that are
    expected to shorten at BFSP"): on the recorded books from STUDY_FROM (re-timed), how much this experiment's price
    adds to the forecast of the SP from the book at each mark, walk-forward by day (research.greyhound.tradeout_study).
    Each day's scores go to ``short_<name>.parquet`` for the paired comparison in ``--collect``."""
    from research.greyhound import score_full, tradeout_study as ts
    days = pred[pred.race_date.astype(str) >= STUDY_FROM]
    if not len(days):
        return {"skipped": f"no test days from {STUDY_FROM}"}
    first = date.fromisoformat(STUDY_FROM)
    last = date.fromisoformat(str(days.race_date.max()))
    score_full.score_days({name: days}, first, last, store, out / "pred")
    panel, _ = ts.load(store, first, last, source="retimed", pred_dir=str(out / "pred" / name))
    if not len(panel):
        return {"skipped": "no panel"}
    _, acc = ts.walk_forward(panel, names=["market_micro", "micro"])
    acc = pd.DataFrame(acc)
    if not len(acc):
        return {"skipped": "too few days"}
    acc.assign(day=acc.day.astype(str)).to_parquet(out / f"short_{name}.parquet", index=False)
    return shortening_summary(acc)


def paired_shortening(folder: Path, baseline: str | None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Each experiment's micro cross-entropy less the baseline's on the same days and marks (first traded, 5 and 3
    minutes out: the entry marks with something to forecast), a race, with its 90% interval from a bootstrap over days.
    Negative: its price forecasts the SP better than the baseline's."""
    files = {p.stem.removeprefix("short_"): p for p in folder.rglob("short_*.parquet")}
    if not baseline or baseline not in files:
        return {}

    def day_sums(f):
        a = pd.read_parquet(f)
        a = a[(a.model == "micro") & a["mark"].isin(SHORT_MARKS[:3])]
        return a.assign(x=a.xent * a.races)[["day", "mark", "x", "races"]]

    base = day_sums(files[baseline])
    rng = np.random.default_rng(seed)
    out = {}
    for name, f in files.items():
        d = day_sums(f).merge(base, on=["day", "mark"], suffixes=("_e", "_b"))
        day = d.groupby("day").agg(e=("x_e", "sum"), b=("x_b", "sum"), n=("races_b", "sum"))
        if len(day) < 2:
            continue
        idx = rng.integers(0, len(day), (n_boot, len(day)))
        diff = (day.e.to_numpy()[idx].sum(1) - day.b.to_numpy()[idx].sum(1)) / day.n.to_numpy()[idx].sum(1)
        out[name] = {"diff": round(float((day.e.sum() - day.b.sum()) / day.n.sum()), 5),
                     "ci90": [round(float(np.percentile(diff, 5)), 5), round(float(np.percentile(diff, 95)), 5)],
                     "days": int(len(day))}
    return out


def run_study(store, pred: pd.DataFrame, name: str, out: Path) -> dict:
    """The trade-out study on an experiment's prices: its days from STUDY_FROM joined to the Betfair markets, then
    ``research.greyhound.tradeout_study`` on the re-timed books."""
    from research.greyhound import score_full, tradeout_study
    days = pred[pred.race_date.astype(str) >= STUDY_FROM]
    if not len(days):
        return {"skipped": f"no test days from {STUDY_FROM}"}
    first = date.fromisoformat(STUDY_FROM)
    last = date.fromisoformat(str(days.race_date.max()))
    joined = score_full.score_days({name: days}, first, last, store, out / "pred")
    tradeout_study.main(["--from", STUDY_FROM, "--to", str(last), "--pred-dir", str(out / "pred" / name),
                         "--source", "retimed", "--out", str(out / f"study_{name}")])
    return {"head_to_head": joined.get("head_to_head"), "report": f"study_{name}.md"}


def run_limit(store, pred: pd.DataFrame, name: str, out: Path) -> dict:
    """The limit-order study (``research.greyhound.limit_study``) on an experiment's prices, its days from STUDY_FROM
    joined to the Betfair markets, on the re-timed books."""
    from research.greyhound import limit_study, score_full
    days = pred[pred.race_date.astype(str) >= STUDY_FROM]
    if not len(days):
        return {"skipped": f"no test days from {STUDY_FROM}"}
    last = str(days.race_date.max())
    if not (out / "pred" / name).exists():
        score_full.score_days({name: days}, date.fromisoformat(STUDY_FROM), date.fromisoformat(last), store,
                              out / "pred")
    limit_study.main(["--from", STUDY_FROM, "--to", last, "--pred-dir", str(out / "pred" / name),
                      "--source", "retimed", "--out", str(out / f"limit_{name}")])
    return {"report": f"limit_{name}.md"}


def publish_pred(store, pred: pd.DataFrame, name: str, out: Path, prefix: str) -> dict:
    """An experiment's priced days from STUDY_FROM, joined to the Betfair markets (``score_days``), written to S3 as
    ``<prefix>/<day>.parquet``: out-of-sample prices of the recorded days for a forecast fit to read
    (greyhound.laytest ``day_fit(prices=)``)."""
    from research.greyhound import score_full
    days = pred[pred.race_date.astype(str) >= STUDY_FROM]
    if not len(days):
        return {"skipped": f"no test days from {STUDY_FROM}"}
    last = str(days.race_date.max())
    if not (out / "pred" / name).exists():
        score_full.score_days({name: days}, date.fromisoformat(STUDY_FROM), date.fromisoformat(last), store,
                              out / "pred")
    n = 0
    for f in sorted((out / "pred" / name).glob("*.parquet")):
        store.put(f"{prefix.rstrip('/')}/{f.name}", f.read_bytes())
        n += 1
    return {"prefix": prefix, "days": n, "first": STUDY_FROM, "last": last}


def run_bfsp(store, pred: pd.DataFrame, name: str, out: Path) -> dict:
    """The closing-price study (``research.greyhound.bfsp_close``) with an experiment's prices as our price: how much
    a better pre-race price sharpens the forecast of log(BSP) from the book (the owner, 3 Nov 2026: "the final price /
    log of final price is key")."""
    from research.greyhound import bfsp_close, score_full
    days = pred[pred.race_date.astype(str) >= STUDY_FROM]
    if not len(days):
        return {"skipped": f"no test days from {STUDY_FROM}"}
    last = str(days.race_date.max())
    if not (out / "pred" / name).exists():
        score_full.score_days({name: days}, date.fromisoformat(STUDY_FROM), date.fromisoformat(last), store,
                              out / "pred")
    bfsp_close.main(["--from", STUDY_FROM, "--to", last, "--pred-dir", str(out / "pred" / name),
                     "--source", "retimed", "--out", str(out / f"bfsp_{name}")])
    return {"report": f"bfsp_{name}.md"}


if __name__ == "__main__":
    raise SystemExit(main())
