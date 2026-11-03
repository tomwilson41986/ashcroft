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
3. **The quick recipe** fits on 2024 on with a learning rate of 0.15 and coarse bins (minutes a fit); ``confirm`` fits
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
              "greyhound/data.py", "greyhound/track.py", "greyhound/model.py", "sources/gbgb.py")
KEEP_FROM = "2020-06-01"
BLOCKS = ("parity", "hrb", "lib")
PRICE_YEARS = "2018-2026"
EXTRA_COLS = ("bsp", "bsp_p_norm")
TEST_FROM = "2026-06-01"
STUDY_FROM = "2026-10-06"
RECIPES = {
    "quick": {"train_from": "2024-01-01", "learning_rate": 0.15, "early_stop": 30, "max_bin": 63, "rounds": 3000},
    # batch 2 (section 19): 2021 on at 0.05 ran into its 8,000-round cap in 51 minutes; now 2022 on at 0.08, stopped
    # by early stopping, the cap or a time budget, whichever comes first
    "confirm": {"train_from": "2022-01-01", "learning_rate": 0.08, "early_stop": 100, "max_bin": 255, "rounds": 6000,
                "max_seconds": 1500},
}
BASE_PARAMS = {"num_leaves": 63, "min_child_samples": 300, "feature_fraction": 0.7, "bagging_fraction": 0.8,
               "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 0}
OBJECTIVES = ("softmax", "binary", "pl2", "logbsp")
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
    ``base``, ``base+parity``, ``base+parity+hrb``; each ``-card`` for card-safe."""
    blocks = meta["features"]
    card = name.endswith("-card") or name == "card_safe"
    key = name.removesuffix("-card")
    order = {"base": ["base"], "base+parity": ["base", "parity"], "base+parity+hrb": ["base", "parity", "hrb"],
             "all": list(blocks), "card_safe": list(blocks)}
    if key not in order:
        raise ValueError(f"unknown feature set {name!r}")
    feats = list(dict.fromkeys(f for b in order[key] for f in blocks.get(b, [])))
    if card:
        feats = [f for f in feats if f not in set(meta["card_unsafe"])]
    return feats


def lane(exp: dict, spec: dict) -> str:
    """``slow`` for an experiment off the quick recipe or with the study, ``quick`` otherwise: the CI runs the quick
    lane's table as soon as it is done, without waiting on the slow lane."""
    slow = exp.get("recipe", spec.get("recipe", "quick")) != "quick" or exp.get("study")
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
    return exps


def _starts(race: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(race[1:] != race[:-1]) + 1]


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
    for pre in exp.get("drop", []):
        feats = [f for f in feats if not f.startswith(pre)]
    feats = [f for f in feats if f in df.columns]
    obj = exp.get("objective", "softmax")
    t = df.t.to_numpy().astype("datetime64[ns]")
    lo = np.datetime64(pd.Timestamp(recipe["train_from"]), "ns")
    edge = np.datetime64(pd.Timestamp(test_from), "ns")
    hi = np.datetime64(pd.Timestamp(test_to) + pd.Timedelta(days=1), "ns") if test_to else None
    fit_i = np.flatnonzero((t >= lo) & (t < edge))
    test_i = np.flatnonzero((t >= edge) & ((t < hi) if hi is not None else True))
    if obj == "logbsp":
        fit_i = fit_i[df.bsp.to_numpy(float)[fit_i] > 1]
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
    params = dict(BASE_PARAMS, learning_rate=recipe["learning_rate"], max_bin=recipe["max_bin"])
    params.update(exp.get("params", {}))
    dtr = lgb.Dataset(matrix(tr_i), y[tr_i], feature_name=feats, free_raw_data=True,
                      params={"max_bin": recipe["max_bin"]})
    des = lgb.Dataset(matrix(es_i), y[es_i], reference=dtr, free_raw_data=True)
    es = lgb.early_stopping(recipe["early_stop"], verbose=False)
    stop = [es, time_budget(recipe.get("max_seconds"), es)]
    if obj == "binary":
        m = lgb.train(dict(params, objective="binary"), dtr, recipe["rounds"], valid_sets=[des], callbacks=stop)
    elif obj == "logbsp":
        m = lgb.train(dict(params, objective="regression"), dtr, recipe["rounds"], valid_sets=[des], callbacks=stop)
    else:
        second = (df.position.to_numpy(float) == 2).astype(float) if obj == "pl2" else None
        fobj, _ = race_objective(race[tr_i], won[tr_i], None if second is None else second[tr_i])
        _, feval = race_objective(race[es_i], won[es_i])
        m = lgb.train(dict(params, objective=fobj, metric="None"), dtr, recipe["rounds"], valid_sets=[des],
                      feval=feval, callbacks=stop)
    fit_s = time.monotonic() - t0
    budget = recipe.get("max_seconds")
    stopped = ("cap" if m.current_iteration() >= recipe["rounds"] else
               "time" if budget and fit_s > budget else "early_stop")
    del dtr, des
    gc.collect()
    raw = m.predict(matrix(test_i), num_iteration=m.best_iteration)
    keep = ["race_id", "t", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "field", "won",
            "sp_decimal", "sp_p_norm", "bsp", "bsp_p_norm"]
    pred = df.iloc[test_i][[c for c in keep if c in df.columns]].reset_index(drop=True)
    if obj == "binary":
        pred["p_model"] = race_norm(pd.Series(raw), pred.race_id).to_numpy()
    elif obj == "logbsp":
        pred["pred_bsp"] = np.exp(raw)
        pred["p_model"] = race_norm(1.0 / pred.pred_bsp, pred.race_id).to_numpy()
    else:
        pred["p_model"] = _race_softmax(raw.astype(float), _starts(race[test_i]))
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
    if obj == "logbsp":
        ok = pred.bsp > 1
        e = np.log(pred.pred_bsp[ok]) - np.log(pred.bsp[ok])
        res["log_bsp_rmse"] = round(float(np.sqrt(np.mean(e ** 2))), 4)
    res["seconds"] = round(time.monotonic() - t0)
    return res, pred


# --------------------------------------------------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------------------------------------------------

def _vs(r: dict, base: dict) -> str:
    pr = r.get("paired")
    if pr and r["name"] != base["name"]:
        return f"{pr['diff']} {pr['ci90']}"
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
        out[name] = {"diff": round(float((day.e.sum() - day.b.sum()) / day.n.sum()), 5),
                     "ci90": [round(float(np.percentile(diff, 5)), 5), round(float(np.percentile(diff, 95)), 5)],
                     "runners": int(day.n.sum()), "days": int(len(day))}
    return out


def table(results: list[dict], baseline: str | None = None) -> str:
    if not results:
        return "(no results)"
    base = next((r for r in results if r["name"] == baseline), results[0])
    lines = [f"Baseline: {base['name']}. Test {base['test_from']} to {base['test_to']}. Lower log-loss is better; "
             "the Benter gain is the blend's log-loss less the SP's alone on the second half of the test "
             "(negative: the model adds to the SP).", "",
             "| Experiment | Features | Objective | Rounds | Log-loss | vs baseline (90%, paired, by day) | "
             "On BSP races: model / BSP | Benter gain | Model weight | Seconds |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        b = r.get("benter_vs_bsp") or {}
        ll2 = b.get("logloss_second_half") or {}
        gain = round(ll2["blend"] - ll2["market"], 5) if ll2 else None
        ob = r.get("on_bsp_races") or {}
        lines.append(f"| {r['name']} | {r['features']} | {r['spec'].get('objective', 'softmax')} | {r['rounds']} | "
                     f"{r['logloss']} | {_vs(r, base)} | {ob.get('model')} / {ob.get('bsp')} | "
                     f"{gain} | {(b.get('weights') or {}).get('model')} | {r['seconds']} |")
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
        for r in rs:
            r["paired"] = paired.get(r["name"])
        md = table(rs, spec.get("baseline"))
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
        _log(f"experiment {e['name']}")
        res, pred = run_experiment(df, meta, e, recipe, spec.get("test_from", TEST_FROM), spec.get("test_to"))
        res["cache"] = {k: meta[k] for k in ("hash", "last_day", "rows")}
        if e.get("study"):
            res["study"] = run_study(store, pred, e["name"], out)
        (out / f"result_{e['name']}.json").write_text(json.dumps(res, indent=1, default=str))
        # each runner's log-loss contribution, for the paired comparison against the baseline in --collect
        pred[["race_id", "race_date", "dog_id", "won", "p_model"]].to_parquet(out / f"pred_{e['name']}.parquet", index=False)
        results.append(res)
        _log(json.dumps({k: res[k] for k in ("name", "logloss", "rounds", "seconds")}))
    print(table(results, spec.get("baseline")))
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
