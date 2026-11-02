"""The card-safe full greyhound model, scored out of sample on the trade-out study's days (the owner, 2 Nov 2026:
"score the card-safe full model and rerun the study").

The trade-out study's first reading (reports/greyhound_model.md section 13) priced each dog with the forward test's
model of 5 Oct: the base 135 features, none of the horse model's families (pace and race shape, the normalised finish
position, the form windows, the market's view; ``greyhound/parity.py``) nor the feature library (Elo, the Kalman
rating, form lines, first-bend crossings; ``greyhound/lib.py``). The best research model (section 11: base, parity,
HorseRaceBase and the library, 342 features, fitted on the race's conditional logit) reads the dog's weight on the
day, which the morning card does not know. This is that model without those features (``greyhound.live.card_safe``):

1. Every GBGB run from 2018 to ``--to`` (``greyhound.track.runs_through``: the year tables and the raw days since),
   the metrics with the parity, HRB and library blocks and the Betfair SP history (the price-file tables).
2. One walk-forward fold: fitted on every modelling race from 2019 to the day before ``--from`` (6 Oct 2026, the first
   recorded greyhound books), early-stopped on its last tenth, the race's softmax objective; every race from ``--from``
   on predicted by it, so every price is out of sample.
3. Each day's predictions joined to the recorded Betfair greyhound markets (``greyhound.track.join_markets``) and
   written a day a file (the forward test's ``pred`` layout) to ``--pred-dir``, for
   ``research.greyhound.tradeout_study --pred-dir``.
4. **The log-BSP model** (the owner, 2 Nov 2026: "add the log-BSP target like the horse model"): the horse model's
   recipe (``train_bfsp.py``: LightGBM regression, L2, on log BFSP) on the same card-safe features and rows, the
   target the Betfair SP from the price files (``greyhound.parity.attach_bsp``), fitted on every priced run before
   ``--from`` and early-stopped on its last tenth (up to ``BSP_ROUNDS``, 20,000: the first run hit its 3,000 cap still
   improving). Its predicted SP is our price the way the horses' closing model reads
   one: the chance 1 / predicted SP, normalised within the race. Written to ``--pred-dir``/logbsp beside the win
   model's ``--pred-dir``/full.
5. The head to head on the same runners: log-loss against the result of the 5 Oct base model, the full win model, the
   log-BSP model and the Betfair SP (normalised within the race), on the races all price whole; and the log-BSP
   model's error against the SP itself (RMSE of log SP, and the no-change error of the race's mean log SP).

Read-only on the store: everything goes to local files (the job's artifact).

    python -m research.greyhound.score_full --from 2026-10-06 --to 2026-11-01 --pred-dir out/pred_full \\
        --out out/greyhound_full_score.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound import tradeout as to

log = logging.getLogger(__name__)

VARIANT = "all+lib-card:softmax"
BLOCKS = ("parity", "hrb", "lib")
FIT_FROM = "2019-01-01"
PRICE_YEARS = "2018-2026"


def _log(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


def build(store, last: date):
    """The metrics on every GBGB run to ``last``, slimmed to the card-safe features."""
    from greyhound.live import card_safe
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices, slim
    from greyhound.track import runs_through
    t0 = time.monotonic()
    runs = runs_through(store, last)
    _log(f"runs: {len(runs):,} to {runs.race_date.max()} ({time.monotonic() - t0:.0f} s)")
    eng = GreyhoundMetricsEngine(blocks=BLOCKS)
    df = eng.calculate_all(runs, prices=load_prices(store, PRICE_YEARS))
    del runs
    gc.collect()
    feats = card_safe(eng.features)
    _log(f"features: {len(eng.features)} built, {len(feats)} card-safe; {eng.timings} ({time.monotonic() - t0:.0f} s)")
    return slim(df, eng.features + ["bsp"]), feats, eng


BSP_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_child_samples=300,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1)


#: the log-BSP fit's cap: the first run stopped at 3,000 rounds still improving (report section 14), so the owner asked
#: for more rounds (2 Nov 2026); the learning rate and everything else unchanged, early stopping after BSP_PATIENCE
BSP_ROUNDS = 20000
BSP_PATIENCE = 200


def fit_log_bsp(df: pd.DataFrame, features: list[str], first: str, date_from: str = FIT_FROM,
                rounds: int = BSP_ROUNDS, patience: int = BSP_PATIENCE) -> tuple[pd.DataFrame, dict]:
    """LightGBM on log BSP (the horse model's target), fitted on every priced modelling run in [date_from, first),
    early-stopped on its last tenth; every modelling run from ``first`` priced: ``pred_bsp`` and ``p_model`` (1 /
    pred_bsp normalised within the race)."""
    import lightgbm as lgb
    from greyhound.model import race_norm
    mask = ~df.is_trial.to_numpy(bool) & (df.field.to_numpy() >= 4) & (df.t >= date_from).to_numpy()
    if "is_card" in df:
        mask &= ~df.is_card.to_numpy(bool)
    rows = np.flatnonzero(mask)
    t = df.t.to_numpy()[rows].astype("datetime64[ns]")
    bsp = df.bsp.to_numpy(float)[rows]
    edge = np.datetime64(pd.Timestamp(first), "ns")
    fit_i = np.flatnonzero((t < edge) & (bsp > 1))
    test_i = np.flatnonzero(t >= edge)
    X = np.empty((len(rows), len(features)), dtype=np.float32)
    for j, f in enumerate(features):
        X[:, j] = df[f].to_numpy(np.float32, na_value=np.nan)[rows]
    y = np.log(np.clip(bsp, 1.01, 1000.0))
    cut = np.quantile(t[fit_i].astype("int64"), 0.9).astype("datetime64[ns]")
    tr_i, es_i = fit_i[t[fit_i] < cut], fit_i[t[fit_i] >= cut]
    _log(f"log-BSP: fit {len(tr_i):,} + {len(es_i):,} priced rows, test {len(test_i):,}, {len(features)} features")
    dtr = lgb.Dataset(X[tr_i], y[tr_i], feature_name=list(features), free_raw_data=True)
    des = lgb.Dataset(X[es_i], y[es_i], reference=dtr, free_raw_data=True)
    curve: dict = {}
    m = lgb.train(BSP_PARAMS, dtr, rounds, valid_sets=[des], valid_names=["es"],
                  callbacks=[lgb.early_stopping(patience, verbose=False), lgb.record_evaluation(curve)])
    l2 = curve.get("es", {}).get("l2", [])
    keep = ["race_id", "t", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "won", "sp_decimal",
            "field"]
    out = df.iloc[rows[test_i]][[c for c in keep if c in df.columns]].reset_index(drop=True)
    out["pred_bsp"] = np.exp(m.predict(X[test_i], num_iteration=m.best_iteration))
    out["p_model"] = race_norm(1.0 / out.pred_bsp, out.race_id)
    es_err = m.predict(X[es_i], num_iteration=m.best_iteration) - y[es_i]
    imp = pd.Series(m.feature_importance("gain"), index=features)
    info = {"rounds": m.best_iteration, "cap": rounds, "hit_cap": bool(m.best_iteration >= rounds),
            "learning_rate": BSP_PARAMS["learning_rate"], "patience": patience, "fit_rows": int(len(fit_i)),
            # the early-stopping slice's RMSE of log BSP every 1,000 rounds: where the fit stopped gaining
            "es_rmse_by_round": {i + 1: round(float(np.sqrt(v)), 5) for i, v in enumerate(l2)
                                 if (i + 1) % 1000 == 0 or i + 1 == len(l2)},
            "early_stop_rmse_log_bsp": round(float(np.sqrt(np.mean(es_err ** 2))), 4),
            "top_features": {k: round(float(v), 4) for k, v in (imp / imp.sum()).sort_values(ascending=False)
                             .head(30).items()}}
    del X
    return out, info


def score_days(preds: dict, first: date, last: date, store, pred_dir: Path) -> dict:
    """Each model's day predictions joined to the day's Betfair markets and written (``pred_dir``/<model>/<day>); the
    head to head on the same runners against the 5 Oct model and the SP."""
    from betfair_recorder import _read_day_csv
    from greyhound.model import logloss
    from greyhound.track import join_markets
    keep = ["race_id", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "field", "won",
            "sp_decimal", "p_model", "pred_bsp", "market_id", "selection_id"]
    names = list(preds)
    rows, days, d = [], [], first
    while d <= last:
        ds = f"{d:%Y-%m-%d}"
        markets = _read_day_csv(d, "markets_greyhound.csv")
        books = _read_day_csv(d, "books_greyhound.csv")
        row = {"day": ds}
        joined = {}
        for name, pred in preds.items():
            p = pred[pred.race_date.astype(str) == ds]
            row[f"races_{name}"] = int(p.race_id.nunique())
            if not len(p) or markets is None:
                continue
            j = join_markets(p, markets)
            j = j[[c for c in keep if c in j.columns]]
            (pred_dir / name).mkdir(parents=True, exist_ok=True)
            j.to_parquet(pred_dir / name / f"{ds}.parquet", index=False)
            row[f"races_on_betfair_{name}"] = int(j.race_id.nunique())
            joined[name] = j.assign(market_id=j.market_id.astype(str),
                                    selection_id=pd.to_numeric(j.selection_id, errors="coerce"))
        days.append(row)
        base = store.get_parquet(f"{to.TRACK_PRED}/{ds}.parquet")
        if len(joined) == len(names) and base is not None and len(base) and books is not None:
            m = base.assign(market_id=base.market_id.astype(str),
                            selection_id=pd.to_numeric(base.selection_id, errors="coerce"))[
                ["market_id", "selection_id", "won", "field", "p_model"]].rename(columns={"p_model": "p_base"})
            for name, j in joined.items():
                cols = ["market_id", "selection_id", "p_model"] + (["pred_bsp"] if "pred_bsp" in j else [])
                m = m.merge(j[cols].rename(columns={"p_model": f"p_{name}"}), on=["market_id", "selection_id"],
                            how="inner")
            m = m.merge(to.closing_prices(books, markets, d), on=["market_id", "selection_id"], how="inner")
            whole = m.groupby("market_id").selection_id.transform("size") == m.field
            rows.append(m[whole & (m.bsp > 1)].assign(day=ds))
        d += timedelta(days=1)
    out = {"days": days}
    if rows:
        h = pd.concat(rows, ignore_index=True).reset_index(drop=True)
        q = (1.0 / h.bsp).groupby(h.market_id).transform(lambda s: s / s.sum())
        cols = {"base_5oct": "p_base", **{n: f"p_{n}" for n in names}}
        hh = {"runners": int(len(h)), "races": int(h.market_id.nunique()), "days": int(h.day.nunique()),
              "logloss": {**{n: round(logloss(h[c], h.won), 5) for n, c in cols.items()},
                          "bsp": round(logloss(q, h.won), 5)},
              "by_day": {k: {**{n: round(logloss(g[c], g.won), 5) for n, c in cols.items()},
                             "bsp": round(logloss(q[g.index], g.won), 5), "races": int(g.market_id.nunique())}
                         for k, g in h.groupby("day")}}
        if "pred_bsp" in h:
            y = np.log(h.bsp.to_numpy(float))
            e = np.log(h.pred_bsp.to_numpy(float)) - y
            nochange = y - pd.Series(y).groupby(h.market_id).transform("mean").to_numpy()
            hh["log_bsp_model_vs_sp"] = {"rmse_log": round(float(np.sqrt(np.mean(e ** 2))), 4),
                                         "mean_error_log": round(float(np.mean(e)), 4),
                                         "rmse_log_race_mean_only": round(float(np.sqrt(np.mean(nochange ** 2))), 4),
                                         "median_ape%": round(100 * float(np.median(np.abs(np.exp(e) - 1))), 2)}
        out["head_to_head"] = hh
    return out


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from greyhound.model import score, walk_forward
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--from", dest="first", default=str(to.FIRST_BOOK_DAY), help="the first day predicted")
    ap.add_argument("--to", dest="last", default=None, help="the last day (default yesterday, UK)")
    ap.add_argument("--pred-dir", default="out/pred_full")
    ap.add_argument("--out", default="out/greyhound_full_score.json")
    ap.add_argument("--bsp-rounds", type=int, default=BSP_ROUNDS, help="the log-BSP fit's round cap")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    store = Store(root=a.store)
    first = date.fromisoformat(a.first)
    last = date.fromisoformat(a.last) if a.last else uk_today() - timedelta(days=1)
    df, feats, eng = build(store, last)
    _, _, objective = VARIANT.partition(":")
    pred = walk_forward(df, feats, [f"{first:%Y-%m-%d}"], FIT_FROM, objective=objective)
    importance = pred.attrs.pop("importance")
    pred = pred[pred.race_date.astype(str) <= f"{last:%Y-%m-%d}"]
    _log(f"predicted {len(pred):,} runners in {pred.race_id.nunique():,} races, {pred.race_date.min()} to "
         f"{pred.race_date.max()} ({pred.rounds.iloc[0] if len(pred) else 0} rounds)")
    gc.collect()
    try:                                       # the win model's scores stand even if this fit fails (memory, time)
        bsp_pred, bsp_info = fit_log_bsp(df, feats, f"{first:%Y-%m-%d}", rounds=a.bsp_rounds)
        bsp_pred = bsp_pred[bsp_pred.race_date.astype(str) <= f"{last:%Y-%m-%d}"]
    except Exception as exc:
        _log(f"log-BSP model failed: {exc!r}")
        bsp_pred, bsp_info = None, {"error": repr(exc)[:500]}
    res = {"variant": VARIANT, "blocks": list(BLOCKS), "features": len(feats), "fit_from": FIT_FROM,
           "fit_before": f"{first}", "predicted": [f"{first}", f"{last}"], "timings": eng.timings,
           "score_vs_sp": score(pred), "top_features": {k: round(float(v), 4) for k, v in importance.head(30).items()},
           "log_bsp_model": bsp_info}
    del df
    gc.collect()
    preds = {"full": pred} if bsp_pred is None else {"full": pred, "logbsp": bsp_pred}
    res.update(score_days(preds, first, last, store, Path(a.pred_dir)))
    res["seconds"] = round(time.monotonic() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "days"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
