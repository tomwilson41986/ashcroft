"""The Betfair-native closing-price model tuned inside the walk-forward: trees, leaves, ridge penalty, the blend.

racing² football research, 24 Nov 2026 (the owner: "Tune the model inside the walk-forward: the number of trees, the
ridge penalty, and a fitted way to combine the two models in place of a plain average"). Registered as
``football-bf-tune-1124`` in reports/research_ledger.jsonl before its run.

The model (``football_bf_close_model``) fits a ridge (penalty ``RIDGE_ALPHA``) and a LightGBM (300 trees, learning
rate 0.03, 15 leaves) a role a month and averages them, settings never tuned. Here every test month is forecast twice
on the same rows and features:

- ``default``: the model as it is;
- ``tuned``: settings chosen on the training months alone. The last training month is held back as an inner
  validation month; on the earlier months a LightGBM is fitted for each of 15, 31 and 63 leaves with up to 2,000
  trees, stopping when the inner month's error has not improved for 100 trees, and a ridge for each penalty in
  ``ALPHAS``. The leaves and trees with the least inner error are kept, and the ridge penalty likewise; the blend
  weight w (forecast = w x LightGBM + (1 - w) x ridge, w on a grid of 0 to 1 in tenths) is the one with the least
  inner error. Both are then refitted on all the training months with those settings and the month forecast.

Nothing about the test month enters a choice. The change in skill against "no move", paired by match day (1,000
resamples), overall and by checkpoint; the settings chosen each month are reported.

    python -m research.open_close.football_bf_tune --out-json out/bf_tune.json [--root <warehouse folder>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.open_close import football_bf_close_model as bf

log = logging.getLogger(__name__)

TAG = "football-bf-tune-1124"
LEAVES = (15, 31, 63)
ALPHAS = (1.0, 10.0, 50.0, 200.0, 1000.0, 5000.0)
WEIGHTS = tuple(np.round(np.arange(0, 1.01, 0.1), 1))
MAX_TREES, PATIENCE = 2000, 100


def ridge_with(alpha: float, tr, te, cols, y) -> np.ndarray:
    old = bf.RIDGE_ALPHA
    bf.RIDGE_ALPHA = alpha
    try:
        return bf.ridge_fit_predict(tr, te, cols, y)
    finally:
        bf.RIDGE_ALPHA = old


def gbm_with(leaves: int, trees: int, tr, cols, y, valid=None):
    import lightgbm as lgb
    g = lgb.LGBMRegressor(n_estimators=trees, learning_rate=0.03, num_leaves=leaves, min_child_samples=200,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0,
                          n_jobs=bf.THREADS)
    if valid is None:
        g.fit(tr[cols], y)
    else:
        va, yv = valid
        g.fit(tr[cols], y, eval_set=[(va[cols], yv)], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return g


def choose(tr: pd.DataFrame, cols: list[str]) -> dict:
    """Settings from the training rows alone: the last month held back as the inner validation month."""
    last = tr["month"].max()
    inner, va = tr[tr["month"] < last], tr[tr["month"] == last]
    y, yv = inner["y_logit"].to_numpy(float), va["y_logit"].to_numpy(float)
    best_g = None
    for lv in LEAVES:
        g = gbm_with(lv, MAX_TREES, inner, cols, y, valid=(va, yv))
        it = int(g.best_iteration_ or MAX_TREES)
        p = g.predict(va[cols], num_iteration=it)
        err = float(np.mean((yv - p) ** 2))
        if best_g is None or err < best_g[0]:
            best_g = (err, lv, it, p)
    best_r = None
    for a in ALPHAS:
        p = ridge_with(a, inner, va, cols, y)
        err = float(np.mean((yv - p) ** 2))
        if best_r is None or err < best_r[0]:
            best_r = (err, a, p)
    w = min(WEIGHTS, key=lambda w: float(np.mean((yv - (w * best_g[3] + (1 - w) * best_r[2])) ** 2)))
    return {"leaves": best_g[1], "trees": max(best_g[2], 50), "alpha": best_r[1], "w_gbm": float(w)}


def walk_forward_tuned(d: pd.DataFrame, cols: list[str], min_train_months: int = 4):
    """As ``bf.walk_forward`` (same months, same training rows) with settings chosen inside each fold."""
    pred = np.full(len(d), np.nan)
    chosen = []
    months = sorted(d["month"].unique())
    for m in months[min_train_months:]:
        start = pd.Timestamp(m + "-01", tz="UTC")
        for role in bf.ROLES:
            r = d["role"] == role
            tr = d[r & d["usable"] & (d["kickoff"] < start - pd.Timedelta(days=1))]
            te_idx = np.flatnonzero((r & (d["month"] == m)).to_numpy())
            if len(tr) < 2000 or not len(te_idx) or tr["month"].nunique() < 2:
                continue
            s = choose(tr, cols)
            te = d.iloc[te_idx]
            y = tr["y_logit"].to_numpy(float)
            g = gbm_with(s["leaves"], s["trees"], tr, cols, y)
            pred[te_idx] = s["w_gbm"] * g.predict(te[cols]) + (1 - s["w_gbm"]) * ridge_with(s["alpha"], tr, te, cols,
                                                                                            y)
            chosen.append({"month": m, "role": role, **s})
            log.info("%s %s %s", m, role, s)
    return pred, chosen


def run(store) -> dict:
    d, fam = bf.load(store)
    if d.empty:
        return {"tag": TAG, "status": "no data found: gold/move is not built"}
    d = d[d["cp_fresh"]].reset_index(drop=True)
    cols = [c for v in fam.values() for c in v]
    res = {"tag": TAG, "rows": int(len(d)), "features": len(cols),
           "note": "development evidence: last traded prices; nothing is traded"}
    if d["month"].nunique() < 5:
        res["status"] = "no data found: fewer than five months"
        return res
    p0, _ = bf.walk_forward(d, cols, min_train_months=4)
    p1, chosen = walk_forward_tuned(d, cols)
    ok = np.isfinite(p0) & np.isfinite(p1) & d["schedule_fixed"].to_numpy() & d["usable"].to_numpy()
    sc = d[ok]
    y, day = sc["y_logit"].to_numpy(float), sc["day"].to_numpy()
    a, b = p0[ok], p1[ok]
    res.update({"scored_rows": int(len(sc)), "test_months": sorted(sc["month"].unique()),
                "default": bf.skill_ci(day, y, a), "tuned": {**bf.skill_ci(day, y, b, base=a), **bf.describe(y, b)},
                "by_horizon": {}, "chosen": chosen})
    for cp in bf.HORIZONS:
        m = (sc["cp_min"] == cp).to_numpy()
        if m.sum() >= 500:
            res["by_horizon"][str(cp)] = bf.skill_ci(day[m], y[m], b[m], base=a[m])
    ch = pd.DataFrame(chosen)
    if len(ch):
        res["chosen_summary"] = {"leaves": ch["leaves"].value_counts().to_dict(),
                                 "trees_median": float(ch["trees"].median()),
                                 "alpha": ch["alpha"].value_counts().to_dict(),
                                 "w_gbm_mean": round(float(ch["w_gbm"].mean()), 3)}
    res["keeps_its_place"] = bool(res["tuned"]["vs_base_lo95"] > 0)
    every = d[np.isfinite(p0) & np.isfinite(p1) & d["schedule_fixed"].to_numpy()]
    res["trades_at_last_traded_prices_0.05"] = {
        "default": bf.trades(every.assign(pred=p0[every.index]), 0.05),
        "tuned": bf.trades(every.assign(pred=p1[every.index]), 0.05)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a local warehouse folder in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    res = run(Store(root=a.root) if a.root else Store())
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k in ("tag", "status", "default", "tuned", "chosen_summary",
                                                             "keeps_its_place")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
