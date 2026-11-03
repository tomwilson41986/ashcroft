"""The closing price: the most accurate forecast of each dog's Betfair SP from the book before the off (the owner,
3 Nov 2026: "we need to focus on improving our BFSP model. The final price / log of final price is key. Once we can
get the most accurate BFSP model then we can trade around it").

Target: log(BSP). At each entry mark (first traded, 5, 3 and 1 minute out), every runner of every race with a whole
book and an SP, models from the simplest up, each fitted only on the days before the day it forecasts:

- ``mid``: log of the market's price now (the geometric mid where the spread is tight, else the best back).
- ``mid_book``: the market's chances normalised in the race and scaled to the SP book's usual total (the overround
  the BSP carries), so the level is right as well as the order.
- ``cl``: the conditional logit on the book and our price (``research.greyhound.tradeout_study`` ``micro``), the
  forecast the live trader uses today.
- ``gbm``: gradient-boosted trees on the gap between log(BSP) and log(mid), from the book's depth and weight of money
  over three levels, the micro-price, the spread, the move into the mark, the last trade, the money matched, the
  minutes to the off, the trap, the field and our pre-race price; then rescaled in the race to the SP book.

Scores, by mark, over every scored day: RMSE and MAE of log(BSP), the mean error (bias), the race-relative RMSE (log
BSP less its race mean: the order and spacing within the race), and the cross-entropy of the SP book's shares; each
against ``mid`` with a 90% interval from a bootstrap over days.

    python -m research.greyhound.bfsp_close --source retimed --out out/greyhound_bfsp_close
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound import tradeout as to
from research.greyhound import limit_study as ls, tradeout_study as ts

log = logging.getLogger(__name__)

MARKS = ("first_traded", "off_5", "off_3", "off_1")
MODELS = ("mid", "mid_book", "cl", "gbm", "gbm_deep", "gbm_pooled", "blend")
GBM_DEEP = dict(num_leaves=31, min_data_in_leaf=30, learning_rate=0.03)
GBM_DEEP_ROUNDS = 900
GBM_PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
              "seed": 0, "num_threads": 2}
GBM_ROUNDS = 400


def _log(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


def book_features(d: pd.DataFrame) -> pd.DataFrame:
    """The book's own measures beside greyhound.tradeout.add_features': weight of money over one and three levels,
    the micro-price against the mid, the depth, the spread in ticks, the last trade and the move, all logged."""
    x = d.copy()
    num = lambda c: pd.to_numeric(x.get(c), errors="coerce") if c in x else pd.Series(np.nan, index=x.index)
    b1, l1 = num("back1"), num("lay1").where(num("lay1") > 1)
    bs = [num(f"back{i}_size").fillna(0) for i in (1, 2, 3)]
    ls_ = [num(f"lay{i}_size").fillna(0) for i in (1, 2, 3)]
    x["wom1"] = bs[0] / (bs[0] + ls_[0]).replace(0, np.nan)
    x["wom3"] = sum(bs) / (sum(bs) + sum(ls_)).replace(0, np.nan)
    micro = (b1 * ls_[0] + l1 * bs[0]) / (bs[0] + ls_[0]).replace(0, np.nan)
    x["log_micro_vs_mid"] = np.log(micro / x["mid"])
    x["log_depth_back"] = np.log1p(sum(bs))
    x["log_depth_lay"] = np.log1p(sum(ls_))
    x["log_mid"] = np.log(x["mid"])
    x["log_last_vs_mid"] = np.log(num("last_traded").where(num("last_traded") > 1) / x["mid"])
    x["log_model_price"] = -np.log(x["p_model"].clip(lower=1e-6)) if "p_model" in x else np.nan
    x["log_model_vs_mid"] = x["log_model_price"] - x["log_mid"]
    return x


GBM_FEATURES = ["log_mid", "lq_market", "lq_model", "log_model_vs_mid", "last_vs_mid", "log_last_vs_mid", "move",
                "imbalance", "wom1", "wom3", "log_micro_vs_mid", "log_spread", "log_depth_back", "log_depth_lay",
                "log_vol", "log_depth", "overround", "rank", "minutes_to_off", "trap", "field"]


def _race(d: pd.DataFrame) -> np.ndarray:
    return to._codes(d)


def rescale(log_price: np.ndarray, d: pd.DataFrame, book: float) -> np.ndarray:
    """Prices whose chances are normalised in the race and scaled to the SP book's total ``book``."""
    g = _race(d)
    p = np.exp(-np.asarray(log_price, float))
    s = np.bincount(g, weights=p)[g]
    return -np.log(book * p / s)


def sp_book(d: pd.DataFrame) -> float:
    """The SP book's usual total (sum of 1/BSP a race), the median over the races given."""
    g = _race(d)
    return float(np.median(np.bincount(g, weights=1.0 / d.bsp.to_numpy(float))))


def forecast_day(train: pd.DataFrame, test: pd.DataFrame, pooled: pd.DataFrame | None = None,
                 importance: dict | None = None) -> pd.DataFrame:
    """Every model's log(BSP) forecast for ``test`` from ``train`` (days before it only). ``pooled``: the rows of
    every mark of those days (one model for all marks, minutes to the off among its inputs); ``gbm_deep``: larger
    trees, more rounds; ``blend``: the mean of ``cl`` and ``gbm``. ``importance`` collects the trees' gain."""
    import lightgbm as lgb
    out = test[["race_id", "race_date", "market_id", "selection_id", "mark", "bsp", "won"]].copy()
    out["y"] = np.log(test.bsp.to_numpy(float))
    book = sp_book(train)
    out["mid"] = test["log_mid"].to_numpy(float)
    out["mid_book"] = rescale(out["mid"].to_numpy(), test, book)
    f = ts.fit_cl(train, ts.CL_SETS["micro"])
    q, b = ts.predict(test, f)
    out["cl"] = -np.log(b * np.asarray(q, float))
    feats = [c for c in GBM_FEATURES if c in train.columns]
    resid = np.log(train.bsp.to_numpy(float)) - train["log_mid"].to_numpy(float)
    m = lgb.train(GBM_PARAMS, lgb.Dataset(train[feats].astype(float), resid), GBM_ROUNDS)
    raw = test["log_mid"].to_numpy(float) + m.predict(test[feats].astype(float))
    out["gbm"] = rescale(raw, test, book)
    if importance is not None:
        for k, v in zip(feats, m.feature_importance("gain")):
            importance[k] = importance.get(k, 0.0) + float(v)
    deep = lgb.train(dict(GBM_PARAMS, **GBM_DEEP), lgb.Dataset(train[feats].astype(float), resid), GBM_DEEP_ROUNDS)
    out["gbm_deep"] = rescale(test["log_mid"].to_numpy(float) + deep.predict(test[feats].astype(float)), test, book)
    if pooled is not None and len(pooled):
        pr = np.log(pooled.bsp.to_numpy(float)) - pooled["log_mid"].to_numpy(float)
        pm = lgb.train(GBM_PARAMS, lgb.Dataset(pooled[feats].astype(float), pr), GBM_ROUNDS)
        out["gbm_pooled"] = rescale(test["log_mid"].to_numpy(float) + pm.predict(test[feats].astype(float)), test, book)
    else:
        out["gbm_pooled"] = out["gbm"]
    out["blend"] = 0.5 * (out["cl"] + out["gbm"])
    return out


def scores(f: pd.DataFrame, days: list[str], n_boot: int = 1000) -> list[dict]:
    """By mark and model: RMSE, MAE, bias, race-relative RMSE, the SP book's cross-entropy a race; and each model's
    RMSE against ``mid``'s with a 90% interval from a bootstrap over days (negative: more accurate)."""
    rng = np.random.default_rng(0)
    rows = []
    for mark, g in f.groupby("mark"):
        g = g.reset_index(drop=True)
        codes = pd.factorize(g.race_id.astype(str))[0]
        y = g.y.to_numpy(float)
        yr = y - pd.Series(y).groupby(codes).transform("mean").to_numpy()
        q = 1.0 / g.bsp.to_numpy(float)
        q = q / np.bincount(codes, weights=q)[codes]
        day = g.race_date.astype(str).to_numpy()
        ud = np.array(sorted(set(day)))
        se_mid = (g["mid"].to_numpy(float) - y) ** 2
        for model in MODELS:
            p = g[model].to_numpy(float)
            e = p - y
            pr = p - pd.Series(p).groupby(codes).transform("mean").to_numpy()
            ph = np.exp(-p)
            ph = ph / np.bincount(codes, weights=ph)[codes]
            xent = -np.bincount(codes, weights=q * np.log(np.clip(ph, 1e-12, 1)))
            diff = pd.DataFrame({"d": day, "a": e ** 2, "b": se_mid}).groupby("d")[["a", "b"]].agg(["sum", "count"])
            sa, sb, n = diff[("a", "sum")].reindex(ud).fillna(0).to_numpy(), diff[("b", "sum")].reindex(ud).fillna(
                0).to_numpy(), diff[("a", "count")].reindex(ud).fillna(0).to_numpy()
            boots = []
            for _ in range(n_boot):
                i = rng.integers(0, len(ud), len(ud))
                nn = n[i].sum()
                if nn:
                    boots.append(np.sqrt(sa[i].sum() / nn) - np.sqrt(sb[i].sum() / nn))
            rows.append({"mark": mark, "model": model, "runners": int(len(g)), "races": int(codes.max() + 1),
                         "days": int(len(ud)), "rmse": round(float(np.sqrt(np.mean(e ** 2))), 4),
                         "mae": round(float(np.mean(np.abs(e))), 4), "bias": round(float(np.mean(e)), 4),
                         "race_rel_rmse": round(float(np.sqrt(np.mean((pr - yr) ** 2))), 4),
                         "xent_a_race": round(float(np.mean(xent)), 4),
                         "rmse_vs_mid": round(float(np.sqrt(np.mean(e ** 2)) - np.sqrt(np.mean(se_mid))), 4),
                         "rmse_vs_mid_90%": [round(float(np.quantile(boots, 0.05)), 4),
                                             round(float(np.quantile(boots, 0.95)), 4)] if boots else None})
    return rows


def report_md(res: dict) -> str:
    L = [f"# The closing price: forecasting log(BSP) from the book ({res['first_day']} to {res['last_day']})", "",
         f"Books: {res['source']}. Panel: {res.get('rows')} runner-marks; scored days {res.get('scored_days')} "
         f"(walk-forward: each day forecast from the days before it only). Lower is better throughout; rmse_vs_mid "
         "is each model's RMSE less the market mid's, with a 90% interval from a bootstrap over days.", "",
         "| Mark | Model | Runners | Races | RMSE log(BSP) | MAE | Bias | Race-relative RMSE | SP-book xent a race "
         "| RMSE vs mid | 90% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {m: i for i, m in enumerate(MARKS)}
    for r in sorted(res.get("scores", []), key=lambda r: (order.get(r["mark"], 9), MODELS.index(r["model"]))):
        L.append(f"| {r['mark']} | {r['model']} | {r['runners']} | {r['races']} | {r['rmse']} | {r['mae']} | "
                 f"{r['bias']} | {r['race_rel_rmse']} | {r['xent_a_race']} | {r['rmse_vs_mid']} | "
                 f"{r['rmse_vs_mid_90%']} |")
    if res.get("importance"):
        L += ["", "What the boosted model (gbm) leans on, its gain summed over every fit:", "",
              "| Feature | Share of gain |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in res["importance"].items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--from", dest="first", default=str(to.FIRST_BOOK_DAY))
    ap.add_argument("--to", dest="last", default=None, help="default: yesterday (UK)")
    ap.add_argument("--source", choices=["recorder", "retimed", "live"], default="retimed")
    ap.add_argument("--min-train-days", type=int, default=ls.MIN_TRAIN_DAYS)
    ap.add_argument("--out", default="out/greyhound_bfsp_close")
    a = ap.parse_args(argv)
    t0 = time.time()
    store = Store(root=a.store)
    last = date.fromisoformat(a.last) if a.last else uk_today() - timedelta(days=1)
    panel = ls.load(store, date.fromisoformat(a.first), last, source=a.source)
    res = {"first_day": a.first, "last_day": f"{last}", "source": a.source, "rows": int(len(panel))}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(panel):
        panel = book_features(panel[panel["mark"].isin(MARKS) & (panel.bsp > 1) & (panel["mid"] > 1)])
        days = sorted(panel.race_date.astype(str).unique())
        scored = days[a.min_train_days:]
        res["scored_days"] = f"{scored[0]} to {scored[-1]} ({len(scored)})" if scored else None
        fc, imp = [], {}
        for day in scored:
            pooled = panel[panel.race_date.astype(str) < day]
            for mark, g in panel.groupby("mark"):
                tr, te = g[g.race_date.astype(str) < day], g[g.race_date.astype(str) == day]
                if len(tr) and len(te):
                    fc.append(forecast_day(tr, te, pooled=pooled, importance=imp))
            _log(f"{day}: forecast")
        f = pd.concat(fc, ignore_index=True) if fc else pd.DataFrame()
        res["scores"] = scores(f, scored) if len(f) else []
        tot = sum(imp.values()) or 1.0
        res["importance"] = {k: round(v / tot, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    res["seconds"] = round(time.time() - t0)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    md = report_md(res)
    out.with_suffix(".md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
