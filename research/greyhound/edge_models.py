"""Models of the edge at the price on offer (the owner, 18 Nov 2026: "Try any model that could help"; the brief's
Phase 3, greyhounds; ledger greyhound-edge-models-1118, registered before the run).

The take and close's refit (greyhound-take-close-refit-1118) showed the failure: a forecast of the SP book's shares
plus a fixed bar (8%) picks the runners where the forecast is most wrong; on them the SP came in 16-19% over the
forecast. These candidates model the trade itself - how far the SP will come in under the back price on offer now -
and calibrate that prediction out of fold, so the bar is set on an honest expected edge:

- ``cl_cal``: the refit's conditional logit (live-key rows) with its predicted edge log(back1 / forecast SP)
  calibrated by isotonic regression on out-of-fold predictions within the training days (two halves by day);
- ``gbm_live``: LightGBM (``tradeout_study.GBM_PARAMS``, fixed, not tuned here) predicting log(back1 / BSP) directly,
  fitted on the live-key race-traded rows of the training days;
- ``gbm_both``: the same, fitted on the live-key rows and the recorder's re-timed first-traded rows together (a
  source flag among the features), for more days of training.

Each is walk-forward by day on the live-key race-traded rows (the book the fixed live trader decides on), the rule
fixed before the run: take the best three backs while the predicted edge at the price is at least ``BAR`` (3%), up to
GBP100, closed by laying the winnings at the SP, 2% on each market's positive net (greyhound/take_close.py plan).
Also reported, judged by nothing: Benter's test at the decision time - a conditional logit on the result with the
market's price at the decision book and our price, against the market's price alone - so the money question (does
our price add to the price on offer) is answered beside the trades.

    python -m research.greyhound.edge_models --out out/study
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-edge-models-1118"
BAR = 0.03
N_TRIALS = 243                  # the trade-out study's 240 rules, the refit and these three arms
ARMS = ("cl_cal", "gbm_live", "gbm_both")
EXTRA = ["log_back1", "src_live"]


def _features(d: pd.DataFrame) -> pd.DataFrame:
    from research.greyhound.tradeout_study import GBM_FEATURES
    x = d[GBM_FEATURES].astype(float).copy()
    x["log_back1"] = np.log(d.back1.astype(float))
    x["src_live"] = (d["mark"].astype(str) == "race_traded").astype(float)
    return x


def _target(d: pd.DataFrame) -> np.ndarray:
    return np.log(d.back1.to_numpy(float) / d.bsp.to_numpy(float))       # + : the SP came in under the price


def fit_gbm(tr: pd.DataFrame):
    import lightgbm as lgb
    from research.greyhound.tradeout_study import GBM_PARAMS, GBM_ROUNDS
    return lgb.train(GBM_PARAMS, lgb.Dataset(_features(tr), _target(tr)), GBM_ROUNDS)


def cl_edge(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """The conditional logit's predicted edge log(back1 / forecast SP), as the live trader forecasts."""
    from greyhound import tradeout as to
    from greyhound.early_lay import vol_in_fit
    from research.greyhound import tradeout_study as ts
    f = ts.fit_cl(tr, ts.CL_SETS["micro"])
    x = te if vol_in_fit(f) else to.add_features(te.assign(runner_total_matched=np.nan))
    q, book = ts.predict(x, f)
    return np.log(te.back1.to_numpy(float) * book * np.asarray(q, float))


def cl_calibrated(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """The CL edge on the test rows, calibrated by isotonic regression of the realised edge on the out-of-fold edge
    of the training days (fitted on alternate days, predicting the others)."""
    from sklearn.isotonic import IsotonicRegression
    days = sorted(tr.race_date.astype(str).unique())
    a, b = days[0::2], days[1::2]
    e_oof, r_oof = [], []
    for fit_days, pred_days in ((a, b), (b, a)):
        f_tr = tr[tr.race_date.astype(str).isin(fit_days)]
        p_tr = tr[tr.race_date.astype(str).isin(pred_days)]
        if len(f_tr) and len(p_tr):
            e_oof.append(cl_edge(f_tr, p_tr))
            r_oof.append(_target(p_tr))
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.concatenate(e_oof), np.concatenate(r_oof))
    return iso.predict(cl_edge(tr, te))


def benter(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    """Log-loss on the result a race, test rows: the market's price at the decision book alone against the market
    with our price (conditional logits fitted on the training rows)."""
    from greyhound import tradeout as to
    from model import race_book

    def design(d, cols):
        return d[cols].to_numpy(float)

    def fit_ll(cols):
        g_tr = to._codes(tr)
        q_tr = race_book.normalise_within(tr.won.to_numpy(float) + 1e-12, g_tr)
        X = design(tr, cols)
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-9] = 1.0
        beta = to.fit_clogit((X - mu) / sd, q_tr, g_tr)
        g = to._codes(te)
        lq = race_book.group_log_softmax(((design(te, cols) - mu) / sd) @ beta, g)
        won = te.won.to_numpy(float) == 1
        return -float(lq[won].sum()), int(won.sum()), (beta / sd).tolist()

    ll_m, n, _ = fit_ll(["lq_market"])
    ll_b, _, beta = fit_ll(["lq_market", "lq_model"])
    return {"races": n, "ll_market": ll_m, "ll_blend": ll_b, "beta_market_model": beta}


def run(rec: pd.DataFrame, live: pd.DataFrame, min_train_days: int = 7) -> dict:
    from greyhound.take_close import CONFIG
    from model.race_book import ratio_interval
    from research.greyhound.take_close_refit import takes
    from research.overfit import dsr_check
    days = sorted(live.race_date.astype(str).unique())
    cfg = dict(CONFIG, bar=BAR)
    per = {a: [] for a in ARMS}
    bent = []
    scored = []
    for i, day in enumerate(days):
        if i < min_train_days:
            continue
        te = live[live.race_date.astype(str) == day]
        tr = live[live.race_date.astype(str) < day]
        trr = rec[rec.race_date.astype(str) < day] if len(rec) else rec
        if not len(te) or not len(tr):
            continue
        scored.append(day)
        b = benter(tr, te)
        bent.append({"day": day, **b})
        preds = {"cl_cal": cl_calibrated(tr, te), "gbm_live": fit_gbm(tr).predict(_features(te))}
        both = pd.concat([tr, trr], ignore_index=True) if len(trr) else tr
        preds["gbm_both"] = fit_gbm(both).predict(_features(te))
        for arm, e in preds.items():
            f_bsp = te.back1.to_numpy(float) / np.exp(np.asarray(e, float))        # the predicted SP
            t = takes(te, f_bsp, 0.02, cfg=cfg)
            if len(t):
                per[arm].append(t.assign(pred_edge=np.log(t.price / t.f_bsp)))
    out = {"tag": TAG, "scored_days": scored, "bar": BAR, "arms": {}}
    bt = pd.DataFrame(bent)
    if len(bt):
        dd = (bt.ll_blend - bt.ll_market).to_numpy() / bt.races.to_numpy()
        lo, hi = ratio_interval((bt.ll_blend - bt.ll_market).to_numpy(), bt.races.to_numpy(float), n=2000)
        out["benter_at_decision"] = {"races": int(bt.races.sum()),
                                     "ll_market_per_race": round(float(bt.ll_market.sum() / bt.races.sum()), 5),
                                     "ll_blend_per_race": round(float(bt.ll_blend.sum() / bt.races.sum()), 5),
                                     "gain_per_race": round(float((bt.ll_blend - bt.ll_market).sum() / bt.races.sum()), 5),
                                     "gain_90%": [round(lo, 5), round(hi, 5)], "days_better": int((dd < 0).sum()),
                                     "beta_market_model_last": bent[-1]["beta_market_model"]}
    for arm in ARMS:
        t = pd.concat(per[arm], ignore_index=True) if per[arm] else pd.DataFrame()
        if not len(t):
            out["arms"][arm] = {"backs": 0}
            continue
        t["clv_value"] = t.clv * t.stake
        d_ = t.groupby("race_date").agg(stake=("stake", "sum"), clv_value=("clv_value", "sum")).reindex(scored, fill_value=0.0)
        mk = t.drop_duplicates("market_id")
        dn = mk.groupby("race_date").market_net.sum().reindex(scored, fill_value=0.0)
        lo, hi = ratio_interval(d_.clv_value.to_numpy(), d_.stake.to_numpy(), n=2000)
        nlo, nhi = ratio_interval(dn.to_numpy(), d_.stake.to_numpy(), n=2000)
        mret = (mk.set_index("market_id").market_net / t.groupby("market_id").stake.sum()).to_numpy(float)
        realised = np.log(t.price / t.bsp)
        out["arms"][arm] = {
            "backs": int(len(t)), "markets": int(t.market_id.nunique()), "staked": round(float(t.stake.sum()), 2),
            "drifted": int((t.bsp > t.price).sum()), "shortened": int((t.bsp < t.price).sum()),
            "clv_by_stake_%": round(100 * float(t.clv_value.sum() / t.stake.sum()), 2),
            "clv_90%": [round(100 * lo, 2), round(100 * hi, 2)] if np.isfinite(lo) else None,
            "pred_edge_taken_%": round(100 * float(t.pred_edge.mean()), 2),
            "realised_edge_taken_%": round(100 * float(realised.mean()), 2),
            "net": round(float(mk.market_net.sum()), 2),
            "roi_%": round(100 * float(mk.market_net.sum() / t.stake.sum()), 2),
            "roi_90%": [round(100 * nlo, 2), round(100 * nhi, 2)] if np.isfinite(nlo) else None,
            "dsr": dsr_check(mret, n_trials=N_TRIALS)}
    for arm, r in out["arms"].items():
        ci = r.get("clv_90%")
        calib = r.get("pred_edge_taken_%") is not None and \
            abs(r["pred_edge_taken_%"] - r["realised_edge_taken_%"]) <= 3.0
        r["reading"] = {"clv_interval_above_zero": bool(ci and ci[0] > 0), "calibrated_on_taken": bool(calib),
                        "net_positive": bool((r.get("net") or 0) > 0),
                        "dsr_passed": bool((r.get("dsr") or {}).get("passed")),
                        "min_backs_met": bool((r.get("backs") or 0) >= 100)}
        r["reading"]["pass"] = all(r["reading"].values())
    return out


def main(argv=None) -> int:
    from greyhound import tradeout as to
    from research.greyhound.take_close_refit import load
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="first", default=str(to.FIRST_BOOK_DAY))
    ap.add_argument("--to", dest="last", default=str(date(2026, 11, 17)))
    ap.add_argument("--out", default="out/edge_models")
    a = ap.parse_args(argv)
    rec, live, notes = load(Store(), date.fromisoformat(a.first), date.fromisoformat(a.last))
    res = run(rec, live) if len(live) and len(rec) else {"note": "no data found: no live or recorder rows"}
    res.update({"recorder_rows": int(len(rec)), "live_rows": int(len(live))})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({**res, "notes": notes}, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
