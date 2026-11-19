"""The ante-post calibration and price stages (docs/antepost_plan.md s.5; ledger antepost-price-1119).

Two parts on the turn-up panel (antepost/turnup.py build_panel), each fitted on 2024-25 and scored once on 2026:

  calibration  The turn-up model fitted on races to 30 Jun 2025 (early-stopped on Jul-Dec 2025, as registered in
               antepost-turnup-1119) predicts Jul-Dec 2025; a Dirichlet calibrator (a multinomial logistic on the
               log probabilities) is fitted on those predictions only, then applied to the same model's 2026
               predictions. Compared on 2026 with the same model uncalibrated.

  price        For horses that ran, the race-day SP from what was known at the mark: E[log BSP | ran]. LightGBM
               regression on the turn-up features (fitted to 30 Jun 2025, early-stopped on Jul-Dec 2025, refit on
               both), against three baselines fitted on 2024-25: P0 the ante-post price itself, P1 the price shifted
               by the median log(BSP / price) for its mark and price band, P2 a linear model on the log price, its
               square, log days out and their product. Scored on the error against the BSP (the owner's rule,
               18 Nov 2026): absolute log error paired by row, its 90% race-bootstrap interval, the bias, and the
               rank concordance within each race. The spread of the errors by mark (from Jul-Dec 2025) is kept for
               E[price / BSP], which the EV needs.

    python -m antepost.price --panel turnup_panel.parquet --out out/antepost_price
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from antepost import turnup as tu

log = logging.getLogger(__name__)
PRICE_PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 50,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                "verbose": -1, "seed": 7}


def splits(df: pd.DataFrame):
    rd = pd.to_datetime(df["race_date"])
    return df[rd <= tu.TRAIN_END], df[(rd > tu.TRAIN_END) & (rd <= tu.VALID_END)], df[rd > tu.VALID_END]


# ----------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------

def fit_dirichlet(p: np.ndarray, y: np.ndarray):
    """A multinomial logistic on the log probabilities (Dirichlet calibration), fitted on held-out predictions."""
    from sklearn.linear_model import LogisticRegression
    x = np.log(np.clip(p, 1e-6, 1.0))
    m = LogisticRegression(max_iter=2000, C=1.0).fit(x, y)
    order = np.argsort(m.classes_)
    return lambda q: m.predict_proba(np.log(np.clip(q, 1e-6, 1.0)))[:, order]


def calibration(df: pd.DataFrame) -> dict:
    tr, va, te = splits(df)
    if tr.empty or va.empty or te.empty:
        return {"skipped": "a split is empty"}
    model, rounds = tu.fit_gbm(tr, va)
    p_va, p_te = model.predict(va[tu.FEATURES]), model.predict(te[tu.FEATURES])
    cal = fit_dirichlet(p_va, tu.y_index(va["y"]))
    p_cal = cal(p_te)
    yi = tu.y_index(te["y"])
    ran = (te["y"] == "ran").to_numpy().astype(float)
    groups = te["market_id"].to_numpy()
    out = {"rounds": rounds, "fit_rows": int(len(tr)), "calibrator_rows": int(len(va)), "test_rows": int(len(te))}
    ll = {}
    for name, p in (("uncalibrated", p_te), ("calibrated", p_cal)):
        ll[name] = tu.row_logloss(p, yi)
        out[name] = {"logloss": round(float(ll[name].mean()), 5), "ece_ran": round(tu.ece(p[:, 1], ran), 5),
                     "brier_ran": round(float(((p[:, 1] - ran) ** 2).mean()), 5),
                     "ece_void": round(tu.ece(p[:, 2], (te["y"] == "void").to_numpy().astype(float)), 5),
                     "deciles_ran": deciles(p[:, 1], ran)}
    d = ll["calibrated"] - ll["uncalibrated"]
    out["calibrated_less_uncalibrated"] = {"mean": round(float(d.mean()), 5),
                                           "interval_90": tu.race_bootstrap(d, groups)}
    return out


def deciles(p: np.ndarray, obs: np.ndarray) -> list[dict]:
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    return [{"decile": int(b), "pred": round(float(p[q == b].mean()), 4), "obs": round(float(obs[q == b].mean()), 4),
             "n": int((q == b).sum())} for b in sorted(set(q))]


# ----------------------------------------------------------------------------------------------------------------
# Price: E[log BSP | ran]
# ----------------------------------------------------------------------------------------------------------------

def price_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Horses that ran, priced at the mark, with a BSP above 1."""
    bsp = pd.to_numeric(df["bsp"], errors="coerce")
    r = df[(df["y"] == "ran") & (df["has_price"] == 1) & (bsp > 1)].copy()
    r["z"] = np.log(pd.to_numeric(r["bsp"], errors="coerce"))
    return r


def fit_shift(train: pd.DataFrame):
    """P1: the price moved by the median log(BSP / price) for its mark and price band (the mark's median where a
    band is thin)."""
    key = tu.band_of(train)
    gap = train["z"] - train["log_ltp"]
    by_key = gap.groupby(key).agg(["median", "size"])
    by_mark = gap.groupby(train["mark"]).median()
    overall = float(gap.median())

    def predict(df):
        k = tu.band_of(df)
        shift = [by_key.loc[x, "median"] if x in by_key.index and by_key.loc[x, "size"] >= 20
                 else by_mark.get(m, overall) for x, m in zip(k, df["mark"])]
        return df["log_ltp"].to_numpy() + np.asarray(shift, dtype=float)
    return predict


def _lin_x(df: pd.DataFrame) -> np.ndarray:
    lp = df["log_ltp"].to_numpy()
    d = np.log(df["days_out"].to_numpy())
    return np.column_stack([np.ones(len(df)), lp, lp ** 2, d, lp * d])


def fit_linear(train: pd.DataFrame):
    """P2: least squares on the log price, its square, log days out and their product."""
    beta, *_ = np.linalg.lstsq(_lin_x(train), train["z"].to_numpy(), rcond=None)
    return lambda df: _lin_x(df) @ beta


def fit_price_gbm(train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int | None = None):
    import lightgbm as lgb
    dtr = lgb.Dataset(train[tu.FEATURES], train["z"])
    if rounds is None:
        dva = lgb.Dataset(valid[tu.FEATURES], valid["z"], reference=dtr)
        b = lgb.train(PRICE_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return b, int(b.best_iteration or b.current_iteration())
    return lgb.train(PRICE_PARAMS, dtr, num_boost_round=rounds), rounds


def concordance(df: pd.DataFrame, pred: np.ndarray) -> float:
    """The mean, over races and marks with at least 3 runners, of the Spearman correlation between the predicted
    and the actual BSP within the race."""
    d = df.assign(_p=pred)
    vals = []
    for _, g in d.groupby(["market_id", "mark"]):
        if len(g) >= 3 and g["z"].nunique() > 1 and g["_p"].nunique() > 1:
            vals.append(g["_p"].rank().corr(g["z"].rank()))
    return round(float(np.nanmean(vals)), 4) if vals else float("nan")


def price(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = price_rows(df)
    tr, va, te = splits(rows)
    out = {"rows": {"train": int(len(tr)), "valid": int(len(va)), "test": int(len(te))},
           "test_races": int(te["market_id"].nunique()) if len(te) else 0}
    if tr.empty or va.empty or te.empty:
        return dict(out, skipped="a split is empty"), pd.DataFrame()
    _, rounds = fit_price_gbm(tr, va)
    # the spread of the errors by mark, from the model fitted to 30 Jun 2025 on Jul-Dec 2025 (for E[price / BSP])
    m_tr, _ = fit_price_gbm(tr, None, rounds)
    res_va = va["z"].to_numpy() - m_tr.predict(va[tu.FEATURES])
    out["residual_sd_by_mark_valid"] = {m: round(float(np.std(res_va[(va["mark"] == m).to_numpy()])), 4)
                                        for m in tu.ORDER if (va["mark"] == m).any()}
    trva = pd.concat([tr, va])
    model, _ = fit_price_gbm(trva, None, rounds)
    preds = {"model": model.predict(te[tu.FEATURES]), "p0_price": te["log_ltp"].to_numpy(),
             "p1_shift": fit_shift(trva)(te), "p2_linear": fit_linear(trva)(te)}
    z = te["z"].to_numpy()
    groups = te["market_id"].to_numpy()
    err = {k: np.abs(p - z) for k, p in preds.items()}
    out["rounds"] = rounds
    out["test"] = {k: {"mae_log": round(float(err[k].mean()), 5),
                       "rmse_log": round(float(np.sqrt(((preds[k] - z) ** 2).mean())), 5),
                       "bias_log": round(float((preds[k] - z).mean()), 5),
                       "concordance": concordance(te, preds[k])} for k in preds}
    out["paired_abs_error"] = {f"model_less_{b}": {"mean": round(float((err["model"] - err[b]).mean()), 5),
                                                   "interval_90": tu.race_bootstrap(err["model"] - err[b], groups)}
                               for b in ("p0_price", "p1_shift", "p2_linear")}
    out["by_mark_mae"] = {m: {"rows": int(sel.sum()), **{k: round(float(e[sel].mean()), 5) for k, e in err.items()}}
                          for m in tu.ORDER if (sel := (te["mark"] == m).to_numpy()).sum()}
    imp = model.feature_importance("gain")
    out["importance_gain"] = {f: round(float(g), 1) for f, g in sorted(zip(tu.FEATURES, imp), key=lambda x: -x[1])}
    keep = te[["market_id", "selection_id", "race_date", "mark", "log_ltp", "z"]].copy()
    for k, p in preds.items():
        keep[f"pred_{k}"] = p
    return out, keep


def reading(cal: dict, pr: dict) -> dict:
    """The registered reading (ledger antepost-price-1119), applied as written."""
    out = {}
    if "calibrated" in cal:
        c, u = cal["calibrated"], cal["uncalibrated"]
        ok = (c["ece_ran"] < u["ece_ran"] and c["ece_ran"] <= 0.02
              and cal["calibrated_less_uncalibrated"]["interval_90"][1] <= 0.002)
        out["calibration"] = {"verdict": "PASS" if ok else "FAIL", "ece_ran": [u["ece_ran"], c["ece_ran"]],
                              "logloss_interval_upper": cal["calibrated_less_uncalibrated"]["interval_90"][1]}
    else:
        out["calibration"] = {"verdict": "NOT RUN", "why": cal.get("skipped")}
    if "paired_abs_error" in pr:
        p = pr["paired_abs_error"]
        beats = all(p[f"model_less_{b}"]["interval_90"][1] < 0 for b in ("p1_shift", "p2_linear"))
        conc = pr["test"]["model"]["concordance"] >= pr["test"]["p1_shift"]["concordance"]
        out["price"] = {"verdict": "PASS" if beats and conc else "FAIL", "beats_p1_and_p2_interval_below_0": beats,
                        "concordance_not_below_p1": conc}
    else:
        out["price"] = {"verdict": "NOT RUN", "why": pr.get("skipped")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", default="out/antepost_price")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.panel)
    rep = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "panel_rows": int(len(df))}
    rep["calibration"] = calibration(df)
    rep["price"], preds = price(df)
    rep["reading"] = reading(rep["calibration"], rep["price"])
    if not preds.empty:
        preds.to_parquet(out / "price_test_predictions.parquet", index=False)
    (out / "antepost_price.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
