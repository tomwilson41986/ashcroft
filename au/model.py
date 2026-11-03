"""The Betfair-only Australian model, walk-forward: does a market-blind forecast built from Betfair's own history say
anything the BSP does not, and how close does it come to the BSP?

For each test year Y: fit LightGBM on every runner before 1 Jan Y (the logit of its normalised BSP chance, the UK
BFSP model's target), price year Y, never refit inside it. Scored on the out-of-sample years:

  - error: mean |log p_model - log p_bsp| (both normalised within the race), the UK model's measure (its development
    window 0.3981-0.4029, reports/serving_note_2026-09-27.md); with two baselines (every runner 1/field; the
    horse's last market chance) and R^2 of the logit;
  - the winner: race log-loss of the model, of the BSP, and of a blend p ~ p_bsp^a * p_model^b fitted on the earlier
    years' out-of-sample prices only, with a 90% race-bootstrap interval on blend minus BSP (below zero: the model
    holds something the BSP does not);
  - value at the BSP: a level back at the BSP wherever the model's chance beats the BSP's by t, after 2% commission
    on winnings, 90% race-bootstrap intervals (the market-only scan found nothing there; reports/au_nz_market_scan.md).

The early-price test (CLV, the UK edge) waits on Betfair Historic Data (au/README.md, gate A); the out-of-sample
prices are stored so it can be run without refitting.

    python -m au.model --targets s3 --out-key au/model_report.json
    python -m au.model --targets local.parquet --years 2023-2026 --rounds 300      # a quick local run
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from au import config
from au.features import build_features, leak_check

log = logging.getLogger(__name__)
COMMISSION = 0.02
PARAMS = {"objective": "huber", "alpha": 1.0, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
          "seed": 42, "num_threads": 0}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _norm_within(market: pd.Series, p: np.ndarray) -> np.ndarray:
    s = pd.Series(p, index=market.index).groupby(market).transform("sum").to_numpy()
    return p / s


def walk_forward(d: pd.DataFrame, feats: list[str], years: list[int], rounds: int) -> pd.DataFrame:
    import lightgbm as lgb
    d = d[d["p_norm"].notna()].copy()
    d["y"] = _logit(d["p_norm"].to_numpy())
    d["year"] = d["date"].dt.year
    out = []
    for y in years:
        tr, te = d[d["year"] < y], d[d["year"] == y]
        if len(tr) < 50_000 or te.empty:
            log.info("year %s skipped: %d train rows, %d test rows", y, len(tr), len(te))
            continue
        booster = lgb.train(PARAMS, lgb.Dataset(tr[feats].astype("float32"), tr["y"]), num_boost_round=rounds)
        te = te.copy()
        raw = 1.0 / (1.0 + np.exp(-booster.predict(te[feats].astype("float32"))))
        te["p_model"] = _norm_within(te["market_id"], raw)
        te["pred_logit"] = _logit(raw)
        imp = pd.Series(booster.feature_importance("gain"), index=feats)
        te.attrs["importance"] = (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()
        log.info("year %s: trained on %d rows, priced %d", y, len(tr), len(te))
        out.append(te)
    res = pd.concat(out, ignore_index=True)
    res.attrs["importance_last_year"] = out[-1].attrs["importance"] if out else {}
    return res


def _race_boot(df: pd.DataFrame, col: str, n: int = 1000, seed: int = 7) -> tuple[float, float]:
    """90% interval of the mean of col over races (resampled whole)."""
    per = df.groupby("market_id")[col].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n, len(per)))
    s, c = per["sum"].to_numpy()[idx].sum(1), per["size"].to_numpy()[idx].sum(1)
    m = s / c
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def _fit_blend(df: pd.DataFrame) -> tuple[float, float]:
    """p ~ p_bsp^a * p_model^b within each race: a and b by maximum likelihood on the winners."""
    from scipy.optimize import minimize
    lb, lm = np.log(df["p_norm"].to_numpy()), np.log(df["p_model"].to_numpy())
    mk = pd.factorize(df["market_id"])[0]
    won = df["won"].to_numpy() == 1

    def nll(th):
        z = th[0] * lb + th[1] * lm
        zmax = pd.Series(z).groupby(mk).transform("max").to_numpy()
        e = np.exp(z - zmax)
        lse = np.log(pd.Series(e).groupby(mk).transform("sum").to_numpy()) + zmax
        return -(z - lse)[won].sum() / max(won.sum(), 1)
    r = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    return float(r.x[0]), float(r.x[1])


def _apply_blend(df: pd.DataFrame, a: float, b: float) -> np.ndarray:
    z = a * np.log(df["p_norm"].to_numpy()) + b * np.log(df["p_model"].to_numpy())
    z = z - pd.Series(z).groupby(df["market_id"].to_numpy()).transform("max").to_numpy()
    e = np.exp(z)
    return e / pd.Series(e).groupby(df["market_id"].to_numpy()).transform("sum").to_numpy()


def score(res: pd.DataFrame) -> dict:
    r = res.copy()
    r["err_model"] = np.abs(np.log(r["p_model"]) - np.log(r["p_norm"]))
    r["err_uniform"] = np.abs(np.log(1.0 / r["field"]) - np.log(r["p_norm"]))
    last = np.exp(r["h_last_lp"].fillna(np.log(1.0 / r["field"])))
    r["p_last"] = _norm_within(r["market_id"], last.to_numpy())
    r["err_last"] = np.abs(np.log(r["p_last"]) - np.log(r["p_norm"]))
    one = r.groupby("market_id")["won"].transform("sum") == 1          # races with one winner, all runners priced
    complete = r.groupby("market_id")["p_norm"].transform(lambda s: s.notna().all())
    w = r[one & complete & (r["won"] == 1)]

    by_year = {}
    for y, g in r.groupby("year"):
        wy = w[w["year"] == y]
        ss_res = ((g["pred_logit"] - g["y"]) ** 2).sum()
        ss_tot = ((g["y"] - g["y"].mean()) ** 2).sum()
        by_year[int(y)] = {"runners": int(len(g)), "races": int(g["market_id"].nunique()),
                           "mae_log_model": round(float(g["err_model"].mean()), 4),
                           "mae_log_last_run": round(float(g["err_last"].mean()), 4),
                           "mae_log_uniform": round(float(g["err_uniform"].mean()), 4),
                           "r2_logit": round(float(1 - ss_res / ss_tot), 4),
                           "logloss_bsp": round(float(-np.log(wy["p_norm"]).mean()), 4),
                           "logloss_model": round(float(-np.log(wy["p_model"]).mean()), 4)}

    # the blend: fitted on the earlier years' out-of-sample prices only, applied to the next year
    years = sorted(r["year"].unique())
    rows = []
    blend_by_year = {}
    for i, y in enumerate(years[1:], start=1):
        fit = r[(r["year"] < y) & one & complete]
        a, b = _fit_blend(fit)
        g = r[(r["year"] == y) & one & complete].copy()
        g["p_blend"] = _apply_blend(g, a, b)
        g = g[g["won"] == 1]
        g["d_ll"] = -np.log(g["p_blend"]) + np.log(g["p_norm"])
        blend_by_year[int(y)] = {"a_bsp": round(a, 4), "b_model": round(b, 4),
                                 "d_logloss_x1000": round(1000 * float(g["d_ll"].mean()), 3)}
        rows.append(g)
    bl = pd.concat(rows) if rows else pd.DataFrame()
    blend = {}
    if len(bl):
        lo, hi = _race_boot(bl, "d_ll")
        blend = {"years": [int(years[1]), int(years[-1])], "races": int(len(bl)),
                 "d_logloss_x1000": round(1000 * float(bl["d_ll"].mean()), 3),
                 "ci90_x1000": [round(1000 * lo, 3), round(1000 * hi, 3)], "by_year": blend_by_year}

    # value at the BSP
    v = r[r["bsp"] > 1].copy()
    v["edge"] = v["p_model"] / v["p_norm"] - 1
    v["ret"] = np.where(v["won"] == 1, (v["bsp"] - 1) * (1 - COMMISSION), -1.0)
    value = {}
    for t in (0.0, 0.1, 0.2, 0.3, 0.5):
        s = v[v["edge"] > t]
        if len(s) < 100:
            continue
        lo, hi = _race_boot(s, "ret")
        value[f"edge_over_{t:.1f}"] = {"bets": int(len(s)), "roi_pct": round(100 * float(s["ret"].mean()), 2),
                                       "ci90_pct": [round(100 * lo, 2), round(100 * hi, 2)],
                                       "mean_bsp": round(float(s["bsp"].mean()), 2)}
    allv = v
    lo, hi = _race_boot(allv, "ret")
    value["every_runner"] = {"bets": int(len(allv)), "roi_pct": round(100 * float(allv["ret"].mean()), 2),
                             "ci90_pct": [round(100 * lo, 2), round(100 * hi, 2)]}
    pooled = {"runners": int(len(r)), "mae_log_model": round(float(r["err_model"].mean()), 4),
              "mae_log_last_run": round(float(r["err_last"].mean()), 4),
              "mae_log_uniform": round(float(r["err_uniform"].mean()), 4),
              "logloss_bsp": round(float(-np.log(w["p_norm"]).mean()), 4),
              "logloss_model": round(float(-np.log(w["p_model"]).mean()), 4)}
    return {"pooled": pooled, "by_year": by_year, "blend_vs_bsp": blend, "value_at_bsp": value,
            "importance_last_year": res.attrs.get("importance_last_year", {})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", default="s3", help="'s3' (sources/au/bsp_targets.parquet) or a local parquet")
    ap.add_argument("--years", default="2015-2026", help="test years, e.g. 2015-2026")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--leak-days", default="2019-06-15,2023-03-04,2026-09-12",
                    help="days the leak check rebuilds the features for (comma separated)")
    ap.add_argument("--out-key", help="store the report (and the out-of-sample prices beside it) under sources/")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    store = Store()
    if args.targets == "s3":
        t = store.get_parquet(config.TARGETS_KEY)
    else:
        t = pd.read_parquet(args.targets)
    rep = {"built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "targets_rows": int(len(t)),
           "targets_dates": [str(t["meeting_date"].min()), str(t["meeting_date"].max())]}
    days = [x for x in args.leak_days.split(",") if x]
    held = set(t["meeting_date"].astype(str))
    days = [x for x in days if x in held]
    rep["leak_check"] = leak_check(t, days) if days else "no leak days in the data"
    if days and not all(v.get("ok") for v in rep["leak_check"].values()):
        print(json.dumps(rep, indent=1, default=str))
        raise SystemExit("leak check failed: a feature read the future")
    d, feats = build_features(t)
    rep["features"] = feats
    rep["horses"] = int(d["horse_key"].nunique())
    rep["same_day_name_clashes"] = d.attrs.get("same_day_name_clashes")
    lo, hi = (int(x) for x in args.years.split("-"))
    res = walk_forward(d, feats, list(range(lo, hi + 1)), args.rounds)
    rep["params"] = {**PARAMS, "rounds": args.rounds}
    rep["score"] = score(res)
    if args.out_key:
        store.put(args.out_key, json.dumps(rep, indent=1, default=str).encode())
        keep = ["market_id", "selection_id", "meeting_date", "off_uk", "track", "bsp", "won", "p_norm", "p_model"]
        store.put_parquet(args.out_key.replace(".json", "_oos.parquet"), res[keep])
        rep["written"] = [store.where(args.out_key), store.where(args.out_key.replace(".json", "_oos.parquet"))]
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
