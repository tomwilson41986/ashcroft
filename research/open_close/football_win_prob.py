"""The most accurate win probability: home, draw and away chances scored against results, not against the move.

racing² football research, 24 Nov 2026 (the owner: "I don't care about accuracy against the move. In the end we want
the most accurate possible win probability for each team. Then we can trade around that"). Registered as
``football-win-prob-1124`` in reports/research_ledger.jsonl before its run.

Every Betfair match in the warehouse with a result, at each checkpoint (360, 120, 60 and 15 minutes before the
off; fresh checkpoint book, schedule known). Four forecasts of the result, each a set of three chances summing to one:

- ``market``: Betfair's normalised chances at the checkpoint (the bar to beat at the moment a bet is placed);
- ``market_cal``: the same, recalibrated: a logistic regression a role on the log odds and the book's overround,
  fitted on earlier months (the favourite-longshot bias, whatever there is of it, removed);
- ``market_move``: the checkpoint's log odds plus the Betfair-native model's forecast of the move to the close
  (``football_bf_close_model.walk_forward``), so the move model's skill read as win-probability accuracy;
- ``benter``: a LightGBM a role trained on the result directly, starting from the checkpoint's log odds
  (``init_score``) and learning only the correction, from every feature the Betfair-native model reads (path,
  implied goals, the market's ratings, move memory); Benter's model-plus-market, fitted on earlier months.

Each row's three chances are renormalised to sum to one. Walk-forward by calendar month from the fifth: a month is
forecast by models fitted on the matches that kicked off at least a day before it began. Scored by log loss and Brier
score against the result, per match, with the paired change against ``market`` (1,000 resamples of match days),
overall and by checkpoint; Betfair's closing chances (``gold/implied`` at 0 minutes) on the same matches are reported
as the ceiling the market itself reaches by the off.

    python -m research.open_close.football_win_prob --out-json out/win_prob.json [--root <warehouse folder>]
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

TAG = "football-win-prob-1124"
MIN_TRAIN_MONTHS = 4
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def renorm(d: pd.DataFrame, col: str) -> pd.Series:
    """A forecast column's three chances an event-checkpoint, scaled to sum to one."""
    s = d.groupby(["event_id", "cp_min"])[col].transform("sum")
    return d[col] / s


def complete_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Event-checkpoints with all three outcomes present, a known result (one winner) and a fresh book."""
    d = d[d["cp_fresh"] & d["schedule_fixed"] & d["won"].notna()].copy()
    g = d.groupby(["event_id", "cp_min"])
    ok = (g["role"].transform("nunique") == 3) & (g["won"].transform("sum") == 1) & (g["role"].transform("size") == 3)
    d = d[ok].copy()
    d["p_mkt"] = renorm(d, "p_cp")
    return d.reset_index(drop=True)


def walk(d: pd.DataFrame, fit_predict) -> np.ndarray:
    """Out-of-sample values for every row from the fifth month; ``fit_predict(tr, te, role)`` a role a month."""
    out = np.full(len(d), np.nan)
    months = sorted(d["month"].unique())
    for m in months[MIN_TRAIN_MONTHS:]:
        start = pd.Timestamp(m + "-01", tz="UTC")
        for role in bf.ROLES:
            r = d["role"] == role
            tr = d[r & (d["kickoff"] < start - pd.Timedelta(days=1))]
            te_idx = np.flatnonzero((r & (d["month"] == m)).to_numpy())
            if len(tr) < 2000 or not len(te_idx):
                continue
            out[te_idx] = fit_predict(tr, d.iloc[te_idx], role)
    return out


def cal_fit_predict(tr, te, role):
    from sklearn.linear_model import LogisticRegression
    X = lambda t: np.c_[logit(t["p_mkt"].to_numpy(float)), t["cp_book_sum"].to_numpy(float) - 1.0]   # noqa: E731
    m = LogisticRegression(C=1e4, max_iter=500).fit(X(tr), tr["won"].to_numpy(int))
    return m.predict_proba(X(te))[:, 1]


def benter_fit_predict(cols):
    def f(tr, te, role):
        import lightgbm as lgb
        y = tr["won"].to_numpy(int)
        base, base_te = logit(tr["p_mkt"].to_numpy(float)), logit(te["p_mkt"].to_numpy(float))
        ds = lgb.Dataset(tr[cols], y, init_score=base, free_raw_data=False)
        params = {"objective": "binary", "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 400,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
                  "verbose": -1, "seed": 0, "num_threads": bf.THREADS}
        g = lgb.train(params, ds, num_boost_round=200)
        return 1.0 / (1.0 + np.exp(-(base_te + g.predict(te[cols], raw_score=True))))
    return f


def scores(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per event-checkpoint: log loss and Brier of a forecast column (three rows each)."""
    p = renorm(d, col).clip(EPS, 1.0)
    w = d["won"].to_numpy(float)
    t = pd.DataFrame({"event_id": d["event_id"], "cp_min": d["cp_min"], "day": d["day"],
                      "ll": -np.log(p) * w, "br": (p - w) ** 2})
    return t.groupby(["event_id", "cp_min", "day"], as_index=False)[["ll", "br"]].sum()


def paired(base: pd.DataFrame, arm: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    j = base.merge(arm, on=["event_id", "cp_min", "day"], suffixes=("_b", "_a"))
    f = j.groupby("day")[["ll_b", "ll_a", "br_b", "br_a"]].sum()
    n = len(j)
    idx = np.random.default_rng(seed).integers(0, len(f), size=(n_boot, len(f)))
    diff = (f["ll_b"] - f["ll_a"]).to_numpy()
    cnt = j.groupby("day").size().to_numpy()
    boot = diff[idx].sum(1) / cnt[idx].sum(1)
    return {"matches_checkpoints": int(n), "days": int(len(f)),
            "log_loss": round(float(j["ll_a"].mean()), 6), "log_loss_market": round(float(j["ll_b"].mean()), 6),
            "brier": round(float(j["br_a"].mean()), 6), "brier_market": round(float(j["br_b"].mean()), 6),
            "log_loss_saved": round(float(diff.sum() / n), 6), "lo95": round(float(np.quantile(boot, 0.025)), 6),
            "hi95": round(float(np.quantile(boot, 0.975)), 6)}


def run(store) -> dict:
    d, fam = bf.load(store)
    if d.empty:
        return {"tag": TAG, "status": "no data found: gold/move is not built"}
    d = complete_rows(d)
    res = {"tag": TAG, "event_checkpoints": int(len(d) // 3), "events": int(d["event_id"].nunique()),
           "months": sorted(d["month"].unique()),
           "note": "development evidence: Betfair's last traded prices; scored on results; nothing is traded"}
    if d["month"].nunique() <= MIN_TRAIN_MONTHS:
        res["status"] = "no data found: too few months"
        return res
    cols = [c for v in fam.values() for c in v]
    d["f_cal"] = walk(d, cal_fit_predict)
    log.info("market_cal done")
    mv, _ = bf.walk_forward(d, cols, min_train_months=MIN_TRAIN_MONTHS)
    d["f_move"] = 1.0 / (1.0 + np.exp(-(logit(d["p_mkt"].to_numpy(float)) + mv)))
    log.info("market_move done")
    d["f_benter"] = walk(d, benter_fit_predict(cols))
    log.info("benter done")
    arms = {"market_cal": "f_cal", "market_move": "f_move", "benter": "f_benter"}
    d["_ok"] = d[list(arms.values())].notna().all(axis=1).astype(int)
    ok = d.groupby(["event_id", "cp_min"])["_ok"].transform("min") == 1
    s = d[ok.to_numpy()].copy()
    res["scored_event_checkpoints"] = int(len(s) // 3)
    res["test_months"] = sorted(s["month"].unique())
    base = scores(s, "p_mkt")
    res["market"] = {"log_loss": round(float(base["ll"].mean()), 6), "brier": round(float(base["br"].mean()), 6)}
    res["arms"] = {a: paired(base, scores(s, c)) for a, c in arms.items()}
    res["by_checkpoint"] = {}
    for cp in bf.HORIZONS:
        sc = s[s["cp_min"] == cp]
        if len(sc) >= 1500:
            b = scores(sc, "p_mkt")
            res["by_checkpoint"][str(cp)] = {a: paired(b, scores(sc, c)) for a, c in arms.items()}
    # the ceiling: Betfair's own close on the same matches
    imp = bf.read_rows(store, "gold/implied", ["event_id", "cp_min", "p_home", "p_draw", "p_away"], horizons=False)
    if imp is not None:
        c = imp.to_pandas()
        c = c[c["cp_min"] == 0].drop(columns="cp_min").dropna()
        role_col = {"home": "p_home", "draw": "p_draw", "away": "p_away"}
        j = s.merge(c, on="event_id", how="inner")
        if len(j):
            j["p_close"] = np.select([j["role"] == k for k in role_col], [j[v] for v in role_col.values()], np.nan)
            res["close_on_the_same_matches"] = paired(scores(j, "p_mkt"), scores(j, "p_close"))
    for a in arms:
        res["arms"][a]["beats_the_market"] = bool(res["arms"][a]["lo95"] > 0)
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
    print(json.dumps({k: v for k, v in res.items() if k in ("tag", "status", "market", "arms",
                                                             "close_on_the_same_matches")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
