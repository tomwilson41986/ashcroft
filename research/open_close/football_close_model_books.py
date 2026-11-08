"""The closing-line model on the recorded books: a move model trained on Betfair's real executable checkpoints, and a
spread-aware entry gate.

racing² football research, 8 Nov 2026 (the owner: "improving our closing line model ... a profitable, automated,
trading solution"). The measured round trip at the recorded books is about 2.3% crossing (reports/
football_model_rethink.md, 8 Nov), so a forecast is worth trading only where it beats the book it would trade on.
This builds the training set the move model will learn from as the weeks of records build up, and scores it.

**The set** (from the football feed's paper book, ``paper_lines``): one row per selection and checkpoint (1440, 360,
120, 60, 15 minutes before the off) of the tradable match-odds and over/under 2.5 books (entry spread at most 3
ticks, a sane two-sided close, price at most 20). Features known at the checkpoint only:

- the book: mid chance, spread (ticks, and as a share of the price), overround, depth on each side and its imbalance;
- the path so far: the log change of the mid chance since the selection's earliest recorded checkpoint, and since the
  checkpoint before;
- context: checkpoint, market type, the selection (home / draw / away, over / under), minutes to the off.

Target: the log change of the mid chance from the checkpoint to the close (y > 0: the selection shortens).

**The model and the gate (fixed before scoring):** LightGBM, walk-forward by match day (each day predicted by a model
fitted on the days before it, from the third day on). A selection is traded when the forecast move's value beats its
own cost: back when exp(y_hat) - 1 exceeds the line's entry half-spread plus the median closing half-spread plus
commission on the gain, lay in the mirror case; closed at the close crossing (the line's own ``green_close``). Reported
beside it: the same gate with a flat one-tick cost, and every line traded (the unconditional cost).

    python -m research.open_close.football_close_model_books --days 30 --out out/close_model_books.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINTS = (1440.0, 360.0, 120.0, 60.0, 15.0)
TYPES = ("MATCH_ODDS", "OVER_UNDER_25")
COMMISSION = 0.02
MIN_TRAIN_DAYS = 2
FEATURES = ["checkpoint_min", "minutes_to_off", "q_mid", "spread_ticks", "spread_rel", "book_over", "log_depth_back",
            "log_depth_lay", "imbalance", "drift_since_first", "drift_since_prev", "is_match_odds", "leg_code"]


def _leg(market_type: str, runner: str, event: str) -> int:
    home, _, away = str(event).partition(" v ")
    r = str(runner)
    if market_type == "MATCH_ODDS":
        return 1 if r == home.strip() else (3 if r == away.strip() else (2 if r == "The Draw" else 0))
    return 4 if r.lower().startswith("over") else (5 if r.lower().startswith("under") else 0)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """The training rows from paper lines (one crossing-back line per selection and checkpoint)."""
    from research.open_close.football_paper_lines_scan import tradable
    c = df[(df["kind"] == "cross") & (df["side"] == "back") & df["market_type"].isin(TYPES)
           & df["checkpoint_min"].isin(CHECKPOINTS)].copy()
    if c.empty:
        return c
    c = c[tradable(c)]
    c["mid"] = (c["best_back"] + c["best_lay"]) / 2
    c["q_mid"] = 1 / c["mid"]
    c["spread_rel"] = (c["best_lay"] - c["best_back"]) / c["mid"]
    c["log_depth_back"] = np.log1p(c["depth_back_gbp"].fillna(0))
    c["log_depth_lay"] = np.log1p(c["depth_lay_gbp"].fillna(0))
    c["imbalance"] = c["depth_back_gbp"].fillna(0) / (c["depth_back_gbp"].fillna(0) + c["depth_lay_gbp"].fillna(0)
                                                     ).replace(0, np.nan)
    c = c.sort_values(["market_id", "selection_id", "checkpoint_min"], ascending=[True, True, False])
    g = c.groupby(["market_id", "selection_id"])["q_mid"]
    c["drift_since_first"] = np.log(c["q_mid"] / g.transform("first"))
    c["drift_since_prev"] = np.log(c["q_mid"] / g.shift(1)).fillna(0.0)
    c["is_match_odds"] = (c["market_type"] == "MATCH_ODDS").astype(int)
    c["leg_code"] = [_leg(m, r, e) for m, r, e in zip(c["market_type"], c["runner"], c["event"])]
    c["y"] = np.log((1 / c["close_mid"]) / c["q_mid"])
    c["close_half_spread"] = (c["close_lay"] - c["close_back"]) / c["close_mid"] / 2
    return c.dropna(subset=["y"])


def walk_forward(rows: pd.DataFrame) -> pd.DataFrame:
    from lightgbm import LGBMRegressor
    days = sorted(rows["day"].unique())
    out = []
    for i, d in enumerate(days):
        if i < MIN_TRAIN_DAYS:
            continue
        tr, te = rows[rows["day"] < d], rows[rows["day"] == d]
        if len(tr) < 300 or te.empty:
            continue
        m = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.9, verbose=-1)
        m.fit(tr[FEATURES], tr["y"])
        out.append(te.assign(y_hat=m.predict(te[FEATURES])))
    return pd.concat(out) if out else pd.DataFrame()


def _summ(x: pd.Series, mk: pd.Series, n: int = 1000) -> dict:
    if len(x) == 0:
        return {"trades": 0}
    s = pd.DataFrame({"x": x.to_numpy(float), "m": mk.to_numpy()}).groupby("m")["x"].agg(["sum", "size"])
    pnl = np.where(s["sum"] > 0, s["sum"] * (1 - COMMISSION), s["sum"])
    w = s["size"].to_numpy(float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(pnl), (n, len(pnl)))
    b = pnl[idx].sum(1) / w[idx].sum(1)
    return {"trades": int(w.sum()), "roi": round(float(pnl.sum() / w.sum()), 4),
            "lo95": round(float(np.quantile(b, 0.025)), 4), "hi95": round(float(np.quantile(b, 0.975)), 4)}


def gate(p: pd.DataFrame, lines: pd.DataFrame, cost: str) -> dict:
    """Trades the gate takes, scored at each chosen line's own green_close."""
    med_close = float(p["close_half_spread"].median())
    if cost == "own":
        c = p["spread_rel"] / 2 + med_close
    else:                                                     # a flat tick each end, as the older gate assumed
        c = pd.Series(0.01, index=p.index)
    gain = np.expm1(p["y_hat"])
    net_back = np.where(gain > 0, gain * (1 - COMMISSION), gain) - c
    net_lay = np.where(-gain > 0, -gain * (1 - COMMISSION), -gain) - c
    side = np.where(net_back > 0, "back", np.where(net_lay > 0, "lay", None))
    pick = p.assign(side=side)[["market_id", "selection_id", "checkpoint_min", "side", "day"]].dropna(subset=["side"])
    got = lines.merge(pick, on=["market_id", "selection_id", "checkpoint_min", "side"], suffixes=("", "_p"))
    got = got.dropna(subset=["green_close"])
    out = _summ(got["green_close"], got["market_id"])
    out["by_day"] = {d: round(float(g["green_close"].mean()), 4) for d, g in got.groupby("day")}
    out["backs"] = int((got["side"] == "back").sum())
    out["lays"] = int((got["side"] == "lay").sum())
    return out


def run(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "no paper lines found"}
    rows = build(df)
    out = {"rows": int(len(rows)), "days": sorted(rows["day"].unique()) if len(rows) else [],
           "markets": int(rows["market_id"].nunique()) if len(rows) else 0}
    p = walk_forward(rows)
    if p.empty:
        out["status"] = f"too few days to walk forward (needs {MIN_TRAIN_DAYS} to train)"
        return out
    out["scored_rows"] = int(len(p))
    out["corr_forecast_move"] = round(float(np.corrcoef(p["y_hat"], p["y"])[0, 1]), 4)
    out["oos_r2_vs_zero"] = round(float(1 - np.sum((p["y"] - p["y_hat"]) ** 2) / np.sum(p["y"] ** 2)), 4)
    out["by_checkpoint_corr"] = {int(k): round(float(np.corrcoef(g["y_hat"], g["y"])[0, 1]), 4)
                                 for k, g in p.groupby("checkpoint_min") if len(g) > 20}
    lines = df[(df["kind"] == "cross") & df["market_type"].isin(TYPES)]
    from research.open_close.football_paper_lines_scan import tradable
    lines = lines[tradable(lines)]
    out["gate_own_cost"] = gate(p, lines, "own")
    out["gate_flat_tick"] = gate(p, lines, "tick")
    every = lines.merge(p[["market_id", "selection_id", "checkpoint_min"]], on=["market_id", "selection_id",
                                                                                "checkpoint_min"])
    out["every_line_in_the_scored_days"] = _summ(every["green_close"].dropna(), every.dropna(
        subset=["green_close"])["market_id"])
    return out


def main(argv=None) -> int:
    from research.open_close.football_paper_lines_scan import load
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(load(Store(root=a.root), a.days))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
