"""What drives the football move model, and where its strongest signals live (racing² football research, 4 Nov 2026).

Read-only over the lab matrix. Three views of the stacked team-feature move model (football_improve):

1. LightGBM gain share by feature, fitted on the same-book (Pinnacle) moves 2013/14-2024/25, home side.
2. Each feature's correlation with Betfair's own 2025/26 home moves.
3. The strongest signals on Betfair 2025/26 (forecast above 0.03, the model fitted on the seasons before), split by
   league tier, price band and direction: bets, return a tick off fair and at a symmetric book, right-way share.

2025/26 was read by earlier rules: development evidence.

    python -m research.open_close.football_model_diagnostics --cache <lab cache> --out-json reports/football_runs/model_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOP5 = ("E0", "D1", "I1", "SP1", "F1")
OTHER_TOP_AND_SECOND = ("E1", "D2", "I2", "SP2", "F2", "N1", "P1", "B1", "SC0", "T1", "G1")


def tier(league: pd.Series) -> np.ndarray:
    return np.where(league.isin(TOP5), "top five", np.where(league.isin(OTHER_TOP_AND_SECOND), "other top and second",
                                                            "lower"))


def run(cache: str) -> dict:
    import lightgbm as lgb
    from research.open_close import football_improve as fi
    from research.open_close.football_stack_betfair_read import betfair_moves, betfair_rows
    d, cols = fi.load(cache)
    same = d[d["same_book"] & d["season"].between(2013, 2024)]
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, importance_type="gain")
    g.fit(same[cols + ["league"]], same["y_h"])
    imp = pd.Series(g.feature_importances_, index=cols + ["league"])
    imp = (imp / imp.sum()).sort_values(ascending=False)
    te = d[(d["test_season"] == 2025)]
    te = te[betfair_rows(te)]
    y = betfair_moves(te)["y_h"]
    cors = {c: float(np.corrcoef(te[c].fillna(te[c].median()), y)[0, 1]) for c in cols if te[c].notna().mean() > 0.5}
    ph = fi.walk_forward(d, cols, "y_h", "same_book", 2025)
    pa = fi.walk_forward(d, cols, "y_a", "same_book", 2025)
    c = fi.candidates(d[d["test_season"] == 2025], {"H": ph, "A": pa}, "MO")
    top = c[c["apred"] > 0.03].copy()
    top["tier"] = tier(top["league"])
    top["band"] = pd.cut(top["price"], [1, 2, 3, 5, 1001], labels=["<2", "2-3", "3-5", "5+"]).astype(str)
    top["direction"] = np.where(top["side"] > 0, "back first", "lay first")
    split = {}
    for k in ("tier", "band", "direction"):
        t = top.groupby(k).agg(bets=("profit_sym", "size"), sym=("profit_sym", "mean"), tick=("profit_tick", "mean"),
                               right_way=("right_way", "mean"))
        split[k] = {str(i): {kk: round(float(v), 4) for kk, v in r.items()} for i, r in t.iterrows()}
    c["tier"] = tier(c["league"])
    return {"gain_share_home_move": {k: round(float(v), 4) for k, v in imp.head(15).items()},
            "corr_with_betfair_2025_home_move": dict(sorted(((k, round(v, 4)) for k, v in cors.items()),
                                                            key=lambda kv: -abs(kv[1]))[:10]),
            "strong_signals_betfair_2025": {"threshold": 0.03, "bets": int(len(top)), **split},
            "candidates_by_tier": {str(k): int(v) for k, v in c.groupby("tier").size().items()},
            "note": "returns are unweighted means per trade before commission; 2025 = 2025/26 to date"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    res = run(a.cache)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
