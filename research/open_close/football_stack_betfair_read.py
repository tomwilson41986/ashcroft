"""The best football model (the stacked team-feature move model) read on Betfair alone, out of sample.

racing² football research, 4 Nov 2026 (the owner: "how is our best model looking?"; Betfair only). The stack
(``football_stack_trial``, arm ``team_all``, the mean of ridge and LightGBM) forecasts each side's early-to-close move
of its chance. Here it is fitted three ways on the seasons to 2024/25 and scored on Betfair's own 2025/26-to-date
moves, which no fit saw:

- ``registered``: every same-book move 2013/14-2024/25, the pre-registered fit. In the matrix these are Pinnacle's
  moves only: 2024/25's stored close is Pinnacle's, so its Betfair early prices never pair with a Betfair close there
  (488 Pinnacle matches stand for 2024/25);
- ``betfair_only``: Betfair's own early-to-close moves of 2024/25, from the bfe_open/bfe_close columns;
- ``pinnacle_only``: Pinnacle's moves 2013/14-2023/24 (the earlier rule's fit).

Scored on: R2 of the home and away chance moves, the right-way share of the strongest tenth of forecasts, and the
trade on Betfair's prices (takeable books, 1.05 or under; back early and lay at the close, or the reverse; greened;
2% commission on each market's net) at thresholds 0.03-0.05, costed a tick off the fair price and at a book
symmetric about fair. 2025/26 was read by earlier pre-registered rules: development evidence, not a holdout. This
read changes nothing of the stack rule's registration (models/football/stack_v1 stay as frozen).

    python -m research.open_close.football_stack_betfair_read --cache <lab cache> --out-json reports/football_runs/stack_betfair_read.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

THRESHOLDS = (0.03, 0.04, 0.05)
TEST_FROM = 2025


def betfair_rows(d: pd.DataFrame) -> pd.Series:
    cols = ["bfe_open_h", "bfe_open_d", "bfe_open_a", "bfe_close_h", "bfe_close_d", "bfe_close_a"]
    return d[cols].gt(1).all(axis=1)


def betfair_moves(te: pd.DataFrame) -> pd.DataFrame:
    """Each side's move of its de-vigged chance on Betfair's own prices, early to close."""
    oo = 1 / te[["bfe_open_h", "bfe_open_d", "bfe_open_a"]].to_numpy(float)
    cc = 1 / te[["bfe_close_h", "bfe_close_d", "bfe_close_a"]].to_numpy(float)
    qo, qc = oo / oo.sum(1, keepdims=True), cc / cc.sum(1, keepdims=True)
    return pd.DataFrame({"y_h": np.log(qc[:, 0] / qo[:, 0]), "y_a": np.log(qc[:, 2] / qo[:, 2]),
                         "over_o": oo.sum(1), "over_c": cc.sum(1)}, index=te.index)


def betfair_training(d: pd.DataFrame, season: int) -> pd.DataFrame:
    """Betfair's own moves of ``season`` as training rows: the targets from Betfair's early and closing columns."""
    tr = d[betfair_rows(d) & (d["season"] == season)].copy()
    mv = betfair_moves(tr)
    tr["y_h"], tr["y_a"] = mv["y_h"], mv["y_a"]
    return tr[np.isfinite(tr["y_h"]) & np.isfinite(tr["y_a"])]


def costed(te: pd.DataFrame, pred_h: np.ndarray, pred_a: np.ndarray, threshold: float) -> dict:
    from football.lab import _brief
    from research.open_close.backtest import hedged_trades, summarise
    from research.open_close.football_exchange import width_label
    from trading.exchange import tick_up
    mv = betfair_moves(te)
    rows = []
    for side, pred in (("h", pred_h), ("a", pred_a)):
        r = pd.DataFrame({"market_id": (te["match_id"] + ":MO").to_numpy(), "day": te["match_date"].astype(str).to_numpy(),
                          "month": te["match_date"].astype(str).str[:7].to_numpy(), "sport": "football",
                          "country": te["country"].to_numpy(), "market_type": "MATCH_ODDS",
                          "open_raw": te[f"bfe_open_{side}"].to_numpy(float),
                          "close_raw": te[f"bfe_close_{side}"].to_numpy(float),
                          "over": mv["over_o"].to_numpy(float), "pred": -pred})
        r["open_price"] = r["open_raw"] * r["over"]
        r["close_price"] = r["close_raw"] * mv["over_c"].to_numpy(float)
        r["y"] = np.log(r["close_price"] / r["open_price"])
        rows.append(r)
    r = pd.concat(rows, ignore_index=True)
    r = r[np.isin(width_label(r["over"]), ["tight <1.02", "medium 1.02-1.05"])].reset_index(drop=True)
    out = {"threshold": threshold}
    t = hedged_trades(r, "pred", threshold, spread=True)
    out["tick_off_fair"] = _brief(summarise(t, n_boot=1000)) if len(t) else {"bets": 0}
    t = hedged_trades(r, "pred", threshold)
    if len(t):
        idx = t["row"].to_numpy()
        ob, cb = r["open_raw"].to_numpy(float)[idx], r["close_raw"].to_numpy(float)[idx]
        of, cf = r["open_price"].to_numpy(float)[idx], r["close_price"].to_numpy(float)[idx]
        sym = lambda raw, fair: np.array([tick_up(v - 1e-9) for v in fair ** 2 / raw])   # noqa: E731
        back = t["side"].to_numpy() > 0
        o = np.where(back, ob, sym(ob, of))
        c = np.where(back, sym(cb, cf), cb)
        t = t.copy()
        t["profit"] = np.where(back, o / c - 1, 1 - o / c)
        out["symmetric_book"] = _brief(summarise(t, n_boot=1000))
    return out


def score(te: pd.DataFrame, pred_h: np.ndarray, pred_a: np.ndarray, base_h: float, base_a: float) -> dict:
    mv = betfair_moves(te)
    out = {"matches": int(len(te))}
    for side, pred, base in (("home", pred_h, base_h), ("away", pred_a, base_a)):
        y = mv[f"y_{side[0]}"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(pred)
        y, p = y[ok], pred[ok]
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - base) ** 2)
        top = np.abs(p) >= np.quantile(np.abs(p), 0.9)
        caught = np.sign(p[top]) * y[top]
        out[side] = {"r2": round(float(r2), 5), "corr": round(float(np.corrcoef(y, p)[0, 1]), 4),
                     "top_tenth_right_way": round(float((caught > 0).mean()), 4),
                     "top_tenth_move_caught": round(float(caught.mean()), 5)}
    out["costed"] = {str(th): costed(te, pred_h, pred_a, th) for th in THRESHOLDS}
    return out


def run(cache: str) -> dict:
    from football import lab
    from research.open_close import football_stack_trial as st
    d, cols = st._matrix(cache)
    test = d[betfair_rows(d) & (d["season"] >= TEST_FROM)].copy()
    same = d[d["same_book"]]
    fits = {
        "registered": same[same["season"].between(st.TRAIN_FROM, st.TRAIN_TO)],
        "betfair_only": betfair_training(d, 2024),
        "pinnacle_only": same[same["season"].between(st.TRAIN_FROM, 2023) & (same["open_1x2_src"] == "PS")],
    }
    out = {"arm": st.ARM, "cols": len(cols), "test": f"Betfair same-venue moves, seasons {TEST_FROM}/26 to date",
           "fits": {}}
    for name, tr in fits.items():
        log.info("%s: %d training rows, %d test", name, len(tr), len(test))
        ph = lab.MODELS["stack"](tr, test, cols, tr["y_h"].to_numpy(float))
        pa = lab.MODELS["stack"](tr, test, cols, tr["y_a"].to_numpy(float))
        res = score(test, ph, pa, float(tr["y_h"].mean()), float(tr["y_a"].mean()))
        res["training_rows"] = int(len(tr))
        out["fits"][name] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run(a.cache)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
