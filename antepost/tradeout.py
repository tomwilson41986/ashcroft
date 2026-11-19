"""The ante-post trade-out rule (ledger antepost-tradeout-1119): back, then lay back in the ante-post market itself.

The owner, 19 Nov 2026: "we want to trade out", "we only want back to lay", "trade out in the ante-post market".
The back laid at the SP (antepost-ev-1119) failed, most of its loss from horses that never ran while the back sat
open to race day. Here the position is closed inside the ante-post market, before it closes about 55 hours out:

  entry  back GBP1 at the first mark of 60, 45, 30, 21 or 14 days out where the turn-up model's P(ran) reaches p,
         at A_fill = the last traded price / (1 + half its band's median spread in the recorded books)
  exit   lay back at the first later mark whose last traded price is at least x shorter than at entry; failing that,
         at the last later mark the horse is listed with a price (the market's last before it closes), at
         L_fill = that price * (1 + half its band's spread); an equal green of A_fill / L_fill - 1 per GBP1, less 2%
         on a win
  void   a horse Betfair removes (balloted out) voids both bets: 0
  none   a horse with no later priced mark is held to settlement as antepost-ev-1119: laid at the SP if it ran,
         the stake lost if not (counted and reported)

p from {0.5, 0.6, 0.7} and x from {10%, 20%, 30%, none (hold to the last mark)}: 12 rules, the pair chosen on
Jul-Dec 2025 with the turn-up model fitted to 30 Jun 2025, then scored once on 2026 with it refit to 31 Dec 2025.

    python -m antepost.tradeout --panel turnup_panel.parquet --books recorded_books.parquet --out out/antepost_tradeout
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from antepost import ev
from antepost import price as pr
from antepost import turnup as tu

log = logging.getLogger(__name__)
ENTRY_MARKS = ["d60", "d45", "d30", "d21", "d14"]
P_GRID = [0.5, 0.6, 0.7]
X_GRID = [0.10, 0.20, 0.30, None]
MIN_VALID_TRADES = 30
PRIOR_TRIALS = 369      # python -m desk.trials --count, horse_racing, 19 Nov 2026 before this registration


def half_spread(price: np.ndarray, spreads: dict[str, float]) -> np.ndarray:
    idx = np.clip(np.searchsorted(tu.BANDS, price, side="left") - 1, 0, len(tu.BANDS) - 2)
    return np.array([spreads.get(f"b{i}", np.nan) for i in idx]) / 2


def paths(df: pd.DataFrame) -> dict[tuple, list[tuple[int, float]]]:
    """Each horse's priced marks, far to near: (mark order, last traded price)."""
    d = df[df["has_price"] == 1]
    order = d["mark"].map({m: i for i, m in enumerate(tu.ORDER)})
    out: dict[tuple, list] = {}
    for key, o, lp in zip(zip(d["market_id"], d["selection_id"]), order, d["log_ltp"]):
        out.setdefault(key, []).append((int(o), float(np.exp(lp))))
    return {k: sorted(v) for k, v in out.items()}


def trades(entries: pd.DataFrame, path: dict, x, spreads: dict) -> pd.DataFrame:
    """The trade of each entry: its exit mark and price, and what it returned per GBP1 backed."""
    rows = []
    for r in entries.itertuples(index=False):
        a = float(np.exp(r.log_ltp))
        a_fill = a / (1 + half_spread(np.array([a]), spreads)[0])
        eo = tu.ORDER.index(r.mark)
        later = [(o, p) for o, p in path.get((r.market_id, r.selection_id), []) if o > eo]
        exit_o, exit_p, how = None, None, "none"
        if later:
            hit = [(o, p) for o, p in later if x is not None and p <= a * (1 - x)]
            (exit_o, exit_p), how = (hit[0], "target") if hit else (later[-1], "last")
        if r.y == "void":
            ret = 0.0
        elif exit_p is not None:
            l_fill = exit_p * (1 + half_spread(np.array([exit_p]), spreads)[0])
            ret = float(ev.net(a_fill / l_fill - 1))
        elif r.y == "ran":
            ret = float(ev.net(a_fill / float(r.bsp) - 1))
        else:
            ret = -1.0
        rows.append({"market_id": r.market_id, "selection_id": r.selection_id, "mark": r.mark, "y": r.y,
                     "p_ran": r.p_ran, "entry_price": a, "exit_mark": tu.ORDER[exit_o] if exit_o is not None else None,
                     "exit_price": exit_p, "exit": how, "ret": ret, "log_ltp": r.log_ltp, "has_price": 1,
                     "ev": r.p_ran})
    return pd.DataFrame(rows)


def entries(d: pd.DataFrame, p: float) -> pd.DataFrame:
    """The first entry mark, far to near, at which each horse's P(ran) reaches p: one trade per horse per race."""
    e = d[d["mark"].isin(ENTRY_MARKS) & (d["has_price"] == 1) & (d["p_ran"] >= p)].copy()
    e["order"] = e["mark"].map({m: i for i, m in enumerate(tu.ORDER)})
    return e.sort_values("order").drop_duplicates(["market_id", "selection_id"])


def scored(df: pd.DataFrame, turn) -> pd.DataFrame:
    d = df.copy()
    p = turn.predict(d[tu.FEATURES])
    d["p_lost"], d["p_ran"], d["p_void"] = p[:, 0], p[:, 1], p[:, 2]
    return d


def describe(t: pd.DataFrame, n_trials: int | None = None, trial_sharpes=None) -> dict:
    out = ev.summary(t, n_trials, trial_sharpes) if len(t) else {"bets": 0, "races": 0}
    if len(t):
        out.pop("pred_ev", None)
        out["exits"] = t["exit"].value_counts().to_dict()
        out["roi_by_exit"] = {k: round(float(g["ret"].mean()), 4) for k, g in t.groupby("exit")}
        out["mean_p_ran"] = round(float(t["p_ran"].mean()), 4)
    return out


def run(df: pd.DataFrame, books: pd.DataFrame) -> dict:
    tr, va, te = pr.splits(df)
    spreads = ev.spread_by_band(books)
    grid = list(product(P_GRID, X_GRID))
    rep = {"spread_by_band": spreads, "grid": [[p, x] for p, x in grid], "entry_marks": ENTRY_MARKS,
           "n_trials": PRIOR_TRIALS + len(grid)}
    if tr.empty or va.empty or te.empty:
        return dict(rep, skipped="a split is empty")
    turn, _ = tu.fit_gbm(tr, va)
    dv, pv = scored(va, turn), paths(va)
    valid = {(p, x): trades(entries(dv, p), pv, x, spreads) for p, x in grid}
    rep["valid"] = {f"{p}|{x}": describe(t) for (p, x), t in valid.items()}
    ok = [k for k in grid if len(valid[k]) >= MIN_VALID_TRADES]
    best = max(ok, key=lambda k: valid[k]["ret"].mean()) if ok else grid[0]
    rep["chosen"] = {"p": best[0], "x": best[1]}
    rounds = tu.fit_gbm(tr, va)[1]
    turn, _ = tu.fit_gbm(pd.concat([tr, va]), None, rounds)
    dt, pt = scored(te, turn), paths(te)
    test = {(p, x): trades(entries(dt, p), pt, x, spreads) for p, x in grid}
    trial_sharpes = [ev.sharpe(t) if len(t) else float("nan") for t in test.values()]
    rep["test"] = describe(test[best], rep["n_trials"], trial_sharpes)
    rep["test_all_rules_for_the_record"] = {f"{p}|{x}": {k: v for k, v in describe(t).items()
                                                         if k in ("bets", "races", "roi", "interval_90", "exits")}
                                            for (p, x), t in test.items()}
    return rep


def reading(rep: dict) -> dict:
    """The registered reading (ledger antepost-tradeout-1119), applied as written."""
    t = rep.get("test") or {}
    if not t.get("bets"):
        return {"verdict": "FAIL", "why": "no trades on the test" if "test" in rep else rep.get("skipped")}
    lo = t["interval_90"][0]
    dsr = (t.get("dsr") or {}).get("passed", False)
    ok = lo > 0 and dsr and t["bets"] >= 100
    return {"verdict": "PASS" if ok else "FAIL", "chosen": rep["chosen"], "trades": t["bets"], "roi": t["roi"],
            "interval_90": t["interval_90"], "dsr": (t.get("dsr") or {}).get("dsr"), "dsr_passed": dsr,
            "at_least_100_trades": t["bets"] >= 100}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--books", required=True)
    ap.add_argument("--out", default="out/antepost_tradeout")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    rep.update(run(pd.read_parquet(a.panel), pd.read_parquet(a.books)))
    rep["reading"] = reading(rep)
    (out / "antepost_tradeout.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
