"""Up to 100 football trading strategies, each scored the same way on Betfair's own prices and the results.

racing² football research, 24 Nov 2026 (the owner: "I want to keep an open mind and test all possible trading
strategies. Up to 100 different"). Registered as ``football-strategy-100-1124`` in reports/research_ledger.jsonl
before its run, with the catalogue below fixed: nothing is added, dropped or retuned after the read.

**Bets.** Every strategy places one-unit bets at last traded prices from the warehouse (``gold/move``,
``gold/event``, ``gold/implied``), each at the price and at a worse one (a back one tick shorter, a lay one tick
longer), 2% commission on net winnings:
- a back held to the result returns (price - 1) x 0.98 or -1 a unit staked;
- a lay held to the result returns 0.98 or -(price - 1) for a unit backer's stake, read per unit of liability;
- a back-to-lay backs at the checkpoint and lays at the close for an equal profit whatever happens: (P1 / P2 - 1),
  less commission when positive, a unit staked; a lay-to-back the reverse, per unit of liability.

**Families** (``catalogue()`` lists every cell; the count is printed in the output):
A. win value: back (or lay) where the win model's chance against the price clears an edge of 2, 5 or 10%, at 360,
   60 and 15 minutes (18);
B. move value: the same with the move model's chance (the price corrected by its forecast of the move), 2 and 5% at
   360 and 60 (8);
C. trade-outs: back-to-lay where the move model forecasts the chance rising by 3, 5 or 8% (log odds), lay-to-back
   where falling, at 360, 120, 60 and 15 minutes (24);
D. the close, model-free: favourite back and lay by price (under 1.5, 1.5-2.2, 2.2-3.0), the draw back and lay by
   price (under 3.2, 3.2-3.8, over 3.8), longshots over 8.0 back and lay, the home underdog back, the away
   favourite lay (16);
E. steam and drift: a move of the chance of 10% or more from 360 to 60 minutes, the steamer backed and laid, the
   drifter backed and laid, at 60 (4);
F. the handicap against the 1X2: where the handicap's supremacy exceeds the 1X2's by 0.15 goals, back the side it
   favours, and lay the other, at 60 (2);
G. the 25 cells of the 12 Nov scan (``football_strategy_scan``: in-play leader, price jumps, over/under and handicap
   at the close, momentum, rating gap), re-run unchanged.

**Validation.** The win model and the move model are walk-forward by calendar month (fitted on matches a day or
more before the month); a cell bets only in months they forecast. Odd kick-off months find, even months confirm. A
cell is a **lead** when both halves are positive at the worse price and the even half's 90% interval is above zero; a
**pass** needs, beyond that, the deflated Sharpe of its daily returns at 0.95 or more with every football trial in the
ledger counted (``research/overfit.py``). Nothing is traded; a pass is a candidate for a paper forward test.

    python -m research.open_close.football_strategy_100 --out-json out/strategy_100.json [--root <warehouse folder>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.open_close import football_bf_close_model as cm
from research.open_close.football_strategy_scan import tick_worse
from research.open_close.football_trade_model import summary, win_walk_forward

log = logging.getLogger(__name__)

TAG = "football-strategy-100-1124"
COM = cm.COMMISSION
EDGES_WIN, CP_WIN = (0.02, 0.05, 0.10), (360, 60, 15)
EDGES_MOVE, CP_MOVE = (0.02, 0.05), (360, 60)
TRADE_THR, CP_TRADE = (0.03, 0.05, 0.08), (360, 120, 60, 15)
FAV_BANDS = ((1.01, 1.5), (1.5, 2.2), (2.2, 3.0))
DRAW_BANDS = ((1.01, 3.2), (3.2, 3.8), (3.8, 1000.0))
LONG, STEAM, SUP_GAP = 8.0, 0.10, 0.15
LADDER = ((2.0, 0.01), (3.0, 0.02), (4.0, 0.05), (6.0, 0.1), (10.0, 0.2), (20.0, 0.5), (30.0, 1.0), (50.0, 2.0),
          (100.0, 5.0), (1000.0, 10.0))


def tick_longer(price) -> np.ndarray:
    p = np.asarray(price, float)
    step = np.full(p.shape, LADDER[-1][1])
    for top, s in reversed(LADDER):
        step = np.where(p < top - 1e-9, s, step)
    return np.minimum(p + step, 1000.0)


def back_ret(price, won) -> np.ndarray:
    p = np.asarray(price, float)
    return np.where(np.asarray(won, bool), (p - 1.0) * (1 - COM), -1.0)


def lay_ret(price, won) -> np.ndarray:
    """Per unit of liability (price - 1 for a unit backer's stake)."""
    p = np.asarray(price, float)
    return np.where(np.asarray(won, bool), -1.0, (1 - COM) / (p - 1.0))


def back_to_lay(p1, p2) -> np.ndarray:
    g = np.asarray(p1, float) / np.asarray(p2, float) - 1.0
    return np.where(g > 0, g * (1 - COM), g)


def lay_to_back(p1, p2) -> np.ndarray:
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    g = 1.0 - p1 / p2
    return np.where(g > 0, g * (1 - COM), g) / (p1 - 1.0)


def catalogue() -> list[str]:
    names = []
    for e in EDGES_WIN:
        for cp in CP_WIN:
            names += [f"A win value back e{e} cp{cp}", f"A win value lay e{e} cp{cp}"]
    for e in EDGES_MOVE:
        for cp in CP_MOVE:
            names += [f"B move value back e{e} cp{cp}", f"B move value lay e{e} cp{cp}"]
    for t in TRADE_THR:
        for cp in CP_TRADE:
            names += [f"C back-to-lay t{t} cp{cp}", f"C lay-to-back t{t} cp{cp}"]
    for lo, hi in FAV_BANDS:
        names += [f"D favourite back {lo}-{hi}", f"D favourite lay {lo}-{hi}"]
    for lo, hi in DRAW_BANDS:
        names += [f"D draw back {lo}-{hi}", f"D draw lay {lo}-{hi}"]
    names += ["D longshot back", "D longshot lay", "D home underdog back", "D away favourite lay",
              "E steamer back", "E steamer lay", "E drifter back", "E drifter lay",
              "F handicap side back", "F handicap other lay"]
    return names


def rules_counted() -> int | None:
    """Every football rule in the ledger (a grid counts each of its cells: ``desk.trials.count_rules``)."""
    try:
        from desk.trials import LEDGER, count_rules, parse
        return count_rules(parse(LEDGER.read_text())[0], "football")
    except Exception:
        return None


def score(name: str, day, ret, ret_worse, n_trials) -> dict:
    day = np.asarray(day)
    odd = pd.to_datetime(pd.Series(day)).dt.month.to_numpy() % 2 == 1
    out = {"all": summary(day, ret_worse, n_trials=n_trials), "all_at_the_price": summary(day, ret),
           "odd_months": summary(day[odd], ret_worse[odd]), "even_months": summary(day[~odd], ret_worse[~odd])}
    a, b = out["odd_months"], out["even_months"]
    out["lead"] = bool("return" in a and "return" in b and a["return"] > 0 and b["return"] > 0 and b["lo90"] > 0)
    dsr = (out["all"].get("deflated_sharpe") or {}).get("dsr")
    out["pass"] = bool(out["lead"] and dsr is not None and dsr >= 0.95)
    return out


def run(store) -> dict:
    d, fam = cm.load(store)
    if d.empty:
        return {"tag": TAG, "status": "no data found: gold/move is not built"}
    d = d[d["cp_fresh"] & d["schedule_fixed"]].reset_index(drop=True)
    cols = [c for v in fam.values() for c in v]
    n_trials = rules_counted()
    res = {"tag": TAG, "catalogue_size": len(catalogue()) + 25, "n_trials_counted": n_trials, "cells": {},
           "note": "development evidence: last traded prices; nothing is traded"}
    d["p_win"] = win_walk_forward(d, cols)                               # the win model (Benter: market + correction)
    mv, _ = cm.walk_forward(d, cols)                                     # the move model
    d["mv"] = mv
    x = np.log(np.clip(d["p_cp"], 1e-6, 1 - 1e-6) / (1 - np.clip(d["p_cp"], 1e-6, 1 - 1e-6)))
    d["p_move"] = 1 / (1 + np.exp(-(x + d["mv"])))
    known = d["won"].notna()
    P = np.array(d["ltp_cp"].to_numpy(float), copy=True)                 # rewritten for the close cells
    W = d["won"].fillna(0).to_numpy(bool)

    def add(name, m, kind, p1=None, p2=None):
        m = np.asarray(m, bool) & known.to_numpy() & np.isfinite(P)
        if kind == "back":
            r, rw = back_ret(P[m], W[m]), back_ret(tick_worse(P[m]), W[m])
        elif kind == "lay":
            r, rw = lay_ret(P[m], W[m]), lay_ret(tick_longer(P[m]), W[m])
        elif kind == "b2l":
            a, b = p1[m], p2[m]
            r, rw = back_to_lay(a, b), back_to_lay(tick_worse(a), tick_longer(b))
        else:
            a, b = p1[m], p2[m]
            r, rw = lay_to_back(a, b), lay_to_back(tick_longer(a), tick_worse(b))
        ok = np.isfinite(r) & np.isfinite(rw)
        res["cells"][name] = score(name, d["day"].to_numpy()[m][ok], r[ok], rw[ok], n_trials)
        log.info("%s: %s", name, res["cells"][name]["all"].get("bets"))

    # A, B: value against the win and the move model
    for fam_name, col, edges, cps in (("A win value", "p_win", EDGES_WIN, CP_WIN),
                                      ("B move value", "p_move", EDGES_MOVE, CP_MOVE)):
        q = d[col].to_numpy(float)
        ev_back = q * (P - 1) * (1 - COM) - (1 - q)
        ev_lay = ((1 - q) * (1 - COM) - q * (P - 1)) / (P - 1)
        for e in edges:
            for cp in cps:
                at = (d["cp_min"] == cp).to_numpy() & np.isfinite(q)
                add(f"{fam_name} back e{e} cp{cp}", at & (ev_back > e), "back")
                add(f"{fam_name} lay e{e} cp{cp}", at & (ev_lay > e), "lay")
    # C: trade-outs on the move model, closing price from a fresh close only
    p2 = np.array(d["ltp_close"].to_numpy(float), copy=True)
    fresh_close = d["usable"].to_numpy(bool)
    for t in TRADE_THR:
        for cp in CP_TRADE:
            at = (d["cp_min"] == cp).to_numpy() & np.isfinite(d["mv"].to_numpy()) & fresh_close
            add(f"C back-to-lay t{t} cp{cp}", at & (d["mv"].to_numpy() > t), "b2l", P, p2)
            add(f"C lay-to-back t{t} cp{cp}", at & (d["mv"].to_numpy() < -t), "l2b", P, p2)
    # D: the close, model-free (one row a runner: the 15-minute checkpoint's row carries the close)
    c = (d["cp_min"] == 15).to_numpy() & fresh_close
    PC = p2
    role = d["role"].to_numpy()
    fav = d.groupby(["event_id", "cp_min"])["ltp_close"].transform("min").to_numpy() == PC
    P_save = P.copy()
    P[:] = PC                                                           # these cells bet the closing price
    for lo, hi in FAV_BANDS:
        m = c & fav & (role != "draw") & (PC >= lo) & (PC < hi)
        add(f"D favourite back {lo}-{hi}", m, "back")
        add(f"D favourite lay {lo}-{hi}", m, "lay")
    for lo, hi in DRAW_BANDS:
        m = c & (role == "draw") & (PC >= lo) & (PC < hi)
        add(f"D draw back {lo}-{hi}", m, "back")
        add(f"D draw lay {lo}-{hi}", m, "lay")
    add("D longshot back", c & (PC >= LONG), "back")
    add("D longshot lay", c & (PC >= LONG), "lay")
    add("D home underdog back", c & (role == "home") & ~fav, "back")
    add("D away favourite lay", c & (role == "away") & fav, "lay")
    P[:] = P_save
    # E: steam and drift, 360 to 60, bet at 60
    w = d[["event_id", "selection_id", "cp_min", "p_cp"]].pivot_table(index=["event_id", "selection_id"],
                                                                     columns="cp_min", values="p_cp")
    if 360 in w.columns and 60 in w.columns:
        ch = (w[60] - w[360]).rename("ch").reset_index()
        e = d.merge(ch, on=["event_id", "selection_id"], how="left")["ch"].to_numpy(float)
        at60 = (d["cp_min"] == 60).to_numpy() & np.isfinite(e)
        add("E steamer back", at60 & (e >= STEAM), "back")
        add("E steamer lay", at60 & (e >= STEAM), "lay")
        add("E drifter back", at60 & (e <= -STEAM), "back")
        add("E drifter lay", at60 & (e <= -STEAM), "lay")
    # F: the handicap against the 1X2 at 60
    if "imp_sup_gap" in d.columns:
        g = d["imp_sup_gap"].to_numpy(float)
        at60 = (d["cp_min"] == 60).to_numpy() & np.isfinite(g)
        side = np.where(g > SUP_GAP, "home", np.where(g < -SUP_GAP, "away", ""))
        other = np.where(side == "home", "away", np.where(side == "away", "home", ""))
        add("F handicap side back", at60 & (side != "") & (role == side), "back")
        add("F handicap other lay", at60 & (other != "") & (role == other), "lay")
    # G: the 12 Nov scan, unchanged
    try:
        from research.open_close import football_strategy_scan as scan
        old = scan.run(store)
        for k, v in old.get("cells", {}).items():
            res["cells"][f"G {k}"] = {"scan_1112": v, "lead": v.get("lead"), "pass": None}
    except Exception as exc:                                             # reported, not fatal
        res["scan_1112_error"] = repr(exc)
    res["cells_run"] = len(res["cells"])
    res["leads"] = sorted(k for k, v in res["cells"].items() if v.get("lead"))
    res["passes"] = sorted(k for k, v in res["cells"].items() if v.get("pass"))
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
    print(json.dumps({k: res.get(k) for k in ("tag", "status", "catalogue_size", "cells_run", "leads", "passes")},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
