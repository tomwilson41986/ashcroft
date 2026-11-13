"""The in-play leader's resting back, scored on the recorded in-play books (paper; nothing reaches Betfair).

Registered as ``football-inplay-leader-maker-1112`` in reports/research_ledger.jsonl before any in-play book was
recorded (the owner, 12 Nov 2026; reports/football_strategy_candidates.md, candidates 1 and 2). The rule, as
registered:

- The sample is what football/exchange_feed.py records in play: match odds, a book a minute.
- Clock minutes are counted from a market's first book in play. Two windows: half time (50-58), late (92-105).
- At the first OPEN book in a window whose three runners all show a back and a lay, the leader is the runner with
  the highest chance 1/mid. It qualifies at a chance of 0.85 to 0.99.
- One order a market a window: a GBP10 back at the best price on the lay side of the book, behind the money there.
- It fills when the money traded at that price or higher in later books exceeds the queue ahead plus the stake. It
  lapses after 5 minutes, or at the first book that is not OPEN (the traded money of that last interval counts).
- A filled bet is held to Betfair's result, 2% commission on a win.

The reading is made once: at 300 filled orders or on 31 Jan 2027, whichever is first. Until then this module
reports counts only (orders, fills, days) and no return, so nothing is seen early. A pass needs the mean return a
unit staked with its 90% interval (resampled match days) above zero and a deflated Sharpe of at least 0.95 with
every football trial in the ledger counted. Under 300 fills on 31 Jan 2027: not decided.

Not registered, so taken as recorded and named here: the traded ladder is Betfair's ``tradedVolume`` at face value
(the football paper book reads it the same way). If Betfair counts both sides of a bet there, fills are overstated.

    python -m research.open_close.football_inplay_leader_trial --out-json out/football_inplay_leader_trial.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from football.exchange_feed import parse_ladder, read_day

log = logging.getLogger(__name__)

TAG = "football-inplay-leader-maker-1112"
FIRST_DAY = date(2026, 11, 12)                # nothing in play was recorded before the registration
READ_BY = date(2027, 1, 31)
FILLS_TO_READ = 300
WINDOWS = {"half_time": (50.0, 58.0), "late": (92.0, 105.0)}
CHANCE = (0.85, 0.99)
STAKE = 10.0
LAPSE_MIN = 5.0
COMMISSION = 0.02
EPS = 1e-9


def settle(price: float, won: bool) -> float:
    """A back's return a unit staked, 2% commission on a win."""
    return (price - 1.0) * (1.0 - COMMISSION) if won else -1.0


def score_market(b: pd.DataFrame) -> list[dict]:
    """The orders of one market from its recorded rows (one a runner a read): at most one a window."""
    b = b.copy()
    b["polled"] = pd.to_datetime(b["polled_utc"], utc=True, errors="coerce")
    b = b[b["polled"].notna()].sort_values("polled")
    b["inplay"] = pd.to_numeric(b["inplay"], errors="coerce").fillna(0).astype(int)
    if not b["inplay"].any():
        return []
    t0 = b.loc[b["inplay"] == 1, "polled"].min()
    b["clock"] = (b["polled"] - t0).dt.total_seconds() / 60.0
    res = b[b["runner_status"].isin(["WINNER", "LOSER"])].drop_duplicates("selection_id", keep="last")
    won = dict(zip(res["selection_id"], res["runner_status"] == "WINNER"))
    books = [(t, g) for t, g in b[b["inplay"] == 1].groupby("polled", sort=True)]
    out = []
    for name, (lo, hi) in WINDOWS.items():
        for i, (t, g) in enumerate(books):
            clock = float(g["clock"].iloc[0])
            if clock < lo:
                continue
            if clock > hi:
                break
            priced = g[(g["back1"] > 1.0) & (g["lay1"] > 1.0)]
            if (g["status"] != "OPEN").any() or len(g) != 3 or len(priced) != 3:
                continue
            chance = 2.0 / (priced["back1"] + priced["lay1"])
            lead = priced.loc[chance.idxmax()]
            c = float(chance.max())
            if not CHANCE[0] <= c <= CHANCE[1]:
                break                             # the first book of the window decides: no order this window
            price, queue = float(lead["lay1"]), float(lead["lay1_size"] or 0.0)
            sid, traded, filled = lead["selection_id"], 0.0, False
            for t2, g2 in books[i + 1:]:
                if (t2 - t).total_seconds() / 60.0 > LAPSE_MIN + EPS:
                    break
                row = g2[g2["selection_id"] == sid]
                if len(row):
                    lad = parse_ladder(row["traded_delta"].iloc[0])
                    traded += sum(v for p, v in lad.items() if p >= price - EPS)
                if traded > queue + STAKE:
                    filled = True
                    break
                if (g2["status"] != "OPEN").any():
                    break
            out.append({"market_id": str(lead["market_id"]), "window": name, "posted_utc": t.isoformat(),
                        "clock_min": round(clock, 1), "selection_id": int(sid), "chance": round(c, 4),
                        "price": price, "back_price": float(lead["back1"]), "queue_ahead": queue,
                        "filled": filled, "won": won.get(sid)})
            break
    return out


def orders_of_day(store, day: str, root=None) -> pd.DataFrame:
    mk, bk = read_day(store, day, "markets", root), read_day(store, day, "books", root)
    if mk is None or bk is None:
        return pd.DataFrame()
    ids = set(mk.loc[mk["market_type"] == "MATCH_ODDS", "market_id"].astype(str))
    bk = bk[bk["market_id"].astype(str).isin(ids)]
    live = bk[pd.to_numeric(bk["inplay"], errors="coerce") == 1].groupby("market_id")["polled_utc"].nunique()
    rows = []
    for mid in live[live > 1].index:              # more than the turn in play: a market of the in-play sample
        rows += score_market(bk[bk["market_id"] == mid])
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "day", day)
    return out


def reading(orders: pd.DataFrame, today: date, n_trials: int | None) -> dict:
    """Counts until the registered moment; then the one reading."""
    from research.open_close.football_trade_model import summary
    n = len(orders)
    filled = orders[orders["filled"]] if n else orders
    settled = filled[filled["won"].notna()] if n else filled
    out = {"tag": TAG, "orders": int(n), "filled": int(len(filled)), "filled_settled": int(len(settled)),
           "filled_no_result_yet": int(len(filled) - len(settled)),
           "days": int(orders["day"].nunique()) if n else 0,
           "by_window": {w: {"orders": int((orders["window"] == w).sum()),
                             "filled": int((filled["window"] == w).sum())} for w in WINDOWS} if n else {}}
    due = len(settled) >= FILLS_TO_READ or today >= READ_BY
    if not due:
        out["gate"] = "running"
        out["note"] = (f"no return is computed before {FILLS_TO_READ} settled fills or {READ_BY.isoformat()} "
                       "(the registered reading, made once)")
        return out
    if len(settled) < FILLS_TO_READ:
        out["gate"] = "not decided"
        return out
    w = settled["won"].astype(bool).to_numpy()
    cash = np.array([settle(p, x) for p, x in zip(settled["price"], w)])
    head = summary(settled["day"].to_numpy(), cash, n_trials=n_trials)
    dsr = (head.get("deflated_sharpe") or {}).get("dsr")
    out["headline"] = head
    out["gate"] = "passed" if head.get("lo90", -1.0) > 0 and dsr is not None and dsr >= 0.95 else "failed"
    out["beside_it"] = {
        "fill_rate": round(len(filled) / n, 4),
        "by_window": {k: summary(g["day"].to_numpy(), cash[(settled["window"] == k).to_numpy()])
                      for k, g in settled.groupby("window")},
        "the_same_bets_at_the_best_back_price": summary(
            settled["day"].to_numpy(), np.array([settle(p, x) for p, x in zip(settled["back_price"], w)])),
    }
    return out


def run(store, today: date | None = None, root=None) -> dict:
    from research.open_close.football_trade_model import football_trials
    today = today or date.today()
    parts, missing, d = [], [], FIRST_DAY
    while d <= today:
        o = orders_of_day(store, d.isoformat(), root)
        if len(o):
            parts.append(o)
        elif read_day(store, d.isoformat(), "markets", root) is None:
            missing.append(d.isoformat())
        d += timedelta(days=1)
    orders = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out = reading(orders, today, football_trials())
    out["days_read"] = {"from": FIRST_DAY.isoformat(), "to": today.isoformat(), "no_data_found": missing}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="a local copy of the recorded days in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from sources.common import Store
    res = run(None if a.root else Store(), root=a.root)
    with open(a.out_json, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
