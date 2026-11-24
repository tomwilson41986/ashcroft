"""The back-to-lay paper test: the four passing cells of the 97-strategy scan, traded at the recorded books' quotes.

racing² football research, 24 Nov 2026 (the owner: "build the paper test on the recorded books"). Registered as
``football-b2l-paper-1124`` in reports/research_ledger.jsonl before its first scoring. Nothing reaches Betfair.

The scan (``football-strategy-100-1124``) found one edge: back where the move model forecasts the chance rising and
lay at the close for an equal profit. It was read at last traded prices, which a trader cannot always get. This test
trades the same signals at the prices that were actually on offer, from the all-games record
(``football/exchange_feed.py``: five levels a side with their sizes, every 10 minutes from 6 hours out, every minute
in the last 15).

**The model, frozen.** The Betfair-native move model (``football_bf_close_model``: level, path, implied, rating and
memory families, plus ``imp_delta``, kept on 24 Nov), ridge and LightGBM averaged a role, fitted once on every usable
warehouse row that kicked off before ``FROM`` (25 Nov 2026) and never refitted. A hash of its training rows is
reported so a later scoring can show it is the same model.

**Signals**, the four cells: a match-odds runner at 360 minutes with a forecast over 0.05 and over 0.03, at 15 minutes
over 0.05 and over 0.08 (log odds of the normalised chance), on a fresh checkpoint book with the kick-off fixed,
kicked off on or after ``FROM``. The forecast reads the warehouse, so a match is scored once Betfair's price file for
its day has been ingested.

**Execution at the quotes**, for stakes of GBP 2, 10 and 50:
- entry: the first recorded pre-off book of the market at or after the checkpoint, no later than the feed's gap there
  plus a minute (``paper_book.tolerance_min``). The back takes the offered back levels in order, best first;
- exit: the last recorded book before the market turned in play. The lay of stake x entry price / exit price takes
  the offered lay levels in order. Equal profit whatever the result where both fill;
- a back that fills only in part trades what filled; a lay that fills only in part leaves the rest of the back held
  to Betfair's result. 2% commission on net winnings. Return a unit staked (the back's filled stake).
- The same signals at last traded prices (the scan's measure) are reported beside them, so the gap between the two is
  measured directly; and the trades by the runner's price updates before the checkpoint (Buchdahl's test,
  ``football-buchdahl-1124``: moves in thin markets partly revert, so an edge there may be stale prices).

**The reading**, made once per cell: at 300 trades at the GBP 10 stake or on 31 Mar 2027, whichever is first. Until
then only counts are shown (signals, books found, fills), no return. A cell passes if its mean return a unit staked
at GBP 10 has a 90% interval (resampled match days) above zero, and the deflated Sharpe of its daily returns is 0.95
or more with every football rule in the ledger counted. Under 300 trades by 31 Mar 2027: not decided.

    python -m research.open_close.football_b2l_paper --out-json out/b2l_paper.json [--root <warehouse folder>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research.open_close import football_bf_close_model as cm

log = logging.getLogger(__name__)

TAG = "football-b2l-paper-1124"
FROM = pd.Timestamp("2026-11-25", tz="UTC")
READ_BY = date(2027, 3, 31)
TRADES_TO_READ = 300
CELLS = {"360 over 0.05": (360, 0.05), "360 over 0.03": (360, 0.03), "15 over 0.05": (15, 0.05),
         "15 over 0.08": (15, 0.08)}
STAKES = (2.0, 10.0, 50.0)
READ_STAKE = 10.0
COM = 0.02
DEPTH = 5


# ---------------------------------------------------------------------------------------------------------------
# execution at the quotes
# ---------------------------------------------------------------------------------------------------------------

def levels(row: pd.Series, side: str) -> list[tuple[float, float]]:
    """The offered (price, size) levels of one side of a recorded runner row, best first."""
    out = []
    for i in range(1, DEPTH + 1):
        p, s = row.get(f"{side}{i}"), row.get(f"{side}{i}_size")
        if pd.notna(p) and pd.notna(s) and float(p) > 1.0 and float(s) > 0:
            out.append((float(p), float(s)))
    return out


def walk(lv: list[tuple[float, float]], stake: float) -> tuple[float | None, float]:
    from football.paper_book import walk as _walk
    return _walk(lv, stake)


def books_of(bk: pd.DataFrame, market_id: str, selection_id: int) -> pd.DataFrame:
    r = bk[(bk["market_id"].astype(str) == str(market_id)) & (bk["selection_id"] == selection_id)].copy()
    r["t"] = pd.to_datetime(r["polled_utc"], utc=True, errors="coerce")
    return r.dropna(subset=["t"]).sort_values("t")


def entry_and_exit(r: pd.DataFrame, cp_time: pd.Timestamp, cp_min: int) -> tuple[pd.Series | None, pd.Series | None]:
    """The entry book (first pre-off book at or after the checkpoint, within the tolerance) and the exit book (the
    last pre-off book before the market turned in play)."""
    from football.paper_book import tolerance_min
    inplay = pd.to_numeric(r["inplay"], errors="coerce").fillna(0).astype(int)
    pre = r[(inplay == 0) & (r["status"] == "OPEN")]
    turned = r.loc[inplay == 1, "t"]
    if len(turned):
        pre = pre[pre["t"] < turned.min()]
    late = cp_time + pd.Timedelta(minutes=tolerance_min(cp_min))
    e = pre[(pre["t"] >= cp_time) & (pre["t"] <= late)]
    entry = e.iloc[0] if len(e) else None
    if not len(turned) or entry is None:
        return entry, None                                    # no turn in play recorded: no close to exit at
    x = pre[pre["t"] > entry["t"]]
    return entry, (x.iloc[-1] if len(x) else None)


def trade(entry: pd.Series, exit_: pd.Series | None, stake: float, won: bool) -> dict:
    """A back at the entry's offers and a lay at the exit's for an equal profit; the cash a unit of filled back."""
    p1, f1 = walk(levels(entry, "back"), stake)
    if p1 is None or f1 <= 0:
        return {"filled": 0.0}
    out = {"filled": f1, "back_price": round(p1, 3)}
    if exit_ is None:
        cash = f1 * (p1 - 1) * (1 - COM) if won else -f1                      # no exit book: held to the result
        out.update({"hedged": 0.0, "cash": cash, "ret": cash / f1})
        return out
    lv = levels(exit_, "lay")
    best = lv[0][0] if lv else None
    want = f1 * p1 / best if best else 0.0
    p2, f2 = walk(lv, want) if best else (None, 0.0)
    if p2 is None or f2 <= 0:
        cash = f1 * (p1 - 1) * (1 - COM) if won else -f1
        out.update({"hedged": 0.0, "cash": cash, "ret": cash / f1})
        return out
    # the back's whole stake against the lay that filled; whatever happens
    gross_win = f1 * (p1 - 1) - f2 * (p2 - 1)                                 # the runner wins
    gross_lose = -f1 + f2                                                     # the runner loses
    gross = gross_win if won else gross_lose
    cash = gross * (1 - COM) if gross > 0 else gross
    out.update({"lay_price": round(p2, 3), "hedged": round(f2 / want, 4) if want else 0.0, "cash": cash,
                "ret": cash / f1})
    return out


# ---------------------------------------------------------------------------------------------------------------
# the frozen model and the signals
# ---------------------------------------------------------------------------------------------------------------

def frozen_forecasts(d: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    """The move model fitted once on usable rows that kicked off before FROM (less a day), a role at a time, and its
    forecasts of every row from FROM on."""
    pred = np.full(len(d), np.nan)
    info = {}
    for role in cm.ROLES:
        r = d["role"] == role
        tr = d[r & d["usable"] & (d["kickoff"] < FROM - pd.Timedelta(days=1))]
        te_idx = np.flatnonzero((r & (d["kickoff"] >= FROM)).to_numpy())
        h = hashlib.sha256(pd.util.hash_pandas_object(tr[["market_id", "selection_id", "cp_min", "y_logit"]],
                                                      index=False).values.tobytes()).hexdigest()[:16]
        info[role] = {"training_rows": int(len(tr)), "training_hash": h, "forecast_rows": int(len(te_idx))}
        if len(tr) < 2000 or not len(te_idx):
            continue
        te = d.iloc[te_idx]
        y = tr["y_logit"].to_numpy(float)
        g = cm.gbm_fit(tr, cols, y)
        pred[te_idx] = 0.5 * (cm.ridge_fit_predict(tr, te, cols, y) + g.predict(te[cols]))
    return pred, info


def signals(d: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    d = d.assign(pred=pred)
    out = []
    for name, (cp, t) in CELLS.items():
        s = d[(d["cp_min"] == cp) & np.isfinite(d["pred"]) & (d["pred"] > t) & (d["kickoff"] >= FROM)]
        out.append(s.assign(cell=name))
    cols = ["cell", "event_id", "market_id", "selection_id", "role", "cp_min", "cp_time", "kickoff", "day", "pred",
            "ltp_cp", "ltp_close", "won", "usable", "n_ticks_before"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def execute(sig: pd.DataFrame, store, root=None) -> pd.DataFrame:
    """Every signal traded at the recorded quotes, a row a signal and stake."""
    from football.exchange_feed import read_day, uk_date
    rows = []
    sig = sig.assign(uk_day=[uk_date(k.to_pydatetime()) for k in pd.to_datetime(sig["kickoff"], utc=True)])
    for day, g in sig.groupby("uk_day"):
        bk = read_day(store, day, "books", root)
        for _, s in g.iterrows():
            base = {"cell": s["cell"], "day": s["day"], "event_id": s["event_id"], "market_id": str(s["market_id"]),
                    "selection_id": int(s["selection_id"]), "pred": float(s["pred"]),
                    "n_ticks_before": float(s["n_ticks_before"])}
            if bk is None:
                rows.append({**base, "status": "no recorded day"})
                continue
            r = books_of(bk, s["market_id"], int(s["selection_id"]))
            entry, exit_ = entry_and_exit(r, pd.Timestamp(s["cp_time"]), int(s["cp_min"])) if len(r) else (None, None)
            if entry is None:
                rows.append({**base, "status": "no entry book"})
                continue
            won = r.loc[r["runner_status"].isin(["WINNER", "LOSER"]), "runner_status"]
            won = (won.iloc[-1] == "WINNER") if len(won) else (bool(s["won"]) if pd.notna(s["won"]) else None)
            if won is None:
                rows.append({**base, "status": "no result"})
                continue
            for st in STAKES:
                t = trade(entry, exit_, st, won)
                rows.append({**base, "status": "traded" if t.get("filled", 0) > 0 else "no back fill", "stake": st,
                             "won": won, **t})
    return pd.DataFrame(rows)


def reading(tr: pd.DataFrame, sig: pd.DataFrame, today: date, n_rules: int | None) -> dict:
    """Counts a cell until its reading is due (300 trades at GBP 10, or READ_BY); then the one reading."""
    from research.open_close.football_strategy_100 import back_to_lay
    from research.open_close.football_trade_model import summary
    out = {}
    for name in CELLS:
        s = sig[sig["cell"] == name]
        t = tr[tr["cell"] == name] if len(tr) else pd.DataFrame(columns=["status", "stake"])
        one = t.drop_duplicates(["event_id", "selection_id"]) if len(t) else t
        at = t[(t["stake"] == READ_STAKE) & (t["status"] == "traded")] if len(t) else t
        c = {"signals": int(len(s)), "status": one["status"].value_counts().to_dict() if len(one) else {},
             "trades_at_10": int(len(at)),
             "fully_filled_back_at_10": int((at["filled"] >= READ_STAKE - 1e-9).sum()) if len(at) else 0,
             "fully_hedged_at_10": int((at.get("hedged", pd.Series(dtype=float)) >= 0.999).sum()) if len(at) else 0}
        if not (len(at) >= TRADES_TO_READ or today >= READ_BY):
            c["gate"] = "running"
        elif len(at) < TRADES_TO_READ:
            c["gate"] = "not decided"
        else:
            head = summary(at["day"].to_numpy(), at["ret"].to_numpy(float), n_trials=n_rules)
            dsr = (head.get("deflated_sharpe") or {}).get("dsr")
            c["headline"] = head
            c["by_stake"] = {str(st): summary(g["day"].to_numpy(), g["ret"].to_numpy(float))
                             for st, g in t[t["status"] == "traded"].groupby("stake")}
            act = pd.cut(at["n_ticks_before"], [0, 20, 60, 150, 10**9], right=False,
                         labels=["under 20", "20-59", "60-149", "150+"])
            c["by_activity_at_10"] = {str(k): summary(g["day"].to_numpy(), g["ret"].to_numpy(float))
                                      for k, g in at.groupby(act, observed=True)}
            c["same_signals_at_last_traded_prices"] = summary(
                s["day"].to_numpy(), back_to_lay(s["ltp_cp"].to_numpy(float), s["ltp_close"].to_numpy(float)))
            c["gate"] = "passed" if head.get("lo90", -1) > 0 and dsr is not None and dsr >= 0.95 else "failed"
        out[name] = c
    return out


def run(store, root=None, today: date | None = None) -> dict:
    from research.open_close import football_bf_features2 as f2
    from research.open_close.football_strategy_100 import rules_counted
    today = today or date.today()
    d, fam = cm.load(store)
    res = {"tag": TAG, "from": str(FROM.date()), "read_by": str(READ_BY), "cells": list(CELLS),
           "note": "paper; nothing reaches Betfair"}
    if d.empty:
        res["status"] = "no data found: gold/move is not built"
        return res
    d = d[d["cp_fresh"] & d["schedule_fixed"]].reset_index(drop=True)
    d, new = f2.add_families(store, d)
    cols = [c for v in fam.values() for c in v] + new.get("imp_delta", [])
    res["latest_kickoff_in_the_warehouse"] = str(d["kickoff"].max())
    if not (d["kickoff"] >= FROM).any():
        res["status"] = ("no data found: no warehouse match from 25 Nov 2026 yet (Betfair's price files for those "
                         "days are not ingested)")
        res["model"] = {"frozen_on_rows_before": str(FROM.date())}
        return res
    pred, info = frozen_forecasts(d, cols)
    res["model"] = info
    sig = signals(d, pred)
    tr = execute(sig, store, root) if len(sig) else pd.DataFrame()
    res["cells_read"] = reading(tr, sig, today, rules_counted())
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a local copy of the recorded days in place of the capture bucket")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    res = run(Store(), a.root)
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
