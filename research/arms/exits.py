"""Arm 6: exits before the off, on the trader's own fills (read-only research).

Every back the live trader matched, closed not at the BSP (the live rule's lay at the SP) but by a lay at the best
lay offered the first time a rule fires: a take-profit (the lay price at least k under the price matched), a
stop-loss (at least s over it), both, or a lay at the last book before a time mark (T-15, T-5, T-1). The hedge lays
``stake x price / lay`` so the result is the same either way: ``stake x (price / lay - 1)``; it needs that much
offered at the best lay, else the poll does not count. Scored against the lay at the BSP on the same fills, after 2%
on the winnings of each bet.
"""

from __future__ import annotations

import bisect
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RULES = [("sp", {}),
         ("tp3", {"tp": 0.03}), ("tp5", {"tp": 0.05}), ("tp8", {"tp": 0.08}), ("tp12", {"tp": 0.12}),
         ("sl5", {"sl": 0.05}), ("sl10", {"sl": 0.10}), ("sl20", {"sl": 0.20}),
         ("tp5_sl10", {"tp": 0.05, "sl": 0.10}), ("tp8_sl20", {"tp": 0.08, "sl": 0.20}),
         ("t15", {"at": 15}), ("t5", {"at": 5}), ("t1", {"at": 1})]


def _fills(D) -> pd.DataFrame:
    led = D.ledger
    if led is None or led.empty:
        return pd.DataFrame()
    b = led[(led["event"] == "back") & (pd.to_numeric(led["matched"], errors="coerce") > 0)].copy()
    for c in ("matched", "avg_price", "selection_id"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["t"] = pd.to_datetime(b["ts"], utc=True)
    b["market_id"] = b["market_id"].astype(str)
    b["selection_id"] = b["selection_id"].astype(int)
    return b[["t", "market_id", "selection_id", "matched", "avg_price"]]


def exit_rule(path: list[tuple[datetime, float, float]], m: float, stake: float, rule: dict, start: datetime):
    """The first poll on which the rule fires: (lay price, time) or None. ``path`` is [(t, best lay, its size)]
    after the fill, open books only."""
    tp, sl, at = rule.get("tp"), rule.get("sl"), rule.get("at")
    if at is not None:
        cut = start - timedelta(minutes=at)
        before = [x for x in path if x[0] <= cut]
        if not before:
            return None
        t, L, size = before[-1]
        return (L, t) if size + 1e-9 >= stake * m / L else None
    for t, L, size in path:
        if size + 1e-9 < stake * m / L:
            continue
        if tp is not None and L <= m * (1.0 - tp) + 1e-12:
            return L, t
        if sl is not None and L >= m * (1.0 + sl) - 1e-12:
            return L, t
    return None


def exits_day(D) -> pd.DataFrame:
    """Every fill under every rule: the exit price (the BSP when the rule never fired), the hedge's result."""
    fills = _fills(D)
    if fills.empty:
        return pd.DataFrame()
    rows = []
    for f in fills.itertuples(index=False):
        key = (f.market_id, f.selection_id)
        if key not in D.settled:
            continue                                          # void (a non-runner) or unpriced
        bsp, result = D.settled[key]
        start = D.starts.get(f.market_id)
        polls = D.polls.get(f.market_id, [])
        times = D.poll_times.get(f.market_id, [])
        i = bisect.bisect_right(times, f.t)
        path = []
        for t, b in polls[i:]:
            if b.status != "OPEN" or b.inplay or (start is not None and t >= start):
                continue
            q = b.runners.get(f.selection_id)
            if q is None or q.status != "ACTIVE" or not q.lay:
                continue
            path.append((t, q.lay[0][0], q.lay[0][1]))
        for name, rule in RULES:
            hit = None if name == "sp" else exit_rule(path, f.avg_price, f.matched, rule, start)
            L = hit[0] if hit else bsp
            when = hit[1] if hit else None
            profit = f.matched * (f.avg_price / L - 1.0)
            rows.append({"day": str(D.day), "rule": name, "market_id": f.market_id, "selection_id": f.selection_id,
                         "matched": f.matched, "price": f.avg_price, "bsp": bsp, "exit": L, "fired": hit is not None,
                         "minutes_to_off": round((start - when).total_seconds() / 60, 1) if when else None,
                         "profit": profit, "profit_net": profit - 0.02 * max(profit, 0.0), "polls": len(path)})
    return pd.DataFrame(rows)


def summarise_exits(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    g = rows.groupby("rule", sort=False).agg(fills=("matched", "size"), staked=("matched", "sum"),
                                             fired=("fired", "mean"), profit=("profit", "sum"),
                                             profit_net=("profit_net", "sum"))
    sp = rows[rows.rule.eq("sp")].set_index(["day", "market_id", "selection_id"])["profit_net"]
    diff = []
    for name in g.index:
        r = rows[rows.rule.eq(name)].set_index(["day", "market_id", "selection_id"])["profit_net"]
        diff.append(float((r - sp.reindex(r.index)).sum()))
    g["vs_sp"] = diff
    g["per_staked"] = g["profit_net"] / g["staked"]
    return g.round(2)
