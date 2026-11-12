"""The recorded Exchange Stream as one row per runner per book change, in each market's last minutes before the off.

Built on research/greyhound/stream_replay.py's parser (``events``, ``apply``): the book is rebuilt message by message
and, for every runner a message changed, the row records the book as it stands after the change, with the money
traded since that runner's previous row split by side.

The side of a trade (who crossed the spread), from the price it traded at against the book just before it:
- at or under the best back price (``atb``, the lowest-probability ask): a backer took the offer, buying probability
  (sign +1);
- at or over the best lay price (``atl``): a layer took it, selling probability (sign -1);
- strictly between the two (not possible on a single price ladder, kept for safety): unsigned (0).

A row is kept from ``window_s`` before the scheduled off until the market turns in play, suspends or closes. Times
are Betfair's publish time ``pt`` (millisecond), else the recorder's receive time.
"""

from __future__ import annotations

import math

import pandas as pd

from research.greyhound import stream_replay as sr

LEVELS = 3


def tick_index(price: float | None) -> float:
    """The price's place on Betfair's ladder (1.01 is 0, each tick 1), so a spread can be counted in ticks."""
    if not price or price < 1.01:
        return math.nan
    bands = ((2.0, 0.01), (3.0, 0.02), (4.0, 0.05), (6.0, 0.1), (10.0, 0.2), (20.0, 0.5), (30.0, 1.0),
             (50.0, 2.0), (100.0, 5.0), (1000.0, 10.0))
    idx, lo = 0.0, 1.01
    for top, step in bands:
        if price <= top + 1e-9:
            return idx + round((price - lo) / step, 6)
        idx += round((top - lo) / step, 6)
        lo = top
    return idx


def _levels(ladder: dict, side: str, n: int = LEVELS) -> list:
    """The best n (price, size) of a ladder: ``atb`` dearest first (best back is the highest price), ``atl`` cheapest
    first."""
    ks = sorted(ladder, reverse=(side == "atb"))[:n]
    return [(p, ladder[p]) for p in ks]


def _signed_trades(prev_trd: dict, trd: dict, bb: float | None, bl: float | None) -> tuple[float, float, float]:
    """Money traded since the last state, split: (by backers, by layers, unsigned)."""
    buy = sell = flat = 0.0
    for p, v in trd.items():
        d = v - prev_trd.get(p, 0.0)
        if d <= 1e-9:
            continue
        if bb is not None and p <= bb + 1e-9:
            buy += d
        elif bl is not None and p >= bl - 1e-9:
            sell += d
        else:
            flat += d
    return buy, sell, flat


def book_rows(lines, window_s: float = 300.0, market_type: str | None = "WIN") -> pd.DataFrame:
    """One row per runner per book change in each market's last ``window_s`` seconds before the scheduled off."""
    markets: dict = {}
    last_trd: dict = {}                    # (market, sid) -> its traded ladder at its previous row
    last_book: dict = {}                   # (market, sid) -> (best back, best lay) at its previous row
    rows = []
    for t, mc in sr.events(lines):
        mid = str(mc.get("id"))
        mid, changed = sr.apply(markets, mc)
        m = markets[mid]
        if market_type and m.defn.get("marketType") and m.defn.get("marketType") != market_type:
            continue
        off = m.off
        if off is None or m.inplay or str(m.defn.get("status", "OPEN")) != "OPEN":
            continue
        to_off = (off - t).total_seconds()
        if to_off > window_s:
            for sid in changed:                          # keep the reference state current before the window opens
                r = m.runners[sid]
                last_trd[(mid, sid)] = dict(r.trd)
                last_book[(mid, sid)] = (r.best_back()[0], r.best_lay()[0])
            continue
        n_active = sum(1 for rr in m.defn.get("runners", []) or [] if rr.get("status") == "ACTIVE")
        book_back = sum(1.0 / rr.best_back()[0] for rr in m.runners.values() if rr.best_back()[0])
        for sid in changed:
            r = m.runners[sid]
            key = (mid, sid)
            pbb, pbl = last_book.get(key, (None, None))
            buy, sell, flat = _signed_trades(last_trd.get(key, {}), r.trd, pbb, pbl)
            last_trd[key] = dict(r.trd)
            backs, lays = _levels(r.atb, "atb"), _levels(r.atl, "atl")
            bb, bbs = backs[0] if backs else (None, 0.0)
            bl, bls = lays[0] if lays else (None, 0.0)
            last_book[key] = (bb, bl)
            row = {"t": t, "market": mid, "sid": sid, "to_off": to_off, "field": n_active, "book_back": book_back,
                   "venue": m.defn.get("venue"), "bb": bb, "bbs": bbs, "bl": bl, "bls": bls,
                   "back_depth": sum(s for _, s in backs), "lay_depth": sum(s for _, s in lays),
                   "buy": buy, "sell": sell, "flat": flat, "tv": r.tv, "ltp": r.ltp, "spn": r.spn}
            for i in range(1, LEVELS):
                row[f"bb{i + 1}s"] = backs[i][1] if len(backs) > i else 0.0
                row[f"bl{i + 1}s"] = lays[i][1] if len(lays) > i else 0.0
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["t"] = pd.to_datetime(out["t"], utc=True)
    return out.sort_values(["market", "sid", "t"], kind="mergesort").reset_index(drop=True)
