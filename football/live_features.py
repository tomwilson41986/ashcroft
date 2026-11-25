"""The historic price path as the live feed would have seen it: Betfair's ticks sampled at the feed's polling times.

racing² football, 25 Nov 2026 (the owner: the back-to-lay strategy live, GBP 2 a bet, GBP 50 a day). The move model
was trained on the warehouse's price path, built from Betfair's BASIC files: a runner's last traded price about once
a minute while it trades. The live football feed (``football/exchange_feed.py``) reads a market every 30 minutes
until six hours before the off, then every 10, 5, 2 and 1 minutes (``CADENCE``), from 24 hours out at most (its
catalogue looks a day ahead). A model trained on the finer path would be fed a coarser one live.

So the model the live strategy uses is trained on the history thinned to the feed's times: for every runner, the last
traded price at each polling time (counted back from the scheduled kick-off with the feed's own cadence), kept as a
"tick" only when it changed since the previous poll. ``gold_path.build`` then computes the same path features from
those ticks as from the live feed's polls (``ticks_from_polls``): one definition, two sources, the same numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.exchange_feed import cadence_seconds

LOOKBACK_MIN = 1440.0                 # the feed's catalogue looks 24 hours ahead: no poll before that


def poll_times(ko_ms: int, first_ms: int, lookback_min: float = LOOKBACK_MIN) -> np.ndarray:
    """The feed's polling times for a market kicking off at ``ko_ms`` (ms), oldest first: counted back from the
    kick-off with the feed's cadence, no earlier than ``first_ms`` or ``lookback_min`` before the off."""
    start = max(int(first_ms), int(ko_ms - lookback_min * 60_000))
    out, t = [], int(ko_ms) - 1                       # the last poll before the off
    while t >= start:
        out.append(t)
        mins = (ko_ms - t) / 60_000.0
        t -= int(cadence_seconds(mins) * 1000)
    return np.array(out[::-1], dtype=np.int64)


def sample_runner(pt: np.ndarray, ltp: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(times, prices) of the polls at which the runner's last traded price had changed since the previous poll.
    ``pt`` sorted ascending (ms)."""
    idx = np.searchsorted(pt, grid, side="right") - 1
    ok = idx >= 0
    g, p = grid[ok], ltp[idx[ok]]
    if not len(g):
        return g, p
    keep = np.ones(len(p), dtype=bool)
    keep[1:] = p[1:] != p[:-1]
    return g[keep], p[keep]


def thin_ticks(tick: pd.DataFrame, kickoff_ms: dict) -> pd.DataFrame:
    """silver/tick rows (market_id, market_type, selection_id, handicap, pt in ms, seq, ltp, pre_off) thinned to the
    feed's polling times; the same columns back. ``kickoff_ms``: market_id -> scheduled kick-off (ms). Markets with no
    kick-off are left out."""
    t = tick[tick["pre_off"].astype(bool) & tick["market_id"].isin(kickoff_ms.keys())]
    t = t.sort_values(["market_id", "selection_id", "handicap", "pt", "seq"], kind="stable")
    out = []
    for (mid, sid, hc), g in t.groupby(["market_id", "selection_id", "handicap"], sort=False):
        pt, ltp = g["pt"].to_numpy(np.int64), g["ltp"].to_numpy(float)
        ko = kickoff_ms[mid]
        times, prices = sample_runner(pt, ltp, poll_times(ko, pt[0]))
        if not len(times):
            continue
        out.append(pd.DataFrame({"market_id": mid, "market_type": g["market_type"].iloc[0], "selection_id": sid,
                                 "handicap": hc, "pt": times, "seq": 0, "ltp": prices, "pre_off": True}))
    cols = ["market_id", "market_type", "selection_id", "handicap", "pt", "seq", "ltp", "pre_off"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def ticks_from_polls(books: pd.DataFrame, market_types: dict) -> pd.DataFrame:
    """The live feed's recorded polls (football/exchange_feed.py book rows: polled_utc, market_id, selection_id,
    last_traded, inplay, status) as ticks: one row a poll at which a runner's last traded price changed, before the
    market turned in play. ``market_types``: market_id -> market type."""
    b = books[(pd.to_numeric(books["inplay"], errors="coerce").fillna(0) == 0)
              & books["market_id"].astype(str).isin(market_types.keys())].copy()
    t = pd.to_datetime(b["polled_utc"], utc=True, errors="coerce")
    b["pt"] = (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)        # ms, whatever the resolution
    b["ltp"] = pd.to_numeric(b["last_traded"], errors="coerce")
    b = b[b["ltp"] > 1.0].sort_values(["market_id", "selection_id", "pt"])
    prev = b.groupby(["market_id", "selection_id"])["ltp"].shift(1)
    b = b[prev.isna() | (b["ltp"] != prev)]
    return pd.DataFrame({"market_id": b["market_id"].astype(str), "market_type": b["market_id"].astype(str).map(market_types),
                         "selection_id": b["selection_id"].astype("int64"), "handicap": 0.0, "pt": b["pt"],
                         "seq": 0, "ltp": b["ltp"], "pre_off": True}).reset_index(drop=True)
