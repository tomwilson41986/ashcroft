"""The in-play price panel: every two-runner cricket Match Odds market, minute by minute, from Betfair BASIC files.

    python -m cricket.market.inplay_panel --root data/cricket/store

For each market (two runners, a winner, no "The Draw"), one row a minute from the minute it turned in play to its
last in-play suspension: each runner's last traded price standing at the minute (and how many minutes old it is),
the implied probability of runner A with the overround removed, whether the market's definition stood SUSPENDED
at that minute, and the result. BASIC writes the last traded price each minute only when it changes, and has no
ladder and no volume: a price here is a price that traded, not one that could be taken.

``holdout`` marks markets scheduled on or after 2026-10-01 (reports/cricket/research_log.md): analyses read them only
for a frozen strategy.

Output: ``cricket/eval/inplay_panel.parquet`` in the store.
"""

from __future__ import annotations

import argparse
import bz2
import datetime as dt
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from sources.common import Store

RAW = "betfair_historic/raw/cricket/"
HOLDOUT_FROM = "2026-10-01"

# Betfair's price ladder: from each lower edge, the tick size up to the next edge
LADDER = ((1.01, 0.01), (2.0, 0.02), (3.0, 0.05), (4.0, 0.1), (6.0, 0.2), (10.0, 0.5), (20.0, 1.0), (30.0, 2.0),
          (50.0, 5.0), (100.0, 10.0))
MAX_PRICE = 1000.0


def tick_up(price: float) -> float:
    """One tick longer: the step of the band the price sits in, from its lower edge."""
    step = [t for lo, t in LADDER if price >= lo - 1e-9][-1]
    return round(min(MAX_PRICE, price + step), 2)


def tick_down(price: float) -> float:
    """One tick shorter: the step of the band just below the price (at a band's edge, the band beneath)."""
    step = [t for lo, t in LADDER if price > lo + 1e-9]
    return round(max(1.01, price - step[-1]), 2) if step else 1.01


def _t(ms) -> dt.datetime:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc)


def read_market(files: list[bytes], market_id: str) -> dict:
    """One market's in-play start, its in-play suspensions and status changes, and every last-traded-price update."""
    out = {"inplay_at": None, "status": [], "ltp": {}}
    for raw in files:
        text = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
        for line in text.splitlines():
            if not line.strip():
                continue
            msg = json.loads(line)
            pt = _t(msg.get("pt", 0))
            for mc in msg.get("mc") or []:
                if mc.get("id") != market_id:
                    continue
                d = mc.get("marketDefinition")
                if d:
                    if d.get("inPlay") and out["inplay_at"] is None:
                        out["inplay_at"] = pt
                    out["status"].append((pt, d.get("status"), bool(d.get("inPlay"))))
                for rc in mc.get("rc") or []:
                    if "ltp" in rc:
                        out["ltp"][(pt, int(rc["id"]))] = float(rc["ltp"])   # files repeat messages: keyed
    return out


def market_grid(m: dict, sel_a: int, sel_b: int) -> pd.DataFrame | None:
    """The per-minute grid of one market, in play only."""
    if m["inplay_at"] is None:
        return None
    start = m["inplay_at"]
    inplay_status = sorted(x for x in m["status"] if x[2])
    susp = [t for t, s, _ in inplay_status if s in ("SUSPENDED", "CLOSED")]
    end = max(susp) if susp else None
    ups = sorted(m["ltp"].items())
    if end is None or not ups:
        return None
    minutes = int((end - start).total_seconds() // 60)
    rows = []
    last = {sel_a: None, sel_b: None}
    seen = {sel_a: None, sel_b: None}
    i = 0
    status_i, cur_status = 0, "OPEN"
    for k in range(minutes + 1):
        t = start + dt.timedelta(minutes=k)
        while i < len(ups) and ups[i][0][0] <= t:
            (pt, sid), p = ups[i]
            if sid in last:
                last[sid], seen[sid] = p, pt
            i += 1
        while status_i < len(inplay_status) and inplay_status[status_i][0] <= t:
            cur_status = inplay_status[status_i][1]
            status_i += 1
        if last[sel_a] is None or last[sel_b] is None:
            continue
        rows.append({"minute": k, "t": t, "ltp_a": last[sel_a], "ltp_b": last[sel_b],
                     "age_a": (t - seen[sel_a]).total_seconds() / 60, "age_b": (t - seen[sel_b]).total_seconds() / 60,
                     "suspended": cur_status != "OPEN"})
    if not rows:
        return None
    g = pd.DataFrame(rows)
    qa, qb = 1 / g.ltp_a, 1 / g.ltp_b
    g["p_a"] = qa / (qa + qb)
    g["overround"] = qa + qb
    return g


def build(store: Store, log=print) -> pd.DataFrame:
    keys = [k for k in store.listing("betfair_historic/") if k.startswith("betfair_historic/markets_cricket_")]
    b = pd.concat([store.get_parquet(k) for k in keys], ignore_index=True)
    mo = b[(b.market_type == "MATCH_ODDS") & b.runner_status.isin(["WINNER", "LOSER"])]
    g = mo.groupby("market_id")
    mo = mo[g.selection_id.transform("size").eq(2) & g.won.transform("sum").eq(1)]
    by_event = defaultdict(list)
    for k in store.listing(RAW):
        parts = k.split("/")
        if len(parts) >= 3:
            by_event[parts[-2]].append(k)
    out, skipped = [], defaultdict(int)
    for mid, m in mo.sort_values(["market_id", "sort_priority"]).groupby("market_id", sort=False):
        a, bb = m.iloc[0], m.iloc[1]
        files = [store.get(k) for k in by_event.get(str(a.event_id), [])]
        if not files:
            skipped["no raw files"] += 1
            continue
        grid = market_grid(read_market(files, mid), int(a.selection_id), int(bb.selection_id))
        if grid is None:
            skipped["no in-play prices"] += 1
            continue
        grid["market_id"] = mid
        grid["event_id"] = str(a.event_id)
        grid["event_name"] = a.event_name
        grid["market_time"] = a.market_time
        grid["runner_a"], grid["runner_b"] = a.runner_name, bb.runner_name
        grid["a_won"] = int(a.won)
        grid["ltp_close_a"], grid["ltp_close_b"] = a.ltp_close, bb.ltp_close
        grid["holdout"] = str(a.market_time) >= HOLDOUT_FROM
        out.append(grid)
    panel = pd.concat(out, ignore_index=True)
    log(f"{panel.market_id.nunique()} markets, {len(panel):,} market-minutes; skipped {dict(skipped)}")
    return panel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    a = ap.parse_args(argv)
    store = Store(root=a.root)
    panel = build(store)
    store.put_parquet("cricket/eval/inplay_panel.parquet", panel)
    d = panel[~panel.holdout]
    print(json.dumps({"markets": int(panel.market_id.nunique()), "discovery_markets": int(d.market_id.nunique()),
                      "holdout_markets": int(panel[panel.holdout].market_id.nunique()),
                      "minutes": int(len(panel)), "median_minutes_a_market":
                      float(panel.groupby("market_id").size().median()),
                      "share_fresh_both_le_1min": float(np.mean((d.age_a <= 1) & (d.age_b <= 1)))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
