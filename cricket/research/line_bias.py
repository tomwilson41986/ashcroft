"""H6 (reports/cricket/research_log.md): are Betfair's cricket over-line markets set too high? Back UNDER at the
closing line minus one run, even money, 5% commission on winnings; settled on Cricsheet.

    python -m cricket.research.line_bias --root data/cricket/store --types 2_INNING_6_OVR_LINE            # discovery
    python -m cricket.research.line_bias --root data/cricket/store \
        --types 1_INNING_6_OVR_LINE,2_INNING_6_OVR_LINE --holdout

A line market ``<k>_INNING_<n>_OVR_LINE`` settles on innings k's runs in its first n overs (the innings' runs if it
ended sooner). Only matched in-scope events (cricket/model/benter.py's matching) are settled: the BASIC files hold
no settled value.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

from cricket.model.benter import market_table, match_markets
from cricket.model.data import load
from cricket.research.inplay_tests import summary
from sources.common import Store

HOLDOUT_FROM = "2026-10-01"
COMMISSION = 0.05


def runs_in(t: dict, ids: set, innings: int, overs: int) -> pd.Series:
    d = t["deliveries"][t["deliveries"].match_id.isin(ids) & (t["deliveries"].innings == innings)]
    return d[d.over < overs].groupby("match_id").runs_total.sum()


def under_bets(rows: pd.DataFrame, stake: float = 10.0, side: str = "UNDER") -> pd.DataFrame:
    b = rows.copy()
    b["line_bet"] = b.line - 1 if side == "UNDER" else b.line + 1
    b["won"] = (b.result < b.line_bet if side == "UNDER" else b.result > b.line_bet).astype(int)
    b["pnl"] = np.where(b.won == 1, stake * (1 - COMMISSION), -stake)
    b["risk"] = stake
    b["price"] = 2.0
    return b


def rows_for(store: Store, types: list[str], holdout: bool) -> pd.DataFrame:
    t = load(store, ("matches", "innings", "deliveries"))
    keys = [k for k in store.listing("betfair_historic/") if k.startswith("betfair_historic/markets_cricket_")]
    bf = pd.concat([store.get_parquet(k) for k in keys], ignore_index=True)
    pre = store.get_parquet("cricket/eval/prematch.parquet")
    j, _ = match_markets(market_table(bf), pre)
    ev = bf[bf.market_id.isin(j.market_id)][["market_id", "event_id"]].drop_duplicates()
    j = j.merge(ev, on="market_id")[["match_id", "event_id"]]
    lines = bf[bf.market_type.isin(types) & bf.ltp_close.notna() & bf.event_id.isin(j.event_id)]
    lines = lines.drop_duplicates("market_id")[["market_id", "event_id", "market_type", "market_time", "ltp_close"]]
    lines = lines.merge(j, on="event_id")
    lines = lines[(lines.market_time >= HOLDOUT_FROM) == holdout]
    out = []
    for mtype, g in lines.groupby("market_type"):
        k, n = (int(x) for x in re.match(r"(\d)_INNING_(\d+)_OVR_LINE", mtype).groups())
        res = runs_in(t, set(g.match_id), k, n)
        g = g[g.match_id.isin(res.index)].assign(line=lambda x: x.ltp_close)
        out.append(g.assign(result=g.match_id.map(res).to_numpy()))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--types", required=True)
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args(argv)
    rows = rows_for(Store(root=a.root), a.types.split(","), a.holdout)
    out = {"holdout": a.holdout}
    for mtype, g in rows.groupby("market_type"):
        out[mtype] = {"markets": int(len(g)), "share_over_close": float((g.result > g.line).mean()),
                      "mean_result_minus_line": float((g.result - g.line).mean()),
                      "under_at_line_minus_1": summary(under_bets(g)),
                      "over_at_line_plus_1": summary(under_bets(g, side="OVER"))}
    if len(rows):
        out["all"] = {"markets": int(len(rows)), "under_at_line_minus_1": summary(under_bets(rows))}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())


def break_rows(store: Store, mtype: str, holdout: bool) -> pd.DataFrame:
    """Chase line markets priced at the innings break (H6 test 1b): the line standing 5 minutes after the break."""
    from collections import defaultdict
    import datetime as dt
    from cricket.model.innings_break import AFTER_MIN, FRESH_MIN, FIRST_INNINGS, read_event
    t = load(store, ("matches", "innings", "deliveries"))
    keys = [k for k in store.listing("betfair_historic/") if k.startswith("betfair_historic/markets_cricket_")]
    bf = pd.concat([store.get_parquet(k) for k in keys], ignore_index=True)
    pre = store.get_parquet("cricket/eval/prematch.parquet")
    j, _ = match_markets(market_table(bf), pre)
    ev = bf[bf.market_id.isin(j.market_id)][["market_id", "event_id"]].drop_duplicates()
    j = j.merge(ev, on="market_id")[["match_id", "event_id"]]
    lines = bf[(bf.market_type == mtype) & bf.event_id.isin(j.event_id)].drop_duplicates("market_id")
    lines = lines[(lines.market_time >= HOLDOUT_FROM) == holdout][["market_id", "event_id", "market_time"]]
    lines = lines.merge(j, on="event_id")
    by_event = defaultdict(list)
    for k in store.listing("betfair_historic/raw/cricket/"):
        by_event[k.split("/")[-2]].append(k)
    k, n = (int(x) for x in re.match(r"(\d)_INNING_(\d+)_OVR_LINE", mtype).groups())
    res = runs_in(t, set(lines.match_id), k, n)
    out = []
    for r in lines.itertuples():
        mk = read_event([store.get(x) for x in by_event[str(r.event_id)]])
        first = [min(s["suspended_inplay"]) for s in mk.values()
                 if s["type"] in FIRST_INNINGS and s["suspended_inplay"]]
        line_mk = mk.get(r.market_id)
        if not first or line_mk is None or r.match_id not in res.index:
            continue
        at = min(first) + dt.timedelta(minutes=AFTER_MIN)
        before = [(tt, p) for tt, _, p in line_mk["ltp"] if tt <= at]
        if not before or (at - before[-1][0]).total_seconds() / 60 > FRESH_MIN:
            continue
        out.append({"market_id": r.market_id, "match_id": r.match_id, "market_type": mtype,
                    "line": before[-1][1], "result": int(res[r.match_id])})
    return pd.DataFrame(out)


def unmatched_rows(store: Store, margin: float = 2.0, fresh_min: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """H7: 6-over lines of events not matched to Cricsheet, priced as S1 prices them, settled by the stand-in."""
    import datetime as dt
    from cricket.model.innings_break import AFTER_MIN, FIRST_INNINGS, FRESH_MIN, read_event
    from cricket.research.line_proxy import last_inplay_trade, raw_index
    keys = [k for k in store.listing("betfair_historic/") if k.startswith("betfair_historic/markets_cricket_")]
    bf = pd.concat([store.get_parquet(k) for k in keys], ignore_index=True)
    pre = store.get_parquet("cricket/eval/prematch.parquet")
    j, _ = match_markets(market_table(bf), pre)
    matched_events = set(bf[bf.market_id.isin(j.market_id)].event_id)
    lines = bf[bf.market_type.isin(["1_INNING_6_OVR_LINE", "2_INNING_6_OVR_LINE"])
               & ~bf.event_id.isin(matched_events)].drop_duplicates("market_id")
    idx = raw_index(store)
    out, why = [], {}
    for r in lines.itertuples():
        if r.market_type == "1_INNING_6_OVR_LINE":
            line = r.ltp_close
        else:
            mk = read_event([store.get(x) for x in idx[str(r.event_id)]])
            first = [min(s["suspended_inplay"]) for s in mk.values()
                     if s["type"] in FIRST_INNINGS and s["suspended_inplay"]]
            line = None
            if first and r.market_id in mk:
                at = min(first) + dt.timedelta(minutes=AFTER_MIN)
                before = [(tt, p) for tt, _, p in mk[r.market_id]["ltp"] if tt <= at]
                if before and (at - before[-1][0]).total_seconds() / 60 <= FRESH_MIN:
                    line = before[-1][1]
        if line is None or pd.isna(line):
            why["no S1 price"] = why.get("no S1 price", 0) + 1
            continue
        proxy, gap = last_inplay_trade(store, idx, r.event_id, r.market_id)
        if proxy is None or gap > fresh_min or abs(proxy - (line - 1)) < margin:
            why["not settleable by the stand-in"] = why.get("not settleable by the stand-in", 0) + 1
            continue
        out.append({"market_id": r.market_id, "event_name": r.event_name, "market_type": r.market_type,
                    "market_time": r.market_time, "line": line, "result": proxy,
                    "women": bool(re.search(r"\bW\b|Women", str(r.event_name)))})
    return pd.DataFrame(out), why
