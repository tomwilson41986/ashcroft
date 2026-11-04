"""H5 (reports/cricket/research_log.md): the simulator's pre-match runs distribution against Betfair's first-innings
runs lines (the 20-over and 6-over line markets' closing lines).

    python -m cricket.research.runs_line --root data/cricket/store --work data/cricket/eval            # discovery
    python -m cricket.research.runs_line --root data/cricket/store --work data/cricket/eval --holdout  # frozen, once

A line market is even money on the result against the line; the line here is the last value traded before the off.
Bets are placed a run worse than that line (over at line + 1, under at line - 1) and pay 5% commission on winnings.
Settled on Cricsheet: the first innings' runs (20-over line) and its runs in overs 1-6 (6-over line).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from cricket.model import simulator as sim
from cricket.model.benter import market_table, match_markets
from cricket.model.data import load
from cricket.model.features import build
from cricket.model.outcomes import extras_runs_distribution
from cricket.model.prematch import start_states
from cricket.research.inplay_tests import summary
from sources.common import Store

LINES = {"1_INNING_20_OVR_LINE": "total", "1_INNING_6_OVR_LINE": 36}
HOLDOUT_FROM = "2026-10-01"
COMMISSION = 0.05


def actuals(t: dict, ids: set) -> pd.DataFrame:
    d = t["deliveries"][t["deliveries"].match_id.isin(ids) & (t["deliveries"].innings == 1)]
    six = d[d.over < 6].groupby("match_id").runs_total.sum().rename("runs_6")
    tot = t["innings"][(t["innings"].innings == 1) & t["innings"].match_id.isin(ids)].set_index("match_id")["runs"]
    return pd.concat([tot.rename("runs_total"), six], axis=1).reset_index().rename(columns={"index": "match_id"})


def model_runs(t: dict, match_ids: list[str], work: Path, paths: int) -> pd.DataFrame:
    f, br, bw, prior = build(t, since="2012-01-01", pop_before="2026-01-01")
    b = xgb.Booster()
    b.load_model(work / "prematch_model_2026.ubj")
    ex_v, ex_p = extras_runs_distribution(f[pd.to_datetime(f.start_date) < "2026-01-01"])
    s = start_states(f, 2026)
    s = s[s.match_id.isin(match_ids)].reset_index(drop=True)
    ids = set(s.match_id)
    batch = sim.prepare(s, t["deliveries"][t["deliveries"].match_id.isin(ids)],
                        t["players"][t["players"].match_id.isin(ids)], br, bw, prior)
    out = sim.first_innings_runs(b, batch, paths, ex_v, ex_p, marks=(36,), seed=2026)
    return pd.DataFrame({"match_id": s.match_id, "paths_total": list(out["total"]), "paths_36": list(out[36])})


def line_bets(rows: pd.DataFrame, edge: float = 0.05, stake: float = 10.0) -> pd.DataFrame:
    """rows: match_id, market_id, line, p_over (the model's P(result > line)), result."""
    over = rows.p_over >= 0.5 + edge
    under = rows.p_over <= 0.5 - edge
    b = rows[over | under].copy()
    b["side"] = np.where(b.p_over >= 0.5, "OVER", "UNDER")
    b["line_bet"] = np.where(b.side == "OVER", b.line + 1, b.line - 1)
    b["won"] = np.where(b.side == "OVER", b.result > b.line_bet, b.result < b.line_bet).astype(int)
    b["stake"] = stake
    b["pnl"] = np.where(b.won == 1, stake * (1 - COMMISSION), -stake)
    b["risk"] = stake
    b["price"] = 2.0
    return b


def run(store: Store, work: Path, paths: int, holdout: bool) -> dict:
    t = load(store)
    keys = [k for k in store.listing("betfair_historic/") if k.startswith("betfair_historic/markets_cricket_")]
    bf = pd.concat([store.get_parquet(k) for k in keys], ignore_index=True)
    pre = store.get_parquet("cricket/eval/prematch.parquet")
    j, _ = match_markets(market_table(bf), pre)
    ev = bf[bf.market_id.isin(j.market_id)][["market_id", "event_id"]].drop_duplicates()
    j = j.merge(ev, on="market_id")[["match_id", "event_id"]]
    lines = bf[bf.market_type.isin(list(LINES)) & bf.ltp_close.notna() & bf.event_id.isin(j.event_id)]
    lines = lines.drop_duplicates("market_id")[["market_id", "event_id", "market_type", "market_time", "ltp_close"]]
    lines = lines.merge(j, on="event_id")
    lines = lines[(lines.market_time >= HOLDOUT_FROM) == holdout]
    mr = model_runs(t, sorted(set(lines.match_id)), work, paths).set_index("match_id")
    act = actuals(t, set(lines.match_id)).set_index("match_id")
    out = {"holdout": holdout}
    for mtype, mark in LINES.items():
        rows = lines[lines.market_type == mtype].copy()
        rows = rows[rows.match_id.isin(mr.index) & rows.match_id.isin(act.index)]
        col = "paths_total" if mark == "total" else "paths_36"
        rows["line"] = rows.ltp_close
        rows["p_over"] = [float((np.asarray(mr.loc[m, col]) > ln).mean()) for m, ln in zip(rows.match_id, rows.line)]
        rows["model_median"] = [float(np.median(mr.loc[m, col])) for m in rows.match_id]
        rows["result"] = act.loc[rows.match_id, "runs_total" if mark == "total" else "runs_6"].to_numpy()
        res = {"markets": int(len(rows)),
               "mae_line": float(np.abs(rows.result - rows.line).mean()) if len(rows) else None,
               "mae_model_median": float(np.abs(rows.result - rows.model_median).mean()) if len(rows) else None,
               "share_over_line": float((rows.result > rows.line).mean()) if len(rows) else None}
        for e in (0.05, 0.03, 0.08):
            res[f"edge_{e}"] = summary(line_bets(rows, e)) if len(rows) else {"bets": 0}
        out[mtype] = res
        store.put_parquet(f"cricket/eval/runs_line_{mtype}_{'holdout' if holdout else 'discovery'}.parquet",
                          rows.drop(columns=[]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--work", default="data/cricket/eval")
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(run(Store(root=a.root), Path(a.work), a.paths, a.holdout), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
