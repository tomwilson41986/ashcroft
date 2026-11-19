"""The early lay on a model's own out-of-sample prices (the owner, 18 Nov 2026: "yes register and run that test on
stack_mlsitd"; ledger greyhound-lay-backtest-mlsitd-1118, registered before the run).

The rule is the forward test's (greyhound/laytest.py ``PRIMARY``: in the first traded book, lay the runners the
forecast says will drift, at 80% of the forecast SP), fixed before this run, read three ways:

- ``live_rule``: matched only when a later book offers our price (``through``), greened at the last book before the
  off: the fill the live reconciliation trusts, the version a live order would get;
- ``spclose``: the same fill, closed at the SP;
- ``traded_ref``: matched when a later traded price reaches it (the running table's optimistic fill), for reference.

The forecast SP is the limit study's conditional logit on the book and our price, refitted for each day on the days
before it only (research/greyhound/limit_study.py ``forecast``); our price is the model's published out-of-sample
day files (``--pred-dir s3:<prefix>``: a model fitted before 6 Oct 2026, every day after priced as served). The
books are the recorder's, re-timed. Only the matched orders are settled (an unmatched order is no position; ``settle_orders`` prices every order, so the
caller filters, as ``rule_summary`` and the forward test do). Each rule's return on the money at risk after commission on each market's net,
with its 90% interval (a bootstrap over days), and the deflated Sharpe of its per-race returns counted against every
rule of the limit study's grid (the family the early lay was chosen from).

    python -m research.greyhound.lay_backtest --pred-dir s3:greyhound/live/oos_prices_stack_mlsitd --out out/lay
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TAG = "greyhound-lay-backtest-mlsitd-1118"


def rules() -> dict:
    from greyhound.laytest import PRIMARY
    return {"live_rule": dict(PRIMARY, fill="through"),
            "spclose": dict(PRIMARY, fill="through", exit="sp"),
            "traded_ref": dict(PRIMARY)}


def race_returns(sel: pd.DataFrame) -> np.ndarray:
    """Each market's net over its money at risk (one period a race), after the commission ``tradeout.settle``
    charges on the market's positive net."""
    from greyhound import tradeout as to
    from research.greyhound.limit_study import KEYS
    m = to.settle(sel, list(KEYS))
    m = m[m.risk > 0]
    return (m.net / m.risk).to_numpy(float)


def evaluate(orders_by_day: dict[str, pd.DataFrame], n_grid: int) -> dict:
    from greyhound.laytest import _select
    from research.greyhound.limit_study import rule_table
    from research.overfit import dsr_check
    days = sorted(orders_by_day)
    allo = pd.concat([orders_by_day[d] for d in days], ignore_index=True) if days else pd.DataFrame()
    out = {"days": len(days), "first": days[0] if days else None, "last": days[-1] if days else None,
           "grid_rules": n_grid, "rules": {}}
    for name, rule in rules().items():
        sel = _select(allo, rule).assign(gate=rule["gate"]) if len(allo) else allo
        if not len(sel):
            out["rules"][name] = {"note": "no orders"}
            continue
        pos = sel[sel.filled.astype(bool)]              # an order the book never reached is not a position
        if not len(pos):
            out["rules"][name] = {"note": "no order matched", "orders": int(len(sel))}
            continue
        t = rule_table(pos, days=days)
        row = t.iloc[0].to_dict() if len(t) else {}
        r = race_returns(pos)
        dsr = dsr_check(r, n_trials=max(n_grid, 1))
        out["rules"][name] = {k: (row.get(k) if not isinstance(row.get(k), (np.floating, np.integer)) else row.get(k).item())
                              for k in ("trades", "per_day", "active_days", "risk", "net", "roi%", "roi_90%",
                                        "clv_mean%", "held_roi%")}
        out["rules"][name].update({"orders": int(len(sel)), "matched": int(sel.filled.sum()), "races": int(len(r)),
                                   "dsr": dsr})
    lr = out["rules"].get("live_rule", {})
    ci = lr.get("roi_90%")
    out["reading"] = {"live_rule_interval_above_zero": bool(ci and ci[0] > 0),
                      "live_rule_dsr_passed": bool(lr.get("dsr", {}).get("passed")),
                      "pass": bool(ci and ci[0] > 0 and lr.get("dsr", {}).get("passed"))}
    return out


def run(store, pred_dir: str, first: date, last: date, source: str) -> dict:
    from research.greyhound import limit_study as ls
    from greyhound.laytest import MARK
    panel = ls.load(store, first, last, source=source, pred_dir=pred_dir)
    if not len(panel):
        return {"note": "no data found: no panel for these prices and books"}
    p = panel[panel["mark"] == MARK]
    fc = ls.forecast(p)
    if not len(fc):
        return {"note": "no data found: too few days to fit the forecast"}
    fc = fc.assign(race_date=fc.race_date.astype(str))
    by_day, grid = {}, set()
    for d, rows in fc.groupby("race_date"):
        o = ls.orders(rows)
        if len(o):
            by_day[d] = o
            ks = [k for k in ls.KEYS if k != "gate"]
            gates = [c for c in o.columns if c.startswith("g_")]
            grid |= {t + (g,) for t in map(tuple, o[ks].drop_duplicates().itertuples(index=False, name=None))
                     for g in gates}
    res = evaluate(by_day, n_grid=len(grid))
    res.update({"pred_dir": pred_dir, "source": source, "panel_rows": int(len(p)), "tag": TAG})
    return res


def report_md(res: dict) -> str:
    L = [f"# The early lay on {res.get('pred_dir')}", "",
         f"Books: {res.get('source')}. Scored days {res.get('first')} to {res.get('last')} ({res.get('days')}); "
         f"deflated against {res.get('grid_rules')} rules. Ledger {TAG}.", "",
         "| Rule | Orders | Matched | Races | ROI % | 90% | CLV % | DSR |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res.get("rules", {}).items():
        if "note" in r:
            L.append(f"| {name} | {r['note']} | | | | | | |")
            continue
        L.append(f"| {name} | {r['orders']} | {r['matched']} | {r['races']} | {r['roi%']} | {r['roi_90%']} | "
                 f"{r['clv_mean%']} | {r['dsr'].get('dsr')} |")
    L += ["", f"Reading: {json.dumps(res.get('reading'))}"]
    return "\n".join(L)


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", required=True, action="append",
                    help="s3:<prefix> of a model's out-of-sample day files (repeatable)")
    ap.add_argument("--from", dest="first", default="2026-10-06")
    ap.add_argument("--to", dest="last", default="2026-11-11")
    ap.add_argument("--source", choices=["retimed", "recorder", "live"], default="retimed")
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/lay_backtest")
    a = ap.parse_args(argv)
    store = Store(root=a.store) if a.store else Store()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allres, md = {}, []
    for pd_ in a.pred_dir:
        r = run(store, pd_, date.fromisoformat(a.first), date.fromisoformat(a.last), a.source)
        allres[pd_] = r
        md.append(report_md(r) if "rules" in r else f"# {pd_}\n\n{r.get('note')}")
    (out / "report.json").write_text(json.dumps(allres, indent=1, default=str))
    (out / "report.md").write_text("\n\n".join(md))
    print(json.dumps(allres, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
