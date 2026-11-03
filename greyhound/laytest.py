"""The early-lay forward test (the owner, 3 Nov 2026: "set up the forward test for the early lay rule"). Paper only:
nothing here can place an order; it scores the recorded books of each new day.

The rule, fixed here before its first day (reports/greyhound_model.md sections 22-24): in the first traded book of
each GB/IE greyhound win market, lay the runners our forecast says will drift, at 80% of the forecast SP (on Betfair's
ladder), left in the book until the off; matched when a later traded price reaches it; closed at the SP as the horse
trader closes. Staked as the horse trader stakes (a lay whose liability wins GBP100, GBP2-50 a bet). It held on the
holdout of the limit-order study twice, on the 5 October model's prices (+1.92%, CLV +6.0%) and on the weigh-in model's
(+1.42%, CLV +9.1%), while the grid of rules as a whole sat at chance, so it is a lead, and this test decides it.

Recorded beside it, and judged by nothing: the same lay matched only when a later book offers the price (``through``),
the same greened up in the last book (lay to back), the short-priced version (section 23: mid under 3, lay to back),
and laying at the best back price now (``join``).

The procedure (research/greyhound/limit_study.py, unchanged): our price is the 5 October forward test's model, as
the greyhound tracker prices each day (``greyhound/track/pred``); the forecast SP is the study's conditional logit on
the first traded book and our price, refitted for each test day on every recorded day before it (6 October on); the
books are the recorder's, re-timed. A day is scored once its books have settled.

The gate, fixed now: judged after READ_DAYS test days and MIN_TRADES primary trades, not before. PASS when the
primary's return over every test day has a 90% interval (a bootstrap over days) above zero and its mean CLV is above
zero. A PASS is not a licence to stake: the size matched is unknown until the live-key books with traded volume have
been read (greyhound/book_record.py, from 3 Nov 2026).

    python -m greyhound.laytest --freeze --days 3 --summary --out out/greyhound_laytest.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from greyhound import tradeout as to

log = logging.getLogger(__name__)

PREFIX = "greyhound/laytest"
FROZEN_KEY = f"{PREFIX}/frozen.json"
START = "2026-11-04"                        # the first test day at the earliest: no book of it existed at the freeze
READ_DAYS = 20
MIN_TRADES = 1000
MIN_COVERAGE = 0.9
REDO_DAYS = 3
MARK = "first_traded"
PRIMARY = {"mark": MARK, "order": "bid", "side": "lay", "param": 0.2, "gate": "drift", "fill": "traded", "exit": "sp"}
SECONDARY = {
    "lay80_drift_through": dict(PRIMARY, fill="through"),
    "lay80_drift_to_back": dict(PRIMARY, exit="last"),
    "lay80_short_to_back": dict(PRIMARY, gate="short", exit="last"),
    "join_drift_sp": dict(PRIMARY, order="join", param=0.0),
}
ORDERS_KEPT = (("bid", 0.2), ("join", 0.0))   # every gate, fill and exit of these two is kept in the ledger


def _rule_key(r: dict) -> tuple:
    from research.greyhound.limit_study import KEYS
    return tuple(r[k] for k in KEYS)


def ledger(panel: pd.DataFrame, days: list[str]) -> dict[str, pd.DataFrame]:
    """Each test day's orders (the kept orders of every runner in the first traded book, matched or not), from the
    walk-forward forecast: each day forecast by a fit on the recorded days before it only."""
    from research.greyhound import limit_study as ls
    p = panel[panel["mark"] == MARK]
    fc = ls.forecast(p)
    out = {}
    if not len(fc):
        return out
    fc = fc.assign(race_date=fc.race_date.astype(str))
    for d in days:
        rows = fc[fc.race_date == d]
        if not len(rows):
            continue
        o = ls.orders(rows)
        keep = pd.Series(False, index=o.index)
        for order, param in ORDERS_KEPT:
            keep |= (o.order == order) & (o.param == param)
        out[d] = o[keep].reset_index(drop=True)
    return out


def freeze(store, today: date) -> dict:
    """The rule, the procedure and the gate, written once. The first test day is START when the freeze runs on or
    before it (the rule is fixed in this file, committed on 3 Nov 2026, before any book of START existed, and no day is
    scored before it has settled), else tomorrow."""
    start = date.fromisoformat(START)
    first = start if today <= start else today + timedelta(days=1)
    out = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "first_test_day": str(first),
           "primary": PRIMARY, "secondary": SECONDARY, "read_days": READ_DAYS, "min_trades": MIN_TRADES,
           "gate": "PASS when the primary's 90% interval (bootstrap over test days) is above zero and its mean CLV "
                   "is above zero; judged only after read_days test days and min_trades primary trades",
           "price": "the 5 October forward test's model (greyhound/track/pred)",
           "forecast": "research.greyhound.limit_study.forecast at the first traded book, refitted for each test day "
                       "on every recorded day before it",
           "books": "the recorder's, re-timed (research.greyhound.limit_study.load, source retimed)",
           "code": __import__("os").environ.get("GITHUB_SHA", "local")}
    store.put(FROZEN_KEY, json.dumps(out, indent=1).encode())
    return out


def track(store, frozen: dict, days: list[date]) -> list[dict]:
    """Score the given test days: the panel from the first recorded day to the last of them read once, each day's
    orders to its ledger with its coverage (the share of the day's priced races in the first traded panel)."""
    from research.greyhound import limit_study as ls
    first = date.fromisoformat(frozen["first_test_day"])
    for d in days:
        if d < first:
            raise ValueError(f"{d} is before the first test day {first}")
    if not days:
        return []
    panel = ls.load(store, to.FIRST_BOOK_DAY, max(days), source="retimed")
    ds = [f"{d:%Y-%m-%d}" for d in days]
    books = ledger(panel, ds) if len(panel) else {}
    out = []
    for d in ds:
        pred = store.get_parquet(f"{to.TRACK_PRED}/{d}.parquet")
        priced = int(pred.dropna(subset=["market_id"]).race_id.nunique()) if pred is not None and len(pred) else 0
        o = books.get(d)
        if o is None or not len(o):
            out.append({"day": d, "note": "no first traded panel yet (prices or settled books missing)",
                        "races_priced": priced})
            continue
        races = int(o.market_id.nunique())
        store.put_parquet(f"{PREFIX}/days/{d}.parquet", o)
        cov = round(races / priced, 3) if priced else 0.0
        store.put(f"{PREFIX}/days/{d}.json", json.dumps({"day": d, "races_priced": priced, "races": races,
                                                         "coverage": cov}).encode())
        prim = _select(o, PRIMARY)
        out.append({"day": d, "races": races, "coverage": cov, "primary_orders": int(len(prim)),
                    "primary_matched": int(prim.filled.sum())})
    return out


def _select(o: pd.DataFrame, rule: dict) -> pd.DataFrame:
    m = (o["mark"] == rule["mark"]) & (o.order == rule["order"]) & (o.side == rule["side"]) & \
        (o.param == rule["param"]) & (o.fill == rule["fill"]) & (o["exit"] == rule["exit"])
    g = f"g_{rule['gate']}"
    return o[m & o[g]] if g in o else o[m]


def _settled(store, day: date, today: date) -> bool:
    ds = f"{day:%Y-%m-%d}"
    if store.get_parquet(f"{PREFIX}/days/{ds}.parquet") is None:
        return False
    raw = store.get(f"{PREFIX}/days/{ds}.json")
    cov = json.loads(raw).get("coverage", 0.0) if raw else 0.0
    return cov >= MIN_COVERAGE or (today - day).days > REDO_DAYS


def summary(store, frozen: dict) -> dict:
    """The test so far over every test day held: the primary and the recorded variants (the limit study's measures),
    and the gate, judged only once READ_DAYS days and MIN_TRADES primary trades are in."""
    from research.greyhound import limit_study as ls
    first = frozen["first_test_day"]
    keys = sorted(k for k in store.listing(f"{PREFIX}/days/") if k.endswith(".parquet") and k[-18:-8] >= first)
    days = [k[-18:-8] for k in keys]
    out = {"first_test_day": first, "days": len(days), "primary": frozen["primary"], "rules": {}}
    parts = [store.get_parquet(k) for k in keys]
    o = pd.concat([x for x in parts if x is not None and len(x)], ignore_index=True) if parts else pd.DataFrame()
    if not len(o):
        out["gate"] = f"not yet judged: 0 of {READ_DAYS} days"
        return out
    t = ls.rule_summary(o.assign(race_date=o.race_date.astype(str)), days)
    rules = {"primary": frozen["primary"], **frozen["secondary"]}
    for name, r in rules.items():
        row = t
        for k, v in r.items():
            row = row[row[k] == v]
        out["rules"][name] = json.loads(row.iloc[0].to_json()) if len(row) else None
    p = out["rules"]["primary"] or {}
    n_trades = int(p.get("trades") or 0)
    if len(days) < READ_DAYS or n_trades < MIN_TRADES:
        out["gate"] = f"not yet judged: {len(days)} of {READ_DAYS} days, {n_trades} of {MIN_TRADES:,} primary trades"
    else:
        lo = (p.get("roi_90%") or [None])[0]
        clv = p.get("clv_mean%")
        out["gate"] = "PASS" if lo is not None and lo > 0 and clv is not None and clv > 0 else "FAIL"
    return out


def markdown(s: dict) -> str:
    cols = ["orders", "filled_pct", "trades", "per_day", "roi%", "roi_90%", "clv_mean%", "clv_unmatched%", "held_roi%"]
    lines = [f"## Greyhound early-lay forward test (from {s['first_test_day']})", "",
             f"Test days: {s['days']}. Gate: **{s.get('gate')}**. Primary: first traded, lay at 80% of the forecast "
             "SP, runners forecast to drift, matched by the traded price, closed at the SP.", "",
             "| Rule | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in (s.get("rules") or {}).items():
        r = r or {}
        lines.append(f"| {name} | " + " | ".join(str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="the sources store root (default S3)")
    ap.add_argument("--freeze", action="store_true", help="fix the rule, the procedure and the gate, once")
    ap.add_argument("--days", type=int, default=0, help="score the last N days (to yesterday, UK) not yet settled")
    ap.add_argument("--date", default=None, help="score this one test day again")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=None, help="write the report JSON here (and markdown beside it)")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    today = uk_today()
    report = {}
    raw = store.get(FROZEN_KEY)
    frozen = json.loads(raw) if raw else None
    if a.freeze:
        if frozen:
            report["frozen"] = "already frozen: " + frozen["frozen_utc"]
        else:
            frozen = freeze(store, today)
            report["frozen"] = frozen
    if not frozen:
        ap.error("not frozen: run --freeze first")
    first = date.fromisoformat(frozen["first_test_day"])
    days = []
    if a.date:
        days = [date.fromisoformat(a.date)]
        if days[0] < first:
            ap.error(f"{a.date} is before the first test day {first}")
    elif a.days:
        days = [d for d in (today - timedelta(days=i) for i in range(a.days, 0, -1)) if d >= first]
        days = [d for d in days if not _settled(store, d, today)]
    if days:
        report["days"] = track(store, frozen, days)
    s = summary(store, frozen)
    store.put(f"{PREFIX}/summary.json", json.dumps(s, indent=1, default=str).encode())
    report["summary"] = s
    txt = json.dumps(report, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
        Path(a.out).with_suffix(".md").write_text(markdown(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
