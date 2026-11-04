"""The early-lay forward test (the owner, 3 Nov 2026: "set up the forward test for the early lay rule"). Paper only:
nothing here can place an order; it scores the recorded books of each new day.

The rule, fixed here before its first day (reports/greyhound_model.md sections 22-24): in the first traded book of
each GB/IE greyhound win market, lay the runners our forecast says will drift, at 80% of the forecast SP (on Betfair's
ladder), left in the book until the off; matched when a later traded price reaches it; greened up at the market in
the runner's last book before the off (backed at the best back price there for the lay's stake times its price over
that price: the same profit whatever the result), or closed at the SP where that book has no price. The owner, 3 Nov
2026, before the first test day: "we want to trade out and take profit whenever we get the movement correct. I dont
care if the dog wins or not. We are trading the price move." (The first version, frozen at 11:41 UTC that day, closed
at the SP instead; it is recorded beside it as ``lay80_drift_sp``.) Staked as the horse trader stakes (a lay whose
liability wins GBP100, GBP2-50 a bet). It held on the
holdout of the limit-order study twice, on the 5 October model's prices (+1.92%, CLV +6.0%) and on the weigh-in model's
(+1.42%, CLV +9.1%), while the grid of rules as a whole sat at chance, so it is a lead, and this test decides it.

Recorded beside it, and judged by nothing: the same lay closed at the SP (the first version), matched only when a
later book offers the price (``through``), the short-priced version (section 23: mid under 3, lay to back), and laying
at the best back price now (``join``).

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
PRIMARY = {"mark": MARK, "order": "bid", "side": "lay", "param": 0.2, "gate": "drift", "fill": "traded",
           "exit": "last"}
SECONDARY = {
    "lay80_drift_sp": dict(PRIMARY, exit="sp"),
    "lay80_drift_through": dict(PRIMARY, fill="through"),
    "lay80_short_to_back": dict(PRIMARY, gate="short"),
    "join_drift_to_back": dict(PRIMARY, order="join", param=0.0),
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


# --------------------------------------------------------------------------------------------------------------------
# Today, race by race (the owner, 3 Nov 2026: "add the early lay to the running table")
# --------------------------------------------------------------------------------------------------------------------

#: the out-of-sample prices of the log-BSP live model (4 Nov 2026): a model fitted before the first recorded day,
#: every recorded day priced (greyhound.fastloop ``publish_pred``), for the forecast fit to read when that model is live
LOGBSP_PRICES = "greyhound/live/oos_prices_logbsp"


def day_fit(store, day: date, mark: str = MARK, prices: str | None = None) -> dict | None:
    """The forecast for ``day``: the conditional logit at the first traded book (or at ``mark``: off_5, off_3, off_1)
    fitted on every recorded day before it (the forward test's procedure), fitted once a day and kept in S3. ``prices``:
    our price from that S3 prefix (a model's out-of-sample day files) instead of the 5 Oct model's."""
    import numpy as np
    from research.greyhound import limit_study as ls, tradeout_study as ts
    tag = "" if not prices else "_" + prices.rstrip("/").split("/")[-1]
    key = f"{PREFIX}/fit{tag}/{day:%Y-%m-%d}.json" if mark == MARK else f"{PREFIX}/fit{tag}_{mark}/{day:%Y-%m-%d}.json"
    raw = store.get(key)
    if raw:
        f = json.loads(raw)
        return {**f, **{k: np.asarray(f[k], float) for k in ("beta", "mu", "sd")}}
    panel = ls.load(store, to.FIRST_BOOK_DAY, day - timedelta(days=1), source="retimed",
                    pred_dir=f"s3:{prices}" if prices else None)
    p = panel[(panel["mark"] == mark) & (panel.race_date.astype(str) < f"{day:%Y-%m-%d}")] if len(panel) else panel
    if not len(p) or p.race_date.nunique() < ls.MIN_TRAIN_DAYS:
        return None
    f = ts.fit_cl(p, ts.CL_SETS["micro"])
    out = {"kind": f["kind"], "cols": list(f["cols"]), "beta": f["beta"].tolist(), "mu": f["mu"].tolist(),
           "sd": f["sd"].tolist(), "book": float(f["book"]), "fit_days": int(p.race_date.nunique()),
           "fit_to": str(p.race_date.max()), "mark": mark, "prices": prices or "the 5 Oct model"}
    store.put(key, json.dumps(out).encode())
    return {**out, **{k: np.asarray(out[k], float) for k in ("beta", "mu", "sd")}}


def live_prices(store, day: date) -> str | None:
    """The prices the forecast must be fitted on to read today's live predictions: the log-BSP model's out-of-sample
    prices when the day's predictions came from that model, else the 5 Oct model's (None)."""
    import io
    raw = store.get(f"greyhound/live/predictions/{day:%Y-%m-%d}.csv")
    if not raw:
        return None
    head = pd.read_csv(io.BytesIO(raw), nrows=5, keep_default_na=False)
    if "model_objective" not in head or str(head.model_objective.iloc[0]) != "logbsp":
        return None
    named = str(head.oos_prices.iloc[0]).strip() if "oos_prices" in head else ""
    if named and named != LOGBSP_PRICES:
        # the model names its own out-of-sample prices; until they are published the log-BSP model's are the nearest
        # (an empty prefix would leave the early lay without a forecast)
        if store.listing(f"{named.rstrip('/')}/"):
            return named
        log.warning("%s has no out-of-sample prices yet: the forecast is fitted on %s", named, LOGBSP_PRICES)
    return LOGBSP_PRICES


def live_fit(store, day: date, mark: str = MARK) -> dict | None:
    """The forecast that matches today's live model (``live_prices``)."""
    return day_fit(store, day, mark=mark, prices=live_prices(store, day))


def today_rows(store, day: date) -> tuple[pd.DataFrame, str]:
    """Today's runners at the first traded book with their later books, settled races only (a BSP is needed): from the
    recorder's greyhound books (re-timed, as the forward test reads them) when they are in, else from the live-key
    record (greyhound/book_record.py), whose settled reads carry the BSP and the result."""
    from betfair_recorder import _read_day_csv
    from research.greyhound import limit_study as ls
    import io
    rec = _read_day_csv(day, "books_greyhound.csv")
    live = _read_day_csv(day, to.ENTRY_BOOKS["live"])
    finals = [b[b.source == "final"] for b in (rec, live) if b is not None and "source" in b]
    pred = store.get_parquet(f"{to.TRACK_PRED}/{day:%Y-%m-%d}.parquet")
    priced_by = "the 5 Oct forward test's model"
    if pred is None or not len(pred):
        # During the day the tracker has not priced today yet: the paper trader's own prices (greyhound/live.py, the
        # card-safe live model, refreshed hourly), with the winner from each settled race's book
        raw = store.get(f"greyhound/live/predictions/{day:%Y-%m-%d}.csv")
        if not raw:
            return pd.DataFrame(), "no prices for today yet"
        pred = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
        fin = pd.concat(finals, ignore_index=True) if finals else pd.DataFrame(
            columns=["market_id", "selection_id", "runner_status"])
        win = fin.assign(market_id=fin.market_id.astype(str),
                         selection_id=pd.to_numeric(fin.selection_id, errors="coerce"))
        win = win[win.runner_status.astype(str) == "WINNER"][["market_id", "selection_id"]].drop_duplicates()
        pred["selection_id"] = pd.to_numeric(pred.selection_id, errors="coerce")
        pred = pred.merge(win.assign(won=1), on=["market_id", "selection_id"], how="left")
        pred["won"] = pred.won.fillna(0).astype(int)
        settled = set(fin.market_id.astype(str)) if len(fin) else set()
        pred = pred[pred.market_id.astype(str).isin(settled)]     # a race counts once its result is in
        priced_by = "the paper trader's live model (the tracker prices today only after racing)"
    markets = _read_day_csv(day, "markets_greyhound.csv")
    if markets is None:
        markets = _read_day_csv(day, "markets_greyhound_live.csv")
    for source, entry in (("recorder, re-timed", rec), ("live-key record", live)):
        if entry is None or markets is None or not len(entry):
            continue
        closing = pd.concat([entry[entry.source != "final"]] + finals, ignore_index=True) if finals else entry
        panel = to.day_panel(pred, entry, markets, day, closing_books=closing, retimed=source.startswith("recorder"))
        panel = panel[panel["mark"] == MARK] if len(panel) else panel
        if len(panel):
            pre = to.pre_off_books(entry, markets, retimed=source.startswith("recorder"))
            pre = pre[pre.market_id.isin(set(panel.market_id))]
            return ls.attach_future(panel, ls.future_extremes(pre)), f"{source}; prices: {priced_by}"
    return pd.DataFrame(), "no settled race with books at the first traded mark yet"


def intraday(store, day: date) -> dict:
    """The primary rule on today's settled races, a row per race: the lays it would have left in the first book, those
    matched, their average price and the BSP, the CLV, how many of the dogs laid won, and the P&L greened up after
    2% commission on the race's net, with the running total. Paper only: no order is placed."""
    import numpy as np
    from research.greyhound import limit_study as ls, tradeout_study as ts
    out = {"day": f"{day:%Y-%m-%d}", "rule": PRIMARY, "markets": []}
    f = day_fit(store, day)
    if f is None:
        out["note"] = "no forecast fit (too few recorded days)"
        return out
    rows, source = today_rows(store, day)
    out["source"] = source
    if not len(rows):
        out["note"] = source
        return out
    q, book = ts.predict(rows, f)
    rows = rows.assign(f_bsp=1.0 / (book * np.asarray(q, float)))
    o = _select(ls.orders(rows), PRIMARY)
    if not len(o):
        out["note"] = "no lays placed on the settled races"
        return out
    info = rows.drop_duplicates("market_id").set_index("market_id")
    total = 0.0
    for mid, g in o.groupby("market_id"):
        m = g[g.filled]
        pnl = float(m.pnl.sum()) if len(m) else 0.0
        net = pnl - to.COMMISSION * max(pnl, 0.0)
        total += net
        out["markets"].append({
            "race_time": str(info.loc[mid, "race_time"]) if "race_time" in info else "", "market_id": mid,
            "track": str(info.loc[mid, "track"]) if "track" in info else "", "lays_placed": int(len(g)),
            "matched": int(len(m)),
            "avg_lay_price": round(float(m.price.mean()), 2) if len(m) else None,
            "greened_at": round(float(m.last_back1[m.closed_at_market].mean()), 2)
            if len(m) and m.closed_at_market.any() else None,
            "bsp": round(float(m.bsp.mean()), 2) if len(m) else None,
            "clv%": round(100 * float(m.clv.mean()), 2) if len(m) else None,
            "laid_dogs_won": int((m.won == 1).sum()) if len(m) else 0,
            "liability": round(float(m.risk.sum()), 2), "pnl": round(net, 2)})
    out["markets"].sort(key=lambda r: r["race_time"])
    run = 0.0
    for r in out["markets"]:
        run += r["pnl"]
        r["running"] = round(run, 2)
    mm = pd.DataFrame(out["markets"])
    out.update({"races": int(len(mm)), "lays_placed": int(mm.lays_placed.sum()), "matched": int(mm.matched.sum()),
                "liability": round(float(mm.liability.sum()), 2), "pnl": round(total, 2),
                "roi%": round(100 * total / float(mm.liability.sum()), 2) if mm.liability.sum() > 0 else None})
    return out


def intraday_markdown(r: dict) -> str:
    lines = ["### Early lay (the forward test's rule, paper): first traded, lay the forecast drifters at 80% of the "
             "forecast SP, matched by the traded price, greened up at the market before the off", ""]
    if not r.get("markets"):
        return "\n".join(lines + [r.get("note", "nothing settled yet")]) + "\n"
    lines += [f"Settled races {r['races']}; lays placed {r['lays_placed']}, matched {r['matched']}; liability "
              f"GBP{r['liability']:,.2f}; P&L **GBP{r['pnl']:+,.2f}** (ROI {r['roi%']}%). Books: {r.get('source')}.", "",
              "| Race (UK) | Track | Lays placed | Matched | Avg lay price | Greened at | BSP | CLV% | Laid dogs won | "
              "Liability | P&L | Running |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in r["markets"]:
        lines.append(f"| {m['race_time']} | {m['track']} | {m['lays_placed']} | {m['matched']} | {m['avg_lay_price']} | "
                     f"{m.get('greened_at')} | {m['bsp']} | {m['clv%']} | {m['laid_dogs_won']} | {m['liability']} | {m['pnl']} | "
                     f"{m['running']} |")
    return "\n".join(lines) + "\n"


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
             "SP, runners forecast to drift, matched by the traded price, greened up at the market before the off.", "",
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
        if frozen and frozen.get("primary") != PRIMARY and today < date.fromisoformat(frozen["first_test_day"]):
            # the rule revised before any test day was scored (the owner, 3 Nov 2026: green up, trade the move):
            # frozen again, the earlier version kept in the record
            before = frozen
            frozen = freeze(store, today)
            frozen["revised_from"] = {k: before.get(k) for k in ("frozen_utc", "primary", "first_test_day")}
            store.put(FROZEN_KEY, json.dumps(frozen, indent=1).encode())
            report["frozen"] = frozen
        elif frozen:
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
