"""The horse paper desk (the owner, 4 Nov 2026: "Keep the current proven profitable strategy as live. Let's paper trade
any others that look promising; Backs, Lays, High Volume Trading each side of the line").

Each strategy is an agent with its own paper account (GBP5,000, its own day limit) and its own ledger. Each racing
day, once its books and Betfair's price file are in S3, all of them trade the day's recorded books together, through
the live trader's code (research/arms), with no order placed anywhere: one shared book (the owner, 4 Nov: "We need to
be careful that they dont conflict with each other e.g. taking the price for one, removes the price for others";
research/arms/core.py run_agents). The live trader's own fills are already out of the books; the live rule's replay
(the control) steps first each minute, as the live trader's orders came first; the lays and the quotes then step in an
order that turns each minute; whatever one takes is gone for the rest, and two quotes at one price queue one behind
the other. Each backs variant is an alternative to the live rule, so it trades a copy of the book in the live
trader's place (WORLDS): three copies of the rule on one book only took each other's leftovers.
Every fill is scored the same way (research/arms/quotes.py fill_pnl): hedged at the SP when matched, after the
non-runners' reduction factors, 2% on each market's net winnings, at the replay's scale (live has run at 0.80 of it).
The ledger, s3://$PRED_BUCKET/paper/horses/shared/: each agent's day (agents/<agent>/<day>/markets.csv, the
dashboard's table, and fills.csv.gz) and the day's rows for all of them (days/<day>.json).

The strategies, frozen on 4 Nov 2026 before the first forward day (5 Nov), from the arms' and the timing study's
reads (ledger arms-full-1103, timing-1104):
  live_rule       the live rule as it trades (trading/config_live.json): the control, and the replay's own check
  backs_floor50   the rule with a GBP50 floor on prices of 21 and over (arms: +GBP403 over 21 days, 13 of 21 ahead)
  backs_win1000   the rule to win GBP1,000 a horse, GBP500 a bet at most (timing: +GBP26 a day at GBP4,000)
  lays            the rule's mirror: lay where the market is 3% or more under the expected close, at 20 or under,
                  GBP400 at risk a horse, GBP4,000 a day, closed by a back at the SP (arms: +GBP39 a day)
  quotes_model3   both sides of the line around the closing model's fair price, 3% each side, GBP10 a quote,
                  GBP100 a runner a side, GBP4,000 matched a day (research/arms/quotes.py)
  quotes_mid2     both sides around the market's own mid, 2% each side, the same sizes: no model at all

The gate, set now, read after 20 forward racing days (5 Nov on), each strategy on its own; promotion to live only on
the owner's word:
  backs variants  the day's result after the cuts minus live_rule's, mean over the days with a paired bootstrap 90%
                  interval wholly above zero, and the variant ahead on at least 12 of the 20 days
  lays, quotes    the day's result after the cuts at 0.80, mean with a bootstrap 90% interval wholly above zero, up on
                  at least 12 of 20 days, and the worst day no worse than GBP-200 at 0.80
The days before 5 Nov (--backfill) are scored and shown apart: the backs and lays were chosen on them.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.arms import core  # noqa: E402

FORWARD_FROM = date(2026, 11, 5)
GATE_DAYS = 20
PREFIX = "paper/horses/shared"     # the agents on one book (4 Nov); the first, standalone run is not used
STRATEGIES = {
    #  name:          (session, policy, cfg overrides)
    "live_rule": ("rule", {}, {}),
    "backs_floor50": ("rule", {"floor_long": (21.0, 50.0)}, {}),
    "backs_win1000": ("rule", {}, {"target": 1000.0, "limits.max_stake": 500.0}),
    "lays": ("lay", {"lay_bar": 0.03, "max_lay_price": 20.0}, {}),
    "quotes_model3": ("quote", {"fair": "model", "margin": 0.03, "stake": 10.0, "max_side": 100.0,
                                "day_matched": 4000.0}, {}),
    "quotes_mid2": ("quote", {"fair": "mid", "margin": 0.02, "stake": 10.0, "max_side": 100.0,
                              "day_matched": 4000.0}, {}),
}
AGENTS = {
    #  name:          (title on the dashboard, what it does)
    "live_rule": ("Live rule, replay", "The live rule replayed on the same book: the control the others are judged by"),
    "backs_floor50": ("Backs, GBP50 floor", "The live rule with at least GBP50 on a horse at 21 or longer"),
    "backs_win1000": ("Backs, to win GBP1,000", "The live rule staking to win GBP1,000 a horse, GBP500 a bet at most"),
    "lays": ("Lays", "Lays a horse 3% or more under its expected SP, at 20 or under, GBP400 at risk; backed at the SP"),
    "quotes_model3": ("Both sides, model", "A resting back 3% above and a lay 3% below the model's expected SP, every "
                      "runner, GBP10 a quote; each fill closed at the SP"),
    "quotes_mid2": ("Both sides, market", "A resting back 2% above and a lay 2% below the market's mid, every runner, "
                    "GBP10 a quote; each fill closed at the SP"),
}
INITIALS = {"live_rule": "LR", "backs_floor50": "F5", "backs_win1000": "W1", "lays": "LY", "quotes_model3": "QM",
            "quotes_mid2": "QD"}
KIND = {"backs_floor50": "backs", "backs_win1000": "backs", "lays": "lays", "quotes_model3": "quotes",
        "quotes_mid2": "quotes", "live_rule": "control"}


def _put(key: str, body: bytes) -> None:
    core.s3().put_object(Bucket=core.bucket(), Key=key, Body=body)


def _keys(prefix: str) -> list[str]:
    out, token = [], None
    while True:
        kw = {"Bucket": core.bucket(), "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = core.s3().list_objects_v2(**kw)
        out += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        token = r.get("NextContinuationToken")


# Which agents share a book (the first shared run, 4 Nov, run 41815919147: with the two backs variants on the live
# rule's book the three copies of the rule chased the same horses, the live rule's replay fell from about GBP6,300 to
# GBP3,965 over 22 days, the GBP50 floor made GBP871 and the GBP1,000 stake lost GBP155: each took the others'
# leftovers). A backs variant is an alternative to the live rule, not an addition, so it trades a book of its own in
# the live trader's place (it stands in for it: the live trader's fills net against its own, as the live rule's replay
# does). The lays and the quotes are additions: they trade the live rule's book beside it, after it each minute.
WORLDS = {
    "live": ["live_rule", "lays", "quotes_model3", "quotes_mid2"],
    "backs_floor50": ["backs_floor50"],
    "backs_win1000": ["backs_win1000"],
}
STANDS_IN = {"live_rule", "backs_floor50", "backs_win1000"}     # trades in the live trader's place


def agent_specs(draws: int, names: list[str] | None = None) -> list[tuple]:
    from research.arms import quotes, sessions
    out = []
    for name, (kind, policy, over) in STRATEGIES.items():
        if names is not None and name not in names:
            continue
        cfg = core.make_cfg(clv_draws=draws, **over)
        cls = {"rule": sessions.ArmSession, "lay": sessions.LaySession, "quote": quotes.QuoteSession}[kind]
        out.append((name, cls, cfg, dict(policy), core.BALANCE, name in STANDS_IN))
    return out


def agent_fills(name: str, led: pd.DataFrame, s, D) -> pd.DataFrame:
    """An agent's fills as one table: ts, market_id, selection_id, side, price, matched, venue, off."""
    kind = STRATEGIES[name][0]
    cols = ["ts", "market_id", "selection_id", "side", "price", "matched", "venue", "off"]
    if kind == "rule":
        if led is None or led.empty or "event" not in led:
            return pd.DataFrame(columns=cols)
        b = led[(led.event == "back") & (pd.to_numeric(led.matched, errors="coerce") > 0)].copy()
        b["side"], b["price"] = "back", pd.to_numeric(b.avg_price, errors="coerce")
    elif kind == "lay":
        b = pd.DataFrame([r for r in s.lay_rows if r["event"] == "lay" and r["matched"] > 0])
        if b.empty:
            return pd.DataFrame(columns=cols)
        b["side"], b["price"] = "lay", b.avg_price
    else:
        b = s.fills_frame()
        if b.empty:
            return pd.DataFrame(columns=cols)
    b["matched"] = pd.to_numeric(b.matched, errors="coerce")
    starts = {m.market_id: m for m in D.markets}
    if "venue" not in b:
        b["venue"] = b.market_id.map(lambda m: starts[m].venue if m in starts else "")
    if "off" not in b:
        b["off"] = b.market_id.map(lambda m: starts[m].start.astimezone(core.UK).strftime("%H:%M") if m in starts else "")
    return b[cols].reset_index(drop=True)


def summary(name: str, fp: pd.DataFrame, D) -> dict:
    from research.arms.quotes import net_of_commission
    r = {"day": str(D.day), "arm": name, "kind": KIND[name], "forward": D.day >= FORWARD_FROM, "fills": 0,
         "horses": 0, "staked": 0.0, "back_staked": 0.0, "lay_staked": 0.0, "lay_liability": 0.0, "locked": 0.0,
         "clv": None, "result": 0.0, "result_reduced": 0.0}
    if fp is None or fp.empty:
        return r
    back = fp.side.eq("back")
    net = fp.groupby("market_id")[["pnl", "pnl_red"]].sum()
    r.update({"fills": int(len(fp)), "horses": int(fp.groupby(["market_id", "selection_id"]).ngroups),
              "staked": round(float(fp.matched.sum()), 2), "back_staked": round(float(fp.matched[back].sum()), 2),
              "lay_staked": round(float(fp.matched[~back].sum()), 2),
              "lay_liability": round(float((fp.matched[~back] * (fp.price[~back] - 1)).sum()), 2),
              "locked": round(float(fp.locked.sum()), 2),
              "result": round(float(net_of_commission(net.pnl).sum()), 2),
              "result_reduced": round(float(net_of_commission(net.pnl_red).sum()), 2)})
    r["clv"] = round(r["locked"] / r["staked"], 4) if r["staked"] else None
    return r


def markets_table(fp: pd.DataFrame) -> pd.DataFrame:
    """The dashboard's table for one agent's day (dashboard/traders.py MARKET_COLS): a row a market, settled, its
    P&L after the cuts and 2% on its net winnings, the value locked at the SP as its expected value, the CLV sums."""
    from research.arms.quotes import net_of_commission
    cols = ["market_id", "label", "off", "staked", "ev", "pnl", "settled", "open_liability",
            "clv_ws", "clv_s", "clv_n", "clv_sum", "clv_sumsq"]
    if fp is None or fp.empty:
        return pd.DataFrame(columns=cols)
    g = fp.assign(_cw=fp.clv * fp.matched, _c2=fp.clv ** 2).groupby("market_id")
    m = g.agg(label=("venue", "first"), off=("off", "first"), staked=("matched", "sum"), ev=("locked", "sum"),
              pnl_red=("pnl_red", "sum"), clv_ws=("_cw", "sum"), clv_s=("matched", "sum"), clv_n=("clv", "size"),
              clv_sum=("clv", "sum"), clv_sumsq=("_c2", "sum"))
    m["pnl"] = net_of_commission(m.pop("pnl_red"))
    m["settled"], m["open_liability"] = True, 0.0
    return m.reset_index()[cols].round(4)


def day_job(day_iso: str, names: list[str], draws: int, write: bool, intraday: bool = False) -> dict:
    """One racing day, every agent on one book. ``intraday``: the day so far on the books recorded to now, the races
    run settled on the recorder's BSPs (provisional); each agent's markets.csv is written, the day's ledger row is
    not (the night's run on Betfair's price file scores it for good)."""
    from research.arms import quotes
    day = date.fromisoformat(day_iso)
    res = {"day": day_iso, "rows": [], "errors": []}
    try:
        D = core.load_day(day, "win", intraday=intraday)
    except Exception:
        res["errors"].append(traceback.format_exc())
        return res
    if D is None:
        res["rows"].append({"day": day_iso, "note": "no books or no price file yet"})
        return res
    core.set_plan_cfg(core.make_cfg(clv_draws=draws), reset=False)
    tic = time.time()
    out = {}
    for world, members in WORLDS.items():
        want = [n for n in members if n in names or n == "live_rule" and world == "live"]
        if not want:
            continue
        specs = agent_specs(draws, want)
        for sp in specs:
            sp[3]["_D"] = D
        try:
            out.update(core.run_agents(D, specs, core.session_start(D, "win"),
                                       t_end=D.last_poll if intraday and D.provisional else None))
        except Exception:
            res["errors"].append(f"{world}: {traceback.format_exc()}")
            return res
    secs = round(time.time() - tic)
    for name, (led, s, x) in out.items():
        fills = agent_fills(name, led, s, D)
        fp = quotes.fill_pnl(fills, D)
        r = summary(name, fp, D)
        r["seconds_all"] = secs
        res["rows"].append(r)
        print({k: v for k, v in r.items() if k != "day"}, flush=True)
        if write:
            base = f"{PREFIX}/agents/{name}/{day_iso}"
            _put(f"{base}/markets.csv", markets_table(fp).to_csv(index=False).encode())
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as g:
                g.write((fp if len(fp) else fills).to_csv(index=False).encode())
            _put(f"{base}/fills.csv.gz", buf.getvalue())
    if write and res["rows"] and not res["errors"] and not D.provisional:
        _put(f"{PREFIX}/days/{day_iso}.json", json.dumps(res["rows"], default=str, indent=1).encode())
    return res


def pending_days(today: date, backfill_from: date | None) -> list[date]:
    """The racing days not yet in the ledger, from the forward start (or a backfill start) to two days ago (a day's
    price file reaches the archive the night after the next)."""
    done = {k.rsplit("/", 1)[-1][:10] for k in _keys(f"{PREFIX}/days/")}
    start = backfill_from or FORWARD_FROM
    last = today - timedelta(days=1)
    days = [start + timedelta(n) for n in range((last - start).days + 1)]
    return [d for d in days if d.isoformat() not in done]


def ledger() -> pd.DataFrame:
    rows = []
    for k in _keys(f"{PREFIX}/days/"):
        raw = core.get(k)
        if raw:
            rows += [r for r in json.loads(raw) if "arm" in r]
    return pd.DataFrame(rows)


def _boot(x: np.ndarray, n: int = 4000, seed: int = 7) -> tuple[float, float]:
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def gate(f: pd.DataFrame) -> pd.DataFrame:
    """Each strategy against its pre-registered gate, on the forward days."""
    fw = f[f.forward.astype(bool)]
    if fw.empty:
        return pd.DataFrame()
    piv = fw.pivot_table(index="day", columns="arm", values="result_reduced")
    out = []
    for name in STRATEGIES:
        if name not in piv or name == "live_rule":
            continue
        kind = KIND[name]
        if kind == "backs":
            if "live_rule" not in piv:
                continue
            d = (piv[name] - piv["live_rule"]).dropna().to_numpy()
            ahead = int((d > 0).sum())
            lo, hi = _boot(d)
            ok = len(d) >= GATE_DAYS and lo > 0 and ahead >= 12
            out.append({"strategy": name, "days": len(d), "mean_vs_live_a_day": round(float(d.mean()), 2),
                        "ci90": f"{lo:+.1f} to {hi:+.1f}", "days_ahead": ahead, "worst_day": None,
                        "gate": "PASSED" if ok else ("running" if len(d) < GATE_DAYS else "failed")})
        else:
            d = (0.8 * piv[name]).dropna().to_numpy()
            up = int((d > 0).sum())
            lo, hi = _boot(d)
            worst = float(d.min()) if len(d) else np.nan
            ok = len(d) >= GATE_DAYS and lo > 0 and up >= 12 and worst >= -200
            out.append({"strategy": name, "days": len(d), "mean_a_day_at_0.80": round(float(d.mean()), 2),
                        "ci90": f"{lo:+.1f} to {hi:+.1f}", "days_up": up, "worst_day": round(worst, 2),
                        "gate": "PASSED" if ok else ("running" if len(d) < GATE_DAYS else "failed")})
    return pd.DataFrame(out)


def report(f: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if f.empty:
        print("the paper ledger is empty")
        return
    f = f.copy()
    f["forward"] = f.forward.astype(bool)
    for label, sub in (("forward (from 5 Nov, the gate's days)", f[f.forward]),
                       ("before 5 Nov (the backs and lays were chosen on these days)", f[~f.forward])):
        if sub.empty:
            continue
        g = sub.groupby("arm").agg(days=("day", "nunique"), result_reduced=("result_reduced", "sum"),
                                   up=("result_reduced", lambda v: int((v > 0).sum())))
        for c in ("staked", "back_staked", "lay_staked", "liability", "lay_liability", "fills"):
            if c in sub:
                g[c] = sub.groupby("arm")[c].sum(min_count=1)
        g["a_day_reduced"] = g.result_reduced / g.days
        g["a_day_at_0.80"] = 0.8 * g.a_day_reduced
        print(f"\n== the paper desk, {label}: after the cuts, scale 1.0 (live has run at 0.80)")
        print(g.reindex([a for a in STRATEGIES if a in g.index]).round(2).to_string())
        print("\n   by day (result after the cuts)")
        print(sub.pivot_table(index="arm", columns="day", values="result_reduced").reindex(
            [a for a in STRATEGIES if a in set(sub.arm)]).round(0).to_string())
    gt = gate(f)
    if len(gt):
        print(f"\n== the gate (pre-registered 4 Nov; read after {GATE_DAYS} forward days; the owner decides)")
        print(gt.to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", default="pending", help="'pending' (the days not yet in the ledger), a..b, or a list")
    ap.add_argument("--backfill-from", default=None, help="with pending: also the days from this one (YYYY-MM-DD)")
    ap.add_argument("--strategies", default="all")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--max-minutes", type=float, default=300.0)
    ap.add_argument("--no-write", action="store_true", help="print only, keep nothing in S3")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--intraday", action="store_true",
                    help="today so far, as if live: the recorder's books to now, the races run settled on its BSPs")
    a = ap.parse_args(argv)
    names = list(STRATEGIES) if a.strategies == "all" else [s for s in a.strategies.split(",") if s]
    bad = [s for s in names if s not in STRATEGIES]
    if bad:
        ap.error(f"unknown strategies {bad}; known {list(STRATEGIES)}")
    if not a.report_only:
        if a.intraday:
            from zoneinfo import ZoneInfo
            days = [datetime.now(ZoneInfo("Europe/London")).date()]
        elif a.days == "pending":
            bf = date.fromisoformat(a.backfill_from) if a.backfill_from else None
            days = pending_days(datetime.now(timezone.utc).date(), bf)
        elif ".." in a.days:
            x, y = (date.fromisoformat(v) for v in a.days.split(".."))
            days = [x + timedelta(n) for n in range((y - x).days + 1)]
        else:
            days = [date.fromisoformat(v) for v in a.days.split(",") if v]
        print(f"the paper desk: {names} on {len(days)} days {[d.isoformat() for d in days]}", flush=True)
        tic = time.time()
        deadline = tic + a.max_minutes * 60
        computed = []
        if days:
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                futs = {ex.submit(day_job, d.isoformat(), names, a.draws, not a.no_write, a.intraday): d for d in days}
                try:
                    for fut in as_completed(futs, timeout=max(60.0, deadline - time.time())):
                        r = fut.result()
                        computed += [row for row in r["rows"] if "arm" in row]
                        for e in r["errors"]:
                            print(f"!! {r['day']}: {e}", flush=True)
                        for row in r["rows"]:
                            if "note" in row:
                                print(f"-- {row['day']}: {row['note']}", flush=True)
                        print(f"-- {futs[fut]} done ({(time.time() - tic) / 60:.0f} min in)", flush=True)
                except TimeoutError:
                    print("!! the time budget ran out; the rest wait for the next run", flush=True)
                    for fut in futs:
                        fut.cancel()
                    ex.shutdown(wait=False, cancel_futures=True)
        if a.no_write:
            report(pd.DataFrame(computed))
            return 0
    report(ledger())
    return 0


if __name__ == "__main__":
    sys.exit(main())
