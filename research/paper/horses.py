"""The horse paper desk (the owner, 4 Nov 2026: "Keep the current proven profitable strategy as live. Let's paper trade
any others that look promising; Backs, Lays, High Volume Trading each side of the line").

Each racing day, once its books and Betfair's price file are in S3, every strategy below is run on the day's recorded
books (the trader's own minute-by-minute reads and the recorder's) through the live trader's code (research/arms),
with no order placed anywhere. Each is scored as the arms are: backs and lays closed at the SP, after the non-runners'
reduction factors, 2% on the winnings, at the replay's scale (live has run at 0.80 of it). The ledger is kept in
s3://$PRED_BUCKET/paper/horses/: a day's rows (days/<day>.json) and fills (fills/<day>/<strategy>.csv.gz).

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
PREFIX = "paper/horses"
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


def run_strategy(D, name: str, draws: int) -> tuple[dict, pd.DataFrame]:
    from research.arms import quotes, sessions
    kind, policy, over = STRATEGIES[name]
    cfg = core.make_cfg(clv_draws=draws, **over)
    core.set_plan_cfg(core.make_cfg(clv_draws=draws), reset=False)
    pol = {**policy, "_D": D}
    t0 = core.session_start(D, "win")
    cls = {"rule": sessions.ArmSession, "lay": sessions.LaySession, "quote": quotes.QuoteSession}[kind]
    led, s, x = core.run_session(D, cls, cfg, pol, t0)
    if kind == "rule":
        r = core.score(led, D, name, s.state.settled_pnl, s.state.turnover >= cfg.limits.max_daily_turnover - 2)
        fills = led[(led.get("event") == "back")] if len(led) and "event" in led else pd.DataFrame()
    elif kind == "lay":
        r = core.score_side(s.lay_positions_frame(), D, name, side="lay")
        fills = pd.DataFrame(s.lay_rows)
    else:
        fills = s.fills_frame()
        r = quotes.score_quotes(fills, D, name)
    return r, fills


def day_job(day_iso: str, names: list[str], draws: int, write: bool) -> dict:
    day = date.fromisoformat(day_iso)
    res = {"day": day_iso, "rows": [], "errors": []}
    try:
        D = core.load_day(day, "win")
    except Exception:
        res["errors"].append(traceback.format_exc())
        return res
    if D is None:
        res["rows"].append({"day": day_iso, "note": "no books or no price file yet"})
        return res
    for name in names:
        tic = time.time()
        try:
            r, fills = run_strategy(D, name, draws)
        except Exception:
            res["errors"].append(f"{name}: {traceback.format_exc()}")
            continue
        r = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
        r.update({"arm": name, "kind": KIND[name], "forward": day >= FORWARD_FROM,
                  "seconds": round(time.time() - tic)})
        res["rows"].append(r)
        print({k: v for k, v in r.items() if k != "day"}, flush=True)
        if write and len(fills):
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as g:
                g.write(fills.to_csv(index=False).encode())
            _put(f"{PREFIX}/fills/{day_iso}/{name}.csv.gz", buf.getvalue())
    if write and res["rows"] and not res["errors"]:
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
    a = ap.parse_args(argv)
    names = list(STRATEGIES) if a.strategies == "all" else [s for s in a.strategies.split(",") if s]
    bad = [s for s in names if s not in STRATEGIES]
    if bad:
        ap.error(f"unknown strategies {bad}; known {list(STRATEGIES)}")
    if not a.report_only:
        if a.days == "pending":
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
                futs = {ex.submit(day_job, d.isoformat(), names, a.draws, not a.no_write): d for d in days}
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
