"""The stream strategy lab (read-only research; the owner, 5 Nov 2026: "just keep an open mind. Our goal is to build a
profitable strategy, i dont care which one it is"). Rules fixed here before their results are seen, scored on every
day the Exchange Stream recorder holds (greyhound/stream_record.py), so each new day is out of sample for all of them.

Each market's book is rebuilt from the stream (research.greyhound.stream_replay) and read at fixed checkpoints before
the scheduled off (``CHECKS`` seconds): each runner's best back and lay with their sizes, the last traded price and
Betfair's projected SP (near). The settled definition gives the SP and the result.

Every rule takes the price on offer at a checkpoint (a back at the best back price, a lay at the best lay price,
GBP2 or more shown there), so the fill is the book's own: no queue is assumed. Each closes at the SP: a back at B
greened by a lay at the SP gives B / SP - 1 a unit staked, a lay at L 1 - L / SP, whatever the result; commission
on a gain. Every rule is scored on two targets (the owner, 6 Nov 2026: "let's always test against two targets; 1.
BFSP 2. Win - winning the bet"): closed at the SP as above, and the same bet left to run to the result (a back wins
B - 1 or loses 1, a lay loses L - 1 or wins 1). The families (``RULES``):
- ``gap``: Betfair's projected SP against the price on offer: back where B / spn - 1 > m, lay where 1 - L / spn > m;
- ``mom``: follow the move since 180 s out: back what has shortened by x or more (mid), lay what has drifted;
- ``rev``: against it: back the drifters, lay the shorteners;
- ``fav_lay``: lay the favourite at 10 s (a control: it should lose about the spread).
Scored by day and pooled over days, with a 90% interval from a bootstrap over races; a rule is a candidate only after
``MIN_DAYS`` days with its pooled lower bound above zero, and the count of rules is in every report (about one in
twenty clears a one-sided 90% bound by chance).

The late family (registered 10 Nov 2026; reports/greyhound_model_review.md): races go off a median 22 s after the
scheduled time, so the rules above act earlier than their names say and never in the last stretch. The same gap,
momentum and reversal rules are also decided at ``LATE_DECIDE``: 0, 10 and 20 s AFTER the scheduled off, wherever the
market is still open then (rule names ``..._late0_``, ``_late10_``, ``_late20_``). They are their own registration:
counted only on days after ``REGISTERED_LATE``, and the first family's rules and counts are untouched.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from research.greyhound import multitest as mt, stream_replay as sr

log = logging.getLogger(__name__)

CHECKS = (180, 120, 60, 30, 10)
DECIDE = (60, 30, 10)
LATE_DECIDE = (0, -10, -20)                 # seconds before the scheduled off: 0, 10 and 20 s after it
ALL_CHECKS = CHECKS + LATE_DECIDE
REGISTERED_LATE = "2026-11-10"
late_name = lambda c: c if c > 0 else f"late{-c}"
GAPS = (0.03, 0.06, 0.10)
MOVES = (0.05, 0.10)
MIN_SIZE = 2.0
from greyhound.costs import COMMISSION as COMM   # the account's rate (greyhound/costs.py)
MIN_DAYS = 10
REGISTERED = "2026-11-05"


def _mid(b, bs, l, ls):
    if b and l and l / b < 1.2:
        return (b * ls + l * bs) / (bs + ls) if (bs + ls) > 0 else (b + l) / 2
    return b or l


def snapshots(lines) -> pd.DataFrame:
    """Each runner of each market at each checkpoint (the book as it stood at off - c), with the SP and result."""
    markets: dict = {}
    nxt: dict = {}                     # market -> index of its next checkpoint
    rows = []

    def take(m, c):
        if m.inplay or str(m.defn.get("status", "OPEN")) != "OPEN":
            return
        for sid, r in m.runners.items():
            b, bs = r.best_back()
            l, ls = r.best_lay()
            rows.append({"market": m.id, "sid": sid, "check": c, "back1": b, "back1_size": bs, "lay1": l,
                         "lay1_size": ls, "ltp": r.ltp, "spn": r.spn, "mid": _mid(b, bs, l, ls)})

    for t, mc in sr.events(lines):
        mid = str(mc.get("id"))
        m = markets.get(mid)
        if m is not None and m.off is not None:
            i = nxt.get(mid, 0)
            while i < len(ALL_CHECKS) and t >= m.off - timedelta(seconds=ALL_CHECKS[i]):
                take(m, ALL_CHECKS[i])                     # the state before this message: the book at the mark
                i += 1
            nxt[mid] = i
        sr.apply(markets, mc)
        m = markets[mid]
        if mid not in nxt and m.off is not None:           # skip the marks already past when the market is first seen
            nxt[mid] = sum(1 for c in ALL_CHECKS if t >= m.off - timedelta(seconds=c))
    snap = pd.DataFrame(rows)
    if not len(snap):
        return snap
    fin = []
    for mid, m in markets.items():
        for r in m.defn.get("runners", []) or []:
            fin.append({"market": mid, "sid": int(r.get("id", -1)), "bsp": r.get("bsp"),
                        "status": r.get("status"), "mtype": m.defn.get("marketType"),
                        "venue": m.defn.get("venue"), "off": m.off})
    fin = pd.DataFrame(fin)
    out = snap.merge(fin, on=["market", "sid"], how="left")
    out["bsp"] = pd.to_numeric(out.bsp, errors="coerce")
    return out[out.status.astype(str).isin(["WINNER", "LOSER", "PLACED", "ACTIVE"])]


def trades(s: pd.DataFrame) -> pd.DataFrame:
    """Every rule's trades (a unit stake each, closed at the SP) from the snapshots."""
    s = s[(s.bsp > 1) & (s.mtype.astype(str) == "WIN")].copy()
    w = s.pivot_table(index=["market", "sid"], columns="check", values="mid", aggfunc="first")
    out = []

    def add(rule, x, side):
        if not len(x):
            return
        if side == "BACK":
            ok = (x.back1 > 1) & (x.back1_size >= MIN_SIZE)
            raw = x.back1 / x.bsp - 1
            price = x.back1
        else:
            ok = (x.lay1 > 1) & (x.lay1_size >= MIN_SIZE)
            raw = 1 - x.lay1 / x.bsp
            price = x.lay1
        x = x.assign(price=price, raw=raw)[ok]
        won = (x.status.astype(str) == "WINNER").astype(int)
        # target 2 (the owner, 6 Nov 2026: "Win - winning the bet"): the same bet left to run to the result
        res = np.where(won == 1, x.price - 1, -1.0) if side == "BACK" else np.where(won == 1, -(x.price - 1), 1.0)
        x = x.assign(pnl=np.where(x.raw > 0, x.raw * (1 - COMM), x.raw), won=won,
                     pnl_win=np.where(res > 0, res * (1 - COMM), res), rule=rule, side=side)
        out.append(x[["rule", "side", "market", "sid", "check", "price", "bsp", "won", "pnl", "pnl_win"]])

    for c in DECIDE + LATE_DECIDE:
        x = s[s.check == c]
        n = late_name(c)
        for g in GAPS:
            add(f"gap_back_{n}_{g}", x[(x.spn > 1) & (x.back1 / x.spn - 1 > g)], "BACK")
            add(f"gap_lay_{n}_{g}", x[(x.spn > 1) & (1 - x.lay1 / x.spn > g)], "LAY")
        if 180 in w.columns and c in w.columns:
            mv = (w[c] / w[180] - 1).rename("move").reset_index()
            xm = x.merge(mv, on=["market", "sid"])
            for mvx in MOVES:
                short, drift = xm[xm.move <= -mvx], xm[xm.move >= mvx]
                add(f"mom_back_{n}_{mvx}", short, "BACK")
                add(f"mom_lay_{n}_{mvx}", drift, "LAY")
                add(f"rev_back_{n}_{mvx}", drift, "BACK")
                add(f"rev_lay_{n}_{mvx}", short, "LAY")
    x10 = s[s.check == 10]
    if len(x10):
        fav = x10.loc[x10.groupby("market").mid.idxmin()]
        add("fav_lay_10", fav, "LAY")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _boot(pnl_by_race: pd.Series, n: int = 2000, seed: int = 0):
    v = pnl_by_race.to_numpy()
    if len(v) < 10:
        return None, None
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(v), size=(n, len(v)))
    m = v[i].mean(1)
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def score(tr: pd.DataFrame, ledger=None) -> pd.DataFrame:
    """Each rule on both targets: closed at the SP (``pnl``) and left to run to the result (``pnl_win``); trades,
    races, days, the mean P&L a unit staked, its 90% interval by race (pooled over days) and its deflated Sharpe by
    race against every test scored (research.overfit: both targets of every rule here, and with a ``ledger`` every
    test the lab has ever scored). A rule is a candidate on a target on its own: ``MIN_DAYS`` days, the lower bound
    above zero and the DSR at or above research.overfit.DSR_THRESHOLD. About one test in twenty clears the bound by
    luck; the DSR is the bar that allows for the count, and q reads the family as one."""
    from research.overfit import deflate_table
    rows, series = [], {}
    # the late family counts only on days after its registration; a day scored before it has no late trades anyway
    tr = tr[~tr.rule.str.contains("_late") | (tr.day.astype(str) > REGISTERED_LATE)]
    for rule, g in tr.groupby("rule"):
        row = {"rule": rule, "trades": len(g), "races": int(g.groupby(["day", "market"]).ngroups),
               "days": g.day.nunique()}
        for col, sfx in (("pnl", ""), ("pnl_win", "_win")):
            v = g[g[col].notna()] if col in g else g.iloc[:0]
            by_race = v.groupby(["day", "market"])[col].mean() if len(v) else pd.Series(dtype=float)
            if len(by_race):
                series[f"{rule}{sfx}"] = by_race.to_numpy()
            lo, hi = _boot(by_race)
            row[f"mean{sfx}%"] = round(100 * float(v[col].mean()), 3) if len(v) else None
            row[f"ci90{sfx}%"] = [round(100 * lo, 3), round(100 * hi, 3)] if lo is not None else None
            row[f"bound{sfx}"] = bool(lo is not None and lo > 0 and v.day.nunique() >= MIN_DAYS)
            row[f"z{sfx}"] = mt.z_by_race(by_race)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    dsr = deflate_table(series, ledger=ledger) if series else {}
    for row in rows:
        for sfx in ("", "_win"):
            d = dsr.get(f"{row['rule']}{sfx}", {})
            row[f"sharpe{sfx}"], row[f"dsr{sfx}"] = d.get("sharpe"), d.get("dsr")
            row[f"candidate{sfx}"] = bool(row.pop(f"bound{sfx}") and d.get("passed", False))
        row["n_trials"] = max((d.get("n_trials", 0) for d in dsr.values()), default=0)
        row["mean_pnl%"] = row.pop("mean%")
        row["candidate_sp"] = row.pop("candidate")
        row["candidate"] = row["candidate_sp"] or row["candidate_win"]
    res = pd.DataFrame(rows).sort_values("mean_pnl%", ascending=False).reset_index(drop=True)
    # the family read as one (research/greyhound/multitest.py; the review of 10 Nov 2026): every rule on both targets,
    # each test's q the share of false discoveries expected among the tests at least as strong
    q = mt.bh(mt.p_one_sided(np.r_[res.z.to_numpy(float), res.z_win.to_numpy(float)]))
    res["q"], res["q_win"] = q[:len(res)], q[len(res):]
    return res


def run_day(lines, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = snapshots(lines)
    if not len(s):
        return s, pd.DataFrame()
    tr = trades(s)
    if len(tr):
        tr["day"] = f"{day}"
    return s, tr


def forward_test_orders(store, day: date) -> list:
    """The early-lay forward test's lays for the day (greyhound/laytest.py, its primary rule, as its intraday table
    builds them: the day's forecast fit on the first traded books), as simulated LAY orders at the time of each book,
    at the rule's limit price, staked as the early lay stakes (GBP20 liability)."""
    from greyhound import early_lay as el, laytest as lt
    from research.greyhound import limit_study as ls, tradeout_study as ts
    f = lt.day_fit(store, day)
    if f is None:
        return []
    rows, _ = lt.today_rows(store, day)
    if not len(rows):
        return []
    q, book = ts.predict(rows, f)
    rows = rows.assign(f_bsp=1.0 / (book * np.asarray(q, float)))
    o = lt._select(ls.orders(rows), lt.PRIMARY).drop_duplicates(["market_id", "selection_id"])
    out = []
    for r in o.itertuples(index=False):
        price = float(r.limit)
        out.append(sr.SimOrder(market=str(r.market_id), sid=int(float(r.selection_id)), side="LAY", price=price,
                               size=el.stake_for(price, el.CONFIG), t=pd.Timestamp(r.polled).to_pydatetime(),
                               tag="forward_test"))
    return out


def forward_check(lines, store, day: date) -> dict:
    """The forward test's lays replayed through the stream with queue-aware fills (x1 and x0.5 volume), closed 5 s
    before the off, at the SP and left to run to the result, each as a return on the matched liability; beside the forward test's own figure
    (its optimistic fill on the recorder's books)."""
    from greyhound import laytest as lt
    out = {"day": f"{day}"}
    try:
        own = lt.intraday(store, day)
        out["forward_test_own"] = {k: own.get(k) for k in ("races", "lays_placed", "matched", "liability", "pnl", "roi%")}
    except Exception as exc:
        out["forward_test_own"] = {"error": str(exc)[:200]}
    for vf in (1.0, 0.5):
        orders = forward_test_orders(store, day)
        if not orders:
            out["note"] = "no data found: no forward-test lays for the day"
            return out
        df = sr.replay(lines, orders, vol_factor=vf, comm=COMM)
        m = df[df.matched > 0]
        liab = float((m.matched * (m.avg - 1)).sum()) if len(m) else 0.0
        green = float(pd.to_numeric(m.pnl, errors="coerce").fillna(0).sum()) if len(m) else 0.0
        msp = m[pd.to_numeric(m.pnl_sp, errors="coerce").notna()] if len(m) else m
        liab_sp = float((msp.matched * (msp.avg - 1)).sum()) if len(msp) else 0.0
        at_sp = float(pd.to_numeric(msp.pnl_sp, errors="coerce").sum()) if len(msp) else 0.0
        mres = m[pd.to_numeric(m.get("pnl_result"), errors="coerce").notna()] if len(m) and "pnl_result" in m else m.iloc[:0]
        liab_res = float((mres.matched * (mres.avg - 1)).sum()) if len(mres) else 0.0
        at_res = float(pd.to_numeric(mres.pnl_result, errors="coerce").sum()) if len(mres) else 0.0
        out[f"stream_x{vf}"] = {**sr.summary(df), "matched_liability": round(liab, 2),
                                "roi_green%": round(100 * green / liab, 2) if liab > 0 else None,
                                "roi_sp%": round(100 * at_sp / liab_sp, 2) if liab_sp > 0 else None,
                                "roi_result%": round(100 * at_res / liab_res, 2) if liab_res > 0 else None}
        # beside it from 10 Nov 2026 (the review): the same lays left in the book until the market really closes,
        # not lapsed 5 s before the scheduled off. A second reading of the same rule; the gate's figure is unchanged
        dc = sr.replay(lines, forward_test_orders(store, day), vol_factor=vf, comm=COMM, to_close=True)
        mc = dc[dc.matched > 0]
        ok = mc[pd.to_numeric(mc.pnl_sp, errors="coerce").notna()] if len(mc) else mc
        lc = float((ok.matched * (ok.avg - 1)).sum()) if len(ok) else 0.0
        okr = mc[pd.to_numeric(mc.pnl_result, errors="coerce").notna()] if len(mc) else mc
        lr = float((okr.matched * (okr.avg - 1)).sum()) if len(okr) else 0.0
        out[f"stream_x{vf}_to_close"] = {
            "matched": int(len(mc)), "matched_liability": round(float((mc.matched * (mc.avg - 1)).sum()), 2) if len(mc) else 0.0,
            "roi_sp%": round(100 * float(pd.to_numeric(ok.pnl_sp, errors="coerce").sum()) / lc, 2) if lc > 0 else None,
            "roi_result%": round(100 * float(pd.to_numeric(okr.pnl_result, errors="coerce").sum()) / lr, 2) if lr > 0 else None}
    return out


def main(argv=None) -> int:
    """Score a day's stream into its trades (s3 stream_lab/trades/<day>.parquet), then pool every day scored."""
    import argparse
    import io
    from betfair_recorder import bucket, s3_client
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", default=None, help="the day to score (default: yesterday, UTC)")
    ap.add_argument("--out", default="out/stream_lab")
    a = ap.parse_args(argv)
    day = date.fromisoformat(a.day) if a.day else date.today() - timedelta(days=1)
    s3, bk = s3_client(), bucket()

    def listing(prefix):
        out, tok = [], None
        while True:
            r = s3.list_objects_v2(Bucket=bk, Prefix=prefix, **({"ContinuationToken": tok} if tok else {}))
            out += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return out
            tok = r["NextContinuationToken"]

    lines = sr.load_lines(lambda k: s3.get_object(Bucket=bk, Key=k)["Body"].read(), listing, day)
    log.info("%s: %d stream lines", day, len(lines))
    snap, tr = run_day(lines, day)
    if len(tr):
        buf = io.BytesIO()
        tr.to_parquet(buf, index=False)
        s3.put_object(Bucket=bk, Key=f"stream_lab/trades/{day}.parquet", Body=buf.getvalue())
    days = []
    for k in sorted(x for x in listing("stream_lab/trades/") if x.endswith(".parquet")):
        t = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=bk, Key=k)["Body"].read()))
        if "pnl_win" not in t:                  # a day scored before the result target (6 Nov): score it again
            d0 = date.fromisoformat(k.rsplit("/", 1)[1][:10])
            _, t = run_day(sr.load_lines(lambda kk: s3.get_object(Bucket=bk, Key=kk)["Body"].read(), listing, d0), d0)
            if len(t):
                buf = io.BytesIO()
                t.to_parquet(buf, index=False)
                s3.put_object(Bucket=bk, Key=k, Body=buf.getvalue())
                log.info("%s scored again with the result target: %d trades", d0, len(t))
        days.append(t)
    pooled = pd.concat(days, ignore_index=True) if days else pd.DataFrame()
    from research.overfit import TrialLedger
    ledger = TrialLedger(f"s3://{bk}/stream_lab/trials.json", programme="greyhound stream lab")
    sc = score(pooled, ledger=ledger) if len(pooled) else pd.DataFrame()
    rep = {"day": f"{day}", "lines": len(lines), "markets": int(snap.market.nunique()) if len(snap) else 0,
           "trades_today": len(tr), "days_pooled": int(pooled.day.nunique()) if len(pooled) else 0,
           "rules": int(len(sc)), "registered": REGISTERED, "candidates": sc[sc.candidate].rule.tolist() if len(sc) else [],
           "scores": sc.to_dict("records") if len(sc) else []}
    try:                                       # the forward test checked with the stream's fills (the owner, 5 Nov)
        from sources.common import Store
        rep["forward_test"] = forward_check(lines, Store(), day)
    except Exception as exc:
        rep["forward_test"] = {"error": str(exc)[:300]}
    s3.put_object(Bucket=bk, Key=f"stream_lab/report_{day}.json", Body=json.dumps(rep, indent=1, default=str).encode())
    from pathlib import Path
    Path(a.out).mkdir(parents=True, exist_ok=True)
    md = [f"## Greyhound stream lab, {day}", "",
          f"{rep['lines']} stream lines, {rep['markets']} markets, {rep['trades_today']} trades today; "
          f"{rep['days_pooled']} days pooled; {rep['rules']} rules on two targets, {2 * rep['rules']} tests (registered "
          f"{REGISTERED}, and the late family, decided 0, 10 and 20 s after the scheduled off, {REGISTERED_LATE}, "
          f"counted on later days; a candidate needs {MIN_DAYS} days, its pooled 90% lower bound above zero on a target "
          f"and its deflated Sharpe by race at least 0.95 against every test scored; about one test in twenty clears "
          f"the bound alone by chance). Candidates: {', '.join(rep['candidates']) or 'none'}.",
          "", "Target 1, closed at the SP (BFSP); target 2, left to run to the result (Win). q: the share of false "
          "discoveries expected among every test at least as strong (under 0.10: the family as a whole supports it).", "",
          "| Rule | Trades | Races | Days | At the SP: mean % a unit | 90% | q | To the result: mean % a unit | 90% | q | "
          "DSR (SP / Win; tests) | Candidate (SP / Win) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    q2 = lambda v: "" if v is None or pd.isna(v) else f"{v:.2f}"
    for r in rep["scores"]:
        md.append(f"| {r['rule']} | {r['trades']} | {r['races']} | {r['days']} | {r['mean_pnl%']} | {r['ci90%']} | "
                  f"{q2(r.get('q'))} | {r.get('mean_win%')} | {r.get('ci90_win%')} | {q2(r.get('q_win'))} | "
                  f"{r.get('dsr')} / {r.get('dsr_win')} ({r.get('n_trials')}) | "
                  f"{r.get('candidate_sp')} / {r.get('candidate_win')} |")
    ft = rep.get("forward_test", {})
    md += ["", "### The early-lay forward test, its lays replayed through the stream", "",
           f"Its own figure (optimistic fill, the recorder's books): {json.dumps(ft.get('forward_test_own'))}", ""]
    for k in ("stream_x1.0", "stream_x0.5"):
        if k in ft:
            md.append(f"- queue-aware, traded volume {k[-4:]}: {json.dumps(ft[k])}")
    for k in ("stream_x1.0_to_close", "stream_x0.5_to_close"):
        if k in ft:
            md.append(f"- the same lays resting until the market really closes (from 10 Nov 2026; not the gate's "
                      f"figure), traded volume {k[7:11]}: {json.dumps(ft[k])}")
    if ft.get("note") or ft.get("error"):
        md.append(f"- {ft.get('note') or ft.get('error')}")
    Path(a.out, "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
