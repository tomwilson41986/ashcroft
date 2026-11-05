"""The stream strategy lab (read-only research; the owner, 5 Nov 2026: "just keep an open mind. Our goal is to build a
profitable strategy, i dont care which one it is"). Rules fixed here before their results are seen, scored on every
day the Exchange Stream recorder holds (greyhound/stream_record.py), so each new day is out of sample for all of them.

Each market's book is rebuilt from the stream (research.greyhound.stream_replay) and read at fixed checkpoints before
the scheduled off (``CHECKS`` seconds): each runner's best back and lay with their sizes, the last traded price and
Betfair's projected SP (near). The settled definition gives the SP and the result.

Every rule takes the price on offer at a checkpoint (a back at the best back price, a lay at the best lay price,
GBP2 or more shown there), so the fill is the book's own: no queue is assumed. Each closes at the SP: a back at B
greened by a lay at the SP gives B / SP - 1 a unit staked, a lay at L 1 - L / SP, whatever the result; commission
on a gain. The families (``RULES``):
- ``gap``: Betfair's projected SP against the price on offer: back where B / spn - 1 > m, lay where 1 - L / spn > m;
- ``mom``: follow the move since 180 s out: back what has shortened by x or more (mid), lay what has drifted;
- ``rev``: against it: back the drifters, lay the shorteners;
- ``fav_lay``: lay the favourite at 10 s (a control: it should lose about the spread).
Scored by day and pooled over days, with a 90% interval from a bootstrap over races; a rule is a candidate only after
``MIN_DAYS`` days with its pooled lower bound above zero, and the count of rules is in every report (about one in
twenty clears a one-sided 90% bound by chance).
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from research.greyhound import stream_replay as sr

log = logging.getLogger(__name__)

CHECKS = (180, 120, 60, 30, 10)
DECIDE = (60, 30, 10)
GAPS = (0.03, 0.06, 0.10)
MOVES = (0.05, 0.10)
MIN_SIZE = 2.0
COMM = 0.02
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
            while i < len(CHECKS) and t >= m.off - timedelta(seconds=CHECKS[i]):
                take(m, CHECKS[i])                         # the state before this message: the book at the mark
                i += 1
            nxt[mid] = i
        sr.apply(markets, mc)
        m = markets[mid]
        if mid not in nxt and m.off is not None:           # skip the marks already past when the market is first seen
            nxt[mid] = sum(1 for c in CHECKS if t >= m.off - timedelta(seconds=c))
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
        x = x.assign(pnl=np.where(x.raw > 0, x.raw * (1 - COMM), x.raw), rule=rule, side=side)
        out.append(x[["rule", "side", "market", "sid", "check", "price", "bsp", "pnl"]])

    for c in DECIDE:
        x = s[s.check == c]
        for g in GAPS:
            add(f"gap_back_{c}_{g}", x[(x.spn > 1) & (x.back1 / x.spn - 1 > g)], "BACK")
            add(f"gap_lay_{c}_{g}", x[(x.spn > 1) & (1 - x.lay1 / x.spn > g)], "LAY")
        if 180 in w.columns and c in w.columns:
            mv = (w[c] / w[180] - 1).rename("move").reset_index()
            xm = x.merge(mv, on=["market", "sid"])
            for mvx in MOVES:
                short, drift = xm[xm.move <= -mvx], xm[xm.move >= mvx]
                add(f"mom_back_{c}_{mvx}", short, "BACK")
                add(f"mom_lay_{c}_{mvx}", drift, "LAY")
                add(f"rev_back_{c}_{mvx}", drift, "BACK")
                add(f"rev_lay_{c}_{mvx}", short, "LAY")
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


def score(tr: pd.DataFrame) -> pd.DataFrame:
    """Each rule: trades, races, days, mean P&L a unit staked, its 90% interval by race (pooled over days)."""
    rows = []
    for rule, g in tr.groupby("rule"):
        by_race = g.groupby(["day", "market"]).pnl.mean()
        lo, hi = _boot(by_race)
        rows.append({"rule": rule, "trades": len(g), "races": len(by_race), "days": g.day.nunique(),
                     "mean_pnl%": round(100 * float(g.pnl.mean()), 3),
                     "ci90%": [round(100 * lo, 3), round(100 * hi, 3)] if lo is not None else None,
                     "candidate": bool(lo is not None and lo > 0 and g.day.nunique() >= MIN_DAYS)})
    return pd.DataFrame(rows).sort_values("mean_pnl%", ascending=False) if rows else pd.DataFrame()


def run_day(lines, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = snapshots(lines)
    if not len(s):
        return s, pd.DataFrame()
    tr = trades(s)
    if len(tr):
        tr["day"] = f"{day}"
    return s, tr


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
        days.append(pd.read_parquet(io.BytesIO(s3.get_object(Bucket=bk, Key=k)["Body"].read())))
    pooled = pd.concat(days, ignore_index=True) if days else pd.DataFrame()
    sc = score(pooled) if len(pooled) else pd.DataFrame()
    rep = {"day": f"{day}", "lines": len(lines), "markets": int(snap.market.nunique()) if len(snap) else 0,
           "trades_today": len(tr), "days_pooled": int(pooled.day.nunique()) if len(pooled) else 0,
           "rules": int(len(sc)), "registered": REGISTERED, "candidates": sc[sc.candidate].rule.tolist() if len(sc) else [],
           "scores": sc.to_dict("records") if len(sc) else []}
    s3.put_object(Bucket=bk, Key=f"stream_lab/report_{day}.json", Body=json.dumps(rep, indent=1, default=str).encode())
    from pathlib import Path
    Path(a.out).mkdir(parents=True, exist_ok=True)
    md = [f"## Greyhound stream lab, {day}", "",
          f"{rep['lines']} stream lines, {rep['markets']} markets, {rep['trades_today']} trades today; "
          f"{rep['days_pooled']} days pooled; {rep['rules']} rules (registered {REGISTERED}; a candidate needs "
          f"{MIN_DAYS} days and its pooled 90% lower bound above zero; about one rule in twenty clears that bound by "
          f"chance). Candidates: {', '.join(rep['candidates']) or 'none'}.", "",
          "| Rule | Trades | Races | Days | Mean P&L % a unit | 90% | Candidate |", "|---|---|---|---|---|---|---|"]
    for r in rep["scores"]:
        md.append(f"| {r['rule']} | {r['trades']} | {r['races']} | {r['days']} | {r['mean_pnl%']} | {r['ci90%']} | "
                  f"{r['candidate']} |")
    Path(a.out, "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
