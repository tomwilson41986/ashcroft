"""Is there a late signal in the horse Exchange Stream? (the owner, 11 Nov: "try the stream data for a late signal";
read-only research, nothing reaches Betfair)

The small book near the off found no edge from our 06:00 forecast (ledger small-book-late-1111): by T-5 the market
knows what it knows. A late book needs a signal the price at the mark has not absorbed. The stream
(research/horse_stream_record.py, live key, from 10 Nov) keeps the whole ladder and the traded volume at 500 ms, so
two candidate signals:
- money flow: a runner's share of the money traded in the last N minutes less its share of the book now
  (money arriving faster than the price shows);
- ladder pressure: money waiting to back within 3 ticks of the best less money waiting to lay, over both.

1. Inventory: every stream day held, parts, markets, and for each market how close to its off the stream reaches
   (the recorder stops at 10:30 UK, so near-off coverage is expected to be thin or none).
2. The test, at each market's last snapshot before the recorder stopped (and at T-5 where the stream reaches it):
   does each signal predict the move from the mid then to the BSP (log of the BSP share over the mid share), beyond
   the price? Slope with a race-clustered se, and the rank correlation. The BSP and result from betfair_live_marks
   'final'. Descriptive: one or two days cannot pass a gate; the reading is whether the signal is worth recording
   the stream to the off.
"""
import gzip
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from research.greyhound import stream_replay as sr  # noqa: E402  (the stream parser; research code, read only)

pd.set_option("display.width", 220)
PREFIX = "betfair_stream/horse"
DAYS = [date(2026, 11, 10), date(2026, 11, 11)]
FLOW_MIN = (10, 30)
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("CAPTURE_BUCKET") or os.environ.get("PRED_BUCKET") or "ashcroft"
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)


def keys(prefix):
    out, tok = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, **({"ContinuationToken": tok} if tok else {})}
        r = s3.list_objects_v2(**kw)
        out += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        tok = r["NextContinuationToken"]


def tick_band(p, n=3):
    """A rough n-tick band: Betfair's ticks are about 1-2% of the price above 2.0; 3 ticks ~ 4%."""
    return p * (1 + 0.013 * n)


def snapshot(m: sr.Market, t, vol_hist):
    """One row per active runner: mid, book share, flow shares over FLOW_MIN, ladder pressure."""
    rows = []
    tv_now = {sid: sum(r.trd.values()) for sid, r in m.runners.items()}
    for sid, r in m.runners.items():
        b, _ = r.best_back()
        l, _ = r.best_lay()
        if not b or not l or l / b > 1.25:
            continue
        mid = float(np.sqrt(b * l))
        near_b = sum(v for p, v in r.atb.items() if p >= b / 1.04)
        near_l = sum(v for p, v in r.atl.items() if p <= tick_band(l))
        row = {"market_id": m.id, "selection_id": sid, "t": t, "mid": mid, "back": b, "lay": l,
               "pressure": (near_b - near_l) / max(near_b + near_l, 1.0), "tv": tv_now[sid]}
        for w in FLOW_MIN:
            past = vol_hist.get(w, {}).get(sid)
            row[f"flow{w}"] = tv_now[sid] - past if past is not None else np.nan
        rows.append(row)
    return rows


all_rows, inv = [], []
for day in DAYS:
    ks = sorted(k for k, _ in keys(f"{PREFIX}/{day:%Y-%m-%d}/") if k.endswith(".jsonl.gz"))
    size = sum(s for k, s in keys(f"{PREFIX}/{day:%Y-%m-%d}/"))
    print(f"{day}: {len(ks)} parts, {size / 1e6:.1f} MB")
    if not ks:
        continue
    lines = []
    for k in ks:
        lines += gzip.decompress(s3.get_object(Bucket=bucket, Key=k)["Body"].read()).decode("utf-8", "replace").splitlines()
    markets, first, last, tv_track = {}, {}, {}, {}
    snaps = {}           # market -> {"last": rows, "t5": rows}
    for t, mc in sr.events(lines):
        mid_, _ = sr.apply(markets, mc)
        m = markets[mid_]
        first.setdefault(mid_, t)
        last[mid_] = t
        off = m.off
        if off is None or m.inplay:
            continue
        # running total volume per runner, every minute, for the flow windows
        tr = tv_track.setdefault(mid_, [])
        if not tr or (t - tr[-1][0]).total_seconds() >= 60:
            tr.append((t, {sid: sum(r.trd.values()) for sid, r in m.runners.items()}))
        hist = {}
        for w in FLOW_MIN:
            older = [v for (tt, v) in tr if (t - tt).total_seconds() >= w * 60]
            if older:
                hist[w] = older[-1]
        mins = (off - t).total_seconds() / 60
        s = snaps.setdefault(mid_, {})
        s["last"] = (mins, snapshot(m, t, hist))
        if 4.0 <= mins <= 6.0 and "t5" not in s:
            s["t5"] = (mins, snapshot(m, t, hist))
    for mid_, m in markets.items():
        off = m.off
        inv.append({"day": str(day), "market_id": mid_, "off": off,
                    "first_min_to_off": (off - first[mid_]).total_seconds() / 60 if off else np.nan,
                    "last_min_to_off": (off - last[mid_]).total_seconds() / 60 if off else np.nan})
        for tag, (mins, rows) in snaps.get(mid_, {}).items():
            for r in rows:
                all_rows.append({**r, "day": str(day), "at": tag, "min_to_off": mins})

inv = pd.DataFrame(inv)
if inv.empty:
    print("no stream data found")
    raise SystemExit
print(f"\n=== inventory: {len(inv)} markets")
for d, g in inv.groupby("day"):
    print(f"  {d}: {len(g)} markets; offs {g.off.min()} to {g.off.max()}; last update before the off: "
          f"median {g.last_min_to_off.median():.0f} min, min {g.last_min_to_off.min():.0f} min; within 15 min of the "
          f"off {int((g.last_min_to_off <= 15).sum())}, within 5 min {int((g.last_min_to_off <= 5).sum())}")

d = pd.DataFrame(all_rows)
fin = pd.read_sql_query("""SELECT market_id, selection_id, runner_status, bsp FROM betfair_live_marks
                           WHERE mark = 'final' AND race_date >= '2026-11-10'""", conn)
fin["selection_id"] = pd.to_numeric(fin.selection_id, errors="coerce")
fin["bsp"] = pd.to_numeric(fin.bsp, errors="coerce")
d = d.merge(fin, on=["market_id", "selection_id"], how="inner")
d = d[(d.bsp > 1) & (d.runner_status != "REMOVED")]
print(f"\njoined to the BSP: {len(d):,} runner snapshots in {d.market_id.nunique()} markets "
      f"(the database's final marks run to the last nightly load)")
if d.empty:
    raise SystemExit


def race_norm(x, g):
    return x / x.groupby(g).transform("sum")


for at, g in d.groupby("at"):
    g = g.copy()
    key = g.market_id
    g["p_mid"] = race_norm(1 / g.mid, key)
    g["p_bsp"] = race_norm(1 / g.bsp, key)
    g["move"] = np.log(g.p_bsp / g.p_mid)
    for w in FLOW_MIN:
        tot = g[f"flow{w}"].groupby(key).transform("sum")
        g[f"flow{w}_excess"] = np.where(tot > 0, g[f"flow{w}"] / tot - g.p_mid, np.nan)
    print(f"\n=== at '{at}': {g.market_id.nunique()} markets, {len(g)} runners, minutes to off median "
          f"{g.min_to_off.median():.0f}; mean |move| to the BSP {g.move.abs().mean():.3f}")
    for sig in [f"flow{w}_excess" for w in FLOW_MIN] + ["pressure"]:
        x = g[[sig, "move", "market_id"]].dropna()
        if len(x) < 30 or x[sig].std() == 0:
            print(f"  {sig:16s} too few ({len(x)})")
            continue
        xs = (x[sig] - x[sig].mean()) / x[sig].std()
        b = float(np.sum(xs * (x.move - x.move.mean())) / np.sum(xs ** 2))
        res = x.move - x.move.mean() - b * xs
        cl = (xs * res).groupby(x.market_id).sum()
        se = float(np.sqrt(np.sum(cl ** 2)) / np.sum(xs ** 2))
        rho = x[sig].rank().corr(x.move.rank())
        print(f"  {sig:16s} slope per sd {b:+.4f} (se {se:.4f}, t {b / se if se else np.nan:+.1f}); rank corr "
              f"{rho:+.3f}; {len(x)} runners, {x.market_id.nunique()} markets")
