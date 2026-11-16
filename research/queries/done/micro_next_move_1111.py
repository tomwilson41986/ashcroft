"""Can the next price move be predicted in the last 5 minutes? Greyhound WIN markets, from the recorded Exchange Stream
(read-only; registered before the run: ledger `micro-next-move-1111`; the owner, 11 Nov 2026: "Developing a model
to predict the next price move ... Kalman, markov chain ... <5 minutes to the off"; docs/agent_desk.md, the
high-frequency track, Stage 1). No order, no strategy: a forecast test only.

Data: s3://$CAPTURE_BUCKET/betfair_stream/greyhound/<day>/ (greyhound/stream_record.py: full ladder, traded by price,
not conflated, Betfair's millisecond publish time), every day held from 5 Nov 2026 to yesterday (UTC). Rows: every
runner's book change in each WIN market's last 300 s before the scheduled off, until it turns in play
(research/micro/events.py); features and targets with no look-ahead (research/micro/features.py).

Targets: the direction of the mid (in probability) at the runner's next change (y_next: up or down), and 1, 5 and 10
seconds on (y_1s/5s/10s: up, none or down).
Models (research/micro/models.py): the baseline (class shares), queue-imbalance logistic, Markov chain on book states,
Kalman fair price, LightGBM on every feature.
Walk-forward by day: each day from the third on is predicted by models fitted on every earlier day.

Registered reading (written before any figure of this run is seen):
- per (target, model): the log-loss gain over the baseline in millinats a row, its mean over the test days and the
  90% day-bootstrap interval; the accuracy on rows that moved; and the same split by seconds to the off (0-60,
  60-120, 120-300) and by price (under 3, 3-6, over 6), descriptive;
- trials counted: 4 targets x 4 models = 16;
- a (target, model) "predicts" only if its interval is above zero on these days AND, later, on the sealed days
  12-18 Nov 2026 read once, unchanged (that read is registered here; it is not run now);
- a forecast that predicts earns Stage 2 (the queue-aware simulator with latency and costs), never an order.
"""
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from betfair_recorder import bucket, s3_client  # noqa: E402
from research.greyhound import stream_replay as sr  # noqa: E402
from research.micro import events, features, models  # noqa: E402

FIRST = date(2026, 11, 5)
LAST = date.today() - timedelta(days=1)
TARGETS = ("y_next", "y_1s", "y_5s", "y_10s")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

s3, bk = s3_client(), bucket()


def listing(prefix):
    out, tok = [], None
    while True:
        r = s3.list_objects_v2(Bucket=bk, Prefix=prefix, **({"ContinuationToken": tok} if tok else {}))
        out += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        tok = r["NextContinuationToken"]


frames, t_all = [], time.time()
day = FIRST
while day <= LAST:
    tic = time.time()
    lines = sr.load_lines(lambda k: s3.get_object(Bucket=bk, Key=k)["Body"].read(), listing, day)
    if not lines:
        print(f"{day}: no data found", flush=True)
        day += timedelta(days=1)
        continue
    rows = events.book_rows(lines, window_s=300.0)
    f = features.build(rows)
    print(f"{day}: {len(lines):,} stream lines, {len(rows):,} book rows, {len(f):,} with a two-sided book, "
          f"{f.market.nunique() if len(f) else 0} markets, {round(time.time() - tic)} s", flush=True)
    if len(f):
        frames.append(f)
    day += timedelta(days=1)

if not frames:
    print("no data found: nothing to read")
    sys.exit(0)
df = pd.concat(frames, ignore_index=True)
print(f"\n== {df.day.nunique()} days ({df.day.min()} .. {df.day.max()}), {len(df):,} rows, "
      f"{df.market.nunique():,} markets; {round((time.time() - t_all) / 60)} minutes to build")
for tg in TARGETS:
    v = df[tg].dropna()
    print(f"  {tg}: {len(v):,} rows, shares " + ", ".join(f"{int(c)}: {(v == c).mean():.3f}" for c in sorted(v.unique())))

print("\n== the walk-forward, a test day at a time (gain over the baseline, millinats a row; positive is better)")
res = []
for tg in TARGETS:
    r = models.walk_forward(df, tg)
    print(r.to_string(index=False), flush=True)
    res.append(r)
res = pd.concat(res, ignore_index=True)

print("\n== the registered reading: mean gain over the test days and its 90% day-bootstrap interval")
summ = []
for (tg, m), g in res[res.model != "baseline"].groupby(["target", "model"], sort=False):
    mean, lo, hi = models.day_interval(g.set_index("day").gain_mnats)
    summ.append({"target": tg, "model": m, "days": len(g), "gain_mnats": round(mean, 3), "lo90": round(lo, 3),
                 "hi90": round(hi, 3), "acc_moved": round(g.acc_moved.mean(), 4),
                 "verdict": "above zero here (needs the sealed days)" if lo > 0 else "no"})
summ = pd.DataFrame(summ)
print(summ.to_string(index=False))
print(f"trials counted: {len(summ)}")

print("\n== descriptive: the gain by seconds to the off and by price, the best model of each target, pooled test days")
days = sorted(df.day.unique())
test_days = days[2:]
for tg in TARGETS:
    s = summ[summ.target == tg].sort_values("gain_mnats", ascending=False)
    if s.empty:
        continue
    best = s.iloc[0].model
    d = df.dropna(subset=[tg]).sort_values(["market", "sid", "t"], kind="mergesort")
    parts = []
    for day_ in test_days:
        tr, te = d[d.day < day_], d[d.day == day_]
        if te.empty:
            continue
        classes = models._classes(d[tg])
        p_b = models.baseline(tr, te, tg, classes)
        p_m = models.FITTERS[best](tr, te, tg, classes)
        y = te[tg].to_numpy()
        gain = models._logloss(p_b, y, classes) - models._logloss(p_m, y, classes)
        parts.append(pd.DataFrame({"gain": gain * 1000, "to_off": te.to_off.to_numpy(), "price": 1 / te.pm.to_numpy()}))
    if not parts:
        continue
    pp = pd.concat(parts, ignore_index=True)
    pp["secs"] = pd.cut(pp.to_off, [0, 60, 120, 300], labels=["0-60", "60-120", "120-300"])
    pp["band"] = pd.cut(pp.price, [1, 3, 6, 1000], labels=["<3", "3-6", ">6"])
    print(f"\n{tg}, {best}:")
    print(pp.groupby("secs", observed=True).gain.agg(["count", "mean"]).round(3).to_string())
    print(pp.groupby("band", observed=True).gain.agg(["count", "mean"]).round(3).to_string())
