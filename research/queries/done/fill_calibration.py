"""Does the closing model's expected CLV come true on the live trader's own fills? (read-only)

For every live day's ledger (trading/live/<day>/ledger.csv in S3, settled: each horse's BSP on its settle row), each
matched back's CLV at its fill price (avg_price / BSP - 1) against the expected CLV the trader backed it on (the
ledger's edge), for each horse's first fill and for the top-ups, by expected-CLV band; the stake-weighted and level
means with 90% intervals from a bootstrap over races; and realised minus expected. CLV before commission.
"""

from __future__ import annotations

import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
FIRST = date.fromisoformat(os.environ.get("FILLS_FROM") or "2026-10-01")
LAST = date.fromisoformat(os.environ["FILLS_TO"]) if os.environ.get("FILLS_TO") else date.today() - timedelta(days=1)
BANDS = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 10.0]
BAND_NAMES = ["3-4%", "4-5%", "5-6%", "6-8%", "8-10%", "10-15%", "15%+"]
KEY = ["market_id", "selection_id"]
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")


def ledger(day):
    try:
        raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day}/ledger.csv")["Body"].read()
    except Exception:
        return pd.DataFrame()
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}, low_memory=False)
    led["t"] = pd.to_datetime(led.ts, utc=True)
    for c in ("selection_id", "matched", "avg_price", "edge", "bsp", "minutes_to_off"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    st = led[(led.event == "settle") & (led.result != "REMOVED") & (led.bsp > 1)][KEY + ["bsp"]]
    st = st.drop_duplicates(KEY)
    m = led[(led.event == "back") & (led.matched > 0)].drop(columns=["bsp"]).merge(st, on=KEY)
    m["clv"] = m.avg_price / m.bsp - 1.0
    m["day"] = str(day)
    return m


parts, day = [], FIRST
while day <= LAST:
    m = ledger(day)
    print(f"{day}: {len(m)} settled fills" if len(m) else f"{day}: no settled ledger")
    if len(m):
        parts.append(m)
    day += timedelta(days=1)
if not parts:
    raise SystemExit("nothing to read")
A = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
A["first"] = ~A.duplicated(["day"] + KEY)
A["band"] = pd.cut(A.edge, BANDS, labels=BAND_NAMES, right=False)
A["race"] = A.day + A.market_id


def tab(g):
    w = g.matched
    return pd.Series({"n": len(g), "expected": g.edge.mean(), "realised_level": g.clv.mean(),
                      "realised_staked": (w * g.clv).sum() / w.sum(), "closed_shorter": (g.clv > 0).mean(),
                      "gbp_staked": w.sum()})


def boot(g, n=5000, seed=1):
    codes, uniq = pd.factorize(g.race)
    wc = np.bincount(codes, weights=g.matched * g.clv)
    ww = np.bincount(codes, weights=g.matched)
    lc = np.bincount(codes, weights=g.clv)
    k = np.bincount(codes)
    le = np.bincount(codes, weights=g.edge)
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n, len(uniq)))
    q = lambda v: np.percentile(v, [5, 95])  # noqa: E731
    return {"races": len(uniq), "staked": (wc.sum() / ww.sum(), q(wc[idx].sum(1) / ww[idx].sum(1))),
            "level": (lc.sum() / k.sum(), q(lc[idx].sum(1) / k[idx].sum(1))),
            "gap": ((lc.sum() - le.sum()) / k.sum(), q((lc[idx].sum(1) - le[idx].sum(1)) / k[idx].sum(1)))}


for label, g in (("first fills", A[A["first"]]), ("top-ups", A[~A["first"]]), ("all fills", A)):
    b = boot(g)
    print(f"\n== {label}: {len(g)} fills in {b['races']} races, {g.day.nunique()} days")
    for k, name in (("staked", "staked CLV"), ("level", "level CLV"), ("gap", "level realised minus expected")):
        v, (lo, hi) = b[k]
        print(f"   {name}: {v:+.2%} (90% {lo:+.2%} to {hi:+.2%})")
    print(g.groupby("band", observed=True).apply(tab).round(4).to_string())
    print("   by day:")
    print(g.groupby("day").apply(tab).round(4).to_string())
f = A[A["first"]]
fit = np.polyfit(f.edge, f.clv.clip(-0.9, 3.0), 1)
print(f"\nfirst fills: realised CLV on expected, slope {fit[0]:.2f}, intercept {fit[1]:+.4f}")
