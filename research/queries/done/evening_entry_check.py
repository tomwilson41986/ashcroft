"""Would the live rule's backs have closed better taken the evening before? (read-only; the owner's ask, 4 Oct)

From 4 Oct the recorder's evening run snapshots tomorrow's GB/IE win markets (live-record.yml, books_evening:
every ten minutes for 45 minutes, after the last race). For each racing day with that evening record:
- every runner: the evening's best back against its BSP (Betfair's price file for the day), by win price band, and
  the size on offer;
- the live rule's horses that day (their first matched back in the trader's ledger): the evening's best back
  against the same BSP, beside the morning fill they actually got, and whether the evening book held the morning's
  stake.
The rule's horses were chosen on the morning's prices, so this reads "the same horses, the evening before"; an
evening rule would choose on the evening's prices with predictions made the evening before. A few days decide
nothing; this starts the record. CLV before commission.
"""

from __future__ import annotations

import gzip
import io
import os
import sys
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402

pd.set_option("display.width", 240)
FIRST = date.fromisoformat(os.environ.get("EVENING_FROM") or "2026-10-05")   # the first day recorded the evening before
LAST = date.fromisoformat(os.environ["EVENING_TO"]) if os.environ.get("EVENING_TO") else date.today() - timedelta(days=1)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")


def _get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return None


def _csv(key, **kw):
    raw = _get(key)
    if raw is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(gzip.decompress(raw) if key.endswith(".gz") else raw), low_memory=False, **kw)


def bsp_file(day):
    """Every runner's BSP and result for the day's GB/IE win markets, keyed by the exchange's ids."""
    parts = []
    for d in (day + timedelta(days=1), day):
        for c in ("uk", "ire"):
            name = f"dwbfprices{c}win{d:%d%m%Y}.csv"
            raw = _get(f"{bp.S3_PREFIX}/{name}")
            if raw is None:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            t = bp.parse_file_text(text, name)
            parts.append(t[t.race_date == day.isoformat()])
    if not parts:
        return pd.DataFrame()
    f = pd.concat(parts, ignore_index=True)
    f["market_id"] = "1." + f.event_id.astype("Int64").astype(str)
    return f[["market_id", "selection_id", "bsp", "win_lose"]].dropna(subset=["bsp"])


band_edges, band_names = [1, 3, 5, 8, 13, 21, 1e9], ["<3", "3-5", "5-8", "8-13", "13-21", "21+"]
every, rule = [], []
day = FIRST
while day <= LAST:                             # a day is read once its racing is over
    ev = _csv(f"betfair_live/{day}/books_evening.csv.gz", dtype={"market_id": str})
    if ev.empty:
        print(f"{day}: no evening record")
        day += timedelta(days=1)
        continue
    ev = ev[ev.runner_status.eq("ACTIVE") & (pd.to_numeric(ev.back1, errors="coerce") > 1)].copy()
    ev["back1"], ev["back1_size"] = (pd.to_numeric(ev[c], errors="coerce") for c in ("back1", "back1_size"))
    ev["t"] = pd.to_datetime(ev.polled_utc, utc=True)
    last = ev.sort_values("t").groupby(["market_id", "selection_id"]).tail(1)       # the evening's last look
    bsp = bsp_file(day)
    print(f"{day}: evening record {ev.t.min() + pd.Timedelta(hours=1):%d %b %H:%M} to {ev.t.max() + pd.Timedelta(hours=1):%H:%M} UK, "
          f"{ev.market_id.nunique()} markets, {len(last)} runners; price file {'yes' if len(bsp) else 'not yet'}")
    if len(bsp):
        e = last.merge(bsp, on=["market_id", "selection_id"])
        e["day"] = str(day)
        every.append(e)
    led = _csv(f"trading/live/{day}/ledger.csv", dtype={"market_id": str})
    if not led.empty:
        bk = led[(led.event == "back") & (pd.to_numeric(led.matched, errors="coerce") > 0)].copy()
        bk["matched"], bk["avg_price"] = (pd.to_numeric(bk[c], errors="coerce") for c in ("matched", "avg_price"))
        bk["t"] = pd.to_datetime(bk.ts, utc=True)
        first = bk.sort_values("t").drop_duplicates(["market_id", "selection_id"])
        stake = bk.groupby(["market_id", "selection_id"]).matched.sum().rename("staked")
        st = led[(led.event == "settle") & (led.result != "REMOVED")][["market_id", "selection_id", "bsp"]].copy()
        st["bsp"] = pd.to_numeric(st.bsp, errors="coerce")
        r = (first[["market_id", "selection_id", "runner", "avg_price", "t"]]
             .merge(stake, left_on=["market_id", "selection_id"], right_index=True)
             .merge(st, on=["market_id", "selection_id"])
             .merge(last[["market_id", "selection_id", "back1", "back1_size"]], on=["market_id", "selection_id"], how="left"))
        r["day"] = str(day)
        rule.append(r)
    day += timedelta(days=1)

if every:
    E = pd.concat(every)
    E["band"] = pd.cut(E.bsp, band_edges, right=False, labels=band_names)
    E["ratio"] = E.back1 / E.bsp
    print("\n== every runner: the evening's best back against the BSP (median ratio; above 1 is a price that closed shorter)")
    print(E.groupby("band", observed=True).agg(runners=("ratio", "size"), back_bsp_median=("ratio", "median"),
                                               clv_level=("ratio", lambda v: (v - 1).mean()),
                                               size_median=("back1_size", "median")).round(3).to_string())
if rule:
    R = pd.concat(rule)
    R = R[R.bsp > 1]
    got = R[R.back1.notna()]
    print(f"\n== the live rule's horses: {len(R)}, {len(got)} of them priced the evening before")
    if len(got):
        w = got.staked
        clv_m = float((w * (got.avg_price / got.bsp - 1)).sum() / w.sum())
        clv_e = float((w * (got.back1 / got.bsp - 1)).sum() / w.sum())
        print(f"   CLV at the morning fill {clv_m:+.2%}; at the evening's best back {clv_e:+.2%} "
              f"(stake-weighted, the same {len(got)} horses)")
        print(f"   the evening's best back above the morning fill for {(got.back1 > got.avg_price).mean():.0%}; "
              f"the evening book held the morning's stake for {(got.back1_size >= got.staked).mean():.0%} "
              f"(size median GBP{got.back1_size.median():.0f} against a stake median GBP{got.staked.median():.0f})")
        print(got.groupby("day").apply(lambda g: pd.Series({
            "horses": len(g), "clv_morning": (g.staked * (g.avg_price / g.bsp - 1)).sum() / g.staked.sum(),
            "clv_evening": (g.staked * (g.back1 / g.bsp - 1)).sum() / g.staked.sum()})).round(4).to_string())
