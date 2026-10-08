"""Audit of the live record (the owner, 8 Oct: "audit the results, why are we losing so much"; read-only).

Every live ledger in S3 (trading/live/<day>/ledger.csv, 30 Sep to today), settled from Betfair's record of the bets:

1. each day: horses, stake matched, the CLV at the settled price, the backs' and the lays' P&L, commission, the result;
2. where the money went: each back is laid at the SP, so a fully closed back locks stake x (price / BSP - 1) whatever
   the result. The result splits into that locked part (the CLV), the part left open (backs with no lay, or a lay
   smaller than the back needed: the results' luck), commission, and void (non-runner) money;
3. the CLV by what the rule saw at entry (expected edge), the price, the UK hour, the minutes to the off, the day;
4. today so far.
"""

from __future__ import annotations

import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)
UK = "Europe/London"
FIRST = date(2026, 9, 30)
TODAY = pd.Timestamp.now(tz=UK).date()


def ledger(day: date) -> pd.DataFrame:
    try:
        raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day:%Y-%m-%d}/ledger.csv")["Body"].read()
    except Exception:
        return pd.DataFrame()
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    d["day"] = f"{day:%Y-%m-%d}"
    return d


frames = []
d = FIRST
while d <= TODAY:
    led = ledger(d)
    print(f"{d}: {len(led)} ledger rows")
    if len(led):
        frames.append(led)
    d += timedelta(days=1)
led = pd.concat(frames, ignore_index=True)
num = ["matched", "avg_price", "edge", "minutes_to_off", "p_model", "p_market", "hedge_liability", "bsp", "clv",
       "pnl_back", "pnl_lay", "pnl", "commission", "target", "asked", "best_back"]
for c in num:
    if c in led:
        led[c] = pd.to_numeric(led[c], errors="coerce")
led["selection_id"] = pd.to_numeric(led["selection_id"], errors="coerce")
led["t"] = pd.to_datetime(led["ts"], utc=True, errors="coerce")
print("events:", led.groupby(["day", "event"]).size().unstack(fill_value=0).to_string())

# ---------------------------------------------------------------------------------- each back's entry, its settle
backs = led[(led.event == "back") & (led.matched > 0)].copy()
backs["stake_x_price"] = backs.matched * backs.avg_price
k = ["day", "market_id", "selection_id"]
entry = backs.sort_values("t").groupby(k).agg(
    first_t=("t", "first"), first_mins=("minutes_to_off", "first"), edge=("edge", "first"), backs=("matched", "size"),
    staked=("matched", "sum"), sxp=("stake_x_price", "sum"), p_model=("p_model", "first"),
    p_market=("p_market", "first")).reset_index()
entry["entry_price"] = entry.sxp / entry.staked
st = led[led.event == "settle"].drop_duplicates(k, keep="last")
h = entry.merge(st[k + ["result", "bsp", "clv", "pnl_back", "pnl_lay", "pnl", "hedge_liability", "avg_price",
                         "error"]].rename(columns={"avg_price": "settled_price"}), on=k, how="left")
comm = led[led.event == "commission"].groupby("day")["pnl"].sum()
h["settled"] = h.result.notna()
h["void"] = h.result.eq("REMOVED")
print(f"\nhorses backed: {len(h)}, settled {int(h.settled.sum())}, void (non-runners) {int(h.void.sum())}, "
      f"unsettled {int((~h.settled).sum())}")

# the locked part: a back closed at the SP returns stake x (price / BSP - 1) whatever the result
live = h[h.settled & ~h.void].copy()
price = live.settled_price.fillna(live.entry_price)
live["locked_full"] = live.staked * (price / live.bsp - 1.0)
# the lay needed to close the back fully: stake x price / BSP, a liability of that x (BSP - 1)
live["lay_needed_liab"] = live.staked * price / live.bsp * (live.bsp - 1.0)
live["closed_share"] = (live.hedge_liability / live.lay_needed_liab).clip(upper=1.5)
live["has_lay"] = live.pnl_lay.fillna(0).ne(0)
live["won"] = live.result.eq("WINNER")
live["luck"] = live.pnl - live.locked_full * np.minimum(live.closed_share.fillna(0), 1.0)
live["hour_uk"] = live.first_t.dt.tz_convert(UK).dt.hour
live["pband"] = pd.cut(price, [1, 3, 5, 8, 13, 21, 1001], labels=["<3", "3-5", "5-8", "8-13", "13-21", "21+"])
live["mband"] = pd.cut(live.first_mins, [0, 30, 60, 120, 240, 480, 2000],
                       labels=["15-30m", "30-60m", "1-2h", "2-4h", "4-8h", "8h+"])
live["eband"] = pd.cut(live.edge, [-1, 0.03, 0.05, 0.08, 0.12, 0.2, 10],
                       labels=["<3%", "3-5%", "5-8%", "8-12%", "12-20%", "20%+"])


def table(g):
    out = g.agg(horses=("staked", "size"), staked=("staked", "sum"), locked=("locked_full", "sum"),
                pnl=("pnl", "sum"), back=("pnl_back", "sum"), lay=("pnl_lay", "sum"),
                exp_edge=("edge", "mean"), winners=("won", "sum"))
    out["clv_staked"] = out.locked / out.staked
    return out.round(3)


print("\n== 1. each day (CLV at the settled price; P&L before commission, then the day's commission)")
day = table(live.groupby("day"))
day["commission"] = comm.reindex(day.index).fillna(0).round(2)
day["result"] = (day.pnl + day.commission).round(2)
print(day.to_string())
print(f"   all days: staked GBP{live.staked.sum():,.2f}, CLV {live.locked_full.sum() / live.staked.sum():+.2%}, "
      f"locked GBP{live.locked_full.sum():+,.2f}, P&L GBP{live.pnl.sum():+,.2f}, commission GBP{comm.sum():+,.2f}, "
      f"result GBP{live.pnl.sum() + comm.sum():+,.2f}")

print("\n== 2. where the result came from")
cov = live.assign(full=live.closed_share >= 0.97, part=(live.closed_share > 0.05) & (live.closed_share < 0.97),
                  none=live.closed_share.fillna(0) <= 0.05)
for name, m in (("closed at the SP (lay >= 97% of what closes it)", cov.full), ("partly closed", cov.part),
                ("not closed (no lay at SP)", cov.none)):
    s = cov[m]
    print(f"   {name}: {len(s)} horses, GBP{s.staked.sum():,.2f} staked, locked GBP{s.locked_full.sum():+,.2f}, "
          f"P&L GBP{s.pnl.sum():+,.2f} ({int(s.won.sum())} winners)")
print(f"   the results' luck on what was left open: GBP{live.luck.sum():+,.2f}; by day: "
      f"{live.groupby('day').luck.sum().round(2).to_dict()}")
notes = live.error.fillna("").str.split("; ").explode()
print("   settle notes:", notes[notes.ne("")].str.replace(r"[0-9.]+", "#", regex=True).value_counts().head(8).to_dict())
print(f"   void (non-runner) horses: {int(h.void.sum())}, GBP{h[h.void].staked.sum():,.2f} staked (returned)")

print("\n== 3. CLV by what the rule saw and when it backed")
for col, title in (("eband", "expected edge at entry"), ("pband", "price"), ("hour_uk", "UK hour of the first back"),
                   ("mband", "minutes to the off at the first back")):
    print(f"-- {title}")
    print(table(live.groupby(col, observed=True))[["horses", "staked", "exp_edge", "clv_staked", "locked", "pnl"]]
          .to_string())
print("-- by day and morning (first back before 11:00 UK) / later")
live["half"] = np.where(live.hour_uk < 11, "by 11", "after 11")
print(table(live.groupby(["day", "half"]))[["horses", "staked", "exp_edge", "clv_staked", "locked"]].to_string())
print("-- expected edge against CLV that came true, staked-weighted, by day")
ew = live.assign(we=live.edge * live.staked).groupby("day").agg(we=("we", "sum"), s=("staked", "sum"),
                                                                 l=("locked_full", "sum"))
print((pd.DataFrame({"expected": ew.we / ew.s, "came_true": ew.l / ew.s})).round(4).to_string())

print("\n== 4. the biggest single results")
show = ["day", "market_id", "selection_id", "first_mins", "edge", "staked", "entry_price", "bsp", "clv",
        "closed_share", "result", "pnl_back", "pnl_lay", "pnl"]
print(live.sort_values("pnl").head(12)[show].round(3).to_string(index=False))
print(live.sort_values("pnl").tail(6)[show].round(3).to_string(index=False))

today = f"{TODAY:%Y-%m-%d}"
t = led[led.day == today]
if len(t):
    tb = t[(t.event == "back") & (t.matched > 0)]
    print(f"\n== today {today}: {len(tb)} backs matched, GBP{tb.matched.sum():,.2f}; last row {t.ts.max()}; "
          f"settled horses {int((t.event == 'settle').sum())}, P&L so far GBP{t[t.event.isin(['settle', 'commission'])].pnl.sum():+,.2f}")
    print("   refusals:", t[(t.event == "back") & (t.status == "FAILURE")].error.fillna("").str[:40].value_counts()
          .head(5).to_dict())
