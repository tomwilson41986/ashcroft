"""The owner's day report (the standard approach, 22 Oct: "never give me inflated expected value, i need real figures
that i can expect and forecast"; read-only). REPORT_DAY picks the day (default: today in the UK). Three figures only:

1. Settled: the races run and settled, as Betfair settled them (the ledger's settled P&L less commission; where the
   ledger holds no commission for a winning market, Betfair's 2% of its net winnings).
2. The mark on the races to run: every position valued now at the mid (the geometric mean of the best back and best
   lay) and at the best lay (Betfair's Cash Out), each fill's price first cut by the reduction factors of the runners
   withdrawn after it was matched, a non-runner's own stakes void; a race that has run but is not settled at its BSP.
3. The forecast: the trader's expected value at the fill (edge x stake) on the races to run, put through the record:
   the ratio of Betfair's settled result to the expected value over every live day from 10 Oct (the owner's rule of
   9 Oct) to the day before, and over the last 7 of them; the range is the 10th-90th percentile of the past days' misses
   around their own calibrated forecast, scaled by the square root of the share of the day's expected value still to
   run. The day's forecast is the settled result plus the forecast on the races to run. The raw expected value is never
   the figure: on 10-21 Oct it ran at 0.61 of the result (forecast-1022).
"""
import io
import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAY = os.environ.get("REPORT_DAY") or datetime.now(ZoneInfo("Europe/London")).date().isoformat()
FIRST = date(2026, 10, 10)
RATE = 0.02
K = ["market_id", "selection_id"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def ledger(day):
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        return None
    led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "edge", "selection_id", "pnl", "commission", "bsp"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    return led


def settled_by_market(led):
    """Each settled market's result after commission (the ledger's, else 2% of a winning market's net)."""
    st = led[led.event == "settle"].drop_duplicates(K, keep="last")
    if st.empty:
        return pd.DataFrame(columns=["market_id", "pnl", "commission", "result"])
    p = st.groupby("market_id").pnl.sum().rename("pnl").reset_index()
    c = led[led.event == "commission"].groupby("market_id").commission.sum().rename("commission").reset_index()
    p = p.merge(c, on="market_id", how="left")
    p["commission"] = np.where(p.commission.fillna(0) > 0, p.commission.fillna(0), RATE * p.pnl.clip(lower=0))
    p["result"] = p.pnl - p.commission
    return p


# ------------------------------------------------------------------ the record: expected value against the result
rec = []
d = FIRST
today = date.fromisoformat(DAY)
while d < today:
    led = ledger(d.isoformat())
    d += timedelta(days=1)
    if led is None:
        continue
    b = led[(led.event == "back") & (led.matched > 0)]
    s = settled_by_market(led)
    rec.append({"day": (d - timedelta(days=1)).isoformat(), "ev": float((b.edge * b.matched).sum()),
                "result": float(s.result.sum())})
rec = pd.DataFrame(rec).set_index("day")
k_all = rec.result.sum() / rec.ev.sum()
last7 = rec.tail(7)
k_7 = last7.result.sum() / last7.ev.sum()
miss = rec.result - rec.ev * k_all
lo, hi = miss.quantile(0.1), miss.quantile(0.9)

# ------------------------------------------------------------------ the day
led = ledger(DAY)
if led is None:
    raise SystemExit(f"no live ledger for {DAY}")
parts = []
for n in ("books_trader", "books"):
    raw = get(f"betfair_live/{DAY}/{n}.csv.gz")
    if raw is not None:
        parts.append(pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False, dtype={"market_id": str},
                                 usecols=lambda c: c in ("polled_utc", "market_id", "selection_id", "status", "inplay",
                                                         "runner_status", "back1", "lay1", "sp_actual",
                                                         "adjustment_factor", "removal_utc")))
bk = pd.concat(parts, ignore_index=True)
for c in ("selection_id", "back1", "lay1", "sp_actual", "adjustment_factor"):
    bk[c] = pd.to_numeric(bk[c], errors="coerce")
bk = bk.dropna(subset=["selection_id"]).sort_values("polled_utc")
bk["selection_id"] = bk.selection_id.astype("int64")
rem = bk[bk.runner_status.eq("REMOVED")]
first_seen = rem.groupby(K).polled_utc.min()
rem = rem.drop_duplicates(K, keep="last")[K + ["adjustment_factor", "removal_utc"]].copy()
rem["removal_t"] = pd.to_datetime(rem.removal_utc, utc=True, errors="coerce")
rem["removal_t"] = rem.removal_t.fillna(pd.to_datetime(rem.set_index(K).index.map(first_seen), utc=True)
                                        .to_series(index=rem.index))
ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
pre = bk[(bk.status == "OPEN") & ~ip & bk.runner_status.fillna("ACTIVE").eq("ACTIVE")]
nowpx = pre.groupby(K).agg(back_now=("back1", "last"), lay_now=("lay1", "last")).reset_index()
sp = bk[bk.sp_actual > 1].groupby(K).sp_actual.last().rename("bsp_now").reset_index()

b = led[(led.event == "back") & (led.matched > 0)].copy()
b["selection_id"] = b.selection_id.astype("int64")
factor = np.ones(len(b))
for i, r in enumerate(b.itertuples()):
    rr = rem[(rem.market_id == r.market_id) & (rem.selection_id != r.selection_id) & (rem.removal_t > r.t)]
    for a in rr.adjustment_factor.fillna(0):
        factor[i] *= max(0.0, 1 - a / 100.0)
b["P_red"] = np.maximum(1.01, b.avg_price * factor)
b = b.merge(rem[K].assign(void=True), on=K, how="left")
b["void"] = b.void.fillna(False).astype(bool)
b = b.merge(nowpx, on=K, how="left").merge(sp, on=K, how="left")
sm = settled_by_market(led)
b["settled"] = b.market_id.isin(sm.market_id)
live = b[~b.void].copy()
ran = live.bsp_now > 1
mid = np.sqrt(live.back_now * live.lay_now)
live["mark_mid"] = live.matched * (live.P_red / np.where(ran, live.bsp_now, mid) - 1)
live["mark_lay"] = live.matched * (live.P_red / np.where(ran, live.bsp_now, live.lay_now) - 1)
live["ev"] = live.matched * live.edge
live["off_uk"] = pd.to_datetime(live.off, format="mixed", errors="coerce").dt.strftime("%H:%M")
m = live.groupby(["off_uk", "venue", "market_id"]).agg(
    horses=("selection_id", "nunique"), staked=("matched", "sum"), mark_mid=("mark_mid", "sum"),
    cash_out=("mark_lay", "sum"), ev=("ev", "sum"), settled=("settled", "any")).reset_index()
m = m.merge(sm[["market_id", "result"]], on="market_id", how="left")
for c in ("mark_mid", "cash_out"):
    m.loc[m.settled, c] = np.nan
m["forecast"] = np.where(m.settled, np.nan, m.ev * k_all)

ev_day = float(live.ev.sum())
ev_left = float(m.loc[~m.settled, "ev"].sum())
share = ev_left / ev_day if ev_day > 0 else 0.0
done_res = float(sm.result.sum())
f_all = done_res + ev_left * k_all
f_7 = done_res + ev_left * k_7
r_lo, r_hi = f_all + lo * np.sqrt(share), f_all + hi * np.sqrt(share)

print(f"== the day report, {DAY}: books read to {bk.polled_utc.max()} UTC")
print(f"   staked GBP{b.matched.sum():,.2f} on {b.groupby(K).ngroups} horses in {b.market_id.nunique()} races; void "
      f"(our non-runners) GBP{b[b.void].matched.sum():,.2f}; prices cut by other non-runners on "
      f"GBP{live[live.P_red < live.avg_price - 1e-9].matched.sum():,.2f}")
print(f"\n1. settled: {len(sm)} races, GBP{done_res:+,.2f} after commission")
left = m[~m.settled]
print(f"2. the mark on the {len(left)} races to run (after the cuts): at the mid GBP{left.mark_mid.sum():+,.2f}; "
      f"Cash Out (the lays) GBP{left.cash_out.sum():+,.2f}")
print(f"3. the forecast for the day: GBP{f_all:+,.2f} on the record ({len(rec)} days, result / expected {k_all:.2f}); "
      f"GBP{f_7:+,.2f} on the last 7 days ({k_7:.2f}); range GBP{r_lo:+,.2f} to GBP{r_hi:+,.2f} (10th-90th)")
print(f"   the record: {len(rec)} days GBP{rec.result.sum():+,.2f}, a day mean GBP{rec.result.mean():+,.2f}, "
      f"{int((rec.result > 0).sum())} up; the last 7 days GBP{last7.result.sum():+,.2f}")
print("\n== every race (GBP; mark and Cash Out on the races to run, the result on the races settled)")
out = m[["off_uk", "venue", "horses", "staked", "mark_mid", "cash_out", "forecast", "result"]].sort_values("off_uk")
print(out.round(2).to_string(index=False))
