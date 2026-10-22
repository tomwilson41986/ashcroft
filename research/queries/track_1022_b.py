"""The tracker on 22 Oct (the owner: "how are we doing today so far?"; read-only), as done/track_1021_d.py: each fill's
price cut by the factors of the runners removed after it was matched, a non-runner's own stakes void, and the value now
against the best back, the mid and the best lay (Betfair's Cash Out). Every market, and the total."""
import io
import os

import boto3
import numpy as np
import pandas as pd

DAY = "2026-10-22"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{DAY}/ledger.csv")["Body"].read()
led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
for c in ("matched", "avg_price", "edge", "selection_id", "pnl", "commission", "bsp"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
COLS = ("polled_utc", "market_id", "selection_id", "status", "inplay", "runner_status", "back1", "lay1", "sp_actual",
        "adjustment_factor", "removal_utc")
parts = []
for n in ("books_trader", "books"):
    try:
        r = s3.get_object(Bucket=bucket, Key=f"betfair_live/{DAY}/{n}.csv.gz")["Body"].read()
        parts.append(pd.read_csv(io.BytesIO(r), compression="gzip", low_memory=False, dtype={"market_id": str},
                                 usecols=lambda c: c in COLS))
    except Exception as e:  # noqa: BLE001
        print("  no", n, type(e).__name__)
bk = pd.concat(parts, ignore_index=True)
for c in ("selection_id", "back1", "lay1", "sp_actual", "adjustment_factor"):
    bk[c] = pd.to_numeric(bk[c], errors="coerce")
bk = bk.dropna(subset=["selection_id"]).sort_values("polled_utc")
bk["selection_id"] = bk.selection_id.astype("int64")
now_read = bk.polled_utc.max()
print(f"== {DAY}: books read to {now_read}")
# the removals: each removed runner's reduction factor and when
rem = bk[bk.runner_status.eq("REMOVED")].sort_values("polled_utc").drop_duplicates(["market_id", "selection_id"], keep="last")
rem = rem[["market_id", "selection_id", "adjustment_factor", "removal_utc"]].copy()
rem["removal_t"] = pd.to_datetime(rem.removal_utc, utc=True, errors="coerce")
first_seen = bk[bk.runner_status.eq("REMOVED")].groupby(["market_id", "selection_id"]).polled_utc.min()
rem["removal_t"] = rem.removal_t.fillna(pd.to_datetime(rem.set_index(["market_id", "selection_id"]).index.map(first_seen),
                                                       utc=True).to_series(index=rem.index))
print(f"   non-runners in our markets: {len(rem[rem.market_id.isin(led.market_id)])}")
ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
pre = bk[(bk.status == "OPEN") & ~ip & bk.runner_status.fillna("ACTIVE").eq("ACTIVE")]
nowpx = pre.groupby(["market_id", "selection_id"]).agg(back_now=("back1", "last"), lay_now=("lay1", "last")).reset_index()
sp = bk[bk.sp_actual > 1].groupby(["market_id", "selection_id"]).sp_actual.last().rename("bsp_now").reset_index()

b = led[(led.event == "back") & (led.matched > 0)].copy()
b["selection_id"] = b.selection_id.astype("int64")
factor = np.ones(len(b))
for i, r in enumerate(b.itertuples()):
    rr = rem[(rem.market_id == r.market_id) & (rem.selection_id != r.selection_id) & (rem.removal_t > r.t)]
    for a in rr.adjustment_factor.fillna(0):
        factor[i] *= max(0.0, 1 - a / 100.0)
b["P_red"] = np.maximum(1.01, b.avg_price * factor)
b["reduced"] = factor < 0.9999
b = b.merge(rem[["market_id", "selection_id"]].assign(void=True), on=["market_id", "selection_id"], how="left")
b["void"] = b.void.fillna(False).astype(bool)
b = b.merge(nowpx, on=["market_id", "selection_id"], how="left").merge(sp, on=["market_id", "selection_id"], how="left")
live = b[~b.void].copy()
mid = np.sqrt(live.back_now * live.lay_now)
off = live.bsp_now > 1
for name, ref in (("back", live.back_now), ("mid", mid), ("lay", live.lay_now)):
    x = np.where(off, live.bsp_now, ref)
    live[f"v_{name}"] = live.matched * (live.P_red / x - 1)
live["v_ev"] = live.matched * live.edge
live["v_asmatched_back"] = live.matched * (live.avg_price / np.where(off, live.bsp_now, live.back_now) - 1)
live["off_uk"] = pd.to_datetime(live.off, errors="coerce").dt.strftime("%H:%M")
print(f"   backs {len(b)}, GBP{b.matched.sum():,.2f}; void (non-runners) GBP{b[b.void].matched.sum():,.2f}; prices cut by a "
      f"reduction factor on {int(b[~b.void].reduced.sum())} backs (GBP{b[~b.void & b.reduced].matched.sum():,.2f})")
m = live.groupby(["off_uk", "venue"]).agg(horses=("selection_id", "nunique"), staked=("matched", "sum"),
                                         ev=("v_ev", "sum"), as_before=("v_asmatched_back", "sum"), vs_back=("v_back", "sum"),
                                         vs_mid=("v_mid", "sum"), vs_lay_cash_out=("v_lay", "sum"),
                                         reduced=("reduced", "any"), run=("bsp_now", lambda s: bool((s > 1).any()))).reset_index()
print("\n== every market: GBP (a market that has run is valued at the BSP in every column)")
print(m.round(2).to_string(index=False))
t = m[["staked", "ev", "as_before", "vs_back", "vs_mid", "vs_lay_cash_out"]].sum()
print(f"\nall: staked GBP{t.staked:,.2f} (void excluded); EV GBP{t.ev:,.2f}; the tracker as before GBP{t.as_before:,.2f}; "
      f"corrected for the reductions: against the best back GBP{t.vs_back:,.2f}, the mid GBP{t.vs_mid:,.2f}, the best lay "
      f"(Cash Out) GBP{t.vs_lay_cash_out:,.2f}")
st = led[led.event == "settle"]
print(f"settled so far: {st.market_id.nunique()} races, result GBP{st.pnl.sum():,.2f} before commission")
