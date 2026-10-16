"""The live tracker (the owner's ask of 16 Oct: track 17 Oct): the day so far, read-only, from the trader's ledger
(copied to S3 after every order) and the recorded books, at any hour of the session. TRACK_DAY=YYYY-MM-DD.

- the money: staked, horses, markets, the pace through the morning (when GBP1,000/2,000/3,000/4,000 went), the last back;
- the shape: first backs against top-ups, top-ups into a drift (the price backed more than 5% above the horse's average
  matched), the ten largest horses' share, the share under 5.0;
- the price taken against the market: before the off, each horse's matched price against its latest best back (a
  horse whose price has since shortened is ahead); after the off, against the BSP Betfair showed (the CLV, before any
  non-runner reduction, which the evening settle applies);
- the yardstick, 10-16 Oct: first backs +7% to +20% at the BSP, top-ups +10% (10 Oct) falling to -9% (15 Oct), the
  morning price's median margin over the BSP +19% (10-13 Oct) falling to +7% (16 Oct)."""
import io
import os

import boto3
import numpy as np
import pandas as pd

DAY = os.environ.get("TRACK_DAY")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
K = ["market_id", "selection_id"]
pd.set_option("display.width", 250)


def csv(key):
    try:
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as e:  # noqa: BLE001
        print("  missing", key, type(e).__name__)
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(raw), compression="gzip" if key.endswith(".gz") else None, low_memory=False,
                       dtype={"market_id": str})


now = pd.Timestamp.now(tz="UTC")
print(f"== the day {DAY}, read at {now:%H:%M} UTC ({now.tz_convert('Europe/London'):%H:%M} UK)")
try:
    h = s3.head_object(Bucket=bucket, Key=f"predictions/{DAY}.csv")
    print(f"predictions/{DAY}.csv written {h['LastModified']}")
except Exception as e:  # noqa: BLE001
    print("predictions not written:", type(e).__name__)
led = csv(f"trading/live/{DAY}/ledger.csv")
if led.empty:
    raise SystemExit("no ledger yet: the session has not backed anything (or has not started)")
for c in ("matched", "avg_price", "selection_id", "edge"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
print("ledger events:", led.event.value_counts().to_dict(), "| first event", led.t.min().tz_convert("Europe/London")
      .strftime("%H:%M UK"), "| last", led.t.max().tz_convert("Europe/London").strftime("%H:%M UK"))
b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").drop(columns=["bsp", "clv", "result"],
                                                                          errors="ignore").copy()
if b.empty:
    raise SystemExit("no back matched yet")
b["uk"] = b.t.dt.tz_convert("Europe/London")
b["k"] = b.groupby(K).cumcount()
b["kind"] = np.where(b.k == 0, "first", "top-up")
b["sxp"] = b.matched * b.avg_price
avg = b.groupby(K).apply(lambda g: (g.sxp.cumsum() / g.matched.cumsum()).shift(1)).reset_index(level=K, drop=True)
b["avg_before"] = avg
b["drift_topup"] = (b.k > 0) & (b.avg_price > b.avg_before * 1.05)
cum = b.matched.cumsum()
pace = {f"GBP{x:,}": (b.uk[cum >= x].min().strftime("%H:%M") if (cum >= x).any() else "-")
        for x in (1000, 2000, 3000, 3990)}
print(f"staked GBP{b.matched.sum():,.2f} on {b.selection_id.nunique()} horses in {b.market_id.nunique()} markets; "
      f"pace (UK): {pace}; last back {b.uk.max():%H:%M} UK")
hs = b.groupby(K).matched.sum()
print(f"shape: first backs GBP{b[b.k == 0].matched.sum():,.2f}, top-ups GBP{b[b.k > 0].matched.sum():,.2f} "
      f"(into a drift GBP{b[b.drift_topup].matched.sum():,.2f}); ten largest horses "
      f"{hs.nlargest(10).sum() / hs.sum():.0%} of the money; staked under 5.0 "
      f"{b[b.avg_price < 5].matched.sum() / b.matched.sum():.0%}; the expected edge at entry "
      f"{(b.edge * b.matched).sum() / b.matched.sum():+.2%}")

bk = pd.concat([csv(f"betfair_live/{DAY}/{n}.csv.gz") for n in ("books_trader", "books")], ignore_index=True)
if bk.empty:
    raise SystemExit("no books recorded yet")
for c in ("sp_actual", "back1", "selection_id"):
    bk[c] = pd.to_numeric(bk[c], errors="coerce")
bk = bk.sort_values("polled_utc")
last = bk.groupby(K).agg(back_now=("back1", "last"), sp=("sp_actual", lambda x: x[x > 1].iloc[-1] if (x > 1).any()
                                                             else np.nan),
                         status=("runner_status", "last"), polled=("polled_utc", "last")).reset_index()
mk = bk.groupby("market_id").agg(inplay=("inplay", "max"), mstatus=("status", "last")).reset_index()
b = b.merge(last, on=K, how="left").merge(mk, on="market_id", how="left")
off = b.sp > 1
b["vs"] = np.where(off, b.avg_price / b.sp - 1, b.avg_price / b.back_now - 1)
b["state"] = np.where(off, "off (at the BSP)", np.where(b.status.eq("REMOVED"), "non-runner", "before the off"))


def summ(g):
    g = g[np.isfinite(g.vs)]
    return pd.Series({"backs": len(g), "horses": g.selection_id.nunique(), "staked": round(g.matched.sum(), 2),
                      "vs_price": round(float((g.vs * g.matched).sum() / g.matched.sum()), 4) if len(g) else np.nan})


print("\n== the price taken against the market now (off: against the BSP, the CLV; before the off: against the best "
      "back now)")
print(b.groupby(["state", "kind"]).apply(summ).to_string())
o = b[off]
if len(o):
    first = o[o.k == 0]
    print(f"\nraced so far: CLV {(o.vs * o.matched).sum() / o.matched.sum():+.2%} on GBP{o.matched.sum():,.2f} "
          f"(first backs {(first.vs * first.matched).sum() / max(first.matched.sum(), 1e-9):+.2%}, top-ups "
          f"{(o[o.k > 0].vs * o[o.k > 0].matched).sum() / max(o[o.k > 0].matched.sum(), 1e-9):+.2%}, "
          f"into a drift {(o[o.drift_topup].vs * o[o.drift_topup].matched).sum() / max(o[o.drift_topup].matched.sum(), 1e-9):+.2%}); "
          f"the median margin of the first back over the BSP {np.log(first.avg_price / first.sp).median():+.3f}")
st = led[led.event == "settle"]
if len(st):
    st = st.assign(pnl=pd.to_numeric(st.get("pnl"), errors="coerce"))
    print(f"settled so far: {st.drop_duplicates(K).shape[0]} horses, P&L before the day's commission "
          f"GBP{st.pnl.sum():,.2f}")
print("\nby UK hour entered:")
print(b.assign(hr=b.uk.dt.hour).groupby(["hr", "kind"]).apply(summ).to_string())
top = b.groupby(K).agg(runner=("runner", "first"), venue=("venue", "first"), off_=("off", "first"),
                       staked=("matched", "sum"), price=("avg_price", "mean"), now=("back_now", "first"),
                       bsp=("sp", "first"), state=("state", "first")).reset_index()
print("\nthe ten largest positions:")
print(top.sort_values("staked", ascending=False).head(10).drop(columns=K).round(2).to_string(index=False))
