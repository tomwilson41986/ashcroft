"""The live tracker on 21 Oct, about 10:30 UK (the owner: "how are we doing today so far?"): the first morning on the
guard (never under our predicted BFSP), the ladder (down the three best backs while they clear the bar and the guard)
and the maximum price of 200, with checks that the trader trades as built (read-only)."""
import io
import os
import runpy
import sys

import boto3
import numpy as np
import pandas as pd

os.environ["TRACK_DAY"] = "2026-10-21"
sys.argv = ["track_day.py"]
try:
    runpy.run_path(os.path.join(os.getcwd(), "research", "queries", "done", "track_day.py"), run_name="__main__")
except SystemExit as e:
    print("tracker:", e)

print("\n== the three changes as built (live from 21 Oct)")
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
raw = s3.get_object(Bucket=os.environ.get("PRED_BUCKET") or "ashcroft", Key="trading/live/2026-10-21/ledger.csv")["Body"].read()
led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
for c in ("matched", "avg_price", "target", "best_back", "selection_id", "move", "edge", "back_size", "asked"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
b = led[(led.event == "back") & (led.matched > 0)].copy()
print(f"backs matched {len(b)}, GBP{b.matched.sum():,.2f} on {b.groupby(['market_id', 'selection_id']).ngroups} horses")
print(f"the guard: backs at a price under our predicted BFSP (move < 0): {int((b.move < -1e-9).sum())} "
      f"(expect 0); the lowest move {b.move.min():+.3f}")
print(f"the cap: backs over 200: {int((b.avg_price > 200.0001).sum())} (expect 0); the highest price backed "
      f"{b.avg_price.max():.2f}")
lad = b[b.reason.astype(str).str.contains("ladder to", na=False)]
print(f"the ladder: backs down the ladder {len(lad)} (GBP{lad.matched.sum():,.2f}, {lad.matched.sum() / b.matched.sum():.0%} "
      f"of the money); their average matched price against the best back {(lad.avg_price / lad.best_back - 1).mean():+.2%}; "
      f"the money above the best back's size {(lad.matched - lad.back_size).clip(lower=0).sum():,.2f}")
print(f"orders asked more than the best back offered (the ladder's extra depth): "
      f"{int((b.matched > b.back_size + 0.01).sum())}")
sk = led[led.event == "skip"]
print("skips by reason (one line a runner and reason):")
print(sk.error.astype(str).str.slice(0, 70).value_counts().head(10).to_string())

print("\n== every market: our money, the expected CLV at entry (EV) and the CLV now (before the off: our average matched "
      "price against the best back now; after the off: against the BSP)")
bk = []
for n in ("books_trader", "books"):
    try:
        r = s3.get_object(Bucket=os.environ.get("PRED_BUCKET") or "ashcroft", Key=f"betfair_live/2026-10-21/{n}.csv.gz")["Body"].read()
        bk.append(pd.read_csv(io.BytesIO(r), compression="gzip", low_memory=False, dtype={"market_id": str},
                              usecols=lambda c: c in ("polled_utc", "market_id", "selection_id", "back1", "lay1",
                                                      "sp_actual", "runner_status", "status", "inplay")))
    except Exception as e:  # noqa: BLE001
        print("  no", n, type(e).__name__)
bk = pd.concat(bk, ignore_index=True)
for c in ("selection_id", "back1", "lay1", "sp_actual"):
    bk[c] = pd.to_numeric(bk[c], errors="coerce")
bk = bk.sort_values("polled_utc")
ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
pre = bk[(bk.status == "OPEN") & ~ip]
now_px = pre.groupby(["market_id", "selection_id"]).agg(back_now=("back1", "last"), lay_now=("lay1", "last"),
                                                         read=("polled_utc", "last")).reset_index()
sp = bk[bk.sp_actual > 1].groupby(["market_id", "selection_id"]).sp_actual.last().rename("bsp").reset_index()
rem = bk.groupby(["market_id", "selection_id"]).runner_status.last().rename("rstatus").reset_index()
b = led[(led.event == "back") & (led.matched > 0)].copy()
b["sxp"], b["exs"] = b.matched * b.avg_price, b.matched * b.edge
h = b.groupby(["market_id", "selection_id"]).agg(venue=("venue", "first"), off=("off", "first"), runner=("runner", "first"),
                                                 S=("matched", "sum"), sxp=("sxp", "sum"), exs=("exs", "sum"),
                                                 backs=("matched", "size")).reset_index()
h["selection_id"] = h.selection_id.astype("int64")
for x in (now_px, sp, rem):
    x["selection_id"] = x.selection_id.astype("int64")
h = h.merge(now_px, on=["market_id", "selection_id"], how="left").merge(sp, on=["market_id", "selection_id"], how="left") \
    .merge(rem, on=["market_id", "selection_id"], how="left")
h["Pm"], h["ev"] = h.sxp / h.S, h.exs / h.S
h["ref"] = np.where(h.bsp > 1, h.bsp, h.back_now)
h["clv_now"] = np.where(h.rstatus.eq("REMOVED"), np.nan, h.Pm / h.ref - 1)
h["state"] = np.where(h.rstatus.eq("REMOVED"), "non-runner", np.where(h.bsp > 1, "run (BSP)", "to run"))
h["off_uk"] = pd.to_datetime(h.off, errors="coerce").dt.strftime("%H:%M")   # the ledger keeps the UK time (run 39577487440 read it as UTC: an hour late)
ok = h.clv_now.notna()


def mkt(g):
    k = g[g.clv_now.notna()]
    return pd.Series({"horses": len(g), "staked": g.S.sum(), "ev": (g.ev * g.S).sum() / g.S.sum(),
                      "ev_gbp": (g.ev * g.S).sum(),
                      "clv_now": (k.clv_now * k.S).sum() / k.S.sum() if len(k) else np.nan,
                      "clv_now_gbp": (k.clv_now * k.S).sum(), "state": ", ".join(sorted(g.state.unique()))})


m = h.groupby(["off_uk", "venue", "market_id"]).apply(mkt).reset_index().sort_values(["off_uk", "venue"])
print(m.drop(columns="market_id").round(4).to_string(index=False))
print(f"\nall: {len(h)} horses in {len(m)} markets, GBP{h.S.sum():,.2f}; EV {(h.ev * h.S).sum() / h.S.sum():+.2%} "
      f"(GBP{(h.ev * h.S).sum():,.2f}); CLV now {(h[ok].clv_now * h[ok].S).sum() / h[ok].S.sum():+.2%} "
      f"(GBP{(h[ok].clv_now * h[ok].S).sum():,.2f}); latest book read {now_px.read.max()}")
print("\n== every horse (by the off)")
print(h.sort_values(["off_uk", "venue", "S"], ascending=[True, True, False])[
    ["off_uk", "venue", "runner", "backs", "S", "Pm", "ev", "back_now", "lay_now", "bsp", "clv_now", "state"]]
      .round(3).to_string(index=False))
