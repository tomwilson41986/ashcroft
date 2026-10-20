"""The owner's questions (20 Oct): "should we have a max price to back? seems like we might lose on big priced horses"
and "13:50 Yarmouth - why did we put GBP15 on a horse priced 1000" (read-only).

1. Today's 13:50 Yarmouth: every ledger row of the race (the backs with the price, the money offered, our predicted
   BFSP, the closing model's expected BSP and CLV, the target and why; the lays at the SP), and every back of today at
   100 or more.
2. Every position of 30 Sep - 19 Oct by the price it was backed at: horses, money, CLV at the BSP, the result as
   settled (each back laid at the SP: a winner nets nothing, a loser S(P - B)/(B - 1)), winners, the drift
   (BSP / price), how many were hedged.
3. Our own lay at the SP at long prices. The lay's liability is the back's winnings (GBP15 at 1000 is GBP14,985), so on
   an outsider our lay can be most of the money laid at the SP and push the BSP out against us. From the trader's books:
   the market's lay liability and back stakes taken at the SP (sp_lay_taken, sp_back_taken), our share of the lay
   liability, and the BSP against the last best back and lay before the off, for our long-priced horses and the
   others at the same prices.
"""
import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)
K = ["day", "market_id", "selection_id"]
TODAY = "2026-10-20"
DAYS = [(date(2026, 9, 30) + timedelta(n)).isoformat() for n in range(21)]
NUM = ["matched", "avg_price", "bsp", "clv", "selection_id", "edge", "pnl", "commission", "minutes_to_off",
       "back_size", "best_back", "target", "asked", "p_model", "p_pool", "p_market", "move", "hedge_liability"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


leds, books = [], []
BOOK = ["polled_utc", "source", "market_id", "selection_id", "status", "inplay", "runner_status", "back1", "back1_size",
        "lay1", "lay1_size", "last_traded", "sp_near", "sp_far", "sp_actual", "sp_back_taken", "sp_lay_taken",
        "market_total_matched", "runner_total_matched"]
for day in DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        print("no ledger", day)
        continue
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    d["day"] = day
    leds.append(d)
    raw = get(f"betfair_live/{day}/books_trader.csv.gz")
    if raw is None:
        continue
    b = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=lambda c: c in BOOK, dtype={"market_id": str},
                    low_memory=False)
    b["day"] = day
    books.append(b)
led = pd.concat(leds, ignore_index=True)
for c in NUM:
    if c in led.columns:
        led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")

print("== 1. today's 13:50 Yarmouth")
td = led[led.day == TODAY].copy()
off = pd.to_datetime(td["off"], utc=True, errors="coerce")
yar = td[td.venue.astype(str).str.contains("Yarm", case=False) & (off.dt.tz_convert("Europe/London").dt.strftime("%H:%M") == "13:50")]
if yar.empty:
    print("   no ledger row for Yarmouth 13:50 today (venues today:", sorted(td.venue.dropna().unique()), ")")
else:
    yar = yar.copy()
    yar["our_bfsp"] = yar.best_back / np.exp(yar.move)
    yar["exp_bsp"] = 1.0 / yar.p_pool
    yar["uk"] = yar.t.dt.tz_convert("Europe/London").dt.strftime("%H:%M:%S")
    cols = ["uk", "event", "runner", "best_back", "back_size", "our_bfsp", "exp_bsp", "edge", "p_model", "p_market",
            "target", "asked", "matched", "avg_price", "status", "error", "hedge_liability", "result", "bsp", "pnl",
            "reason"]
    print(yar[[c for c in cols if c in yar.columns]].to_string(index=False))
big = td[(td.event == "back") & (td.matched > 0) & (td.avg_price >= 100)].copy()
big["our_bfsp"] = big.best_back / np.exp(big.move)
big["exp_bsp"] = 1.0 / big.p_pool
print(f"\n   today's backs at 100 or more: {len(big)}")
if len(big):
    big["uk"] = big.t.dt.tz_convert("Europe/London").dt.strftime("%H:%M")
    print(big[["uk", "venue", "runner", "best_back", "back_size", "our_bfsp", "exp_bsp", "edge", "target", "matched",
               "avg_price"]].to_string(index=False))

print("\n== 2. every position 30 Sep - 19 Oct by the price backed at")
b = led[(led.event == "back") & (led.matched > 0) & (led.day < TODAY)].sort_values("t").copy()
b["sxp"] = b.matched * b.avg_price
b["our_bfsp"] = b.best_back / np.exp(b.move)
pos = b.groupby(K).agg(S=("matched", "sum"), sxp=("sxp", "sum"), backs=("matched", "size"), runner=("runner", "first"),
                       venue=("venue", "first"), first_price=("avg_price", "first"), our_bfsp=("our_bfsp", "first"),
                       edge=("edge", "first"), p_pool=("p_pool", "first")).reset_index()
pos["Pm"] = pos.sxp / pos.S
st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[K + ["result", "bsp", "avg_price", "pnl"]] \
    .rename(columns={"avg_price": "P"})
pos = pos.merge(st, on=K, how="left")
tr = led[led.event == "trade_out"].groupby(K).agg(lay_liab=("hedge_liability", "sum"),
                                                  lay_ok=("status", lambda s: s.isin(["SUCCESS", "PENDING"]).all()),
                                                  hedged=("hedged", "last")).reset_index()
pos = pos.merge(tr, on=K, how="left")
v = pos[pos.result.isin(["WINNER", "LOSER"]) & (pos.bsp > 1) & (pos.P > 1)].copy()
v["clv"] = v.P / v.bsp - 1
v["drift"] = np.log(v.bsp / v.Pm)
BANDS = [1, 5, 10, 21, 50, 100, 200, 1001]
v["band"] = pd.cut(v.Pm, BANDS, right=False)


def band(g):
    return pd.Series({"horses": len(g), "staked": g.S.sum(), "share": g.S.sum() / v.S.sum(),
                      "clv": (g.clv * g.S).sum() / g.S.sum(), "clv_x_stake": (g.clv * g.S).sum(),
                      "result": g.pnl.sum(), "winners": int(g.result.eq("WINNER").sum()),
                      "median_drift": g.drift.median(), "share_drifted": (g.drift > 0).mean(),
                      "median_bsp_over_price": float(np.median(g.bsp / g.Pm)),
                      "median_bsp_over_our_bfsp": float(np.median(g.bsp / g.our_bfsp)),
                      "lay_liability": g.lay_liab.sum(), "all_laid": g.lay_ok.fillna(False).mean()})


print(v.groupby("band", observed=True).apply(band).round(4).to_string())
print(f"   all: {len(v):,} horses, GBP{v.S.sum():,.2f}, CLV {(v.clv * v.S).sum() / v.S.sum():+.2%}, result "
      f"GBP{v.pnl.sum():,.2f}")
long = v[v.Pm >= 50].sort_values("Pm", ascending=False)
print("\n   every horse backed at 50 or more")
print(long[["day", "venue", "runner", "Pm", "our_bfsp", "edge", "S", "bsp", "clv", "result", "pnl", "lay_liab"]]
      .round(3).to_string(index=False))

print("\n== 3. our lay at the SP at long prices, against the market's money at the SP")
if books:
    bk = pd.concat(books, ignore_index=True)
    for c in ["selection_id", "back1", "back1_size", "lay1", "lay1_size", "last_traded", "sp_near", "sp_far", "sp_actual",
              "sp_back_taken", "sp_lay_taken", "market_total_matched", "runner_total_matched"]:
        if c in bk.columns:
            bk[c] = pd.to_numeric(bk[c], errors="coerce")
    bk["t"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
    ip = bk.inplay.astype(str).str.lower().isin(["1", "true", "1.0"])
    pre = bk[(bk.status == "OPEN") & ~ip].sort_values("t").drop_duplicates(K, keep="last")[
        K + ["back1", "lay1", "last_traded", "sp_near", "sp_far", "sp_back_taken", "sp_lay_taken"]].rename(
        columns={"back1": "last_back", "lay1": "last_lay", "sp_back_taken": "pre_sp_back", "sp_lay_taken": "pre_sp_lay"})
    taken = bk[bk.sp_lay_taken.notna() | bk.sp_back_taken.notna()].sort_values("t").drop_duplicates(K, keep="last")[
        K + ["sp_back_taken", "sp_lay_taken", "sp_actual"]]
    allr = pre.merge(taken, on=K, how="left")
    bsp_all = bk[bk.sp_actual > 1].sort_values("t").drop_duplicates(K, keep="last")[K + ["sp_actual"]].rename(
        columns={"sp_actual": "B"})
    allr = allr.merge(bsp_all, on=K, how="left")
    allr = allr[(allr.B > 1) & (allr.last_back > 1)].copy()
    allr["selection_id"] = allr.selection_id.astype("int64")
    v2 = v.copy()
    v2["selection_id"] = v2.selection_id.astype("int64")
    allr = allr.merge(v2[K + ["Pm", "S", "lay_liab"]], on=K, how="left")
    allr["ours"] = allr.S.notna()
    allr["bsp_over_last_back"] = allr.B / allr.last_back
    allr["bsp_over_last_lay"] = allr.B / allr.last_lay
    allr["our_share_sp_lay"] = allr.lay_liab / allr.sp_lay_taken
    print(f"   runners with a book before the off and a BSP: {len(allr):,}; SP money read for "
          f"{allr.sp_lay_taken.notna().sum():,}")
    allr["band"] = pd.cut(allr.last_back, BANDS, right=False)
    print(allr.groupby(["band", "ours"], observed=True).apply(lambda g: pd.Series({
        "runners": len(g), "median_bsp_over_last_back": g.bsp_over_last_back.median(),
        "median_bsp_over_last_lay": g.bsp_over_last_lay.median(),
        "share_bsp_over_last_lay": (g.B > g.last_lay).mean(),
        "median_sp_lay_taken": g.sp_lay_taken.median(), "median_sp_back_taken": g.sp_back_taken.median(),
        "median_our_lay_liab": g.lay_liab.median(), "median_our_share_sp_lay": g.our_share_sp_lay.median()})
    ).round(3).to_string())
    ol = allr[allr.ours & (allr.Pm >= 50)].sort_values("Pm", ascending=False)
    print("\n   our horses backed at 50 or more: the last book before the off, the SP money and the BSP")
    print(ol[["day", "selection_id", "Pm", "last_back", "last_lay", "sp_near", "sp_far", "B", "lay_liab",
              "sp_lay_taken", "sp_back_taken", "our_share_sp_lay"]].round(2).to_string(index=False))
