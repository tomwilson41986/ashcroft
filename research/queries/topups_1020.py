"""The owner's ask (20 Oct): "Breakdown the top ups more granular". (Read-only; done/below_model_1020.py's fills.) The owner's question (20 Oct, sharper): when we take a thin top price, are the next prices we top up at below our own
predicted BFSP? (read-only). Every fill of 10-19 Oct: its price against the model's predicted BFSP for the horse (the
trader's own reading: price / exp(move), move = ln(best back / predicted BFSP)) and against the closing model's
expected BSP (1 / p_pool); the CLV at Betfair's settled price of the fills below the predicted BFSP against those above,
first backs and top-ups apart; the top-ups that followed a thin first back (the best back over 10% above the second);
and a rule that never backs under the predicted BFSP (or under it by a margin): what it drops."""
import io
import os

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAYS = [f"2026-10-{d}" for d in range(10, 20)]
BOOK = ["polled_utc", "market_id", "selection_id", "status", "inplay", "back1", "back1_size", "back2", "back2_size",
        "back3", "back3_size", "lay1", "lay1_size", "market_total_matched", "runner_total_matched", "last_traded"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


fills = []
for day in DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        continue
    led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "bsp", "selection_id", "minutes_to_off", "best_back", "back_size", "edge",
              "target", "move", "p_model", "p_pool"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
    b["day"] = day
    b["sxp"] = b.matched * b.avg_price
    b["k"] = b.groupby(["market_id", "selection_id"]).cumcount()
    hm = b.groupby(["market_id", "selection_id"]).agg(m=("matched", "sum"), sxp=("sxp", "sum")).reset_index()
    hm["Pm"] = hm.sxp / hm.m
    st = led[led.event == "settle"].sort_values("t").drop_duplicates(["market_id", "selection_id"], keep="last")[
        ["market_id", "selection_id", "result", "bsp", "avg_price"]].rename(columns={"avg_price": "Ps", "bsp": "B"})
    b = b.drop(columns=["bsp", "result"], errors="ignore").merge(hm[["market_id", "selection_id", "Pm"]],
                                                                  on=["market_id", "selection_id"]) \
        .merge(st, on=["market_id", "selection_id"], how="left")
    raw = get(f"betfair_live/{day}/books_trader.csv.gz")
    if raw is None:
        print("no trader books", day)
        continue
    bk = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=lambda c: c in BOOK, dtype={"market_id": str},
                     low_memory=False)
    for c in BOOK[3:]:
        if c in bk.columns and c not in ("status", "inplay"):
            bk[c] = pd.to_numeric(bk[c], errors="coerce")
    bk["selection_id"] = pd.to_numeric(bk.selection_id, errors="coerce")
    bk = bk.dropna(subset=["selection_id"])
    bk["t"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
    bk = bk.dropna(subset=["t"]).sort_values("t")
    b = b.dropna(subset=["t"]).sort_values("t")
    b["selection_id"] = b.selection_id.astype("float64")
    bk["selection_id"] = bk.selection_id.astype("float64")
    j = pd.merge_asof(b, bk.drop(columns=["polled_utc"]).rename(columns={"t": "t_book"}), left_on="t",
                      right_on="t_book", by=["market_id", "selection_id"], direction="backward",
                      tolerance=pd.Timedelta("3min"))
    fills.append(j)
f = pd.concat(fills, ignore_index=True)
ok = f.result.isin(["WINNER", "LOSER"]) & (f.B > 1) & (f.Ps > 1)
f = f[ok].copy()
f["price_s"] = f.avg_price * f.Ps / f.Pm                     # as Betfair settled it (a non-runner's reduction)
f["clv"] = f.price_s / f.B - 1
f["locked"] = f.clv * f.matched
f["spread"] = f.lay1 / f.back1 - 1
f["top_gap"] = f.back1 / f.back2 - 1
f["took_share"] = f.matched / f.back1_size
f["above_book"] = f.avg_price / f.back1 - 1                  # the fill against the best back the trader read
f["vs_last"] = f.back1 / f.last_traded - 1                   # the best back against the last traded price
f["kind"] = np.where(f.k == 0, "first", "top-up")
have = f.back1.notna()
print(f"fills 10-19 Oct settled with a BSP: {len(f):,}, GBP{f.matched.sum():,.2f}; read against the trader's book "
      f"within 3 minutes: {int(have.sum()):,} ({have.mean():.0%}); CLV {f.locked.sum() / f.matched.sum():+.2%}, "
      f"CLV x stake GBP{f.locked.sum():,.2f}")
f = f[have].copy()
f = f.sort_values("t")
f["model_bfsp"] = f.best_back / np.exp(f.move)
f["vs_model"] = f.avg_price / f.model_bfsp - 1
first = f[f.k == 0][["day", "market_id", "selection_id", "t", "avg_price", "top_gap"]].rename(
    columns={"t": "t_first", "avg_price": "first_price", "top_gap": "first_gap"})
t = f[f.k > 0].merge(first, on=["day", "market_id", "selection_id"], how="left")
t["sxp"] = t.matched * t.avg_price
fb = f.copy()
fb["sxp"] = fb.matched * fb.avg_price
fb["cm"] = fb.groupby(["day", "market_id", "selection_id"]).matched.cumsum()
fb["cs"] = fb.groupby(["day", "market_id", "selection_id"]).sxp.cumsum()
fb["avg_before"] = (fb.cs - fb.sxp) / (fb.cm - fb.matched)
t = t.merge(fb[["day", "market_id", "selection_id", "k", "avg_before"]], on=["day", "market_id", "selection_id", "k"],
            how="left")
t["vs_first"] = t.avg_price / t.first_price - 1
t["vs_avg"] = t.avg_price / t.avg_before - 1
t["mins_after_first"] = (t.t - t.t_first).dt.total_seconds() / 60
t["rule"] = np.where(t.day >= "2026-10-19", "new (19 Oct)", "old (10-18 Oct)")
T = t.matched.sum()
print(f"top-ups 10-19 Oct with a BSP and a book: {len(t):,} fills, GBP{T:,.2f}, CLV {t.locked.sum() / T:+.2%} "
      f"(first backs in the same set GBP{f[f.k == 0].matched.sum():,.2f} at "
      f"{f[f.k == 0].locked.sum() / f[f.k == 0].matched.sum():+.2%})")


def tab(name, key):
    g = t.groupby(key, observed=True).apply(lambda x: pd.Series({
        "fills": len(x), "staked": x.matched.sum(), "share": x.matched.sum() / T,
        "clv": x.locked.sum() / x.matched.sum(), "locked": x.locked.sum(), "median_price": x.avg_price.median()}))
    print(f"\n== top-ups by {name}")
    print(g.round(4).to_string())


tab("their number on the horse (2 = the first top-up)", pd.cut(t.k + 1, [1, 2, 3, 5, 10, 20, 1000],
                                                             labels=["2nd", "3rd", "4-5th", "6-10th", "11-20th", "21st+"]))
tab("price against the model's predicted BFSP", pd.cut(t.vs_model, [-1, 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50,
                                                                    0.75, 1.0, 100]))
tab("price against the first back's price", pd.cut(t.vs_first, [-1, -0.20, -0.10, -0.05, -0.0001, 0.0001, 0.05, 0.10,
                                                               0.20, 100]))
tab("price against the horse's average matched before it", pd.cut(t.vs_avg, [-1, -0.10, -0.05, -0.0001, 0.0001, 0.05,
                                                                             0.10, 100]))
tab("minutes after the first back", pd.cut(t.mins_after_first, [-0.1, 2, 5, 15, 30, 60, 120, 240, 10000]))
tab("minutes to the off", pd.cut(t.minutes_to_off, [0, 60, 120, 240, 360, 480, 600, 10000]))
tab("UK hour", t.t.dt.tz_convert("Europe/London").dt.hour)
tab("price", pd.cut(t.avg_price, [1, 2, 3, 4, 5, 7, 10, 15, 25, 1001]))
tab("expected CLV at the top-up (edge)", pd.cut(t.edge, [0, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 10]))
tab("the fill's size", pd.cut(t.matched, [0, 2.5, 5, 10, 20, 50, 1000]))
tab("the market's matched money", pd.cut(t.market_total_matched, [-1, 1000, 2500, 5000, 10000, 20000, 1e12]))
tab("the first back a thin top price (best back over 10% above the second)", t.first_gap > 0.10)
tab("rule", t.rule)
print("\n== top-ups by price against the model and minutes after the first back (CLV, staked)")
x = t.assign(vm=pd.cut(t.vs_model, [-1, 0, 0.10, 0.25, 0.50, 100]),
             mf=pd.cut(t.mins_after_first, [-0.1, 5, 30, 120, 10000]))
print(x.groupby(["vm", "mf"], observed=True).apply(lambda g: pd.Series({
    "staked": g.matched.sum(), "clv": g.locked.sum() / g.matched.sum()})).unstack("mf").round(3).to_string())
print("\n== top-ups by price against the model and the expected CLV (CLV, staked)")
x = t.assign(vm=pd.cut(t.vs_model, [-1, 0, 0.10, 0.25, 0.50, 100]), ed=pd.cut(t.edge, [0, 0.04, 0.06, 0.10, 10]))
print(x.groupby(["vm", "ed"], observed=True).apply(lambda g: pd.Series({
    "staked": g.matched.sum(), "clv": g.locked.sum() / g.matched.sum()})).unstack("ed").round(3).to_string())
print("\n== top-ups by price against the model, by day (CLV)")
print(t.assign(vm=pd.cut(t.vs_model, [-1, 0, 0.10, 0.25, 0.50, 100])).groupby(["day", "vm"], observed=True).apply(
    lambda g: g.locked.sum() / g.matched.sum()).unstack("vm").round(3).to_string())
