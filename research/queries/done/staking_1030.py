"""The evening check-in of 30 Oct: done/staking_1029.py with 30 Oct in. The first four days on the owner's staking (19 Oct; PR #155: to win GBP400 at the price now, at least GBP15 and at most
GBP200 a horse, top-ups to that stake) against the old rule's days 10-18 Oct (read-only): the money by first price
band, the median and largest stake, the CLV at Betfair's settled price (the price after any non-runner reduction, so
the trader's own CLV) by band; the horses at the floor; the top-ups at a price above the horse's average."""
import io
import os

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
K = ["day", "market_id", "selection_id"]
DAYS = [f"2026-10-{d}" for d in range(10, 31)]
frames = []
for day in DAYS:
    raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day}/ledger.csv")["Body"].read()
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    d["day"] = day
    frames.append(d)
led = pd.concat(frames, ignore_index=True)
for c in ("matched", "avg_price", "bsp", "target", "selection_id"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
b["k"] = b.groupby(K).cumcount()
b["sxp"] = b.matched * b.avg_price
b["cum"] = b.groupby(K).matched.cumsum()
b["avg_before"] = (b.groupby(K).sxp.cumsum() - b.sxp) / (b.cum - b.matched)
pos = b.groupby(K).agg(S=("matched", "sum"), first_price=("avg_price", "first"), backs=("matched", "size"),
                       runner=("runner", "first"), target_max=("target", "max")).reset_index()
st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[K + ["result", "bsp", "avg_price"]] \
    .rename(columns={"avg_price": "P"})
pos = pos.merge(st, on=K, how="left")
ok = pos.result.isin(["WINNER", "LOSER"]) & (pos.bsp > 1) & (pos.P > 1)
pos["clv"] = np.where(ok, pos.P / pos.bsp - 1, np.nan)
pos["rule"] = np.where(pos.day >= "2026-10-19", "19-22 Oct (new)", "10-18 Oct (old)")
pos["band"] = pd.cut(pos.first_price, [1, 3, 5, 8, 13, 21, 1001], right=False,
                     labels=["<3", "3-5", "5-8", "8-13", "13-21", "21+"])


def summ(g):
    s = g[np.isfinite(g.clv)]
    return pd.Series({"horses_a_day": len(g) / g.day.nunique(), "staked_a_day": g.S.sum() / g.day.nunique(),
                      "share": np.nan, "median_stake": g.S.median(), "max_stake": g.S.max(),
                      "clv": (s.clv * s.S).sum() / s.S.sum() if len(s) else np.nan})


out = pos.groupby(["rule", "band"], observed=True).apply(summ)
for r in out.index.get_level_values(0).unique():
    out.loc[r, "share"] = (out.loc[r, "staked_a_day"] / out.loc[r, "staked_a_day"].sum()).values
print("== the money by first price band, the old rule (a day, 10-18 Oct) against the new (19-22 Oct); CLV at Betfair's "
      "settled price")
print(out.round(3).to_string())
tot = pos[ok].groupby("rule").apply(lambda g: pd.Series({"horses": len(g), "staked": g.S.sum(),
                                                         "clv": (g.clv * g.S).sum() / g.S.sum()}))
print(tot.round(4).to_string())
n = pos[pos.day >= "2026-10-19"]
fl = n[n.first_price >= 400 / 15 + 1]
print(f"\n19-22 Oct: horses first backed at {400 / 15 + 1:.2f} or longer (the floor's stake, GBP15, is more than the to-win "
      f"stake): {len(fl)}, staked GBP{fl.S.sum():.2f}, at GBP15 or more {int((fl.S >= 14.99).sum())}, median "
      f"GBP{fl.S.median():.2f}; their CLV {(fl[np.isfinite(fl.clv)].clv * fl[np.isfinite(fl.clv)].S).sum() / max(fl[np.isfinite(fl.clv)].S.sum(), 1e-9):+.2%}")
print("the largest stakes on 19-22 Oct:")
print(n.sort_values("S", ascending=False).head(10)[["day", "runner", "first_price", "S", "P", "bsp", "clv", "result"]]
      .round(3).to_string(index=False))
bb = b.drop(columns=["bsp"], errors="ignore").merge(pos[K + ["bsp"]], on=K, how="left")
tu = bb[bb.k > 0].copy()
tu["drift"] = tu.avg_price > tu.avg_before * 1.0001
tu["fclv"] = tu.avg_price / tu.bsp - 1
print("\ntop-ups by whether the price was above the horse's average matched (each fill's CLV at the BSP, before a "
      "non-runner's reduction):")
tu["rule"] = np.where(tu.day >= "2026-10-19", "19-22 Oct (new)", "10-18 Oct (old)")
print(tu[tu.bsp > 1].groupby(["rule", "drift"]).apply(lambda g: pd.Series({
    "fills": len(g), "staked": g.matched.sum(), "clv": (g.fclv * g.matched).sum() / g.matched.sum()})).round(4).to_string())
