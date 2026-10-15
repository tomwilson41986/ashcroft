"""Audit of 15 Oct, part 2 (read-only): the top-ups, 10-15 Oct. Each top-up against its horse's first back: did the
trader add to horses that had drifted since (the market moving against the first back) or shortened? CLV at Betfair's
settled price (the settle's own rows), by that move, by price band and by day; and the stake per horse."""
import io
import os

import boto3
import numpy as np
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
K = ["market_id", "selection_id"]
pd.set_option("display.width", 250)
rows = []
for day in ["2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15"]:
    d = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=f"trading/live/{day}/ledger.csv")["Body"].read()),
                    low_memory=False, dtype={"market_id": str})
    for c in ["matched", "avg_price", "bsp", "clv", "selection_id", "edge"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["t"] = pd.to_datetime(d.ts, utc=True)
    b = d[(d.event == "back") & (d.matched > 0)].sort_values("t").copy()
    b["sxp"] = b.matched * b.avg_price
    hm = b.groupby(K).agg(m=("matched", "sum"), sxp=("sxp", "sum")).reset_index()
    hm["mp"] = hm.sxp / hm.m
    st = d[d.event == "settle"].drop_duplicates(K, keep="last")[K + ["bsp", "avg_price", "result"]] \
        .rename(columns={"avg_price": "sp_settled"})
    b = b.drop(columns=["bsp", "clv", "result"], errors="ignore").merge(hm[K + ["mp"]], on=K).merge(st, on=K)
    b = b[b.result.isin(["WINNER", "LOSER"]) & (b.bsp > 1)]
    b["price_s"] = b.avg_price * b.sp_settled / b.mp
    b["clv"] = b.price_s / b.bsp - 1
    b["k"] = b.groupby(K).cumcount()
    first = b[b.k == 0].set_index(K).avg_price.rename("first_price")
    b = b.join(first, on=K)
    b["move"] = b.avg_price / b.first_price - 1
    b["day"] = day
    rows.append(b)
a = pd.concat(rows, ignore_index=True)
a["kind"] = np.where(a.k == 0, "first", "top-up")
a["since_first"] = np.where(a.k == 0, "first back", np.select([a.move > 0.05, a.move < -0.05],
                                                              ["drifted >5%", "shortened >5%"], "within 5%"))
a["band"] = pd.cut(a.avg_price, [1, 3, 5, 8, 13, 1000], labels=["<3", "3-5", "5-8", "8-13", "13+"])


def tab(g):
    out = g.agg(orders=("matched", "size"), staked=("matched", "sum"),
                locked=("clv", lambda x: (x * a.loc[x.index, "matched"]).sum()))
    out["clv"] = out.locked / out.staked
    return out.round(3)


print("== by move since the first back, all days")
print(tab(a.groupby("since_first")).to_string())
print("\n== by day and move")
print(tab(a.groupby(["day", "since_first"])).unstack("since_first")[["staked", "clv"]].round(3).to_string())
print("\n== top-ups by price band and move")
print(tab(a[a.k > 0].groupby(["band", "since_first"], observed=True)).to_string())
print("\n== stake per horse by day: horses, median, the 10 largest's share, top-ups per horse")
h = a.groupby(["day"] + K).agg(staked=("matched", "sum"), n=("matched", "size")).reset_index()
print(h.groupby("day").apply(lambda g: pd.Series({
    "horses": len(g), "median": g.staked.median(), "top10_share": g.nlargest(10, "staked").staked.sum() / g.staked.sum(),
    "orders_per_horse": g.n.mean()})).round(3).to_string())
print("\n== the edge the trader expected at entry against the CLV realised, by kind and day")
e = a.groupby(["day", "kind"]).apply(lambda g: pd.Series({
    "expected": (g.edge * g.matched).sum() / g.matched.sum(), "realised": (g.clv * g.matched).sum() / g.matched.sum()}))
print(e.round(4).unstack("kind").to_string())
