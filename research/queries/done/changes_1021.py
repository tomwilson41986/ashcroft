"""The evening check-in of 21 Oct (read-only): the first day of the guard (min_move 0, PR #158), the ladder (PR #159) and
the maximum price of 200 (PR #160), from the live ledger and the books the trader read.

1. The guard and the cap: every 5 minutes of the trader's own reads from 06:15 UK to the off, each race planned twice by
   the trader's own code, as live (the guard, nothing over 200) and without them (min_move -100, max price 1000); the
   horses only the second plan backs, by why (under the model's price, over 200), the stake the rule would have asked at
   the first such read, and its CLV at that price against the BSP (no reduction applied: an estimate).
2. The ladder: the fills whose reason says "ladder to ...", their money, how far below the best back they went, and their
   CLV at Betfair's settled price against the best-back fills.
3. The cap: the largest price backed.
4. The entry (qa-1021): the first backs' price over the BSP (median log), 21 Oct against 10-13 Oct (+13.3%) and
   14-20 Oct (+6.7%).
"""
import gzip
import io
import os
import sys
from dataclasses import replace

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from trading import config as tc  # noqa: E402
from trading import strategy  # noqa: E402
from trading.exchange import Market  # noqa: E402
from trading.matching import attach_ids  # noqa: E402

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAY = "2026-10-21"
K = ["market_id", "selection_id"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def csv(key):
    raw = get(key)
    if raw is None:
        return pd.DataFrame()
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}, low_memory=False)


led = csv(f"trading/live/{DAY}/ledger.csv")
for c in ("matched", "avg_price", "best_back", "bsp", "edge", "move", "selection_id", "target"):
    led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[K + ["result", "bsp", "avg_price"]] \
    .rename(columns={"avg_price": "P", "bsp": "B"})
b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
b["sxp"] = b.matched * b.avg_price
hm = b.groupby(K).agg(m=("matched", "sum"), sxp=("sxp", "sum")).reset_index()
hm["Pm"] = hm.sxp / hm.m
b = b.drop(columns=["result", "bsp"]).merge(hm[K + ["Pm"]], on=K).merge(st, on=K, how="left")
ok = b.result.isin(["WINNER", "LOSER"]) & (b.B > 1) & (b.P > 1)
b["clv_settled"] = np.where(ok, b.avg_price * b.P / b.Pm / b.B - 1, np.nan)
b["clv_fill"] = np.where(b.B > 1, b.avg_price / b.B - 1, np.nan)
b["first"] = ~b.duplicated(K)
print(f"{DAY}: {len(b):,} fills, GBP{b.matched.sum():,.2f} on {b.groupby(K).ngroups} horses; first back "
      f"{b.t.min().tz_convert('Europe/London'):%H:%M} UK")

print("\n== 2. the ladder")
b["ladder"] = b.reason.astype(str).str.contains("ladder to")
b["below_best"] = np.log(b.best_back / b.avg_price)


def tab(g):
    s = g[np.isfinite(g.clv_settled)]
    return pd.Series({"fills": len(g), "staked": g.matched.sum(),
                      "clv_settled": (s.clv_settled * s.matched).sum() / max(s.matched.sum(), 1e-9),
                      "clv_at_fill": (s.clv_fill * s.matched).sum() / max(s.matched.sum(), 1e-9),
                      "median_log_below_best": g.below_best.median()})


print(b.groupby(["ladder", "first"]).apply(tab).round(4).to_string())
lad = b[b.ladder]
print(f"   ladder fills: {len(lad)}, GBP{lad.matched.sum():,.2f} ({lad.matched.sum() / b.matched.sum():.0%} of the money); "
      f"levels below the best back: median {lad.below_best.median():.2%}, 90th {lad.below_best.quantile(0.9):.2%}, "
      f"max {lad.below_best.max():.2%} (log)")

print("\n== 3. the cap")
print(f"   largest price backed {b.avg_price.max():.2f} (best back {b.best_back.max():.2f}); fills over 200: "
      f"{int((b.avg_price > 200).sum())}")

print("\n== 4. the entry")
f1 = b[b["first"] & ok].copy()
print(f"   first backs with a BSP: {len(f1)}; price over the BSP, median log {np.log(f1.avg_price / f1.B).median():+.2%} "
      f"(10-13 Oct +13.3%, 14-20 Oct +6.7%); expected CLV at the first back, mean {f1.edge.mean():+.2%}")

print("\n== 1. the guard and the cap, from the trader's reads")
books = csv(f"betfair_live/{DAY}/books_trader.csv.gz")
cat = pd.concat([csv(f"betfair_live/{DAY}/{n}.csv.gz") for n in ("markets_trader", "markets")], ignore_index=True)
preds = pd.read_csv(io.BytesIO(get(f"predictions/{DAY}.csv")), dtype={"race_time": str})
cat = cat[cat.market_type.astype(str).str.upper().eq("WIN") & cat.country.isin(["GB", "IE"])]
cat = cat.drop_duplicates(["market_id", "selection_id"])
markets = []
for mid, g in cat.groupby("market_id"):
    start = pd.Timestamp(g.market_start_utc.iloc[0]).tz_convert("UTC").to_pydatetime()
    markets.append(Market(str(mid), str(g.venue.iloc[0]), str(g.country.iloc[0]), start, name=str(g.market_name.iloc[0]),
                          runners={int(s): str(n) for s, n in zip(g.selection_id, g.runner_name)}))
start = {m.market_id: m.start for m in markets}
preds = attach_ids(preds, markets)
preds = preds[preds.market_id.notna()]
price = {(str(r.market_id), int(r.selection_id)): float(r.predicted_bfsp) for r in preds.itertuples()
         if pd.notna(r.predicted_bfsp) and r.predicted_bfsp > 1}
names = {(m.market_id, s): n for m in markets for s, n in m.runners.items()}
cfg = tc.load_config("trading/config_live.json", env={})
cfg_open = replace(cfg, min_move=-100.0, limits=replace(cfg.limits, max_price=1000.0))
books = books[books.market_id.isin(start)].copy()
books["t"] = pd.to_datetime(books.polled_utc, utc=True)
books["selection_id"] = pd.to_numeric(books.selection_id, errors="coerce")
books = books[books.inplay.astype(str).str.lower().isin(["false", "0"]) & books.status.eq("OPEN")]
uk = books.t.dt.tz_convert("Europe/London")
books = books[(uk.dt.strftime("%H:%M") >= "06:15")]
books["slot"] = books.t.dt.floor("5min")
sp = books[pd.to_numeric(books.get("sp_actual"), errors="coerce") > 1].groupby(K).sp_actual.last()
bsp = {**{(r.market_id, int(r.selection_id)): float(r.B) for r in st[st.B > 1].itertuples()},
       **{(k[0], int(k[1])): float(v) for k, v in sp.items()}}
reads = books.sort_values("t").groupby(["market_id", "slot"]).t.max().reset_index()
books = books.merge(reads, on=["market_id", "slot", "t"])
seen = {}
for (mid, t), g in books.groupby(["market_id", "t"], sort=True):
    if (start[mid] - t.to_pydatetime()).total_seconds() < 15 * 60:
        continue
    views = []
    for r in g.itertuples():
        sid = int(r.selection_id)
        views.append(strategy.RunnerView(
            selection_id=sid, name=names.get((mid, sid), str(sid)), model_price=price.get((mid, sid)),
            back=float(r.back1) if pd.notna(r.back1) and r.back1 > 1 else None,
            back_size=float(r.back1_size) if pd.notna(r.back1_size) else 0.0,
            lay=float(r.lay1) if pd.notna(r.lay1) and r.lay1 > 1 else None,
            last_traded=float(r.last_traded) if pd.notna(r.last_traded) else None,
            status=str(r.runner_status) if pd.notna(r.runner_status) else "ACTIVE", traded=0.0))
    live = {d.selection_id for d in strategy.plan_race(views, cfg, 5000.0)}
    for d in strategy.plan_race(views, cfg_open, 5000.0):
        if d.selection_id in live or (mid, d.selection_id) in seen:
            continue
        why = "over 200" if d.price > 200 else "under the model's price" if d.move < 0 else "other"
        seen[(mid, d.selection_id)] = {"market_id": mid, "selection_id": d.selection_id, "runner": d.name, "why": why,
                                       "t": t, "price": d.price, "edge": d.edge, "stake": min(d.target, 200.0),
                                       "bsp": bsp.get((mid, d.selection_id))}
x = pd.DataFrame(seen.values())
if x.empty:
    print("   no horse cleared the bar that the guard or the cap kept out")
else:
    backed = set(map(tuple, hm[K].astype({"selection_id": int}).to_numpy().tolist()))
    x["backed_later"] = [(m, int(s)) in backed for m, s in zip(x.market_id, x.selection_id)]
    x["clv"] = x.price / x.bsp - 1
    print(x.groupby(["why", "backed_later"]).apply(lambda g: pd.Series({
        "horses": len(g), "stake_asked": g.stake.sum(), "mean_edge": g.edge.mean(),
        "clv_at_bsp": (g.clv * g.stake).sum() / g.stake[g.clv.notna()].sum() if g.clv.notna().any() else np.nan,
        "with_bsp": int(g.clv.notna().sum())})).round(4).to_string())
    print("   (a horse kept out at one read can be backed at a later one, at a price over the model's: backed_later)")
    print(x[~x.backed_later].sort_values("stake", ascending=False).head(12)[
        ["runner", "why", "price", "edge", "stake", "bsp", "clv"]].round(3).to_string(index=False))
