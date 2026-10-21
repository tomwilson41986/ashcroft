"""The execution audit's follow-up (21 Oct; read-only; execution-audit-1021): the non-runners' reductions cost GBP1,909 of
CLV on 10-20 Oct (+7.65% at the price matched, +3.08% settled). Which backs do they hit, and would the bar on the CLV
after the expected reduction (walk-forward: each day's expected factor from the days before it, by the price band of the
back) have kept the money that loses to them? Read by price and the field (the removals still to come at the fill), not
time."""
import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAYS = [(date(2026, 10, 10) + timedelta(n)).isoformat() for n in range(12)]
K = ["day", "market_id", "selection_id"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


rows = []
for day in DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    bkr = get(f"betfair_live/{day}/books_trader.csv.gz")
    if raw is None or bkr is None:
        continue
    led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "best_back", "edge", "selection_id", "bsp"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    st = led[led.event == "settle"].sort_values("t").drop_duplicates(["market_id", "selection_id"], keep="last")
    st = st[st.result.isin(["WINNER", "LOSER"]) & (st.bsp > 1) & (st.avg_price > 1)][
        ["market_id", "selection_id", "bsp", "avg_price"]].rename(columns={"bsp": "B", "avg_price": "P"})
    b = led[(led.event == "back") & (led.matched > 0)].drop(columns=["bsp"]).dropna(subset=["t"])
    b = b.merge(st, on=["market_id", "selection_id"])
    if b.empty:
        continue
    bk = pd.read_csv(io.BytesIO(bkr), compression="gzip", low_memory=False, dtype={"market_id": str},
                     usecols=lambda c: c in ("polled_utc", "market_id", "selection_id", "runner_status",
                                             "adjustment_factor", "removal_utc"))
    bk["selection_id"] = pd.to_numeric(bk.selection_id, errors="coerce")
    rm = bk[bk.runner_status.eq("REMOVED")].copy()
    rm["af"] = pd.to_numeric(rm.adjustment_factor, errors="coerce").fillna(0)
    rm["rt"] = pd.to_datetime(rm.removal_utc, utc=True, errors="coerce").fillna(pd.to_datetime(rm.polled_utc, utc=True))
    rm = rm.groupby(["market_id", "selection_id"]).agg(af=("af", "last"), rt=("rt", "min")).reset_index()
    n_run = bk.groupby("market_id").selection_id.nunique()
    to_come, af_to_come = [], []
    for r in b.itertuples():
        x = rm[(rm.market_id == r.market_id) & (rm.selection_id != r.selection_id) & (rm.rt > r.t)]
        to_come.append(len(x))
        af_to_come.append(float(x.af.sum()))
    b["to_come"], b["af_to_come"], b["runners"] = to_come, af_to_come, b.market_id.map(n_run)
    b["day"] = day
    rows.append(b)
f = pd.concat(rows, ignore_index=True)
f["sxp"] = f.matched * f.avg_price
f["factor"] = (f.P / (f.groupby(K).sxp.transform("sum") / f.groupby(K).matched.transform("sum"))).clip(upper=1.0)
f["clv_fill"] = f.avg_price / f.B - 1
f["clv_set"] = f.avg_price * f.factor / f.B - 1
f["band"] = pd.cut(f.best_back, [1, 4, 8, 16, 40, 1001], labels=["<4", "4-8", "8-16", "16-40", "40+"])


def tab(g):
    m = g.matched.sum()
    return pd.Series({"backs": len(g), "staked": m, "clv_fill": (g.clv_fill * g.matched).sum() / m,
                      "clv_settled": (g.clv_set * g.matched).sum() / m, "cost_gbp": ((g.clv_fill - g.clv_set) * g.matched).sum(),
                      "share_cut": float((g.factor < 0.9999).mean())})


print(f"live backs settled with a BSP, {f.day.min()} .. {f.day.max()}: {len(f):,}, GBP{f.matched.sum():,.2f}")
print("\n== by the price of the back")
print(f.groupby("band", observed=True).apply(tab).round(4).to_string())
print("\n== by the removals still to come in the market at the fill")
print(f.groupby(f.to_come.clip(upper=3)).apply(tab).round(4).to_string())
print("\n== by the field at the off (runners listed, removals included)")
print(f.groupby(pd.cut(f.runners, [0, 8, 12, 16, 40])).apply(tab).round(4).to_string())
print("\n== by the reductions to come (sum of the removed runners' factors, %)")
print(f.groupby(pd.cut(f.af_to_come, [-0.01, 0.0, 5, 15, 30, 100])).apply(tab).round(4).to_string())

print("\n== the bar on the CLV after the expected reduction (walk-forward by price band; the trader's edge as its CLV forecast)")
f = f.sort_values("t")
days = sorted(f.day.unique())
out = []
for i, d in enumerate(days[1:], 1):
    past = f[f.day < d]
    ef = (past.factor * past.matched).groupby(past.band, observed=True).sum() / past.matched.groupby(past.band, observed=True).sum()
    x = f[f.day == d].copy()
    x["ef"] = x.band.map(ef).astype(float).fillna(1.0)
    out.append(x)
w = pd.concat(out, ignore_index=True)
w["edge_adj"] = (1 + w.edge) * w.ef - 1
for bar in (0.0, 0.03, 0.05):
    keep = w.edge_adj >= bar
    print(f"   bar {bar:.0%} on the CLV after the expected reduction: kept {keep.mean():.0%} of the backs, "
          f"GBP{w.matched[keep].sum():,.2f} (CLV settled {(w.clv_set * w.matched)[keep].sum() / max(w.matched[keep].sum(), 1e-9):+.2%}); "
          f"dropped GBP{w.matched[~keep].sum():,.2f} (CLV settled {(w.clv_set * w.matched)[~keep].sum() / max(w.matched[~keep].sum(), 1e-9):+.2%})")
print("   (the money dropped would go to the next back that clears the bar while the day limit binds: not replayed here)")
print(f"   the trader's edge against the CLV: at the fill {np.corrcoef(w.edge, w.clv_fill)[0, 1]:.3f}, settled "
      f"{np.corrcoef(w.edge, w.clv_set)[0, 1]:.3f}; mean edge {(w.edge * w.matched).sum() / w.matched.sum():+.2%}, "
      f"after the expected reduction {(w.edge_adj * w.matched).sum() / w.matched.sum():+.2%}, settled CLV "
      f"{(w.clv_set * w.matched).sum() / w.matched.sum():+.2%}")
