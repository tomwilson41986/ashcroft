"""The owner's question (20 Oct): are we taking bad prices because a higher price was there but at lower liquidity?
(read-only). Every back fill of 10-19 Oct read against the trader's own book at that minute (the last book it read for
the runner before the order): the spread (best lay / best back), how far the best back stood above the next price
(best back / second back), the money offered at the best back and the share of it taken, the market's and the
runner's matched money, the minutes to the off. Each fill scored by its CLV at Betfair's settled price (the matched
price scaled as Betfair settled the horse, so a non-runner's reduction counts), stake-weighted, by bands of each; then
what a guard would have cut: the fills it drops, their stake and their CLV x stake (the hedged profit before
commission)."""
import io
import os

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
K = ["day", "market_id", "selection_id"]
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
              "target"):
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


def tab(col, bins, labels=None):
    g = f.assign(band=pd.cut(f[col], bins, labels=labels, include_lowest=True))
    out = g.groupby(["band"], observed=True).apply(lambda x: pd.Series({
        "fills": len(x), "staked": x.matched.sum(), "share": x.matched.sum() / f.matched.sum(),
        "clv": x.locked.sum() / x.matched.sum(), "locked": x.locked.sum(),
        "first_clv": x[x.k == 0].locked.sum() / max(x[x.k == 0].matched.sum(), 1e-9),
        "topup_clv": x[x.k > 0].locked.sum() / max(x[x.k > 0].matched.sum(), 1e-9),
        "median_price": x.avg_price.median()}))
    print(f"\n== by {col}")
    print(out.round(4).to_string())


tab("spread", [-1, 0.02, 0.05, 0.10, 0.20, 0.50, 100])
tab("top_gap", [-1, 0.0001, 0.03, 0.06, 0.10, 0.20, 100])
tab("back1_size", [0, 5, 10, 25, 50, 100, 1e9])
tab("took_share", [0, 0.25, 0.5, 0.9, 1.0001, 1e9])
tab("market_total_matched", [-1, 1000, 5000, 20000, 50000, 1e12])
tab("runner_total_matched", [-1, 100, 500, 2000, 10000, 1e12])
tab("vs_last", [-1, -0.0001, 0.02, 0.05, 0.10, 0.25, 100])
tab("minutes_to_off", [0, 60, 180, 360, 600, 1e9])

print("\n== the price taken against the best back read (above 0: filled at a better price than shown)")
print(f.above_book.describe(percentiles=[0.05, 0.5, 0.95]).round(4).to_string())

print("\n== guards: the fills each would have dropped (CLV x stake is the hedged profit before commission)")
GUARDS = {
    "spread over 10%": f.spread > 0.10,
    "spread over 20%": f.spread > 0.20,
    "best back over 10% above the second": f.top_gap > 0.10,
    "best back over 20% above the second": f.top_gap > 0.20,
    "under GBP10 offered at the best back": f.back1_size < 10,
    "market matched under GBP5,000": f.market_total_matched < 5000,
    "market matched under GBP1,000": f.market_total_matched < 1000,
    "runner matched under GBP100": f.runner_total_matched < 100,
    "best back over 10% above the last traded": f.vs_last > 0.10,
    "spread over 20% or best back over 20% above the second": (f.spread > 0.20) | (f.top_gap > 0.20),
}
rows = {}
for name, m in GUARDS.items():
    m = m.fillna(False)
    d, k = f[m], f[~m]
    rows[name] = {"fills_dropped": int(m.sum()), "staked_dropped": d.matched.sum(),
                  "clv_dropped": d.locked.sum() / max(d.matched.sum(), 1e-9), "locked_dropped": d.locked.sum(),
                  "clv_kept": k.locked.sum() / max(k.matched.sum(), 1e-9),
                  "days_dropped_lost": int((d.groupby("day").locked.sum() < 0).sum()), "days": d.day.nunique()}
print(pd.DataFrame(rows).T.round(4).to_string())
print("\n   (a dropped fill's money would go to other horses at the kept fills' CLV at best; a guard pays only if the "
      "dropped CLV is below the kept)")

print("\n== 19 Oct (the new staking) alone, by spread and by size offered")
n = f[f.day == "2026-10-19"]
for col, bins in (("spread", [-1, 0.05, 0.10, 0.20, 100]), ("back1_size", [0, 10, 25, 100, 1e9])):
    print(n.assign(band=pd.cut(n[col], bins)).groupby("band", observed=True).apply(lambda x: pd.Series({
        "fills": len(x), "staked": x.matched.sum(), "clv": x.locked.sum() / x.matched.sum()})).round(4).to_string())
