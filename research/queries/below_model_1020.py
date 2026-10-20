"""The owner's question (20 Oct, sharper): when we take a thin top price, are the next prices we top up at below our own
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
f["model_bfsp"] = f.best_back / np.exp(f.move)                # the predicted BFSP the trader read
f["exp_bsp"] = 1.0 / f.p_pool                                # the closing model's expected BSP
f["vs_model"] = f.avg_price / f.model_bfsp - 1
f["vs_exp"] = f.avg_price / f.exp_bsp - 1
f = f[np.isfinite(f.vs_model)].copy()
print(f"fills with the model's price read: {len(f):,}, GBP{f.matched.sum():,.2f}")
f["below_model"] = f.vs_model < 0
print("\n== the price taken against our predicted BFSP (below 0: shorter than the model's own price)")


def summ(x):
    return pd.Series({"fills": len(x), "staked": x.matched.sum(), "share": x.matched.sum() / f.matched.sum(),
                      "clv": x.locked.sum() / x.matched.sum(), "locked": x.locked.sum(),
                      "median_vs_exp_bsp": x.vs_exp.median()})


f["band"] = pd.cut(f.vs_model, [-1, -0.3, -0.2, -0.1, 0, 0.1, 0.25, 0.5, 100])
print(f.groupby(["band", "kind"], observed=True).apply(summ).round(4).to_string())
print("\n   all fills, below against at or above the predicted BFSP:")
print(f.groupby(["below_model", "kind"]).apply(summ).round(4).to_string())
print(f"\n   fills below the closing model's expected BSP (should be none: the rule asks 3% over it): "
      f"{int((f.vs_exp < 0).sum())} (GBP{f[f.vs_exp < 0].matched.sum():.2f})")
first = f[f.k == 0][["day", "market_id", "selection_id", "top_gap", "avg_price", "model_bfsp"]].rename(
    columns={"top_gap": "first_gap", "avg_price": "first_price"})
t = f[f.k > 0].merge(first, on=["day", "market_id", "selection_id"], how="left")
t["after_thin"] = t.first_gap > 0.10
t["down_from_first"] = t.avg_price < t.first_price
print("\n== top-ups after the first back, by whether the first back was a thin top price (best back over 10% above the "
      "second) and whether the top-up's price was under the predicted BFSP")
print(t.groupby(["after_thin", "below_model"]).apply(lambda x: pd.Series({
    "fills": len(x), "staked": x.matched.sum(), "clv": x.locked.sum() / x.matched.sum(),
    "median_down_from_first": (x.avg_price / x.first_price - 1).median()})).round(4).to_string())
print("\n== a rule never to back under the predicted BFSP (by a margin): the fills it drops")
rows = {}
for m in (-0.2, -0.1, 0.0, 0.05, 0.10):
    for kind in ("all", "top-ups only"):
        drop = f.vs_model < m
        if kind == "top-ups only":
            drop = drop & (f.k > 0)
        d, k = f[drop], f[~drop]
        rows[(f"price under predicted BFSP x {1 + m:.2f}", kind)] = {
            "fills": int(drop.sum()), "staked": d.matched.sum(), "clv_dropped": d.locked.sum() / max(d.matched.sum(), 1e-9),
            "locked_dropped": d.locked.sum(), "clv_kept": k.locked.sum() / max(k.matched.sum(), 1e-9)}
print(pd.DataFrame(rows).T.round(4).to_string())
print("\n   by day, the fills under the predicted BFSP: staked and CLV")
print(f[f.below_model].groupby("day").apply(lambda x: pd.Series({"staked": x.matched.sum(),
                                                                 "clv": x.locked.sum() / x.matched.sum()}))
      .round(4).to_string())
