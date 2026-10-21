"""The owner's ask (21 Oct): "audit the execution". Live made 0.40 of the replay's pounds on 10-19 Oct (replay-vs-live-1021:
CLV +3.6% against +9.5%). Part 1, the live side, every back of 10-20 Oct (read-only):

- the non-runners: the CLV at the price matched against the price Betfair settled (cut by later withdrawals' reduction
  factors), and the void stakes;
- the price: the matched price against the best back the trader read (a fill-or-kill limit at that price fills at it or
  better), and the decision's best back against the best back the trader read next (did the price move away while the
  order went?);
- the book's age: from the trader's last read of the market to the order;
- the refusals: backs Betfair did not match (EXPIRED and the rest), their money and the CLV they would have had at the
  price asked;
- the CLV by the share of the best back's money the order took (an order that takes the whole level against one that
  leaves some)."""
import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
pd.set_option("display.width", 250)
DAYS = [(date(2026, 10, 10) + timedelta(n)).isoformat() for n in range(11)]
K = ["day", "market_id", "selection_id"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


rows, refused, ages = [], [], []
for day in DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        continue
    led = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    for c in ("matched", "avg_price", "best_back", "back_size", "asked", "edge", "selection_id", "bsp", "pnl"):
        led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
    led["day"] = day
    st = led[led.event == "settle"].sort_values("t").drop_duplicates(["market_id", "selection_id"], keep="last")[
        ["market_id", "selection_id", "result", "bsp", "avg_price"]].rename(columns={"avg_price": "P", "bsp": "B"})
    raw = get(f"betfair_live/{day}/books_trader.csv.gz")
    bk = pd.read_csv(io.BytesIO(raw), compression="gzip", low_memory=False, dtype={"market_id": str},
                     usecols=lambda c: c in ("polled_utc", "market_id", "selection_id", "status", "inplay", "back1",
                                             "back1_size", "sp_actual")) if raw is not None else pd.DataFrame()
    if len(bk):
        bk["selection_id"] = pd.to_numeric(bk.selection_id, errors="coerce").astype("float64")
        bk["back1"] = pd.to_numeric(bk.back1, errors="coerce")
        bk["t"] = pd.to_datetime(bk.polled_utc, utc=True, errors="coerce")
        bk = bk.dropna(subset=["t", "selection_id"]).sort_values("t")
        sp = bk[pd.to_numeric(bk.sp_actual, errors="coerce") > 1].groupby(["market_id", "selection_id"]).sp_actual.last()
    backs = led[led.event == "back"].copy()
    backs["selection_id"] = backs.selection_id.astype("float64")
    backs = backs.dropna(subset=["t"])
    if len(bk):
        mreads = bk[["market_id", "t"]].drop_duplicates().sort_values("t").rename(columns={"t": "t_read"})
        backs = pd.merge_asof(backs.sort_values("t"), mreads, left_on="t", right_on="t_read", by="market_id",
                              direction="backward")
        nxt = bk[["market_id", "selection_id", "t", "back1"]].rename(columns={"t": "t_next", "back1": "back_next"})
        backs = pd.merge_asof(backs.sort_values("t"), nxt, left_on="t", right_on="t_next",
                              by=["market_id", "selection_id"], direction="forward", allow_exact_matches=False)
    backs = backs.merge(st.assign(selection_id=st.selection_id.astype("float64")), on=["market_id", "selection_id"],
                        how="left")
    if len(bk):
        sp = sp.rename("B_book").reset_index()
        sp["selection_id"] = sp.selection_id.astype("float64")
        backs = backs.merge(sp, on=["market_id", "selection_id"], how="left")
        backs["B"] = backs.B.fillna(pd.to_numeric(backs.B_book, errors="coerce"))
    rows.append(backs)
b = pd.concat(rows, ignore_index=True)
f = b[b.matched > 0].copy()
r = b[(b.matched.fillna(0) <= 0)].copy()
print(f"backs sent 10-20 Oct: {len(b):,}; matched {len(f):,} (GBP{f.matched.sum():,.2f}); refused or unmatched {len(r):,}")

print("\n== 1. the non-runners")
ok = f.result.isin(["WINNER", "LOSER"]) & (f.B > 1) & (f.P > 1)
s = f[ok].copy()
s["sxp"] = s.matched * s.avg_price
s["factor"] = s.P / (s.groupby(K).sxp.transform("sum") / s.groupby(K).matched.transform("sum"))
s["clv_fill"] = s.avg_price / s.B - 1
s["clv_settled"] = s.avg_price * s.factor / s.B - 1
void = f[f.result.eq("REMOVED")]
print(f"   settled with a BSP: GBP{s.matched.sum():,.2f}; CLV at the price matched {(s.clv_fill * s.matched).sum() / s.matched.sum():+.2%} "
      f"(GBP{(s.clv_fill * s.matched).sum():,.2f}); at the price settled (after reductions) "
      f"{(s.clv_settled * s.matched).sum() / s.matched.sum():+.2%} (GBP{(s.clv_settled * s.matched).sum():,.2f}); "
      f"backs cut by a reduction: {(s.factor < 0.9999).mean():.0%} of the money, by {1 - (s.factor * s.matched).sum() / s.matched.sum():.2%} on average")
print(f"   void (our horse a non-runner): {void.groupby(K).ngroups} horses, GBP{void.matched.sum():,.2f}")
print(s.groupby("day").apply(lambda g: pd.Series({"staked": g.matched.sum(),
                                                  "clv_fill": (g.clv_fill * g.matched).sum() / g.matched.sum(),
                                                  "clv_settled": (g.clv_settled * g.matched).sum() / g.matched.sum(),
                                                  "cost_gbp": ((g.clv_fill - g.clv_settled) * g.matched).sum()})).round(4).to_string())

print("\n== 2. the price: matched against the best back read; the best back at the next read")
f["slip"] = f.avg_price / f.best_back - 1
f["moved"] = np.log(f.back_next / f.best_back)
print(f"   matched price over the best back read: mean {f.slip.mean():+.3%}, share better {(f.slip > 1e-9).mean():.1%}, "
      f"share worse {(f.slip < -1e-9).mean():.1%} (a back limit cannot fill worse; the ladder from 21 Oct can)")
print(f"   the best back at the trader's next read against the one acted on: median {f.moved.median():+.2%}, mean "
      f"{f.moved.mean():+.2%}, shorter {(f.moved < -1e-9).mean():.0%}, longer {(f.moved > 1e-9).mean():.0%}")

print("\n== 3. the book's age at the order (the trader's last read of the market to the order)")
f["age_s"] = (f.t - f.t_read).dt.total_seconds()
print(f["age_s"].describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())

print("\n== 4. the refusals")
r = r[r.error.notna() | r.status.notna()]
print(r.groupby(r.status.astype(str) + " | " + r.error.astype(str).str.slice(0, 50)).agg(
    backs=("asked", "size"), asked=("asked", "sum")).sort_values("backs", ascending=False).head(10).to_string())
rr = r[(r.B > 1) & (r.best_back > 1)]
print(f"   refused backs with a BSP: {len(rr):,}, GBP{rr.asked.sum():,.2f} asked, their CLV at the price asked "
      f"{((rr.best_back / rr.B - 1) * rr.asked).sum() / max(rr.asked.sum(), 1e-9):+.2%}")

print("\n== 5. the CLV by the share of the best back's money taken")
s["took"] = s.matched / s.back_size
print(s.groupby(pd.cut(s.took, [0, 0.5, 0.95, 1.05, 100])).apply(lambda g: pd.Series({
    "fills": len(g), "staked": g.matched.sum(), "clv_settled": (g.clv_settled * g.matched).sum() / g.matched.sum()})).round(4).to_string())
