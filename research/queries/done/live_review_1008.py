"""The live record, every price, bet and trade, 30 Sep to today (the owner, 8 Oct: "review all prices, bets, trades
today and since the start; do we need to refine our strategy?"; read-only).

From the live ledgers (trading/live/<day>/ledger.csv: every order the trader sent, every lay at the SP, every
settlement from Betfair's record of the bets):

A. the orders: sent, matched, killed (fill-or-kill), refused (funds), top-ups, by day;
B. the prices: what the rule saw (best back, our price, the market's, the expected BSP), what it got (matched price),
   and the BSP; did our price point the right way, and was the expected BSP right, by day and by the model that read
   the feed (the volume model on the live key's feed; "feed without volume" on the delayed key's);
C. the trades: the lays at the SP, asked against matched, how much of each back they closed;
D. what a stricter rule would have kept: the CLV and the result at each bar of expected edge, under a price floor and
   ceiling, by first back only against all backs, and staked level against staked to win;
E. today, horse by horse.

Writes out/live_review.xlsx (every horse, every order, the day table) for the owner.
"""

from __future__ import annotations

import io
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)
UK = "Europe/London"
FIRST = date(2026, 9, 30)
TODAY = pd.Timestamp.now(tz=UK).date()
COMM = 0.02


def ledger(day: date) -> pd.DataFrame:
    try:
        raw = s3.get_object(Bucket=bucket, Key=f"trading/live/{day:%Y-%m-%d}/ledger.csv")["Body"].read()
    except Exception:
        return pd.DataFrame()
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    d["day"] = f"{day:%Y-%m-%d}"
    return d


frames, d = [], FIRST
while d <= TODAY:
    x = ledger(d)
    if len(x):
        frames.append(x)
    d += timedelta(days=1)
led = pd.concat(frames, ignore_index=True)
for c in ["matched", "avg_price", "edge", "minutes_to_off", "p_model", "p_market", "p_pool", "move", "best_back",
          "back_size", "target", "asked", "hedge_liability", "bsp", "clv", "pnl_back", "pnl_lay", "pnl", "commission"]:
    if c in led:
        led[c] = pd.to_numeric(led[c], errors="coerce")
led["selection_id"] = pd.to_numeric(led["selection_id"], errors="coerce")
led["t"] = pd.to_datetime(led["ts"], utc=True, errors="coerce")
led["hour_uk"] = led.t.dt.tz_convert(UK).dt.hour
K = ["day", "market_id", "selection_id"]

# ------------------------------------------------------------------------------------------------------ A. orders
o = led[led.event == "back"].copy()
o["model_feed"] = np.where(o.reason.fillna("").str.contains("without volume"), "no-volume model (delayed feed)",
                           "volume model (live feed)")
o["outcome"] = np.select([o.matched > 0, o.error.fillna("").str.contains("INSUFFICIENT_FUNDS"),
                          o.error.fillna("").str.contains("EXPIRED") | o.status.eq("EXPIRED")],
                         ["matched", "refused: funds", "killed (fill-or-kill)"], "other")
print("== A. orders by day")
a = o.groupby("day").agg(orders=("outcome", "size"), matched=("outcome", lambda s: (s == "matched").sum()),
                         killed=("outcome", lambda s: (s == "killed (fill-or-kill)").sum()),
                         refused=("outcome", lambda s: (s == "refused: funds").sum()),
                         other=("outcome", lambda s: (s == "other").sum()),
                         gbp_matched=("matched", "sum"))
a["horses"] = o[o.matched > 0].drop_duplicates(K).groupby("day").size()
a["orders_per_horse"] = (a.matched / a.horses).round(1)
print(a.to_string())
print("   orders by model:", o.groupby(["day", "model_feed"]).size().unstack(fill_value=0).to_string())
print("   'other' errors:", o[o.outcome == "other"].error.fillna(o.status).astype(str).str[:50].value_counts().head(6)
      .to_dict())

# ------------------------------------------------------------------------------------------- B. horses and prices
m = o[o.matched > 0].sort_values("t").copy()
m["sxp"] = m.matched * m.avg_price
g = m.groupby(K)
h = g.agg(venue=("venue", "first"), off=("off", "first"), runner=("runner", "first"), first_t=("t", "first"),
          first_mins=("minutes_to_off", "first"), model_feed=("model_feed", "first"), edge=("edge", "first"),
          best_back=("best_back", "first"), p_model=("p_model", "first"), p_market=("p_market", "first"),
          p_pool=("p_pool", "first"), first_price=("avg_price", "first"), first_stake=("matched", "first"),
          orders=("matched", "size"), staked=("matched", "sum"), sxp=("sxp", "sum"),
          last_price=("avg_price", "last")).reset_index()
h["price"] = h.sxp / h.staked
h["our_price"] = 1.0 / h.p_model
h["mkt_price"] = 1.0 / h.p_market
h["exp_bsp"] = 1.0 / h.p_pool
st = led[led.event == "settle"].drop_duplicates(K, keep="last")
h = h.merge(st[K + ["result", "bsp", "pnl_back", "pnl_lay", "pnl", "hedge_liability", "avg_price"]]
            .rename(columns={"avg_price": "settled_price"}), on=K, how="left")
lays = led[led.event == "trade_out"].groupby(K).agg(lay_orders=("event", "size"), lay_asked=("asked", "sum"),
                                                   lay_status=("status", lambda s: ",".join(sorted(set(map(str, s))))))
h = h.merge(lays.reset_index(), on=K, how="left")
h["price_used"] = h.settled_price.fillna(h.price)
ok = h.result.isin(["WINNER", "LOSER"]) & (h.bsp > 1)
h["clv"] = np.where(ok, h.price_used / h.bsp - 1.0, np.nan)
h["first_clv"] = np.where(ok, h.first_price / h.bsp - 1.0, np.nan)
h["locked"] = h.staked * h.clv
h["drift"] = np.log(h.bsp / h.first_price)                       # + the price went out before the off
h["model_said"] = np.log(h.our_price / h.first_price)            # + our price longer than the one taken (never)
h["exp_move"] = np.log(h.exp_bsp / h.first_price)                # what the closing model expected the BSP to do
h["band"] = pd.cut(h.price_used, [1, 3, 5, 8, 13, 21, 1001], labels=["<3", "3-5", "5-8", "8-13", "13-21", "21+"])
h["eband"] = pd.cut(h.edge, [-1, 0.05, 0.08, 0.12, 0.2, 10], labels=["3-5%", "5-8%", "8-12%", "12-20%", "20%+"])
s = h[ok].copy()

print("\n== B. prices, every settled horse: the price taken against the BSP, and what the model expected")
b = s.groupby("day").apply(lambda x: pd.Series({
    "horses": len(x), "staked": x.staked.sum(), "clv_staked": (x.locked.sum() / x.staked.sum()),
    "clv_level": x.clv.mean(), "share_beat_bsp": (x.clv > 0).mean(), "exp_edge": (x.edge * x.staked).sum() / x.staked.sum(),
    "exp_move_med": x.exp_move.median(), "drift_med": x.drift.median(), "drifted_10pc": (x.drift > np.log(1.1)).mean(),
    "steamed_10pc": (x.drift < -np.log(1.1)).mean(), "first_clv_staked": (x.staked * x.first_clv).sum() / x.staked.sum(),
    "result": x.pnl.sum()}))
print(b.round(3).to_string())
print("-- by the model that read the feed (the volume model is the live key's, from 6 Oct 15:25 UK)")
mf = s.groupby(["model_feed"]).apply(lambda x: pd.Series({
    "horses": len(x), "staked": x.staked.sum(), "exp_edge": (x.edge * x.staked).sum() / x.staked.sum(),
    "clv_staked": x.locked.sum() / x.staked.sum(), "clv_level": x.clv.mean(), "share_beat_bsp": (x.clv > 0).mean(),
    "corr_expected_vs_actual_move": np.corrcoef(x.exp_move, x.drift)[0, 1] if len(x) > 2 else np.nan}))
print(mf.round(3).to_string())
print("-- the volume model's days only (6 Oct after 15:25 UK, 7 and 8 Oct) against the no-volume model's (1-6 Oct)")
for name, x in s.groupby("model_feed"):
    by = x.groupby("eband", observed=True).apply(lambda y: pd.Series({
        "horses": len(y), "staked": y.staked.sum(), "exp": y.edge.mean(), "clv_staked": y.locked.sum() / y.staked.sum()}))
    print(f"   {name}:\n{by.round(3).to_string()}")
print("-- is the expected BSP right? the BSP over the expected BSP, by day (1 = right; above 1 = it drifted more than "
      "expected)")
s["bsp_over_exp"] = s.bsp / s.exp_bsp
print(s.groupby("day").bsp_over_exp.describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]]
      .round(3).to_string())
print("-- the price taken against the best back the rule saw (slippage) and against the market's price")
s["slip"] = s.first_price / s.best_back - 1.0
s["vs_mkt"] = s.first_price / s.mkt_price - 1.0
print(s.groupby("day")[["slip", "vs_mkt"]].median().round(4).to_string())

print("\n== C. the trades: each back laid at the SP")
c = h[h.result.isin(["WINNER", "LOSER"])].groupby("day").agg(
    horses=("staked", "size"), with_lay=("lay_orders", lambda x: x.notna().sum()),
    lay_orders=("lay_orders", "sum"), lay_liab_asked=("lay_asked", "sum"), hedge_liab_matched=("hedge_liability", "sum"),
    back_pnl=("pnl_back", "sum"), lay_pnl=("pnl_lay", "sum"))
print(c.round(2).to_string())
print("   lay statuses:", led[led.event == "trade_out"].status.astype(str).value_counts().head(6).to_dict())

print("\n== D. what a stricter rule would have kept (settled horses; CLV x stake is the locked result before commission)")


def keep(x, label):
    if not len(x):
        return {"rule": label, "horses": 0}
    net = (x.locked - COMM * np.clip(x.locked, 0, None)).sum()
    return {"rule": label, "horses": len(x), "staked": round(x.staked.sum(), 0),
            "clv_staked": round(x.locked.sum() / x.staked.sum(), 4), "locked": round(x.locked.sum(), 2),
            "after_2pc": round(net, 2), "actual_result": round(x.pnl.sum(), 2)}


rows = [keep(s, "all, as traded")]
for bar in (0.05, 0.08, 0.10, 0.12, 0.15):
    rows.append(keep(s[s.edge >= bar], f"expected edge >= {bar:.0%}"))
rows.append(keep(s[s.price_used >= 3], "price 3.0 or more"))
rows.append(keep(s[(s.price_used >= 3) & (s.price_used <= 21)], "price 3.0-21"))
rows.append(keep(s[(s.edge >= 0.08) & (s.price_used >= 3)], "edge >= 8% and price 3.0+"))
rows.append(keep(s[s.first_t.dt.tz_convert(UK).dt.hour < 11], "first back before 11:00 UK"))
rows.append(keep(s[s.first_mins >= 240], "first back 4 h or more before the off"))
rows.append(keep(s[s.model_feed.str.startswith("no-volume")], "no-volume model only"))
print(pd.DataFrame(rows).to_string(index=False))
lv = s.assign(level_locked=10 * s.clv)
print(f"   staked level GBP10 a horse instead: CLV {s.clv.mean():+.2%}, locked GBP{lv.level_locked.sum():+.2f} on "
      f"GBP{10 * len(s):,.0f}; first back only (no top-ups): CLV {(s.first_stake * s.first_clv).sum() / s.first_stake.sum():+.2%} "
      f"on GBP{s.first_stake.sum():,.0f}")
by_rule = s.assign(rule=np.where(s.day >= "2026-10-07", "GBP400 to win, no limit (7-8 Oct)",
                                 "GBP250 to win, GBP4,000 a day (30 Sep-6 Oct)"))
print(by_rule.groupby(["rule", "band"], observed=True).apply(lambda x: pd.Series({
    "horses": len(x), "staked": x.staked.sum(), "avg_stake": x.staked.mean(), "clv_staked": x.locked.sum() / x.staked.sum(),
    "locked": x.locked.sum()})).round(3).to_string())

print("\n== E. today, horse by horse")
today = f"{TODAY:%Y-%m-%d}"
cols = ["off", "venue", "runner", "first_mins", "model_feed", "edge", "orders", "staked", "first_price", "price",
        "exp_bsp", "bsp", "clv", "result", "pnl"]
t = h[h.day == today].sort_values("first_t")
t = t.assign(model_feed=t.model_feed.str.split(" ").str[0])
print(t[cols].round(3).to_string(index=False))
print(f"   today: {len(t)} horses, GBP{t.staked.sum():,.2f} staked, settled {int(t.result.notna().sum())}, "
      f"CLV {(t.staked * t.clv).sum() / t[t.clv.notna()].staked.sum():+.2%}, P&L GBP{t.pnl.sum():+,.2f}")

os.makedirs("out", exist_ok=True)
with pd.ExcelWriter("out/live_review.xlsx") as xw:
    b.round(4).to_excel(xw, sheet_name="days")
    hh = h.copy()
    hh["first_t"] = hh.first_t.dt.tz_convert(UK).dt.strftime("%Y-%m-%d %H:%M")
    hh.drop(columns=["sxp"]).to_excel(xw, sheet_name="horses", index=False)
    oo = o.copy()
    oo["t"] = oo.t.dt.tz_convert(UK).dt.strftime("%Y-%m-%d %H:%M:%S")
    oo.to_excel(xw, sheet_name="orders", index=False)
    led[led.event.isin(["trade_out", "settle", "commission"])].assign(
        t=lambda x: x.t.dt.tz_convert(UK).dt.strftime("%Y-%m-%d %H:%M:%S")).to_excel(xw, sheet_name="lays_settles",
                                                                                     index=False)
print("written out/live_review.xlsx")
