"""The owner's asks (20 Oct): "Review the results and profit today" and "a holistic and systematic review of our
approach and model" (read-only). Every live day, 30 Sep - 20 Oct (today as far as settled):

A. Today: horses, money, CLV at the BSP, the result and commission of the races settled, what is still open, by UK hour
   and price, first backs against top-ups, and the backs over 100.
B. Every day: horses, money, CLV, CLV x stake (the expected profit of the hedged backs before commission), the result
   before and after commission, the running total, and the rule in force.
C. Skill against luck: each back laid at the SP makes S(P - B)/(B - 1) on a loser and nothing on a winner, so its
   expectation is S(P/B - 1), the CLV x stake; the result less that is the noise of which horses won.
D. The closing model's calibration: the CLV each fill closed at, by the expected CLV it was backed at.
E. Where the CLV comes from, by period: the price band, the UK hour, first backs and top-ups, the market's money.
F. The hedges: the losers whose result is not S(P - B)/(B - 1) (a lay at the SP not wholly matched), and the winners
   whose result is not nothing.
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
K = ["day", "market_id", "selection_id"]
TODAY = "2026-10-20"
DAYS = [(date(2026, 9, 30) + timedelta(n)).isoformat() for n in range(21)]
NUM = ["matched", "avg_price", "bsp", "clv", "selection_id", "edge", "pnl", "commission", "minutes_to_off",
       "back_size", "best_back", "target", "p_pool", "move", "hedge_liability"]


def get(key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # noqa: BLE001
        return None


def era(day):
    if day <= "2026-10-05":
        return "a 30 Sep-5 Oct (GBP250, delayed key)"
    if day <= "2026-10-09":
        return "b 6-9 Oct (live key, volume model, no limit 7-8)"
    if day <= "2026-10-18":
        return "c 10-18 Oct (the rule of 9 Oct)"
    return "d 19-20 Oct (floor 15, cap 200)"


leds = []
for day in DAYS:
    raw = get(f"trading/live/{day}/ledger.csv")
    if raw is None:
        print("no ledger", day)
        continue
    d = pd.read_csv(io.BytesIO(raw), low_memory=False, dtype={"market_id": str})
    d["day"] = day
    leds.append(d)
led = pd.concat(leds, ignore_index=True)
for c in NUM:
    if c in led.columns:
        led[c] = pd.to_numeric(led[c], errors="coerce")
led["t"] = pd.to_datetime(led.ts, utc=True, errors="coerce")
led["uk"] = led.t.dt.tz_convert("Europe/London")

b = led[(led.event == "back") & (led.matched > 0)].sort_values("t").copy()
b["sxp"] = b.matched * b.avg_price
b["k"] = b.groupby(K).cumcount()
hm = b.groupby(K).agg(m=("matched", "sum"), sxp=("sxp", "sum")).reset_index()
hm["Pm"] = hm.sxp / hm.m
st = led[led.event == "settle"].sort_values("t").drop_duplicates(K, keep="last")[
    K + ["result", "bsp", "avg_price", "pnl"]].rename(columns={"avg_price": "P", "bsp": "B"})
comm = led[led.event == "commission"].groupby("day").commission.sum()
b = b.drop(columns=["bsp", "result", "pnl", "clv"], errors="ignore").merge(hm[K + ["Pm"]], on=K).merge(st, on=K, how="left")
ok = b.result.isin(["WINNER", "LOSER"]) & (b.B > 1) & (b.P > 1)
b["clv"] = np.where(ok, b.avg_price * b.P / b.Pm / b.B - 1, np.nan)
b["locked"] = b.clv * b.matched
b["kind"] = np.where(b.k == 0, "first", "top-up")
b["era"] = b.day.map(era)
pos = b.groupby(K).agg(S=("matched", "sum"), Pm=("Pm", "first"), first_price=("avg_price", "first"),
                       runner=("runner", "first"), venue=("venue", "first"), backs=("matched", "size"),
                       edge=("edge", "first")).reset_index().merge(st, on=K, how="left")
pos["settled"] = pos.result.isin(["WINNER", "LOSER", "REMOVED"])
v = pos[pos.result.isin(["WINNER", "LOSER"]) & (pos.B > 1) & (pos.P > 1)].copy()
v["clv"] = v.P / v.B - 1
v["exp"] = v.S * v.clv
v["model_result"] = np.where(v.result == "WINNER", 0.0, v.S * (v.P - v.B) / (v.B - 1))

print("== A. today,", TODAY)
tp = pos[pos.day == TODAY]
tv = v[v.day == TODAY]
tb = b[b.day == TODAY]
print(f"   backed: {len(tp)} horses, GBP{tp.S.sum():,.2f} in {len(tb)} orders, first back "
      f"{tb.uk.min():%H:%M} UK, last {tb.uk.max():%H:%M}")
print(f"   settled: {int(tp.settled.sum())} horses (GBP{tp[tp.settled].S.sum():,.2f}); open {int((~tp.settled).sum())} "
      f"(GBP{tp[~tp.settled].S.sum():,.2f})")
if len(tv):
    print(f"   settled with a BSP: {len(tv)} horses, GBP{tv.S.sum():,.2f}, CLV {tv.exp.sum() / tv.S.sum():+.2%}, "
          f"CLV x stake GBP{tv.exp.sum():,.2f}; result GBP{tv.pnl.sum():,.2f} before commission, commission "
          f"GBP{comm.get(TODAY, 0.0):,.2f}, after GBP{tv.pnl.sum() - comm.get(TODAY, 0.0):,.2f}; winners "
          f"{int(tv.result.eq('WINNER').sum())}")
    s = tb[tb.clv.notna()]
    g = s.groupby([s.uk.dt.hour.rename("uk_hour"), "kind"]).agg(orders=("matched", "size"), staked=("matched", "sum"),
                                                                 locked=("locked", "sum"))
    g["clv"] = g.locked / g.staked
    print(g.round(3).to_string())
    k_ = s.groupby("kind").agg(staked=("matched", "sum"), locked=("locked", "sum"))
    k_["clv"] = k_.locked / k_.staked
    print(k_.round(4).to_string())
    print(tv.groupby(pd.cut(tv.Pm, [1, 5, 10, 21, 50, 100, 200, 1001], right=False), observed=True).apply(
        lambda g: pd.Series({"horses": len(g), "staked": g.S.sum(), "clv": g.exp.sum() / g.S.sum(),
                             "result": g.pnl.sum()})).round(3).to_string())
    print("   the backs at 100 or more today:")
    print(tp[tp.Pm >= 100][["venue", "runner", "Pm", "S", "result", "B", "pnl"]].round(2).to_string(index=False))
    print("   the ten biggest results today:")
    print(tv.reindex(tv.pnl.abs().sort_values(ascending=False).index).head(10)[
        ["venue", "runner", "Pm", "S", "B", "clv", "result", "pnl"]].round(3).to_string(index=False))

print("\n== B. every day")
days = v.groupby("day").agg(horses=("S", "size"), staked=("S", "sum"), clv_x_stake=("exp", "sum"),
                            result=("pnl", "sum"), winners=("result", lambda r: int(r.eq("WINNER").sum())))
days["clv"] = days.clv_x_stake / days.staked
days["commission"] = comm.reindex(days.index).fillna(0.0)
days["after_commission"] = days.result - days.commission
days["cumulative"] = days.after_commission.cumsum()
days["era"] = days.index.map(era)
print(days.round(3).to_string())
e = days.groupby("era").agg(days=("staked", "size"), staked=("staked", "sum"), clv_x_stake=("clv_x_stake", "sum"),
                            result=("result", "sum"), after=("after_commission", "sum"))
e["clv"] = e.clv_x_stake / e.staked
e["per_day_after"] = e.after / e.days
print(e.round(3).to_string())

print("\n== C. skill against luck")
print(f"   all settled: GBP{v.S.sum():,.2f} on {len(v):,} horses; CLV x stake GBP{v.exp.sum():,.2f}; result "
      f"GBP{v.pnl.sum():,.2f} before commission, GBP{v.pnl.sum() - comm.sum():,.2f} after (commission "
      f"GBP{comm.sum():,.2f}); the luck GBP{v.pnl.sum() - v.exp.sum():,.2f}")
print(f"   day by day: the result's sd GBP{days.result.std():,.2f}, CLV x stake's sd GBP{days.clv_x_stake.std():,.2f}, "
      f"their correlation {days.result.corr(days.clv_x_stake):.2f}; losing days after commission "
      f"{int((days.after_commission < 0).sum())} of {len(days)}; worst GBP{days.after_commission.min():,.2f}")

print("\n== D. the closing model's calibration (fills, 10-20 Oct: the no-volume model on every feed)")
c = b[(b.day >= "2026-10-10") & b.clv.notna() & b.edge.notna()]
g = c.groupby(pd.cut(c.edge, [0, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.25, 10]), observed=True).apply(
    lambda g: pd.Series({"fills": len(g), "staked": g.matched.sum(), "mean_expected": (g.edge * g.matched).sum() / g.matched.sum(),
                         "realised": g.locked.sum() / g.matched.sum()}))
print(g.round(4).to_string())

print("\n== E. where the CLV comes from, by period (fills)")
s = b[b.clv.notna()]
for name, key in (("price", pd.cut(s.avg_price, [1, 3, 5, 10, 21, 50, 100, 1001], right=False)),
                  ("UK hour", s.uk.dt.hour), ("kind", s.kind),
                  ("market money", pd.cut(pd.to_numeric(s.get("back_size"), errors="coerce"), [0, 5, 10, 25, 100, 1e9]))):
    t = s.groupby([key, s.era], observed=True).apply(lambda g: pd.Series({
        "staked": g.matched.sum(), "clv": g.locked.sum() / g.matched.sum()})).unstack("era")
    print(f"\n   by {name}" + (" (the money offered at the best back)" if name == "market money" else ""))
    print(t.round(3).to_string())

print("\n== F. the hedges")
v["gap"] = v.pnl - v.model_result
bad = v[v.gap.abs() > 1.0]
print(f"   settled horses whose result is more than GBP1 from the hedged result: {len(bad)} of {len(v)}; their gap "
      f"GBP{bad.gap.sum():,.2f} (positive: the account did better)")
if len(bad):
    print(bad.sort_values("gap")[["day", "venue", "runner", "Pm", "S", "B", "result", "pnl", "model_result", "gap"]]
          .round(2).head(25).to_string(index=False))
