"""The morning models: hold to the result, or trade out at the BSP? (16 Nov 2026, read-only)

The owner, 16 Nov 2026: "evaluate whether it is worth trading the steamers and quick returners from 10:00 to BFSP."
Pre-registered: reports/research_ledger.jsonl, tag morning-exit-bsp-1116.

Held-out 2022 onward. Steamer: the CLEAN model fitted on 2018-21 with its training top 5% threshold, field 5+; quick
return: back within 1-4 days, flat/AW/hurdle/chase, field 5+. Each with a morning price (morningwap, GBP50+ matched)
and a BSP. Stake GBP50 to win at the morning price, at most GBP100 (the live rule). Arms:
  HOLD   held to the result
  TRADE  the back laid at the BSP for stake x price / BSP: P&L = stake x (price / BSP - 1), win or lose
  COND   a lay at the SP with a limit of the morning price (LIMIT_ON_CLOSE): traded where BSP <= price, else held
2% on each market's positive net. Sensitivity: the price at 95% of the WAP. Then the paper forward test's real fills.
Prints only; writes nothing but this table's rules to research/trials/morning_exit.json.
"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "research/strat")
import strat_lib as L  # noqa: E402
from research.overfit import TrialLedger, deflate_table  # noqa: E402
from research.paper import morning_features as MF  # noqa: E402

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
TO_WIN, MAX_STAKE, RATE = 50.0, 100.0, 0.02
RNG = np.random.default_rng(0)

df = L.load()
print("table", df.race_date.min(), "to", df.race_date.max(), "rows", len(df))
booster, thr, n_fit = MF.train_steamer(df, cutoff=L.TRAIN_END)
print(f"steamer fitted on 2018-01-01..{L.TRAIN_END}: {n_fit} rows, threshold {thr:.4f}")
x = MF.build(df)
x = L.priced(x)
x = x[x["race_date"] > L.TRAIN_END].copy()
x["pred"] = booster.predict(x[MF.CLEAN])
x["am"] = x["morningwap"].where((x["morning_vol"] >= L.MIN_AM_VOL) & (x["morningwap"] > 1))
sel = {"steamer": x[(x["pred"] >= thr) & (x["field"] >= 5)], "quick_return": x[MF.rule_qr(x)]}
for k, v in sel.items():
    print(f"{k}: {len(v)} selections 2022+, {int(v['am'].notna().sum())} with a morning price "
          f"({len(v) - int(v['am'].notna().sum())} dropped: under GBP50 matched in the morning); duplicates "
          f"{int(v.duplicated(['rk', 'horse_name']).sum())}")


def arms(b: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Per bet: stake and P&L before commission for each arm."""
    p, bsp, won = price.to_numpy(float), b["bsp"].to_numpy(float), b["won"].fillna(0).to_numpy(float)
    s = np.minimum(TO_WIN / (p - 1), MAX_STAKE)
    hold = np.where(won == 1, s * (p - 1), -s)
    trade = s * (p / bsp - 1)
    cond = np.where(bsp <= p, trade, hold)
    return pd.DataFrame({"day": b["race_date"].to_numpy(), "mk": b["rk"].astype(str).to_numpy(), "stake": s,
                         "HOLD": hold, "TRADE": trade, "COND": cond, "clv": p / bsp - 1, "won": won, "p": p,
                         "bsp": bsp})


def daily(a: pd.DataFrame, arm: str) -> pd.DataFrame:
    m = a.groupby(["day", "mk"]).agg(pnl=(arm, "sum"), stake=("stake", "sum"))
    m["net"] = m.pnl - RATE * m.pnl.clip(lower=0)
    return m.groupby("day").agg(net=("net", "sum"), stake=("stake", "sum"))


def summary(a: pd.DataFrame, arm: str) -> dict:
    d = daily(a, arm)
    eq = d.net.cumsum()
    boot = [RNG.choice(d.net.to_numpy(), len(d)).mean() for _ in range(2000)]
    r = d.net / d.stake
    return {"arm": arm, "bets": len(a), "days": len(d), "staked": round(a.stake.sum()), "pnl": round(d.net.sum()),
            "roi%": round(100 * d.net.sum() / a.stake.sum(), 2), "per_bet": round(d.net.sum() / len(a), 2),
            "day_mean": round(d.net.mean(), 2), "day_lo95": round(np.quantile(boot, 0.025), 2),
            "day_hi95": round(np.quantile(boot, 0.975), 2), "day_sd": round(d.net.std(), 2),
            "sharpe_d": round(r.mean() / r.std(), 4), "max_dd": round((eq.cummax() - eq).max()),
            "losing_days%": round(100 * (d.net < 0).mean(), 1)}


series, out = {}, []
for model, b in sel.items():
    b = b[b["am"].notna() & (b["bsp"] > 1)]
    a = arms(b, b["am"])
    won, e_am, e_bsp = a.won.sum(), (1 / a.p).sum(), (1 / a.bsp).sum()
    print(f"\n=== {model}: {len(a)} bets, {a.day.nunique()} days, {a.day.min()}..{a.day.max()} ===")
    print(f"winners {int(won)}; A/E at the morning price {won / e_am:.3f}, at the BSP {won / e_bsp:.3f}; "
          f"CLV (price/BSP - 1) mean {a.clv.mean():+.4f}, median {a.clv.median():+.4f}, "
          f"share steamed (BSP < price) {100 * (a.bsp < a.p).mean():.1f}%")
    t = pd.DataFrame([summary(a, k) for k in ("HOLD", "TRADE", "COND")])
    print(t.to_string(index=False))
    out.append(t.assign(model=model, price="WAP"))
    for k in ("HOLD", "TRADE", "COND"):
        d = daily(a, k)
        series[f"{model}:{k}"] = (d.net / d.stake).to_numpy()
    print("by year (P&L GBP):")
    a["yr"] = a.day.str[:4]
    print(pd.DataFrame([{"year": y, "bets": len(g), **{k: round(daily(g, k).net.sum()) for k in ("HOLD", "TRADE", "COND")}}
                        for y, g in a.groupby("yr")]).to_string(index=False))
    print("by CLV band (bets, HOLD and TRADE P&L before commission, GBP):")
    a["band"] = pd.cut(a.clv, [-1, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 10])
    print(a.groupby("band", observed=True).agg(bets=("stake", "size"), hold=("HOLD", "sum"), trade=("TRADE", "sum"),
                                               won=("won", "mean")).round(2).to_string())
    a95 = arms(b, b["am"] * 0.95)
    t95 = pd.DataFrame([summary(a95, k) for k in ("HOLD", "TRADE", "COND")])
    print("sensitivity, the price at 95% of the WAP:")
    print(t95[["arm", "pnl", "roi%", "per_bet", "day_mean", "day_lo95", "day_hi95", "sharpe_d", "max_dd"]].to_string(index=False))

# the deflated Sharpe on daily returns on stake, every strategy-study rule counted
bk = os.environ.get("PRED_BUCKET")
base = TrialLedger(f"s3://{bk}/research/trials/strat_am.json", programme="strategy studies (am)") if bk else None
prior = base.n_trials if base is not None else 0
own = TrialLedger(f"s3://{bk}/research/trials/morning_exit.json", programme="morning exit") if bk else None
res = deflate_table(series, n_trials=prior + len(series), ledger=own)
print(f"\ndeflated Sharpe (daily return on stake; strategy-study rules already scored {prior}, plus this table):")
print(pd.DataFrame(res).T[["n_obs", "n_trials", "sharpe", "dsr", "passed", "reason"]].to_string())

# the paper forward test's real fills (GBP100 to win; halved here to the live GBP50)
if bk:
    import boto3
    s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION", "eu-west-2"))
    keys = [o["Key"] for o in s3.list_objects_v2(Bucket=bk, Prefix="paper/morning_edges/bets/").get("Contents", [])]
    fr = []
    for k in keys:
        day = k.rsplit("/", 1)[-1][:10]
        if day < "2026-11-05":
            continue
        f = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bk, Key=k)["Body"].read()), dtype={"market_id": str})
        fr.append(f.assign(day=day))
    if fr:
        f = pd.concat(fr, ignore_index=True)
        f = f[(f.get("market") == "win") & (f.get("event") == "back") & (pd.to_numeric(f.matched, errors="coerce") > 0)]
        f = f[pd.to_numeric(f.bsp, errors="coerce") > 1]
        print(f"\n=== paper forward test, real fills: {f.day.min()}..{f.day.max()}, {len(f)} win backs (halved to GBP50 to win) ===")
        for strat, model in (("steam_am", "steamer"), ("qr_am", "quick_return")):
            g = f[f.strategy == strat]
            if g.empty:
                print(model, "no data found"); continue
            s = pd.to_numeric(g.matched) / 2
            p, bsp = pd.to_numeric(g.price), pd.to_numeric(g.bsp)
            won = (g.result == "WINNER").astype(float).to_numpy()
            a = pd.DataFrame({"day": g.day.to_numpy(), "mk": g.market_id.to_numpy(), "stake": s.to_numpy(),
                              "HOLD": (pd.to_numeric(g.pnl) / 2).to_numpy(), "TRADE": (s * (p / bsp - 1)).to_numpy()})
            a["COND"] = np.where(bsp.to_numpy() <= p.to_numpy(), a.TRADE, a.HOLD)
            rows = []
            for k in ("HOLD", "TRADE", "COND"):
                d = daily(a, k)
                rows.append({"arm": k, "bets": len(a), "days": len(d), "pnl": round(d.net.sum(), 2),
                             "roi%": round(100 * d.net.sum() / a.stake.sum(), 2), "day_sd": round(d.net.std(), 2)})
            print(f"{model}: winners {int(won.sum())} vs {(1 / p).sum():.2f} at the price; CLV mean "
                  f"{(p / bsp - 1).mean():+.4f}")
            print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("\npaper forward test: no data found")
