"""The model's top pick, backed at Betfair's morning price: closing-line value and profit.

The standing goal is a profitable rank-1 selection. At BSP the model's rank-1 loses
about 4% (iterations 17-20), as the favourite does: the closing market is efficient
against every block tested. The one edge that has held up is earlier in the day -- the
morning price, where the model's forecast of BSP beats the morning market
(reports/clv_betfair_2026q1.md, +4.83% CLV under the pre-registered rule). So: what does
the model's rank-1 do when it is backed at the morning price?

Descriptive, on data already scored: data/oos_predictions.csv (the walk-forward forecast
behind the Q1 test, ending 21 Mar 2026) against Betfair's win-market files for Jan-Mar
2026 (no price on or after 1 Apr is read). Nothing here is a new rule; the pre-registered
rule stays the one the forward test scores.

For each selection: net CLV (morning WAP / BSP - 1, 5% commission on a positive), and
profit backing at the morning WAP and at BSP (5% commission on winnings), each with a
90% race-bootstrap interval.
"""
import sqlite3

import numpy as np
import pandas as pd

COMM = 0.05
KEY = ["race_date", "track", "race_time", "horse_name"]
pd.set_option("display.width", 220)

o = pd.read_csv("data/oos_predictions.csv", dtype={"race_time": str})
o = o[(o.race_date >= "2026-01-01") & (o.race_date < "2026-04-01")]
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
ex = pd.read_sql_query("""
    SELECT r.race_date, r.track, r.race_time, r.horse_name, b.morningwap, b.morning_vol, b.bsp
    FROM betfair_prices b JOIN race_results r ON r.id = b.race_results_id
    WHERE b.market_type = 'win' AND r.race_date >= '2026-01-01' AND r.race_date < '2026-04-01'""", conn)
d = o.merge(ex, on=KEY)
d["race"] = d.race_date + "|" + d.track + "|" + d.race_time
d = d[(d.morningwap > 1) & (d.bsp > 1) & (d.predicted_bfsp > 1)].copy()
# whole races only: the ranks mean nothing in a race missing runners
d = d[d.groupby("race").horse_name.transform("size") == d.number_of_runners]
d["won"] = d.won.astype(str).str.lower().isin(["true", "1", "1.0"])
d["rank_model"] = d.groupby("race").predicted_bfsp.rank(method="first")
d["rank_morning"] = d.groupby("race").morningwap.rank(method="first")
d["rank_bsp"] = d.groupby("race").bsp.rank(method="first")
d["move"] = np.log(d.morningwap / d.predicted_bfsp)
clv = d.morningwap / d.bsp - 1
d["clv"] = np.where(clv > 0, clv * (1 - COMM), clv)
d["pl_morning"] = np.where(d.won, (d.morningwap - 1) * (1 - COMM), -1.0)
d["pl_bsp"] = np.where(d.won, (d.bsp - 1) * (1 - COMM), -1.0)
print(f"{len(d):,} runners in {d.race.nunique():,} whole races, {d.race_date.min()} to {d.race_date.max()}")


def interval(frame, col, n=2000, seed=0):
    if len(frame) < 30:
        return f"too few ({len(frame)})"
    g = frame.groupby("race")[col].agg(["sum", "size"])
    s, c = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    b = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return f"{100 * frame[col].mean():+6.2f}% ({100 * np.percentile(b, 5):+.2f} to {100 * np.percentile(b, 95):+.2f})"


liquid = d[d.morning_vol >= 100]
r1 = liquid[liquid.rank_model == 1]
sel = {
    "every runner (GBP100+ matched in the morning)": liquid,
    "the model's rank-1": r1,
    "rank-1 that is also the morning favourite": r1[r1.rank_morning == 1],
    "rank-1 that is NOT the morning favourite": r1[r1.rank_morning != 1],
    "rank-1 with the model >= 11% shorter than the morning price": r1[r1.move >= 0.1],
    "rank-1 with the model >= 22% shorter (the pre-registered threshold)": r1[r1.move >= 0.2],
    "rank-1 with the model LONGER than the morning price": r1[r1.move < 0],
    "the pre-registered rule, any rank": liquid[liquid.move >= 0.2],
    "the morning favourite": liquid[liquid.rank_morning == 1],
}
rows = []
for name, x in sel.items():
    rows.append({"selection": name, "bets": len(x), "strike": f"{x.won.mean():.3f}" if len(x) else "",
                 "net CLV": interval(x, "clv"), "profit at morning WAP": interval(x, "pl_morning"),
                 "profit at BSP": interval(x, "pl_bsp")})
print(pd.DataFrame(rows).set_index("selection").to_string())

print("\nthe model's rank-1 by month (profit at the morning WAP)")
for m, x in r1.groupby(r1.race_date.str[:7]):
    print(f"  {m}: {len(x):,} bets, strike {x.won.mean():.3f}, CLV {interval(x, 'clv')}, "
          f"profit {interval(x, 'pl_morning')}")
print("\nthe model's rank-1 by morning price band")
for band, x in r1.groupby(pd.cut(r1.morningwap, [1, 2, 3, 4, 6, 10, 1000]), observed=True):
    print(f"  {str(band):14s} {len(x):5,} bets, strike {x.won.mean():.3f}, CLV {interval(x, 'clv')}, "
          f"profit {interval(x, 'pl_morning')}")
