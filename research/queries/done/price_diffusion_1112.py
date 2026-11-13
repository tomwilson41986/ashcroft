"""The price as a diffusion: does an option-pricing view of the morning price hold? (read-only; ledger
`price-diffusion-1112`; the owner, 12 Nov 2026: "research where we can use ... Black Scholes")

The trade backs in the morning and lays at the Betfair SP, so its result is the move of the price from the entry to
the close. Black and Scholes price a claim on a quantity that diffuses without drift and with a known volatility.
Here the quantity is x = logit of the runner's implied chance (the middle of back and lay). If x moves to the close
without drift and with a spread that depends only on the time to the off and the price, then the chance that the SP
ends shorter than a back price B is N((x - logit(1/B)) / sigma): a digital option, and the spread of the trade's
result is known when the bet is struck, so a stake can be sized by it.

The recorded marks (betfair_live_marks, WIN markets): the book at 08:00-12:00 UK and at 120..1 minutes before the
off, and the BSP. For each runner and mark: d = logit(1/BSP) - x.
1. Drift: the mean of d and of (1/BSP - p) by time to the off.
2. Volatility: the standard deviation of d by time to the off and price band; the fit ln var = a + b ln hours.
3. The test, out of sample: sigma by cell (time band x price band) from the first half of the days; on the second
   half, the cell's realised sd over the forecast, and the calibration of N(.) for "the SP ends shorter than the
   best back now" by fifths of the forecast.
Registered pass: every cell with 200 readings has realised/forecast sd within 0.85-1.15, and every fifth's realised
frequency is within 3 points of its forecast. A pass goes on to a stake sized by the forecast spread, in the replay.
"""
import sqlite3
from math import erf, sqrt

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
mk = pd.read_sql_query("SELECT race_date, market_id, selection_id, mark, runner_status, back1, lay1, bsp "
                       "FROM betfair_live_marks", con)
cat = pd.read_sql_query("SELECT DISTINCT market_id, market_type, market_start_utc FROM betfair_live_markets", con)
cat = cat[cat.market_type.fillna("WIN").str.upper().eq("WIN")].drop_duplicates("market_id")
mk = mk.merge(cat[["market_id", "market_start_utc"]], on="market_id")
for c in ("back1", "lay1", "bsp"):
    mk[c] = pd.to_numeric(mk[c], errors="coerce")
fin = mk[(mk.mark == "final") & (mk.bsp > 1)][["market_id", "selection_id", "bsp"]].drop_duplicates(["market_id", "selection_id"])
b = mk[(mk.mark != "final") & (mk.mark != "last") & (mk.back1 > 1) & (mk.lay1 > 1)].drop(columns="bsp").merge(
    fin, on=["market_id", "selection_id"])
off = pd.to_datetime(b.market_start_utc, utc=True, errors="coerce")
is_clock = b.mark.str.startswith("clock_")
clock = pd.to_datetime(b.race_date.str[:10] + " " + b.mark.str.replace("clock_", "", regex=False).where(is_clock, "00:00"),
                       errors="coerce").dt.tz_localize("Europe/London", nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC")
mins = pd.to_numeric(b.mark.str.replace("off_", "", regex=False).where(~is_clock), errors="coerce")
b["hours"] = np.where(is_clock, (off - clock).dt.total_seconds() / 3600.0, mins / 60.0)
b = b[(b.hours > 0) & (b.lay1 / b.back1 <= 1.5)].copy()
b["p"] = (1 / b.back1 + 1 / b.lay1) / 2
logit = lambda q: np.log(q / (1 - q))  # noqa: E731
b["x"], b["d"] = logit(b.p), logit(1 / b.bsp) - logit(b.p)
b["dp"] = 1 / b.bsp - b.p
b["day"] = b.race_date.str[:10]
b["tb"] = pd.cut(b.hours, [0, 0.05, 0.1, 0.3, 0.6, 1.2, 2.5, 5, 8, 24], labels=["<3m", "3-6m", "6-18m", "18-36m", "36-72m",
                                                                                  "1.2-2.5h", "2.5-5h", "5-8h", "8h+"])
b["pb"] = pd.cut(b.p, [0, 0.05, 0.15, 0.35, 1], labels=["<5%", "5-15%", "15-35%", "35%+"])
print(f"readings {len(b):,}; runners {b.groupby(['market_id', 'selection_id']).ngroups:,}; markets {b.market_id.nunique():,}; "
      f"days {b.day.nunique()} ({b.day.min()} .. {b.day.max()}); kept where lay/back <= 1.5")
print("\n1-2. by time to the off: drift and spread")
t = b.groupby("tb", observed=True).agg(n=("d", "size"), hours=("hours", "median"), mean_d=("d", "mean"), sd_d=("d", "std"),
                                       mean_dp=("dp", "mean"))
t["se_d"] = t.sd_d / np.sqrt(t.n)
print(t.round(4).to_string())
fit = np.polyfit(np.log(t.hours), np.log(t.sd_d ** 2), 1)
print(f"ln var = {fit[1]:+.3f} {fit[0]:+.3f} ln hours (a pure diffusion in clock time has slope +1)")
print("\nsd of d by price band and time band")
print(b.pivot_table(index="pb", columns="tb", values="d", aggfunc="std", observed=True).round(3).to_string())

days = sorted(b.day.unique())
cut = days[len(days) // 2]
A, B = b[b.day < cut], b[b.day >= cut].copy()
cell = A.groupby(["tb", "pb"], observed=True).d.std().rename("sigma")
B = B.merge(cell, left_on=["tb", "pb"], right_index=True, how="inner")
r = B.groupby(["tb", "pb"], observed=True).agg(n=("d", "size"), real=("d", "std"), fc=("sigma", "first"))
r["ratio"] = r.real / r.fc
big = r[r.n >= 200]
print(f"\n3. out of sample (sigma from days before {cut}; scored {cut} on): realised sd over forecast, cells with 200+")
print(big.round(3).to_string())
N = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))
B["fc"] = N((B.x - logit(1 / B.back1)) / B.sigma)
B["hit"] = (B.bsp < B.back1).astype(float)
B["q"] = pd.qcut(B.fc, 5, labels=False, duplicates="drop")
c = B.groupby("q").agg(n=("hit", "size"), forecast=("fc", "mean"), realised=("hit", "mean"))
c["gap"] = c.realised - c.forecast
print("\nthe SP ends shorter than the best back now: forecast against realised, by fifths of the forecast")
print(c.round(4).to_string())
ok = bool(((big.ratio >= 0.85) & (big.ratio <= 1.15)).all() and (c.gap.abs() <= 0.03).all())
print("verdict:", "PASS" if ok else "does not pass")
