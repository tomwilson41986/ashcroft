"""How well the Betfair SP prices the winner, over the last 10 and 5 years (the owner, 7 Oct; read-only).

Every GB/IE race in race_results from April 2016 to March 2026 (the holdout from 1 Apr 2026 is not read) where every
runner has a win BSP and there is exactly one winner. Each runner's chance is 1/BSP, scaled so the race sums to 1
(the book is about 1.0 at the BSP, the overround tiny). Reported:

- calibration: runners grouped by that chance, the share that won against the chance, with a 95% interval;
- R-squared three ways, since they answer different questions:
  * across the groups (20 equal-count bins): how closely the win rate follows the BSP's chance (calibration);
  * per runner (Efron's: 1 - sum (won - p)^2 / sum (won - mean)^2): how much of the win/lose outcome the BSP explains;
  * per race (McFadden's pseudo R-squared against each runner equally likely): the information in the BSP's ranking;
- the Brier score, log loss, the conditional-logit exponent (1.0 = the BSP's chances need no stretching; above 1 the
  favourites win more than their price says), how often the winner is shorter than a random loser (AUC), the BSP
  favourite's strike rate, and the return from backing every runner at the BSP by price band;
- every measure by year and for the two windows.
"""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

t0 = time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
COMM = 0.02
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, horse_name, placing_numerical, bfsp
    FROM race_results WHERE race_date >= '2016-04-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
for c in ("bfsp", "placing_numerical"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
n_all, r_all = len(d), d.raceid.nunique()
g = d.groupby("raceid")
whole = g.bfsp.transform(lambda s: bool((s > 1).all()))
winners = (d.placing_numerical == 1).groupby(d.raceid).transform("sum")
d = d[whole & (winners == 1)].copy()
print(f"{n_all:,} runners in {r_all:,} races; kept {len(d):,} runners in {d.raceid.nunique():,} races (every runner "
      f"with a BSP, one winner; dead heats and races missing a BSP left out) ({time.time() - t0:.0f}s)")

d["won"] = (d.placing_numerical == 1).astype(float)
d["p_raw"] = 1.0 / d.bfsp
d["book"] = d.groupby("raceid").p_raw.transform("sum")
d["p"] = d.p_raw / d.book
d["n"] = d.groupby("raceid").p.transform("size")
d = d[d.n >= 2].copy()                                           # no walkovers
d["year"] = d.race_date.str[:4]
d["season"] = np.where(d.race_date.str[5:7] >= "04", d.year, (d.year.astype(int) - 1).astype(str))   # Apr-Mar years


def r2_bins(x: pd.DataFrame, bins: int = 20) -> tuple[float, pd.DataFrame]:
    """Calibration across equal-count bins: the win rate against the mean chance, weighted R-squared of y = p."""
    q = pd.qcut(x.p.rank(method="first"), bins, labels=False)
    t = x.groupby(q).agg(runners=("p", "size"), chance=("p", "mean"), won=("won", "mean"))
    w, y, f = t.runners.to_numpy(float), t.won.to_numpy(), t.chance.to_numpy()
    ybar = np.average(y, weights=w)
    return 1 - np.sum(w * (y - f) ** 2) / np.sum(w * (y - ybar) ** 2), t


def beta_fit(x: pd.DataFrame) -> float:
    """The conditional-logit exponent: chances p^b / sum p^b fitted to the winners (1 = calibrated as it stands)."""
    lp, rid = np.log(x.p.to_numpy()), x.raceid.to_numpy()
    codes = pd.factorize(rid)[0]
    won = x.won.to_numpy() == 1

    def nll(b):
        z = np.exp(b * lp)
        den = np.bincount(codes, weights=z)
        return -(b * lp[won]).sum() + np.log(den).sum()
    return float(minimize_scalar(nll, bounds=(0.5, 2.0), method="bounded").x)


def measures(x: pd.DataFrame) -> dict:
    y, p = x.won.to_numpy(), x.p.to_numpy()
    ybar = y.mean()
    brier, brier_u = np.mean((y - p) ** 2), np.mean((y - 1.0 / x.n.to_numpy()) ** 2)
    win = x[x.won == 1]
    ll, ll_u = -np.log(win.p).mean(), np.log(win.n).mean()            # per race: the winner's chance
    rank = x.groupby("raceid").p.rank(method="average")
    auc = ((rank[x.won == 1] - 1) / (x.n[x.won == 1] - 1)).mean()      # the winner shorter than a random loser
    fav = x.loc[x.groupby("raceid").p.idxmax()]
    r2b, _ = r2_bins(x)
    return {"races": x.raceid.nunique(), "runners": len(x), "book": round(x.groupby("raceid").book.first().mean(), 4),
            "r2_bins": round(r2b, 4), "r2_efron": round(1 - np.sum((y - p) ** 2) / np.sum((y - ybar) ** 2), 4),
            "r2_mcfadden": round(1 - ll / ll_u, 4), "brier": round(brier, 5),
            "brier_skill_vs_equal": round(1 - brier / brier_u, 4), "log_loss_race": round(ll, 4),
            "beta": round(beta_fit(x), 3), "auc": round(auc, 4), "fav_win": round(fav.won.mean(), 4),
            "fav_chance": round(fav.p.mean(), 4)}


def by_band(x: pd.DataFrame) -> pd.DataFrame:
    """Backing every runner at the BSP, GBP1 level: return before and after 2% commission on winnings."""
    edges = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 101.0, 1e9]
    lab = ["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-51", "51-101", "101+"]
    band = pd.cut(x.bfsp, edges, labels=lab, right=False)
    pnl = np.where(x.won == 1, (x.bfsp - 1), -1.0)
    pnl_c = np.where(x.won == 1, (x.bfsp - 1) * (1 - COMM), -1.0)
    t = pd.DataFrame({"band": band, "p": x.p, "won": x.won, "pnl": pnl, "pnl_c": pnl_c}).groupby(
        "band", observed=True).agg(runners=("p", "size"), chance=("p", "mean"), won=("won", "mean"),
                                   roi=("pnl", "mean"), roi_after_2pc=("pnl_c", "mean"))
    t["a_over_e"] = t.won / t.chance
    return t.round(4)


def calib_table(x: pd.DataFrame) -> pd.DataFrame:
    edges = [0, .01, .02, .03, .05, .075, .10, .15, .20, .25, .30, .40, .50, .60, .70, .80, 1.0001]
    lab = [f"{a:.1%}-{b:.0%}" if b < 1 else f"{a:.0%}+" for a, b in zip(edges[:-1], edges[1:])]
    t = x.groupby(pd.cut(x.p, edges, labels=lab, right=False), observed=True).agg(
        runners=("p", "size"), chance=("p", "mean"), won=("won", "mean"))
    se = np.sqrt(t.won * (1 - t.won) / t.runners)
    t["lo95"], t["hi95"] = (t.won - 1.96 * se).clip(lower=0), t.won + 1.96 * se
    t["a_over_e"] = t.won / t.chance
    return t.round(4)


windows = {"10 years (Apr 2016-Mar 2026)": d, "5 years (Apr 2021-Mar 2026)": d[d.race_date >= "2021-04-01"]}
for name, x in windows.items():
    print(f"\n==== {name}")
    print(pd.Series(measures(x)).to_string())
    print("\n-- calibration by the BSP's chance")
    print(calib_table(x).to_string())
    r2b, t20 = r2_bins(x)
    print(f"\n-- the 20 equal-count bins (R-squared {r2b:.4f}): chance, win rate, runners")
    for r in t20.itertuples():
        print(f"BIN|{name[:2].strip()}|{r.chance:.5f}|{r.won:.5f}|{r.runners}")
    print("\n-- backing every runner at the BSP")
    print(by_band(x).to_string())
    print(f"({time.time() - t0:.0f}s)")

print("\n==== by season (April to March)")
rows = []
for s, x in d.groupby("season"):
    m = measures(x)
    m["season"] = f"{s}-{str(int(s) + 1)[2:]}"
    rows.append(m)
print(pd.DataFrame(rows).set_index("season").to_string())
print(f"done ({time.time() - t0:.0f}s)")
