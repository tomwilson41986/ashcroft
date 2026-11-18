"""HorseRaceBase's Ratings Machine and System Builder components on our own results: how predictive each is, and the
weights each rating set should carry (the owner, 18 Nov 2026; ledger hrb-machine-1118, registered before the run;
read-only, nothing asked of HRB).

model/blocks/hrb_machine.py builds the components from race_results (2018 on, earlier days only). Then, on races with
a winner and every runner priced at the BSP, development window only (nothing from 1 Apr 2026, the locked holdout):
fit on 2021-2024, scored on 1 Jan 2025 - 31 Mar 2026.

1. Each component alone: a conditional logit (softmax within the race) on its z-score; the scored log-loss gain per
   race over an equal-chance field, and over the BSP when it is added to log p_BSP (the System Builder question:
   does it carry anything the market has not priced?), with race-clustered bootstrap intervals.
2. Each set (form, class, speed, connections, horse, weight, market) as one rating: the fitted weights (per standard
   deviation: the Ratings Machine's multipliers, put on one scale) and its gain alone and over the BSP.
3. All sets together over the BSP.
Missing values are read as the training mean (z = 0), as a points rating gives a missing component no score.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.getcwd())
from model.blocks import hrb_machine as hm  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
TRAIN = ("2021-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-03-31")
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
have = {r[1] for r in conn.execute("PRAGMA table_info(race_results)")}
want = [c for c in hm.READS if c in have and c != "raceid"]
print("race_results columns read:", len(want), "; not in the table:", sorted(set(hm.READS) - have - {"raceid"}))
df = pd.read_sql_query(f"SELECT {', '.join(want)} FROM race_results WHERE race_date >= '2018-01-01' "
                       f"AND race_date <= '{TEST[1]}'", conn)
print(f"rows {len(df):,}, {df.race_date.min()} to {df.race_date.max()}")
df = hm.build(df)

d = df[(df.race_date >= TRAIN[0]) & (df.race_date <= TEST[1])].copy()
d["race"] = d.race_date.astype(str) + "|" + d.track.astype(str) + "|" + d.race_time.astype(str)
d["bsp"] = pd.to_numeric(d.bfsp, errors="coerce")
d["pos"] = pd.to_numeric(d.placing_numerical, errors="coerce")
g = d.groupby("race")
ok = (g.bsp.transform(lambda s: (s > 1).all())) & (g.pos.transform(lambda s: (s == 1).sum() == 1)) \
    & (g.race.transform("size") >= 3)
d = d[ok].copy()
d["won"] = (d.pos == 1).astype(float)
p = 1 / d.bsp
d["lp_bsp"] = np.log(p / p.groupby(d.race).transform("sum"))
d = d.sort_values(["race_date", "race"]).reset_index(drop=True)
tr = (d.race_date <= TRAIN[1]).to_numpy()
te = (d.race_date >= TEST[0]).to_numpy()
print(f"races: train {d.loc[tr, 'race'].nunique():,} ({tr.sum():,} runners), test {d.loc[te, 'race'].nunique():,} "
      f"({te.sum():,} runners)")


def z(cols):
    X = d[cols].astype(float).to_numpy()
    mu = np.nanmean(X[tr], axis=0)
    sd = np.nanstd(X[tr], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0)


def fit(X, idx, l2=1e-3):
    """Conditional logit (the softmax within each race) on rows idx; returns the weights."""
    Xs, y = X[idx], d.won.to_numpy()[idx]
    r = pd.factorize(d.race.to_numpy()[idx])[0]
    nr = r.max() + 1

    def f(w):
        s = Xs @ w
        m = np.zeros(nr)
        np.maximum.at(m, r, s)
        e = np.exp(s - m[r])
        den = np.bincount(r, e, nr)
        ll = (y * (s - m[r])).sum() - np.log(den).sum()
        pr = e / den[r]
        grad = Xs.T @ (y - pr)
        return -ll + l2 * w @ w, -grad + 2 * l2 * w

    w0 = np.zeros(X.shape[1])
    return minimize(f, w0, jac=True, method="L-BFGS-B").x


def race_ll(X, w, idx):
    """Each scored race's log-likelihood of its winner, and the race ids."""
    s = X[idx] @ w
    r = pd.factorize(d.race.to_numpy()[idx])[0]
    nr = r.max() + 1
    m = np.zeros(nr)
    np.maximum.at(m, r, s)
    e = np.exp(s - m[r])
    lp = s - m[r] - np.log(np.bincount(r, e, nr))[r]
    return np.bincount(r, lp * d.won.to_numpy()[idx], nr)


def gain(ll_with, ll_base, n_boot=1000, seed=0):
    diff = ll_with - ll_base
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return diff.mean(), np.quantile(b, 0.05), np.quantile(b, 0.95)


n_te = np.bincount(pd.factorize(d.race.to_numpy()[te])[0])
ll_uniform = -np.log(n_te.astype(float))
LB = d[["lp_bsp"]].to_numpy()
w_b = fit(LB, tr)
ll_bsp = race_ll(LB, w_b, te)
print(f"\nthe BSP alone (weight {w_b[0]:.3f}): log-likelihood a race {ll_bsp.mean():.4f}; equal chances "
      f"{ll_uniform.mean():.4f}")

rows = []
for s, cols in hm.SETS.items():
    for c in cols:
        Zc = z([c])
        cover = float(d.loc[te, c].notna().mean())
        w1 = fit(Zc, tr)
        g1 = gain(race_ll(Zc, w1, te), ll_uniform)
        X2 = np.column_stack([LB, Zc])
        w2 = fit(X2, tr)
        g2 = gain(race_ll(X2, w2, te), ll_bsp)
        rows.append({"set": s, "feature": c, "coverage": cover, "w_alone": w1[0], "gain_alone": g1[0],
                     "w_over_bsp": w2[1], "gain_over_bsp": g2[0], "lo90": g2[1], "hi90": g2[2]})
R = pd.DataFrame(rows)
print("\n== 1. each component (gain: scored log-likelihood a race, x1000; weights per sd)")
show = R.assign(gain_alone=R.gain_alone * 1000, gain_over_bsp=R.gain_over_bsp * 1000, lo90=R.lo90 * 1000,
                hi90=R.hi90 * 1000)
print(show.round(3).to_string(index=False))
print(f"\n   components whose gain over the BSP has its 90% interval above zero: "
      f"{int((R.lo90 > 0).sum())} of {len(R)} (by chance about {0.05 * len(R):.1f})")

print("\n== 2. each set as one rating: weights per sd (alone; over the BSP), and the scored gain x1000")
srows = []
for s, cols in hm.SETS.items():
    Zs = z(cols)
    w1 = fit(Zs, tr)
    g1 = gain(race_ll(Zs, w1, te), ll_uniform)
    X2 = np.column_stack([LB, Zs])
    w2 = fit(X2, tr)
    g2 = gain(race_ll(X2, w2, te), ll_bsp)
    srows.append({"set": s, "n": len(cols), "gain_alone": g1[0] * 1000, "gain_over_bsp": g2[0] * 1000,
                  "lo90": g2[1] * 1000, "hi90": g2[2] * 1000})
    wt = pd.DataFrame({"feature": cols, "weight_alone": w1, "weight_over_bsp": w2[1:]})
    wt["share_of_rating"] = (wt.weight_alone.abs() / wt.weight_alone.abs().sum()).round(3)
    print(f"\n  {s}:")
    print(wt.round(3).to_string(index=False))
print("\n  the sets:")
print(pd.DataFrame(srows).round(3).to_string(index=False))

allc = [c for v in hm.SETS.values() for c in v]
X = np.column_stack([LB, z(allc)])
w = fit(X, tr)
ga = gain(race_ll(X, w, te), ll_bsp)
Zall = z(allc)
wa = fit(Zall, tr)
gu = gain(race_ll(Zall, wa, te), ll_uniform)
print(f"\n== 3. every component together: alone {gu[0] * 1000:.2f} x1000 a race over equal chances (the BSP: "
      f"{(ll_bsp - ll_uniform).mean() * 1000:.2f}); over the BSP {ga[0] * 1000:.3f} ({ga[1] * 1000:.3f} to "
      f"{ga[2] * 1000:.3f}); the BSP's weight then {w[0]:.3f}")
