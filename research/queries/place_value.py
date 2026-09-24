"""Is the Betfair place market priced consistently with the win market?

Benter priced the exotic pools from the win pool's probabilities; Ziemba and Hausch
made a system of the show pools disagreeing with the win pool. The win market has
beaten every fundamental block this project has tried (research ledger, iterations
1-19), so take it as the truth and ask whether the PLACE market agrees with it.

Development data only: 2021-01-01 to 2026-03-31 (the holdout is not read). The
ordering exponents and the combination are fitted on 2021-22 and everything is
scored on 2023-01 onwards.

0. Coverage: place BSP and places paid, by year.
1. Benter / Lo-Bacon-Shone ordering exponents (gamma, delta) per field band, 2021-22.
2. Forecasting "placed" on 2023+: the place chance the WIN BSPs imply (discounted
   Harville; four places simulated), the place market's own (1 / place BSP, scaled so
   the race sums to the places paid), and a logistic combination of the two fitted on
   2021-22. Log loss, and calibration of each by bins.
3. Value: back to place at place BSP where the expected return after commission
   exceeds a threshold -- ROI with race-clustered 90% intervals, by year, field band
   and win price band.
4. The simplest rule: the win-market favourite backed to place, by price and field.

Both prices are struck at the off, so this is a first look at whether the markets
disagree systematically; a bet placed before the off sees pre-off prices instead.
"""
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, ".")
from model.ordering import position_probabilities, simulate_finishing_orders  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 220)
COMMISSION = (0.05, 0.02)
BANDS = ((2, 7), (8, 11), (12, 15), (16, 40))

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, race_type, number_of_runners, horse_name, placing_numerical,
           bfsp, bfsp_place, bf_plcs_paid, plcs_paid
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
for c in ("bfsp", "bfsp_place", "bf_plcs_paid", "plcs_paid", "placing_numerical", "number_of_runners"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["year"] = d.race_date.str[:4]
print(f"{len(d):,} rows, {d.raceid.nunique():,} races ({time.time() - t0:.0f}s)")

# --- 0. coverage --------------------------------------------------------------------------
g = d.groupby("raceid")
d["runners"] = g.horse_name.transform("size")
d["win_ok"] = (d.bfsp > 1).groupby(d.raceid).transform("all")
d["plc_ok"] = (d.bfsp_place > 1).groupby(d.raceid).transform("all")
d["paid"] = d.bf_plcs_paid.where(d.bf_plcs_paid > 0, d.plcs_paid)
cov = d.groupby("year").agg(rows=("horse_name", "size"), win_bsp=("win_ok", "mean"), place_bsp=("plc_ok", "mean"),
                            bf_paid=("bf_plcs_paid", lambda s: s.gt(0).mean()), paid=("paid", lambda s: s.gt(0).mean()))
print("\n== 0. coverage by year (share of rows in races where every runner has the price)")
print(cov.round(3).to_string())
print("\nplaces paid (Betfair) by field size")
print(pd.crosstab(pd.cut(d.runners, [1, 4, 7, 11, 15, 40]), d.bf_plcs_paid.fillna(-1)).to_string())

r = d[d.win_ok & d.plc_ok & d.paid.gt(0) & d.placing_numerical.notna().groupby(d.raceid).transform("any")].copy()
r["paid"] = r.groupby("raceid").paid.transform("max").astype(int)
r = r[r.paid < r.runners]
r["pi"] = (1 / r.bfsp) / (1 / r.bfsp).groupby(r.raceid).transform("sum")
r["placed"] = (r.placing_numerical <= r.paid).astype(float)
r["won"] = (r.placing_numerical == 1).astype(float)
print(f"\nraces with both markets and places paid: {r.raceid.nunique():,} ({len(r):,} runners)")

# --- 1. ordering exponents, 2021-22 ---------------------------------------------------------
races = [x for _, x in r.groupby("raceid", sort=False)]


def padded(rs):
    n = max(len(x) for x in rs)
    P = np.zeros((len(rs), n))
    F = np.full((len(rs), 3), -1)
    for k, x in enumerate(rs):
        P[k, :len(x)] = x.pi.to_numpy()
        pos = x.placing_numerical.to_numpy()
        for j in range(3):
            w = np.flatnonzero(pos == j + 1)
            if len(w) == 1:
                F[k, j] = w[0]
    return P, F


def nll(theta, P, F):
    gm, dl = theta
    with np.errstate(divide="ignore"):
        S = np.where(P > 0, P ** gm, 0.0); S /= S.sum(1, keepdims=True)
        T = np.where(P > 0, P ** dl, 0.0); T /= T.sum(1, keepdims=True)
    ok = (F >= 0).all(1)
    a, b, c = F[ok, 0], F[ok, 1], F[ok, 2]
    i = np.flatnonzero(ok)
    l2 = np.log(np.clip(S[i, b] / (1 - S[i, a]), 1e-12, None))
    l3 = np.log(np.clip(T[i, c] / np.clip(1 - T[i, a] - T[i, b], 1e-12, None), 1e-12, None))
    return -(l2 + l3).sum()


fit = {}
early = [x for x in races if x.race_date.iat[0] < "2023-01-01"]
print("\n== 1. ordering exponents fitted on 2021-22")
for lo, hi in BANDS:
    sub = [x for x in early if lo <= len(x) <= hi]
    P, F = padded(sub)
    res = minimize(nll, np.array([0.8, 0.65]), args=(P, F), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-3})
    fit[(lo, hi)] = tuple(res.x)
    print(f"  {lo}-{hi} runners: gamma {res.x[0]:.3f} delta {res.x[1]:.3f} on {len(sub):,} races "
          f"(nll {res.fun:,.0f} vs Harville {nll((1.0, 1.0), P, F):,.0f})")


def band_of(n):
    for lo, hi in BANDS:
        if lo <= n <= hi:
            return (lo, hi)
    return BANDS[-1]


def top3(p, gm, dl):
    s = p ** gm; s /= s.sum()
    t = p ** dl; t /= t.sum()
    A = p[:, None] * s[None, :] / (1 - s[:, None]); np.fill_diagonal(A, 0.0)
    B = 1.0 / np.clip(1 - t[:, None] - t[None, :], 1e-12, None); np.fill_diagonal(B, 0.0)
    M = A * B
    p2 = A.sum(0)
    p3 = t * (M.sum() - M.sum(1) - M.sum(0))
    return p, p2, p3


rng = np.random.default_rng(0)
q_win = np.empty(len(r))
pos = 0
t1 = time.time()
for x in races:
    p = x.pi.to_numpy()
    gm, dl = fit[band_of(len(p))]
    k = int(x.paid.iat[0])
    if k <= 3:
        p1, p2, p3 = top3(p, gm, dl)
        q = p1 + (p2 if k >= 2 else 0) + (p3 if k >= 3 else 0)
    else:
        o = simulate_finishing_orders(p, n_sims=4000, gamma=gm, delta=dl, rng=rng)
        q = position_probabilities(o, len(p), n_positions=k).sum(axis=1)
    q_win[pos:pos + len(p)] = q
    pos += len(p)
r = pd.concat(races, ignore_index=True)
r["q_win"] = np.clip(q_win, 1e-4, 1 - 1e-4)
inv = 1 / r.bfsp_place
r["q_mkt"] = np.clip(inv * r.paid / inv.groupby(r.raceid).transform("sum"), 1e-4, 1 - 1e-4)
r["q_mkt_raw"] = np.clip(inv, 1e-4, 1 - 1e-4)
r["book"] = inv.groupby(r.raceid).transform("sum") / r.paid
print(f"win-implied place chances in {time.time() - t1:.0f}s")

# --- 2. which forecasts "placed" better? ------------------------------------------------------
lg = lambda q: np.log(q / (1 - q))  # noqa: E731
tr = r.race_date < "2023-01-01"
te = ~tr
X = np.c_[np.ones(len(r)), lg(r.q_win), lg(r.q_mkt)]
y = r.placed.to_numpy()


def ll(beta, X, y):
    z = X @ beta
    return -(y * z - np.logaddexp(0, z)).sum()


beta = minimize(ll, np.array([0.0, 0.5, 0.5]), args=(X[tr.to_numpy()], y[tr.to_numpy()]), method="BFGS").x
r["q_comb"] = 1 / (1 + np.exp(-(X @ beta)))
print(f"\n== 2. forecasting 'placed', 2023 on ({te.sum():,} runners, {r[te].raceid.nunique():,} races)")
print(f"combination fitted on 2021-22: logit q = {beta[0]:+.3f} + {beta[1]:.3f} logit q_win + {beta[2]:.3f} logit q_mkt")
for name in ("q_win", "q_mkt", "q_comb"):
    q = r.loc[te, name].to_numpy(); yy = y[te.to_numpy()]
    L = -(yy * np.log(q) + (1 - yy) * np.log(1 - q))
    print(f"  {name:7s} log loss {L.mean():.5f}   Brier {np.mean((q - yy) ** 2):.5f}")
diff = (-(y * np.log(r.q_win) + (1 - y) * np.log(1 - r.q_win)) + (y * np.log(r.q_mkt) + (1 - y) * np.log(1 - r.q_mkt)))
per_race = diff[te].groupby(r.raceid[te]).sum()
print(f"  q_win less q_mkt log loss per race: {per_race.mean() * 1000:+.2f} mnats (t {per_race.mean() / per_race.std() * np.sqrt(len(per_race)):+.2f}); "
      "negative = the win market forecasts placing better than the place market")
print(f"  place book (sum of 1/place BSP / places paid), 2023 on: median {r.loc[te].groupby('raceid').book.first().median():.3f}")

print("\ncalibration, 2023 on: realised place rate by decile of each forecast")
for name in ("q_win", "q_mkt"):
    b = pd.qcut(r.loc[te, name], 10, labels=False, duplicates="drop")
    t = r.loc[te].groupby(b).agg(pred=(name, "mean"), real=("placed", "mean"), n=("placed", "size"))
    print(f"  {name}: " + "  ".join(f"{a:.3f}/{c:.3f}" for a, c in zip(t.pred, t.real)))

# --- 3. value bets ------------------------------------------------------------------------------
def roi_table(sel, q, label, comm):
    ret = np.where(sel.placed == 1, (sel.bfsp_place - 1) * (1 - comm), -1.0)
    s = pd.Series(ret, index=sel.index)
    by_race = s.groupby(sel.raceid).sum()
    n_bets = len(s)
    if n_bets < 30:
        return {"rule": label, "bets": n_bets}
    m = s.mean()
    se = by_race.std() * np.sqrt(len(by_race)) / n_bets
    return {"rule": label, "bets": n_bets, "races": len(by_race), "ROI": m, "lo90": m - 1.645 * se, "hi90": m + 1.645 * se,
            "strike": sel.placed.mean(), "avg_placeBSP": sel.bfsp_place.mean()}


s = r[te].copy()
print("\n== 3. back to place at place BSP where the expected return clears a threshold (2023 on)")
for comm in COMMISSION:
    rows = []
    for name in ("q_win", "q_comb"):
        ev = s[name] * (s.bfsp_place - 1) * (1 - comm) - (1 - s[name])
        for th in (0.0, 0.05, 0.10, 0.20):
            rows.append(roi_table(s[ev > th], None, f"{name} EV>{th:.2f}", comm))
    rows.append(roi_table(s, None, "every runner", comm))
    print(f"\ncommission {comm:.0%}")
    print(pd.DataFrame(rows).set_index("rule").round(4).to_string())

comm = COMMISSION[0]
ev = s.q_win * (s.bfsp_place - 1) * (1 - comm) - (1 - s.q_win)
v = s[ev > 0.05].copy()
v["fband"] = pd.cut(v.runners, [1, 7, 11, 15, 40])
v["pband"] = pd.cut(v.bfsp, [1, 2, 3, 5, 8, 13, 21, 1000])
for col in ("year", "fband", "pband"):
    t = pd.DataFrame([roi_table(x, None, str(k), comm) for k, x in v.groupby(col, observed=True)]).set_index("rule")
    print(f"\nq_win EV>0.05 at 5% commission, by {col}")
    print(t.round(4).to_string())

# --- 4. the favourite to place -------------------------------------------------------------------
s["rank"] = s.groupby("raceid").bfsp.rank(method="first")
f = s[s["rank"] == 1].copy()
f["fband"] = pd.cut(f.runners, [1, 7, 11, 15, 40])
f["pband"] = pd.cut(f.bfsp, [1, 1.5, 2, 2.5, 3, 4, 6, 1000])
print("\n== 4. the win favourite backed to place at place BSP, 5% commission (2023 on)")
print(pd.DataFrame([roi_table(f, None, "all favourites", comm)]).set_index("rule").round(4).to_string())
for col in ("pband", "fband"):
    t = pd.DataFrame([roi_table(x, None, str(k), comm) for k, x in f.groupby(col, observed=True)]).set_index("rule")
    print(f"\nby {col}")
    print(t.round(4).to_string())
rows = [roi_table(x, None, f"{fb} x {pb}", comm) for (fb, pb), x in f.groupby(["fband", "pband"], observed=True)]
print("\nby field band x price band")
print(pd.DataFrame(rows).set_index("rule").round(4).to_string())
print(f"\ndone in {time.time() - t0:.0f}s")
