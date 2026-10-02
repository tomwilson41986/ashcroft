"""Market edges, 2: where the Betfair SP itself is mispriced, win and place, UK and Ireland, 2010 to 2026 Q1 (read-only).

The owner's research programme (2 Oct), first step: evaluate each market for edges before any model. The SP is the
price the live trader lays at, so a bias in it is an edge on its own (a back or a lay at the SP, no model) and a
cost or a gift to every other strategy. From race_results (HRB, with each runner's win and place BSP, 2010 on):

A. the place pocket as gated (24 Sep; passed at 2% on 2 Oct): back to place at the place BSP where the win BSP is
   5.0 or shorter and the win-implied place chance gives an expected return above 5% after commission. The rule was
   chosen on 2023-26Q1 and gated on 2021-22; the years before 2021 have never been looked at, so they are a clean
   test of it, year by year.
B. backing and laying at the SP, win and place, by SP band x race group x field: cells chosen on 2010-17, read on
   2018-26Q1, 2% commission on each runner's net.
C. the place market against the win market in general: by win-SP band x the gap between the win-implied place
   chance and the place SP, backs and lays at the place SP, chosen on 2010-17, read on 2018-26Q1.

The holdout (from 1 Apr 2026) is not read. Standard errors by race.
"""

from __future__ import annotations

import sqlite3
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, ".")
from model.bet_analysis import broad_race_type  # noqa: E402
from model.ordering import position_probabilities, simulate_finishing_orders  # noqa: E402

t0 = time.time()
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 500)
COMM, SPLIT = 0.02, "2018-01-01"
BANDS = ((2, 7), (8, 11), (12, 15), (16, 40))

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT race_date, race_time, track, horse_name, placing_numerical, bfsp, bfsp_place, bf_plcs_paid, plcs_paid,
           race_type
    FROM race_results WHERE race_date >= '2010-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
for c in ("bfsp", "bfsp_place", "bf_plcs_paid", "plcs_paid", "placing_numerical"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["year"] = d.race_date.str[:4]
print(f"{len(d):,} runners, {d.raceid.nunique():,} races ({time.time() - t0:.0f}s)")
print("share with a win BSP / place BSP by year:")
print(d.groupby("year").agg(runners=("bfsp", "size"), win_bsp=("bfsp", lambda s: (s > 1).mean()),
                            place_bsp=("bfsp_place", lambda s: (s > 1).mean())).round(3).T.to_string())

d = d[d.placing_numerical.notna().groupby(d.raceid).transform("any")].copy()
d["won"] = (d.placing_numerical == 1).astype(float)
d["runners"] = d.groupby("raceid").horse_name.transform("size")
d["paid"] = d.bf_plcs_paid.where(d.bf_plcs_paid > 0, d.plcs_paid)
d["paid"] = d.groupby("raceid").paid.transform("max")
d["placed"] = (d.placing_numerical <= d.paid).astype(float)
d["group"] = d.race_type.map(broad_race_type)
d["period"] = np.where(d.race_date < SPLIT, "choose", "test")
d["field_band"] = pd.cut(d.runners, [0, 7, 11, 15, 99], labels=["2-7", "8-11", "12-15", "16+"])
d["sp_band"] = pd.cut(d.bfsp, [1, 2, 3, 5, 8, 13, 21, 50, 1001], right=False,
                      labels=["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-50", "50+"])
d["psp_band"] = pd.cut(d.bfsp_place, [1, 1.25, 1.5, 2, 3, 5, 10, 1001], right=False,
                       labels=["1-1.25", "1.25-1.5", "1.5-2", "2-3", "3-5", "5-10", "10+"])


def net(x):
    return np.where(x > 0, x * (1 - COMM), x)


def summary(pnl: pd.Series, races: pd.Series) -> dict:
    n = len(pnl)
    if n < 30:
        return {"n": n}
    by_race = pnl.groupby(races).sum()
    m = pnl.mean()
    se = by_race.std(ddof=1) * np.sqrt(len(by_race)) / n
    return {"n": n, "races": len(by_race), "roi": m, "lo90": m - 1.645 * se, "hi90": m + 1.645 * se, "t": m / se}


# --- ordering exponents, fitted on 2021-22 as the gate was --------------------------------------------------
def padded(rs):
    n = max(len(x) for x in rs)
    P = np.zeros((len(rs), n)); F = np.full((len(rs), 3), -1)
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
    S = np.where(P > 0, P ** gm, 0.0); S /= S.sum(1, keepdims=True)
    T = np.where(P > 0, P ** dl, 0.0); T /= T.sum(1, keepdims=True)
    ok = (F >= 0).all(1); i = np.flatnonzero(ok)
    a, b, c = F[ok, 0], F[ok, 1], F[ok, 2]
    l2 = np.log(np.clip(S[i, b] / (1 - S[i, a]), 1e-12, None))
    l3 = np.log(np.clip(T[i, c] / np.clip(1 - T[i, a] - T[i, b], 1e-12, None), 1e-12, None))
    return -(l2 + l3).sum()


def band_of(n):
    return next((b for b in BANDS if b[0] <= n <= b[1]), BANDS[-1])


def top3(p, gm, dl):
    s = p ** gm; s /= s.sum(); t = p ** dl; t /= t.sum()
    A = p[:, None] * s[None, :] / (1 - s[:, None]); np.fill_diagonal(A, 0.0)
    B = 1.0 / np.clip(1 - t[:, None] - t[None, :], 1e-12, None); np.fill_diagonal(B, 0.0)
    M = A * B
    return p, A.sum(0), t * (M.sum() - M.sum(1) - M.sum(0))


def place_chance(frame, fit, rng):
    out = np.full(len(frame), np.nan)
    pos = {ix: i for i, ix in enumerate(frame.index)}
    for _, x in frame.groupby("raceid", sort=False):
        p = x.pi.to_numpy(); k = int(x.paid.iat[0])
        gm, dl = fit[band_of(len(p))]
        if k <= 3:
            p1, p2, p3 = top3(p, gm, dl)
            q = p1 + (p2 if k >= 2 else 0) + (p3 if k >= 3 else 0)
        else:
            o = simulate_finishing_orders(p, n_sims=4000, gamma=gm, delta=dl, rng=rng)
            q = position_probabilities(o, len(p), n_positions=k).sum(axis=1)
        out[[pos[i] for i in x.index]] = np.clip(q, 1e-4, 1 - 1e-4)
    return out


pl = d[d.paid.gt(0) & d.paid.lt(d.runners)]
pl = pl[(pl.bfsp > 1).groupby(pl.raceid).transform("all") & (pl.bfsp_place > 1).groupby(pl.raceid).transform("all")]
pl = pl.copy()
pl["pi"] = (1 / pl.bfsp) / (1 / pl.bfsp).groupby(pl.raceid).transform("sum")
early = [x for _, x in pl[(pl.race_date >= "2021-01-01") & (pl.race_date < "2023-01-01")].groupby("raceid", sort=False)]
fit = {}
for lo, hi in BANDS:
    P, F = padded([x for x in early if lo <= len(x) <= hi])
    fit[(lo, hi)] = tuple(minimize(nll, np.array([0.8, 0.65]), args=(P, F), method="Nelder-Mead",
                                   options={"xatol": 1e-4, "fatol": 1e-3}).x)
print("\nordering exponents (2021-22, as gated):", {f"{k[0]}-{k[1]}": tuple(round(v, 3) for v in g) for k, g in fit.items()})
pl["q"] = place_chance(pl, fit, np.random.default_rng(0))
print(f"place chances for {pl.raceid.nunique():,} races with every runner's win and place BSP ({time.time() - t0:.0f}s)")

# --- A. the place pocket on the years never looked at -------------------------------------------------------
print("\n== A. the place pocket as gated, every year (2010-20 never looked at; 2021-22 the gate; 2023-26Q1 chosen)")
pl["ev_back"] = pl.q * (pl.bfsp_place - 1) * (1 - COMM) - (1 - pl.q)
rule = (pl.bfsp <= 5.0) & (pl.ev_back > 0.05)
pnl_place_back = pd.Series(net(np.where(pl.placed == 1, pl.bfsp_place - 1, -1.0)), index=pl.index)
rows = [{"year": y, **summary(pnl_place_back[rule & (pl.year == y)], pl.raceid[rule & (pl.year == y)])}
        for y in sorted(pl.year.unique())]
m_old = rule & (pl.race_date < "2021-01-01")
rows.append({"year": "2010-20 (clean)", **summary(pnl_place_back[m_old], pl.raceid[m_old])})
print(pd.DataFrame(rows).set_index("year").round(4).to_string())

# --- B. backs and lays at the SP by segment -----------------------------------------------------------------
print("\n== B. backing and laying at the SP, chosen on 2010-17 (n >= 3,000, t >= 2.5), read on 2018-26Q1")
w = d[d.bfsp > 1].copy()
w["back_win"] = net(np.where(w.won == 1, w.bfsp - 1, -1.0))
w["lay_win"] = net(np.where(w.won == 1, -(w.bfsp - 1), 1.0)) / (w.bfsp - 1)        # per GBP1 of liability
p2 = d[(d.bfsp_place > 1) & d.paid.gt(0)].copy()
p2["back_place"] = net(np.where(p2.placed == 1, p2.bfsp_place - 1, -1.0))
p2["lay_place"] = net(np.where(p2.placed == 1, -(p2.bfsp_place - 1), 1.0)) / (p2.bfsp_place - 1)
out = []
for frame, band_col, strats in ((w, "sp_band", ("back_win", "lay_win")), (p2, "psp_band", ("back_place", "lay_place"))):
    for keys, g in frame.groupby([band_col, "group", "field_band"], observed=True):
        for s in strats:
            ch = g[g.period == "choose"]; te = g[g.period == "test"]
            a = summary(pd.Series(ch[s].to_numpy()), pd.Series(ch.raceid.to_numpy()))
            if a.get("n", 0) < 3000 or not (a.get("t", 0) >= 2.5):
                continue
            b = summary(pd.Series(te[s].to_numpy()), pd.Series(te.raceid.to_numpy()))
            out.append({"strategy": s, "band": keys[0], "group": keys[1], "field": keys[2],
                        **{f"choose_{k}": v for k, v in a.items() if k in ("n", "roi", "t")},
                        **{f"test_{k}": v for k, v in b.items() if k in ("n", "roi", "lo90", "t")}})
ob = pd.DataFrame(out)
if ob.empty:
    print("no cell passed on 2010-17")
else:
    print(ob.sort_values("choose_t", ascending=False).round(4).to_string(index=False))
print("\nfor scale, every runner (2018-26Q1): " + ", ".join(
    f"{s} {frame.loc[frame.period == 'test', s].mean():+.4f}" for frame, s in
    ((w, "back_win"), (w, "lay_win"), (p2, "back_place"), (p2, "lay_place"))))
print("by SP band (2018-26Q1), every race group:")
print(w[w.period == "test"].groupby("sp_band", observed=True)[["back_win", "lay_win"]].mean().round(4).T.to_string())
print(p2[p2.period == "test"].groupby("psp_band", observed=True)[["back_place", "lay_place"]].mean().round(4).T.to_string())

# --- C. the place SP against the win-implied place chance ---------------------------------------------------
print("\n== C. the place SP against the win-implied place chance: back where EV_back > x, lay where EV_lay > x,")
print("      by win-SP band, chosen on 2010-17 (n >= 2,000, t >= 2.5), read on 2018-26Q1")
pl["ev_lay"] = (1 - pl.q) * (1 - COMM) - pl.q * (pl.bfsp_place - 1)                 # per GBP1 laid
pnl_place_lay = pd.Series(net(np.where(pl.placed == 1, -(pl.bfsp_place - 1), 1.0)), index=pl.index)
out = []
for band, g in pl.groupby("sp_band", observed=True):
    for side, ev_col, pnl in (("back", "ev_back", pnl_place_back), ("lay", "ev_lay", pnl_place_lay)):
        for x in (0.0, 0.02, 0.05, 0.10):
            sel = g[g[ev_col] > x]
            ch = sel[sel.period == "choose"]; te = sel[sel.period == "test"]
            a = summary(pnl.loc[ch.index], ch.raceid)
            b = summary(pnl.loc[te.index], te.raceid)
            out.append({"win_sp": band, "side": side, "ev_over": x, "choose_n": a.get("n"), "choose_roi": a.get("roi"),
                        "choose_t": a.get("t"), "test_n": b.get("n"), "test_roi": b.get("roi"),
                        "test_lo90": b.get("lo90"), "test_t": b.get("t")})
oc = pd.DataFrame(out)
print(oc.round(4).to_string(index=False))
passed = oc[(oc.choose_n.fillna(0) >= 2000) & (oc.choose_t >= 2.5)]
print(f"\nchosen on 2010-17: {len(passed)} rules; positive on 2018-26Q1: {int((passed.test_roi > 0).sum())}; with the "
      f"test 90% interval above zero: {int((passed.test_lo90 > 0).sum())}")
print(f"\ndone in {time.time() - t0:.0f}s")
