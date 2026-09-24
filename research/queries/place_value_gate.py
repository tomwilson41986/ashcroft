"""Gate for the one pocket research/queries/done/place_value.py found.

That query (research-query run 25) found the Betfair place market prices placing BETTER
than the win market's implied place chances, with a fair book at BSP and no value rule
beating commission overall. The one pocket was chosen after looking at 2023-26Q1:

    back to place at place BSP where the win BSP is 5.0 or shorter and the place chance
    the win BSPs imply gives an expected return above 0.05 after 5% commission

(+1.6% on 10,413 bets, intervals touching zero). Development data only; the holdout is
not read.

1. The same rule, unchanged, on 2021 and on 2022 -- years not used to choose it. (The
   ordering exponents are fitted on 2021-22, which fits finishing orders, not returns.)
2. At pre-off prices, where Betfair's files hold both markets inside the development
   window (Jan-Mar 2026): the place chance from the win PPWAPs, value at the place PPWAP,
   returns at the place PPWAP and at the place BSP.

It passes only if 2021 and 2022 are both positive and the pooled 90% interval is above zero.
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
COMM, EV_MIN, WIN_MAX = 0.05, 0.05, 5.0
BANDS = ((2, 7), (8, 11), (12, 15), (16, 40))

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("""
    SELECT id AS rr_id, race_date, race_time, track, number_of_runners, horse_name, placing_numerical,
           bfsp, bfsp_place, bf_plcs_paid, plcs_paid
    FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
d["raceid"] = d.race_date + "|" + d.track + "|" + d.race_time.astype(str)
d = d.drop_duplicates(["raceid", "horse_name"]).reset_index(drop=True)
for c in ("bfsp", "bfsp_place", "bf_plcs_paid", "plcs_paid", "placing_numerical"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["runners"] = d.groupby("raceid").horse_name.transform("size")
d["paid"] = d.bf_plcs_paid.where(d.bf_plcs_paid > 0, d.plcs_paid)
d["paid"] = d.groupby("raceid").paid.transform("max")
d = d[d.paid.gt(0) & d.paid.lt(d.runners) & d.placing_numerical.notna().groupby(d.raceid).transform("any")].copy()
d["placed"] = (d.placing_numerical <= d.paid).astype(float)
d["year"] = d.race_date.str[:4]
print(f"{len(d):,} rows, {d.raceid.nunique():,} races with places paid ({time.time() - t0:.0f}s)")


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
    """Win-implied place chance for every row of `frame` (column pi, paid), race by race."""
    out = pd.Series(np.nan, index=frame.index)
    for _, x in frame.groupby("raceid", sort=False):
        p = x.pi.to_numpy(); k = int(x.paid.iat[0])
        gm, dl = fit[band_of(len(p))]
        if k <= 3:
            p1, p2, p3 = top3(p, gm, dl)
            q = p1 + (p2 if k >= 2 else 0) + (p3 if k >= 3 else 0)
        else:
            o = simulate_finishing_orders(p, n_sims=4000, gamma=gm, delta=dl, rng=rng)
            q = position_probabilities(o, len(p), n_positions=k).sum(axis=1)
        out.loc[x.index] = np.clip(q, 1e-4, 1 - 1e-4)
    return out


def summary(sel, price_col, label):
    ret = np.where(sel.placed == 1, (sel[price_col] - 1) * (1 - COMM), -1.0)
    s = pd.Series(ret, index=sel.index)
    by_race = s.groupby(sel.raceid).sum()
    n = len(s)
    if n < 30:
        return {"rule": label, "bets": n}
    m = s.mean(); se = by_race.std() * np.sqrt(len(by_race)) / n
    return {"rule": label, "bets": n, "races": len(by_race), "ROI": m, "lo90": m - 1.645 * se, "hi90": m + 1.645 * se,
            "strike": sel.placed.mean(), "avg_price": sel[price_col].mean()}


# --- 1. the rule on 2021 and 2022 ---------------------------------------------------------------
b = d[(d.bfsp > 1).groupby(d.raceid).transform("all") & (d.bfsp_place > 1).groupby(d.raceid).transform("all")].copy()
b["pi"] = (1 / b.bfsp) / (1 / b.bfsp).groupby(b.raceid).transform("sum")
early = [x for _, x in b[b.race_date < "2023-01-01"].groupby("raceid", sort=False)]
fit = {}
for lo, hi in BANDS:
    P, F = padded([x for x in early if lo <= len(x) <= hi])
    fit[(lo, hi)] = tuple(minimize(nll, np.array([0.8, 0.65]), args=(P, F), method="Nelder-Mead",
                                   options={"xatol": 1e-4, "fatol": 1e-3}).x)
print("ordering exponents (2021-22):", {f"{k[0]}-{k[1]}": tuple(round(v, 3) for v in g) for k, g in fit.items()})
rng = np.random.default_rng(0)
b["q"] = place_chance(b, fit, rng)
b["ev"] = b.q * (b.bfsp_place - 1) * (1 - COMM) - (1 - b.q)
rule = (b.bfsp <= WIN_MAX) & (b.ev > EV_MIN)
print(f"\n== 1. the rule (win BSP <= {WIN_MAX}, EV > {EV_MIN} at {COMM:.0%} commission), by year, at place BSP")
rows = [summary(b[rule & (b.year == y)], "bfsp_place", y) for y in sorted(b.year.unique())]
rows.append(summary(b[rule & b.year.isin(["2021", "2022"])], "bfsp_place", "2021-22 (gate)"))
rows.append(summary(b[rule & (b.race_date >= "2023-01-01")], "bfsp_place", "2023-26Q1 (where it was chosen)"))
print(pd.DataFrame(rows).set_index("rule").round(4).to_string())
print("\nthe same years, every EV > 0.05 bet at any price, and every short-priced runner, for scale")
for label, m in (("EV>0.05 any price", b.ev > EV_MIN), (f"win BSP <= {WIN_MAX}, any EV", b.bfsp <= WIN_MAX)):
    rows = [summary(b[m & b.year.isin(["2021", "2022"])], "bfsp_place", f"{label}, 2021-22"),
            summary(b[m & (b.race_date >= "2023-01-01")], "bfsp_place", f"{label}, 2023-26Q1")]
    print(pd.DataFrame(rows).set_index("rule").round(4).to_string())

# --- 2. pre-off prices, Jan-Mar 2026 ---------------------------------------------------------
tabs = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if "betfair_prices" not in tabs:
    print("\n== 2. no betfair_prices table")
else:
    bf = pd.read_sql_query("""
        SELECT race_results_id AS rr_id, market_type, ppwap, bsp, pp_vol FROM betfair_prices
        WHERE race_results_id IS NOT NULL AND race_date >= '2026-01-01' AND race_date < '2026-04-01'""", conn)
    print(f"\n== 2. Betfair price files, Jan-Mar 2026: {len(bf):,} rows; by market:",
          bf.market_type.value_counts().to_dict())
    w = bf[bf.market_type == "win"].drop_duplicates("rr_id").set_index("rr_id")
    pl = bf[bf.market_type == "place"].drop_duplicates("rr_id").set_index("rr_id")
    q = d[(d.race_date >= "2026-01-01")].copy()
    q["win_pp"] = q.rr_id.map(w.ppwap)
    q["plc_pp"] = q.rr_id.map(pl.ppwap)
    ok = (q.win_pp > 1).groupby(q.raceid).transform("all") & (q.plc_pp > 1).groupby(q.raceid).transform("all")
    q = q[ok].copy()
    print(f"races with both pre-off prices for every runner: {q.raceid.nunique():,} ({len(q):,} runners)")
    if len(q):
        q["pi"] = (1 / q.win_pp) / (1 / q.win_pp).groupby(q.raceid).transform("sum")
        q["q"] = place_chance(q, fit, rng)
        q["ev"] = q.q * (q.plc_pp - 1) * (1 - COMM) - (1 - q.q)
        m = (q.win_pp <= WIN_MAX) & (q.ev > EV_MIN)
        rows = [summary(q[m], "plc_pp", "rule at pre-off prices, paid at place PPWAP"),
                summary(q[m & (q.bfsp_place > 1)], "bfsp_place", "rule at pre-off prices, paid at place BSP"),
                summary(q[q.win_pp <= WIN_MAX], "plc_pp", f"every runner with win PPWAP <= {WIN_MAX}, at place PPWAP"),
                summary(q, "plc_pp", "every runner, at place PPWAP")]
        print(pd.DataFrame(rows).set_index("rule").round(4).to_string())
print(f"\ndone in {time.time() - t0:.0f}s")
