"""The place pocket again, at the commission the account pays: 2%, not the 5% it was gated at.

research/queries/done/place_value_gate.py (24 Sep, ledger place-pocket-gate) gated one rule chosen after looking
at 2023-26Q1:

    back to place at the place BSP where the win BSP is 5.0 or shorter and the place chance the win BSPs imply
    gives an expected return above 0.05 after commission

At 5% it FAILED (2021 -0.43%, 2022 +2.64%, pooled 2021-22 +1.06%, 90% interval -0.70 to +2.82) and the ledger
said: "the commission rate the account pays decides whether it is worth a forward test, and the place-market price
files would be needed to test it at prices available before the off". On 1 Oct the account was found to pay 2%
(the day's settled result at 2% met its balance to the penny). The rate is a fact about the account, not a choice,
so the same gate is read again at 2%: the same rule (expected return after commission at the rate charged), the
same ordering exponents (fitted on 2021-22 finishing orders), the same criterion set before the first run (2021
and 2022 both positive and the pooled 2021-22 90% interval above zero). 5% is shown beside it.

Then, at prices a bettor could take before the off, where Betfair's place files are loaded for the development
window: the place chance from the win market's morning (MORNINGWAP) and pre-play (PPWAP) prices, value at the
place market's own, returns at those prices and at the place BSP, and what traded on the runners chosen (the
liquidity a stake would need). Development data only (before 1 Apr 2026); the holdout is not read. Read-only.
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
EV_MIN, WIN_MAX = 0.05, 5.0
RATES = (0.05, 0.02)                 # the rule as gated on 24 Sep, and at the account's rate (1 Oct)
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


def summary(sel, price_col, label, comm):
    ret = np.where(sel.placed == 1, (sel[price_col] - 1) * (1 - comm), -1.0)
    s = pd.Series(ret, index=sel.index)
    by_race = s.groupby(sel.raceid).sum()
    n = len(s)
    if n < 30:
        return {"rule": label, "bets": n}
    m = s.mean(); se = by_race.std() * np.sqrt(len(by_race)) / n
    return {"rule": label, "bets": n, "races": len(by_race), "ROI": m, "lo90": m - 1.645 * se, "hi90": m + 1.645 * se,
            "strike": sel.placed.mean(), "avg_price": sel[price_col].mean()}



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

# --- 1. the gate, at each rate --------------------------------------------------------------------
for comm in RATES:
    b["ev"] = b.q * (b.bfsp_place - 1) * (1 - comm) - (1 - b.q)
    rule = (b.bfsp <= WIN_MAX) & (b.ev > EV_MIN)
    print(f"\n== 1. the rule at {comm:.0%} commission (win BSP <= {WIN_MAX}, EV > {EV_MIN}), by year, at place BSP")
    rows = [summary(b[rule & (b.year == y)], "bfsp_place", y, comm) for y in sorted(b.year.unique())]
    gate_row = summary(b[rule & b.year.isin(["2021", "2022"])], "bfsp_place", "2021-22 (gate)", comm)
    rows.append(gate_row)
    rows.append(summary(b[rule & (b.race_date >= "2023-01-01")], "bfsp_place", "2023-26Q1 (where it was chosen)", comm))
    print(pd.DataFrame(rows).set_index("rule").round(4).to_string())
    y21 = summary(b[rule & (b.year == "2021")], "bfsp_place", "2021", comm)
    y22 = summary(b[rule & (b.year == "2022")], "bfsp_place", "2022", comm)
    passed = y21.get("ROI", -1) > 0 and y22.get("ROI", -1) > 0 and gate_row.get("lo90", -1) > 0
    print(f"gate at {comm:.0%}: {'PASSES' if passed else 'FAILS'} (2021 {y21.get('ROI', float('nan')):+.4f}, "
          f"2022 {y22.get('ROI', float('nan')):+.4f}, pooled 90% low {gate_row.get('lo90', float('nan')):+.4f})")
    base = summary(b[(b.bfsp <= WIN_MAX) & b.year.isin(["2021", "2022"])], "bfsp_place", "every runner, win BSP <= 5", comm)
    print(f"  for scale, every runner at win BSP <= {WIN_MAX} in 2021-22: {base.get('ROI', float('nan')):+.4f}")

# --- 2. at prices before the off -------------------------------------------------------------------
bf = pd.read_sql_query("""
    SELECT race_results_id AS rr_id, market_type, morningwap, ppwap, bsp, morning_vol, pp_vol FROM betfair_prices
    WHERE race_results_id IS NOT NULL AND race_date < '2026-04-01'""", conn)
print(f"\n== 2. Betfair's price files in the development window: {len(bf):,} rows;",
      bf.market_type.value_counts().to_dict())
w = bf[bf.market_type == "win"].drop_duplicates("rr_id").set_index("rr_id")
pl = bf[bf.market_type == "place"].drop_duplicates("rr_id").set_index("rr_id")
for when, col in (("morning", "morningwap"), ("pre-play", "ppwap")):
    q = d.copy()
    q["win_px"], q["plc_px"] = q.rr_id.map(w[col]), q.rr_id.map(pl[col])
    q["plc_vol"] = q.rr_id.map(pl["morning_vol" if when == "morning" else "pp_vol"])
    ok = (q.win_px > 1).groupby(q.raceid).transform("all") & (q.plc_px > 1).groupby(q.raceid).transform("all")
    q = q[ok].copy()
    if q.empty:
        print(f"{when}: no race with both markets' {col} for every runner yet (the place files load nightly)")
        continue
    print(f"\n{when} ({col}): {q.raceid.nunique():,} races, {len(q):,} runners, {q.race_date.min()} to {q.race_date.max()}")
    q["pi"] = (1 / q.win_px) / (1 / q.win_px).groupby(q.raceid).transform("sum")
    q["q"] = place_chance(q, fit, rng)
    for comm in RATES:
        q["ev"] = q.q * (q.plc_px - 1) * (1 - comm) - (1 - q.q)
        m = (q.win_px <= WIN_MAX) & (q.ev > EV_MIN)
        rows = [summary(q[m], "plc_px", f"rule, paid at the place {col}", comm),
                summary(q[m & (q.bfsp_place > 1)], "bfsp_place", "rule, paid at the place BSP", comm),
                summary(q[q.win_px <= WIN_MAX], "plc_px", f"every runner with win {col} <= {WIN_MAX}", comm),
                summary(q, "plc_px", f"every runner, at the place {col}", comm)]
        print(f"  at {comm:.0%}:")
        print(pd.DataFrame(rows).set_index("rule").round(4).to_string())
        sel = q[m]
        if len(sel):
            v = sel.plc_vol.dropna()
            print(f"  what traded on the runners chosen in the place market ({when}): median GBP{v.median():,.0f}, "
                  f"10th percentile GBP{v.quantile(0.1):,.0f}, 90th GBP{v.quantile(0.9):,.0f} ({len(v):,} runners)")
print(f"\ndone in {time.time() - t0:.0f}s")
