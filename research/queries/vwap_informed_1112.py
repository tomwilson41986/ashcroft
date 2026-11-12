"""Is the volume-weighted price nearer the truth than the Betfair SP, and what does an unsettled book say?
(read-only; ledger `vwap-informed-1112`; from the owner's reading of 12 Nov 2026; the price files' table)

Two claims to test on our own records.
1. "VWAP is more accurate than BFSP." Betfair's price files hold, for every runner, the BSP and the pre-play
   volume-weighted average price (PPWAP), and the morning's (MORNINGWAP). Each is turned into a share of its race's
   book. Scored on winners, win markets, 1 Jan 2022 - 31 Mar 2026 (no holdout): the logarithmic public R2 of each,
   and a two-price model of the winner (a conditional logit by race on ln BSP share and ln PPWAP share) fitted on
   2022-23 and scored on 2024-26 against the BSP alone fitted the same way.
   Registered pass: the two-price model's log loss on 2024-26 is below the BSP-alone model's, the 90% interval of
   the difference (days bootstrapped) below zero. A pass means the pre-play WAP carries something the SP does not.
2. An unsettled book. For each race, how concentrated the book is: 1 - entropy(shares) / ln(runners) (0: every
   runner the same price; 1: one certainty), at the morning WAP and at the BSP, and how far the book moved between
   them (the mean absolute change of ln share). Descriptive: the BSP's R2 by thirds of each, and the size of the
   morning-to-SP move by thirds of the morning's concentration.
"""
import sqlite3

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
d = pd.read_sql_query("SELECT race_date, event_id, selection_id, win_lose, bsp, ppwap, morningwap FROM betfair_prices "
                      "WHERE market_type = 'win' AND race_date >= '2022-01-01' AND race_date <= '2026-03-31'", con)
for c in ("win_lose", "bsp", "ppwap", "morningwap"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["day"] = d.race_date.str[:10]
g = d.groupby("event_id")
d["n"] = g.bsp.transform("size")
for c in ("bsp", "ppwap", "morningwap"):
    ok = d[c] > 1
    inv = np.where(ok, 1 / d[c].where(ok, 1), np.nan)
    d["p_" + c] = inv / pd.Series(inv).groupby(d.event_id.to_numpy()).transform("sum").to_numpy()
    d["full_" + c] = pd.Series(ok.to_numpy()).groupby(d.event_id.to_numpy()).transform("all").to_numpy()
d["wins"] = g.win_lose.transform("sum")
A = d[(d.wins == 1) & (d.n >= 2) & d.full_bsp & d.full_ppwap].copy()
print(f"races with one winner and every runner holding a BSP and a pre-play WAP: {A.event_id.nunique():,} "
      f"({A.day.min()} .. {A.day.max()}); runners {len(A):,}")
W = A[A.win_lose == 1]
r2 = lambda p, n: 1 - np.log(p).sum() / np.log(1 / n).sum()  # noqa: E731
print(f"public R2 on the winners: BSP {r2(W.p_bsp, W.n):.4f}; pre-play WAP {r2(W.p_ppwap, W.n):.4f}; "
      f"mean ln share of the winner: BSP {np.log(W.p_bsp).mean():.4f}, pre-play WAP {np.log(W.p_ppwap).mean():.4f}")
M = A[A.full_morningwap]
Wm = M[M.win_lose == 1]
print(f"the same on the {M.event_id.nunique():,} races that also hold a morning WAP for every runner: BSP {r2(Wm.p_bsp, Wm.n):.4f}; "
      f"pre-play WAP {r2(Wm.p_ppwap, Wm.n):.4f}; morning WAP {r2(Wm.p_morningwap, Wm.n):.4f}")


def fit(X, y, race):
    """Conditional logit by race, Newton's method. X: runners x k."""
    b = np.zeros(X.shape[1])
    b[0] = 1.0
    for _ in range(25):
        s = X @ b
        s = s - pd.Series(s).groupby(race).transform("max").to_numpy()
        e = np.exp(s)
        P = e / pd.Series(e).groupby(race).transform("sum").to_numpy()
        grad = X.T @ (y - P)
        mean = pd.DataFrame(X * P[:, None]).groupby(race).transform("sum").to_numpy()
        H = (X * P[:, None]).T @ X - (mean * P[:, None]).T @ mean
        b = b + np.linalg.solve(H + 1e-9 * np.eye(len(b)), grad)
    return b


def loss(X, y, race, b):
    s = X @ b
    s = s - pd.Series(s).groupby(race).transform("max").to_numpy()
    e = np.exp(s)
    P = e / pd.Series(e).groupby(race).transform("sum").to_numpy()
    return -np.log(P[y == 1])


A = A.sort_values(["event_id"]).reset_index(drop=True)
A["lb"], A["lw"] = np.log(A.p_bsp), np.log(A.p_ppwap)
tr, te = A[A.day < "2024-01-01"], A[A.day >= "2024-01-01"]
b1 = fit(tr[["lb"]].to_numpy(), tr.win_lose.to_numpy(), tr.event_id.to_numpy())
b2 = fit(tr[["lb", "lw"]].to_numpy(), tr.win_lose.to_numpy(), tr.event_id.to_numpy())
l1 = loss(te[["lb"]].to_numpy(), te.win_lose.to_numpy(), te.event_id.to_numpy(), b1)
l2 = loss(te[["lb", "lw"]].to_numpy(), te.win_lose.to_numpy(), te.event_id.to_numpy(), b2)
days = te[te.win_lose == 1].day.to_numpy()
diff = pd.Series(l2 - l1).groupby(days).agg(["sum", "size"])
rng = np.random.default_rng(1112)
bs = [diff.iloc[rng.integers(0, len(diff), len(diff))].pipe(lambda z: z["sum"].sum() / z["size"].sum()) for _ in range(2000)]
lo, hi = np.percentile(bs, [5, 95])
print(f"\n1. fitted on 2022-23 ({tr.event_id.nunique():,} races): BSP alone exponent {b1[0]:.3f}; two prices: BSP {b2[0]:.3f}, "
      f"pre-play WAP {b2[1]:.3f}")
print(f"   scored on 2024-26 ({te.event_id.nunique():,} races): log loss a race BSP alone {l1.mean():.5f}, two prices {l2.mean():.5f}; "
      f"difference {np.mean(l2 - l1):+.5f} (90% {lo:+.5f} to {hi:+.5f})")
print("   verdict:", "PASS" if hi < 0 else "does not pass")

print("\n2. an unsettled book (races with a morning WAP for every runner)")


def conc(p, race, n):
    h = -(p * np.log(p)).groupby(race).transform("sum")
    return 1 - h / np.log(n)


M = M.copy()
M["c_m"], M["c_b"] = conc(M.p_morningwap, M.event_id, M.n), conc(M.p_bsp, M.event_id, M.n)
M["mv"] = np.abs(np.log(M.p_bsp / M.p_morningwap))
R = M.groupby("event_id").agg(c_m=("c_m", "first"), c_b=("c_b", "first"), move=("mv", "mean"), n=("n", "first"))
R = R[R.n >= 5]
Wr = M[(M.win_lose == 1)].merge(R, left_on="event_id", right_index=True)
for col, name in (("c_m", "concentration of the morning book"), ("c_b", "concentration of the SP book"),
                  ("move", "morning-to-SP move of the book")):
    Wr["t"] = pd.qcut(Wr[col], 3, labels=["lowest third", "middle", "highest third"])
    t = Wr.groupby("t", observed=True).apply(lambda x: pd.Series({"races": len(x), "BSP R2": r2(x.p_bsp, x.n_x), "morning R2": r2(x.p_morningwap, x.n_x),
                                                                  "mean move": x.move.mean(), "runners": x.n_x.mean()}))
    print(f"\nby {name}")
    print(t.round(4).to_string())
