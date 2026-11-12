"""Where does a withdrawn horse's share go? (read-only; reports/discovery_1111.md 4.5; ledger `substitution-1112`)

Betfair's reduction factor and a pro-rata renormalisation give every survivor the same proportional lift. If money
leaving a withdrawn horse goes to what resembles it, the survivors that resemble it close shorter than pro rata.

Every removal with a cut (factor >= 2.5%) from 2 Oct, read off the recorded books (the trader's and the recorder's):
the last read of the market before the removal with the horse still active and priced, shares from the middle of
back and lay (the back alone where there is no lay), over the active runners. Kept when the horses with a BSP are
exactly that read's runners less the removed one. For each survivor i of removal j:

    e_i = ln(share_i at the BSP / share_i before) - ln(1 / (1 - p_j))        zero under pro rata

regressed on three kernels, each centred within the removal: the same trainer as j, next to j in the price order,
drawn within two stalls of j (races with stalls). The projected run style is not held on the live record and is
left out. One coefficient each; standard errors by bootstrap over markets. Registered pass: one kernel with the
same sign and |t| > 2.5 in both halves of the days. 1 trial (three kernels read together). No block until then.
"""
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
path = os.path.join(os.getcwd(), "research", "queries", "done", "closing_model_by_hour_1007.py")
src = open(path, encoding="utf-8").read()
G = {"__name__": "substitution", "__file__": path}
exec(compile(src[:src.index("\nrows = []\n")], path, "exec"), G)
books, catalogue, settled, KEY, FIRST, LAST = (G[k] for k in ("books", "catalogue", "settled", "KEY", "FIRST", "LAST"))
KERNELS = ["trainer", "adjacent", "stall"]
rng = np.random.default_rng(1112)

rows, seen, kept, day = [], 0, 0, FIRST
while day <= LAST:
    parts = [books(f"betfair_live/{day}/{n}.csv.gz") for n in ("books_trader", "books")]
    parts = [p for p in parts if not p.empty]
    cat = catalogue(day)
    bsp, rem = settled(day)
    day += timedelta(days=1)
    if not parts or cat.empty or bsp.empty or rem.empty:
        continue
    b = pd.concat(parts, ignore_index=True)
    b = b[b.runner_status.fillna("ACTIVE").eq("ACTIVE") & (b.back1 > 1)]
    c = cat.drop_duplicates(KEY).copy()
    c["selection_id"] = pd.to_numeric(c.selection_id, errors="coerce")
    tr = {(str(r.market_id), int(r.selection_id)): str(r.trainer).strip().lower() for r in c.dropna(
        subset=["selection_id"]).itertuples() if isinstance(r.trainer, str) and r.trainer.strip()}
    st = {(str(r.market_id), int(r.selection_id)): float(r.stall_draw) for r in c.dropna(
        subset=["selection_id"]).itertuples() if pd.notna(pd.to_numeric(r.stall_draw, errors="coerce"))
        and float(pd.to_numeric(r.stall_draw, errors="coerce")) > 0}
    final = {m: g.set_index("selection_id").bsp for m, g in bsp[bsp.bsp > 1].groupby("market_id")}
    for r in rem[(rem.rf >= 2.5) & rem.removed_at.notna()].itertuples():
        seen += 1
        m = b[(b.market_id == r.market_id) & (b.t < r.removed_at)]
        mine = m[m.selection_id == r.selection_id]
        if mine.empty or r.market_id not in final:
            continue
        last = mine.sort_values("t").iloc[-1]
        s = m[(m.polled_utc == last.polled_utc) & (m.source == last.source)].drop_duplicates("selection_id")
        f = final[r.market_id]
        surv = s[s.selection_id != r.selection_id]
        if len(surv) < 3 or set(surv.selection_id.astype(int)) != set(f.index.astype(int)):
            continue
        lay = s.lay1.where(s.lay1 > 1)
        q = ((1 / s.back1 + 1 / lay.fillna(s.back1)) / 2).to_numpy(float)
        q = q / q.sum()
        sid = s.selection_id.astype(int).to_numpy()
        pj = float(q[sid == int(r.selection_id)][0])
        if not 0 < pj < 0.9:
            continue
        rank = pd.Series(q).rank(ascending=False, method="first").to_numpy()
        rj = float(rank[sid == int(r.selection_id)][0])
        fs = (1 / f).to_numpy(float)
        fshare = dict(zip(f.index.astype(int), fs / fs.sum()))
        kj = (r.market_id, int(r.selection_id))
        kept += 1
        for i, qi, rk in zip(sid, q, rank):
            if i == int(r.selection_id):
                continue
            ki = (r.market_id, int(i))
            rows.append({"day": str(day - timedelta(days=1)), "event": f"{r.market_id}|{r.selection_id}",
                         "market_id": r.market_id, "pj": pj,
                         "e": np.log(fshare[int(i)] / qi) - np.log(1 / (1 - pj)),
                         "trainer": float(kj in tr and ki in tr and tr[ki] == tr[kj]),
                         "adjacent": float(abs(rk - rj) == 1),
                         "stall": float(abs(st[ki] - st[kj]) <= 2) if ki in st and kj in st else np.nan})
if not rows:
    raise SystemExit("no data found: no cut removal with a book before it and the survivors' BSPs")
E = pd.DataFrame(rows)
print(f"cut removals seen {seen:,}; kept (a read before, the survivors' BSPs, no other change of the field) {kept:,}; "
      f"survivors {len(E):,}; days {E.day.nunique()} ({E.day.min()} .. {E.day.max()})")
print(f"e (zero under pro rata): mean {E.e.mean():+.4f}, sd {E.e.std():.4f}; removed share mean {E.pj.mean():.3f}")
print("survivors by kernel:", {k: int(E[k].fillna(0).sum()) for k in KERNELS},
      "| removals with a stall for both:", E[E.stall.notna()].event.nunique())
for k in KERNELS:
    g = E[E[k].notna()]
    print(f"  {k:9s} mean e where 1: {g[g[k] == 1].e.mean():+.4f} ({int((g[k] == 1).sum()):,}); where 0: "
          f"{g[g[k] == 0].e.mean():+.4f} ({int((g[k] == 0).sum()):,})")


def fit(d):
    X = d[KERNELS].copy()
    X["stall"] = X.stall.fillna(0.0)
    X = X - X.groupby(d.event).transform("mean")
    y = d.e - d.e.groupby(d.event).transform("mean")
    return np.linalg.lstsq(X.to_numpy(float), y.to_numpy(float), rcond=None)[0]


def read(d, label):
    th = fit(d)
    by = {k: v for k, v in d.groupby("market_id")}
    ms = list(by)
    bs = np.array([fit(pd.concat([by[m] for m in rng.choice(ms, len(ms))])) for _ in range(400)])
    se = bs.std(axis=0)
    print(f"{label}: {d.event.nunique():,} removals, {len(d):,} survivors")
    for k, t, s in zip(KERNELS, th, se):
        print(f"   {k:9s} theta {t:+.4f}  se {s:.4f}  t {t / s if s > 0 else float('nan'):+.2f}")
    return th, th / np.where(se > 0, se, np.nan)


read(E, "all days")
days = sorted(E.day.unique())
half = days[len(days) // 2]
(t1, z1), (t2, z2) = read(E[E.day < half], f"first half (before {half})"), read(E[E.day >= half], f"second half ({half} on)")
ok = [k for k, a, b, c, d in zip(KERNELS, t1, t2, z1, z2) if a * b > 0 and abs(c) > 2.5 and abs(d) > 2.5]
print("verdict:", f"PASS on {ok}" if ok else "does not pass (no kernel with the same sign and |t| > 2.5 in both halves)")
