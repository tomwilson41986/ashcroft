"""The reliability-weighted book, the first test (reports/discovery_1111.md 4.4; read-only).

In the thin morning book a runner's quote can be wide and small. The idea: read each runner's chance as the middle of
its back and lay, and give the book's missing or surplus probability to the runners whose quotes are least reliable
(wide, small), not in proportion. Is that nearer the BSP book than the closing model's own market read
(race_book.market_now, normalised)?

betfair_live_marks at 08:00, 09:00 and 10:00 UK, WIN markets, days from 20 Oct 2026; races where every runner still
running has a back price and a BSP. Per runner: m = (1/back + 1/lay)/2 (no lay: 0.5/back and the race's largest v);
v = ((1/back - 1/lay)/2)^2 / (1 + min(back size, lay size)/d0); x = m + (1 - sum m) v / sum v, floor 0.001,
renormalised. d0 is the median smaller size on the days before 20 Oct, fixed before the scored days. The marks do not
carry how old a quote is, so the report's age term is left out (said here, not hidden).
Score: mean |ln x - ln BSP share| less mean |ln market_now share - ln BSP share|, paired by race; the 90% interval by
race bootstrap. Pass (the report's): the interval below zero. 1 trial.
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model import race_book as rb  # noqa: E402

con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
mk = pd.read_sql_query(
    "SELECT m.race_date, m.market_id, m.selection_id, m.mark, m.runner_status, m.back1, m.back1_size, m.lay1, "
    "m.lay1_size, m.bsp FROM betfair_live_marks m WHERE m.mark IN ('clock_08:00', 'clock_09:00', 'clock_10:00', 'final')", con)
cat = pd.read_sql_query("SELECT DISTINCT market_id, market_type FROM betfair_live_markets", con)
win = set(cat[cat.market_type.fillna("WIN").str.upper().eq("WIN")].market_id)
mk = mk[mk.market_id.isin(win)]
for c in ("back1", "back1_size", "lay1", "lay1_size", "bsp"):
    mk[c] = pd.to_numeric(mk[c], errors="coerce")
print("rows by mark:", mk.mark.value_counts().to_dict(), "| WIN markets", len(win))
print("status at the clock marks:", mk[mk.mark != "final"].runner_status.value_counts().to_dict())
print("non-null share: back1 %.2f lay1 %.2f back1_size %.2f lay1_size %.2f; final bsp > 1: %d of %d" % (
    mk.back1.notna().mean(), mk.lay1.notna().mean(), mk.back1_size.notna().mean(), mk.lay1_size.notna().mean(),
    int((mk[mk.mark == "final"].bsp > 1).sum()), int((mk.mark == "final").sum())))
print("days:", mk.race_date.min(), mk.race_date.max())
fin = mk[mk.mark == "final"].set_index(["market_id", "selection_id"])
bsp = fin.bsp[fin.bsp > 1]
m = mk[(mk.mark != "final") & (mk.runner_status == "ACTIVE")].copy()
m["bsp"] = pd.MultiIndex.from_frame(m[["market_id", "selection_id"]]).map(bsp)
m["size"] = m[["back1_size", "lay1_size"]].min(axis=1)
d0 = float(m[(m.race_date < "2026-10-20") & (m["size"] > 0)]["size"].median())
print(f"d0 (the median smaller size before 20 Oct): GBP{d0:.2f}")
m = m[m.race_date >= "2026-10-20"]
rows = []
for (mid, mark), g in m.groupby(["market_id", "mark"]):
    b, la, s, sp = g.back1.to_numpy(float), g.lay1.to_numpy(float), g["size"].fillna(0).to_numpy(float), g.bsp.to_numpy(float)
    if len(g) < 2 or not (b > 1).all() or not np.isfinite(sp).all():
        continue
    has = la > 1
    mm = np.where(has, (1 / b + 1 / np.where(has, la, b)) / 2, 0.5 / b)
    v = np.where(has, ((1 / b - 1 / np.where(has, la, b)) / 2) ** 2 / (1 + s / d0), np.nan)
    if not np.isfinite(v).any() or np.nansum(v) <= 0:
        continue
    v = np.where(np.isfinite(v), v, np.nanmax(v))
    x = np.maximum(mm + (1 - mm.sum()) * v / v.sum(), 0.001)
    x = x / x.sum()
    base = 1 / rb.market_now(b, np.where(has, la, np.nan))
    base = base / base.sum()
    t = (1 / sp) / (1 / sp).sum()
    pm = mm / mm.sum()                                       # the control: the plain middle, shared in proportion
    rows.append({"day": g.race_date.iloc[0], "market_id": mid, "mark": mark, "n": len(g),
                 "err_x": np.abs(np.log(x) - np.log(t)).mean(), "err_base": np.abs(np.log(base) - np.log(t)).mean(),
                 "err_mid": np.abs(np.log(pm) - np.log(t)).mean(),
                 "over": (1 / b).sum(), "no_lay": int((~has).sum())})
r = pd.DataFrame(rows)
if r.empty:
    raise SystemExit("no data found: no whole race at the marks")
r["d"] = r.err_x - r.err_base
rng = np.random.default_rng(1112)


def read(x, label):
    per = x.groupby("market_id").d.mean()
    bs = [per.sample(len(per), replace=True, random_state=int(rng.integers(1 << 31))).mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [5, 95])
    print(f"  {label:12s} races {len(per):5d}  market_now {x.err_base.mean():.4f}  reliability {x.err_x.mean():.4f}  "
          f"change {per.mean():+.4f} (90% {lo:+.4f} to {hi:+.4f})")
    return lo, hi


print(f"{r.day.min()} to {r.day.max()}: {r.market_id.nunique():,} races, {len(r):,} race-marks; "
      f"book sum of 1/back median {r.over.median():.3f}; race-marks with a runner lacking a lay {int((r.no_lay > 0).sum()):,}")
lo, hi = read(r, "all marks")
for k in ("clock_08:00", "clock_09:00", "clock_10:00"):
    read(r[r.mark == k], k)
half = sorted(r.day.unique())
read(r[r.day.isin(half[:len(half) // 2])], "first half")
read(r[r.day.isin(half[len(half) // 2:])], "second half")
print("the control (added after the first reading, so not part of the registered pass): the plain middle of back and "
      "lay, normalised in proportion")
r["d"] = r.err_x - r.err_mid
r["err_base"] = r.err_mid
read(r, "vs plain mid")
for k in ("clock_08:00", "clock_09:00", "clock_10:00"):
    read(r[r.mark == k], k)
print("verdict:", "PASS (the interval is below zero)" if hi < 0 else "does not pass")
