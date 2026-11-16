"""Attention share of the day: is a neglected race's morning price further from its close? (read-only first read of
reports/discovery_1111.md 4.7; ledger `attention-share-1112`)

A race's share of the day's turnover is largely known before any trade: the track, the race type, the weekday and
what else is on. If the morning price of a race that will draw little money sits further from the Betfair SP, the
closing model should lean less on the market there.

Betfair's price files (betfair_prices, win markets, 1 Jan 2023 - 31 Mar 2026; the locked holdout from 1 Apr 2026 is
not read). For each race: vol = the pre-play volume summed over its runners. E = the expected ln vol of its cell
(track x race type x weekday), the mean over days at least two before, shrunk to the track's mean with weight
K = 20; neglect = -ln(exp(E) / the sum of exp(E) over the day's races). y = the race's mean |ln(morning WAP / BSP)|
over its runners with both. Scored from 1 Jan 2024 (2023 only feeds E).

    y = b0 + b1 neglect + b2 E + controls (runners, mean ln BSP, jumps)        days clustered by bootstrap

Registered pass for this first read: b1 above zero with its 90% interval above zero in 2024 and in 2025-26 apart.
Only then the closing-model arms of the specification. 1 trial.
"""
import sqlite3

import numpy as np
import pandas as pd

K = 20.0
con = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
p = pd.read_sql_query(
    "SELECT p.race_date, p.event_id, p.track_bf, p.bsp, p.morningwap, p.pp_vol, r.race_type "
    "FROM betfair_prices p LEFT JOIN race_results r ON r.id = p.race_results_id "
    "WHERE p.market_type = 'win' AND p.race_date >= '2023-01-01' AND p.race_date <= '2026-03-31'", con)
for c in ("bsp", "morningwap", "pp_vol"):
    p[c] = pd.to_numeric(p[c], errors="coerce")
p["day"] = pd.to_datetime(p.race_date.str[:10], errors="coerce")
p = p.dropna(subset=["day", "event_id"])
print(f"rows {len(p):,}; races {p.event_id.nunique():,}; days {p.day.nunique():,} ({p.day.min():%Y-%m-%d} .. {p.day.max():%Y-%m-%d}); "
      f"with a race type {p.race_type.notna().mean():.1%}; with a morning WAP {(p.morningwap > 1).mean():.1%}")
p["dev"] = np.where((p.morningwap > 1) & (p.bsp > 1), np.abs(np.log(p.morningwap / p.bsp)), np.nan)
jump = p.race_type.fillna("").str.lower().str.contains("hurdle|chase|bumper|nh")
p["code"] = np.where(p.race_type.isna(), "unknown", np.where(jump, "jumps", "flat"))
R = p.groupby("event_id").agg(day=("day", "first"), track=("track_bf", "first"), code=("code", "first"),
                              vol=("pp_vol", "sum"), y=("dev", "mean"), n=("bsp", "size"), priced=("dev", "count"),
                              lbsp=("bsp", lambda s: float(np.log(s[s > 1]).mean()) if (s > 1).any() else np.nan))
R = R[(R.vol > 0) & R.track.notna()].copy()
R["lv"] = np.log(R.vol)
R["cell"] = R.track.astype(str) + "|" + R.code + "|" + R.day.dt.dayofweek.astype(str)


def lagged(key):
    """Sum and count of ln vol over the key's races on days at least two before each race's day."""
    d = R.groupby([key, "day"]).lv.agg(["sum", "count"]).reset_index().sort_values([key, "day"])
    d["cs"], d["cn"] = d.groupby(key)["sum"].cumsum(), d.groupby(key)["count"].cumsum()
    out_s, out_n = [], []
    for _, g in d.groupby(key, sort=False):
        days = g.day.to_numpy()
        i = np.searchsorted(days, days - np.timedelta64(2, "D"), side="right") - 1
        out_s.append(np.where(i >= 0, g.cs.to_numpy()[np.clip(i, 0, None)], 0.0))
        out_n.append(np.where(i >= 0, g.cn.to_numpy()[np.clip(i, 0, None)], 0.0))
    d["s"], d["c"] = np.concatenate(out_s), np.concatenate(out_n)
    return R[[key, "day"]].merge(d[[key, "day", "s", "c"]], on=[key, "day"], how="left").set_index(R.index)


c, t = lagged("cell"), lagged("track")
tm = np.where(t.c > 0, t.s / t.c.where(t.c > 0, 1), np.nan)
R["E"] = (c.s + K * tm) / (c.c + K)
R = R[np.isfinite(R.E)].copy()
R["neglect"] = -np.log(np.exp(R.E) / np.exp(R.E).groupby(R.day).transform("sum"))
S = R[(R.day >= "2024-01-01") & (R.priced >= 3) & np.isfinite(R.y) & np.isfinite(R.lbsp)].copy()
S["jumps"] = (S.code == "jumps").astype(float)
print(f"scored races {len(S):,} over {S.day.nunique():,} days; E against the realised ln vol: correlation "
      f"{np.corrcoef(S.E, S.lv)[0, 1]:.3f}; y mean {S.y.mean():.4f}, sd {S.y.std():.4f}")
S["third"] = pd.qcut(S.neglect, 3, labels=["most attention", "middle", "least attention"])
print(S.groupby("third", observed=True).agg(races=("y", "size"), y=("y", "mean"), vol=("vol", "median")).round(4).to_string())
rng = np.random.default_rng(1112)
COLS = ["neglect", "E", "n", "lbsp", "jumps"]


def fit(d):
    X = np.column_stack([np.ones(len(d))] + [d[k].to_numpy(float) for k in COLS])
    return np.linalg.lstsq(X, d.y.to_numpy(float), rcond=None)[0][1:]


ok = []
for label, d in (("2024", S[S.day < "2025-01-01"]), ("2025-26", S[S.day >= "2025-01-01"]), ("all", S)):
    b = fit(d)
    by = {k: v for k, v in d.groupby("day")}
    days = list(by)
    bs = np.array([fit(pd.concat([by[k] for k in rng.choice(len(days), len(days))] if False else
                                 [by[days[i]] for i in rng.integers(0, len(days), len(days))])) for _ in range(400)])
    lo, hi = np.percentile(bs, [5, 95], axis=0)
    print(f"{label}: {len(d):,} races")
    for k, v, a, z in zip(COLS, b, lo, hi):
        print(f"   {k:8s} {v:+.5f} (90% {a:+.5f} to {z:+.5f})")
    if label != "all":
        ok.append(lo[0] > 0)
print("verdict:", "PASS (go on to the closing-model arms)" if all(ok) else "does not pass")
