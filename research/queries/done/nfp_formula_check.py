"""The owner's NFP formula against ours (6 Oct 2026, read-only).

Ours (model/custom_metrics.py _calc_nfp): NFP = (N - F)/(N - 1), 1 for the winner and 0 for last whatever the field.
The owner's: NFPz = (N + 1 - 2F) / (3 sqrt((1/3)(N + 1)/(N - 1)) (N - 1)), the finishing position as a z-score over
the field, divided by 3: each race's runners average 0 with a spread of 1/3, and a place in a big field counts for
more than the same place in a small one (a win scores 0.33 in a two-runner race and 0.58 in a large field). Per race
NFPz = (2 NFP - 1) sqrt((N - 1)/(3 (N + 1))): the same order within a race, a different weight across races, which
is what a horse's averages over its runs feel.

Both are built over the same windows of a horse's earlier runs (last run, the mean of the last 3 and 5, the last 5
weighted 5..1, exponential in runs with a half-life of 3, career) and compared on GB/IE races from 2022 whose every
runner has a BSP and one winner: (1) on their own, the conditional logit's out-of-sample log-likelihood gain on the
winner over picking at random (millinats a race); (2) beside the BSP (Benter's test: the market's log chance and
its square, then the feature), the gain over the market alone; (3) the within-race rank correlation with where the
horse finishes. Two folds, each fitted on one period and scored on the other (2022-23, 2024-26Q1).
"""

from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from residual_screen import OutcomeScreen, Part, feature_design, gain_summary, market_columns  # noqa: E402

pd.set_option("display.width", 220)
DB = os.environ.get("DB_PATH") or "horse_racing.db"
FROM = "2018-01-01"                                 # history for the windows
EVAL_FROM, SPLIT = "2022-01-01", "2024-01-01"

con = sqlite3.connect(DB)
d = pd.read_sql_query(
    "SELECT race_date, race_time, track, horse_name, number_of_runners, placing_numerical, bfsp "
    f"FROM race_results WHERE race_date >= '{FROM}'", con)
con.close()
for c in ("number_of_runners", "placing_numerical", "bfsp"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["date"] = pd.to_datetime(d.race_date, errors="coerce")
d = d.dropna(subset=["date"]).copy()
d["race"] = d.race_date.astype(str) + "|" + d.track.astype(str) + "|" + d.race_time.astype(str)
N = d.number_of_runners.where(d.number_of_runners >= 2)
F = d.placing_numerical.where((d.placing_numerical >= 1) & (d.placing_numerical <= N))
d["nfp"] = (N - F) / (N - 1)
d["nfpz"] = (N + 1 - 2 * F) / (3 * np.sqrt((1 / 3) * (N + 1) / (N - 1)) * (N - 1))
print(f"{len(d):,} runs from {FROM}; NFP known on {d.nfp.notna().mean():.1%}")
print("NFPz by field size for the winner and last:",
      {int(n): (round(float((n - 1) / (3 * np.sqrt((n + 1) / (3 * (n - 1))) * (n - 1))), 3),
                round(float(-(n - 1) / (3 * np.sqrt((n + 1) / (3 * (n - 1))) * (n - 1))), 3))
       for n in (2, 5, 8, 12, 16, 20)})
chk = d.dropna(subset=["nfpz"]).groupby("race").nfpz.agg(["mean", "std", "size"])
chk = chk[chk["size"] >= 5]
print(f"per race (5+ finishers): NFPz mean {chk['mean'].mean():+.4f}, sd {chk['std'].mean():.4f}")

# a horse's earlier runs only: the windows of each measure, read before the race
d = d.sort_values(["horse_name", "date", "race_time"], kind="stable")
g = d.groupby("horse_name", sort=False)
for m in ("nfp", "nfpz"):
    prev = g[m].shift(1)
    d[f"{m}_l1"] = prev
    pg = prev.groupby(d.horse_name, sort=False)
    for w in (3, 5):
        d[f"{m}_m{w}"] = pg.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    d[f"{m}_car"] = pg.expanding().mean().reset_index(level=0, drop=True)
    lags = [g[m].shift(i) for i in range(1, 6)]
    wts = np.array([5, 4, 3, 2, 1], float)
    num = sum(np.nan_to_num(lag.to_numpy(float)) * wt for lag, wt in zip(lags, wts))
    den = sum(np.isfinite(lag.to_numpy(float)) * wt for lag, wt in zip(lags, wts))
    d[f"{m}_w5"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    d[f"{m}_e3"] = pg.ewm(halflife=3, ignore_na=True).mean().reset_index(level=0, drop=True)
WINDOWS = ("l1", "m3", "m5", "w5", "e3", "car")

# the sample: whole races from 2022 with a BSP for every runner and one winner
e = d[d.date >= EVAL_FROM].copy()
ok = e.groupby("race").agg(n=("bfsp", "size"), good=("bfsp", lambda s: bool((s > 1).all())),
                           wins=("placing_numerical", lambda s: int((s == 1).sum())))
e = e[e.race.isin(ok.index[ok.good & (ok.wins == 1) & (ok.n >= 3)])].copy()
e["y"] = (e.placing_numerical == 1).astype(float)
inv = 1 / e.bfsp
e["ln_pi"] = np.log(inv / inv.groupby(e.race).transform("sum"))
e = e.sort_values(["date", "race"], kind="stable").reset_index(drop=True)
codes = pd.factorize(e.race)[0]
y = e.y.to_numpy(float)
M = market_columns(e.ln_pi.to_numpy(float))
first = (e.date < SPLIT).to_numpy()
pos = np.arange(len(e))
print(f"sample: {e.race.nunique():,} races, {len(e):,} runners ({first.sum():,} before {SPLIT})")


def part(mask):
    return Part(pos[mask], codes[mask], y[mask], M[mask])


folds = [(part(first), part(~first)), (part(~first), part(first))]
scr = OutcomeScreen(folds)
print(f"the market's McFadden R2 on the held-out folds: {scr.market_r2():.4f}")

rows = []
for w in WINDOWS:
    for m, label in (("nfp", "ours (N-F)/(N-1)"), ("nfpz", "owner's z/3")):
        col = f"{m}_{w}"
        r = {"window": w, "formula": label, "known": float(e[col].notna().mean())}
        for both in (False, True):
            gains = []
            for k, (tr, te) in enumerate(folds):
                train_mask = np.zeros(len(e), bool)
                train_mask[tr.idx] = True
                X, _ = feature_design(e[col].to_numpy(float), train_mask)
                one = OutcomeScreen([(tr, te)])
                gains.append(one.gains(X, with_market=both))
            s = gain_summary(np.concatenate(gains), np.concatenate([te.null for _, te in folds]))
            tag = "beside_bsp" if both else "alone"
            r[f"{tag}_mnats"], r[f"{tag}_t"] = s["dll_mnats"], s["t"]
        x = e[[col, "placing_numerical", "race"]].dropna()
        x = x[x.groupby("race")[col].transform("size") >= 3]
        rk = x.groupby("race")[[col, "placing_numerical"]].rank()
        r["rank_corr_with_finish"] = float(-rk[col].corr(rk["placing_numerical"]))
        rows.append(r)
out = pd.DataFrame(rows)
print("\n== each window, ours against the owner's: the gain on the winner alone and beside the BSP "
      "(millinats a race, out of sample), and the rank correlation with the finish")
print(out.round(4).to_string(index=False))
piv = out.pivot(index="window", columns="formula", values="alone_mnats")
piv["owner_minus_ours"] = piv["owner's z/3"] - piv["ours (N-F)/(N-1)"]
print("\nalone, owner's minus ours (millinats a race):")
print(piv.round(3).to_string())
