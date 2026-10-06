"""Horseracebase's system-builder ideas the model lacked, screened on the winner (6 Oct 2026, read-only).

The owner's list (horseracebase's system builder v4) is mapped in reports/hrb_system_builder_review.md; nearly all
of it is already the model's. Two blocks carry the rest: model/blocks/inrunning.py (each past run's in-running low
and high from Betfair's price files, "BF In Play (Min)/(Max)") and model/blocks/hrb_extras.py (same surname, the
share of seconds, the last race's winner's price and non-finishers, the biggest field won in, last year's run at the
course and trip). Each feature, and each block together, is scored on GB/IE races from 2022 to 31 Mar 2026 (the
locked holdout from 1 Apr is not read) whose every runner has a BSP and one winner: the conditional logit's
out-of-sample gain on the winner on its own and beside the BSP (the market's log chance and its square), millinats a
race, two folds each fitted on one period and scored on the other (2022-23, 2024 to Mar 2026).
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
from model.blocks import hrb_extras, inrunning  # noqa: E402

pd.set_option("display.width", 220)
DB = os.environ.get("DB_PATH") or "horse_racing.db"
inrunning.DB_PATH = os.path.abspath(DB)
FROM = "2018-01-01"                                 # history for the windows (the price files start in 2018)
EVAL_FROM, SPLIT, END = "2022-01-01", "2024-01-01", "2026-03-31"

con = sqlite3.connect(DB)
d = pd.read_sql_query(
    "SELECT race_date, race_time, track, horse_name, jockey_name, trainer, number_of_runners, placing_numerical, "
    f"bfsp, dist_furlongs FROM race_results WHERE race_date >= '{FROM}' AND race_date <= '{END}'", con)
n_pf = con.execute("SELECT COUNT(*), MIN(race_date), MAX(race_date) FROM betfair_prices WHERE market_type = 'win'").fetchone()
con.close()
print(f"price files: {n_pf[0]:,} win-market rows, {n_pf[1]} to {n_pf[2]}")
for c in ("number_of_runners", "placing_numerical", "bfsp", "dist_furlongs"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
d["date"] = pd.to_datetime(d.race_date, errors="coerce")
d = d.dropna(subset=["date"]).reset_index(drop=True)
d["raceid"] = d.race_date.astype(str) + "|" + d.track.astype(str) + "|" + d.race_time.astype(str)
print(f"{len(d):,} runs {FROM} to {END}")

d = inrunning.build(d)
d = hrb_extras.build(d)
FEATS = list(inrunning.FEATURES) + list(hrb_extras.FEATURES)
r = inrunning.readings(d)
print(f"runs with an in-running record: {r['pf_ipmin'].notna().mean():.1%}; winners' low <= 1.05: "
      f"{(r.loc[d.placing_numerical == 1, 'pf_ipmin'] <= 1.05).mean():.1%}")

# the sample: whole races from 2022 with a BSP for every runner and one winner
e = d[(d.date >= EVAL_FROM) & (d.date <= END)].copy()
ok = e.groupby("raceid").agg(n=("bfsp", "size"), good=("bfsp", lambda s: bool((s > 1).all())),
                             wins=("placing_numerical", lambda s: int((s == 1).sum())))
e = e[e.raceid.isin(ok.index[ok.good & (ok.wins == 1) & (ok.n >= 3)])].copy()
e["y"] = (e.placing_numerical == 1).astype(float)
inv = 1 / e.bfsp
e["ln_pi"] = np.log(inv / inv.groupby(e.raceid).transform("sum"))
e = e.sort_values(["date", "raceid"], kind="stable").reset_index(drop=True)
codes = pd.factorize(e.raceid)[0]
y = e.y.to_numpy(float)
M = market_columns(e.ln_pi.to_numpy(float))
first = (e.date < SPLIT).to_numpy()
pos = np.arange(len(e))
print(f"sample: {e.raceid.nunique():,} races, {len(e):,} runners ({first.sum():,} before {SPLIT})")
print("known in the sample:", {f: round(float(e[f].notna().mean()), 3) for f in FEATS})
print("same surname bookings:", f"{(e.hx_same_surname == 1).mean():.2%} of runners;",
      "their win rate", f"{e.loc[e.hx_same_surname == 1, 'y'].mean():.3f} against the market's",
      f"{np.exp(e.loc[e.hx_same_surname == 1, 'ln_pi']).mean():.3f}")


def part(mask):
    return Part(pos[mask], codes[mask], y[mask], M[mask])


folds = [(part(first), part(~first)), (part(~first), part(first))]
print(f"the market's McFadden R2 on the held-out folds: {OutcomeScreen(folds).market_r2():.4f}")


def score(cols: list[str]) -> dict:
    out = {}
    for both in (False, True):
        gains = []
        for tr, te in folds:
            train_mask = np.zeros(len(e), bool)
            train_mask[tr.idx] = True
            X = np.hstack([feature_design(e[c].to_numpy(float), train_mask)[0] for c in cols])
            gains.append(OutcomeScreen([(tr, te)]).gains(X, with_market=both))
        s = gain_summary(np.concatenate(gains), np.concatenate([te.null for _, te in folds]))
        tag = "beside_bsp" if both else "alone"
        out[f"{tag}_mnats"], out[f"{tag}_t"] = s["dll_mnats"], s["t"]
    return out


rows = [{"feature": f, "known": float(e[f].notna().mean()), **score([f])} for f in FEATS]
rows += [{"feature": "ALL inrunning", "known": np.nan, **score(list(inrunning.FEATURES))},
         {"feature": "ALL hrb_extras", "known": np.nan, **score(list(hrb_extras.FEATURES))}]
out = pd.DataFrame(rows)
print("\n== each feature and each block: the gain on the winner alone and beside the BSP (millinats a race, "
      "out of sample; t over races)")
print(out.round(3).to_string(index=False))

# what the in-running low says, beside the finishing position it hides: beaten last time, by how short it traded
b = e[e.ir_low_l1.notna()].copy()
b["last_low"] = pd.cut(np.exp(b.ir_low_l1), [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1000.0])
tab = b.groupby("last_low", observed=True).agg(runners=("y", "size"), won=("y", "mean"),
                                               market=("ln_pi", lambda s: float(np.exp(s).mean())))
tab["a_over_e"] = tab.won / tab.market
print("\n== by the last run's in-running low: the win rate against the market's chance (A/E)")
print(tab.round(4).to_string())
