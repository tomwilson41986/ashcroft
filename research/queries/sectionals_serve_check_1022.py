"""The sectionals blend against the served blend on 22 Oct's card, priced the same way (read-only; the serving check of
iteration 113, the owner's ask of 22 Oct to bring the sectionals through).

Both priced by predict-now.yml on 22 Oct's card after the racing (the card's pre-race fields; nothing from the
results): the served blend from data/models (run 39872671163) and the candidate, its three members retrained to 13 Oct
with the sectionals block (runs 39843011851, 39843027319, 39843043011; predict-now 39869609141), gated as served. Each
against the BSP as the trader read it (betfair_live_marks' final mark; the price file for 22 Oct is not out), with the
06:00 file (the served blend on the morning's card) beside them:

- the error of log(predicted / BSP), the error after each race is centred (what the closing model reads), the rank
  correlation, by code (Flat/AW against jumps: the sectionals cover Flat and all-weather only);
- how far the candidate moves the price, and whether those moves went the BSP's way.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from betfair_prices import normalise_horse  # noqa: E402

DAY = "2026-10-22"
HERE = os.path.join("research", "data", "serve_cmp_1022")
cols = ["race_date", "race_time", "track", "horse_name", "race_name", "predicted_bfsp"]
new = pd.read_csv(os.path.join(HERE, "candidate.csv"), usecols=cols)
old = pd.read_csv(os.path.join(HERE, "served.csv"), usecols=["horse_name", "predicted_bfsp"])
new["h"] = new.horse_name.map(normalise_horse)
old["h"] = old.horse_name.map(normalise_horse)
m = new.merge(old[["h", "predicted_bfsp"]].rename(columns={"predicted_bfsp": "served"}), on="h", how="inner")
m = m.rename(columns={"predicted_bfsp": "cand"})

import boto3  # noqa: E402
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
os.makedirs("live_q", exist_ok=True)
s3.download_file(os.environ.get("PRED_BUCKET") or "ashcroft", f"predictions/{DAY}.csv", "live_q/p0600.csv")
am = pd.read_csv("live_q/p0600.csv")
hcol = "horse_name" if "horse_name" in am.columns else "runner_name"
am["h"] = am[hcol].map(normalise_horse)
m = m.merge(am.drop_duplicates("h")[["h", "predicted_bfsp"]].rename(columns={"predicted_bfsp": "at_0600"}), on="h",
            how="left")

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
bsp = pd.read_sql_query(f"""SELECT k.horse_norm AS h, MAX(f.bsp) AS bsp FROM betfair_live_markets k
                            JOIN betfair_live_marks f ON f.market_id = k.market_id AND f.selection_id = k.selection_id
                            WHERE k.race_date = '{DAY}' AND f.mark = 'final' AND f.bsp > 1 GROUP BY k.horse_norm""",
                        conn)
m = m.merge(bsp, on="h", how="inner")
m = m[(m.cand > 1) & (m.served > 1)].copy()
m["race"] = m.track + "|" + m.race_time.astype(str)
m["code"] = np.where(m.race_name.astype(str).str.contains("Hurdle|Chase|Bumper|NH Flat|National Hunt|Hunters", case=False),
                     "jumps", "flat/aw")
print(f"joined: {len(m)} runners in {m.race.nunique()} races ({(m.code == 'flat/aw').sum()} Flat/AW); "
      f"06:00 price for {m.at_0600.notna().sum()}")

lb = np.log(m.bsp)
for c in ("cand", "served", "at_0600"):
    e = np.log(m[c]) - lb
    m[f"e_{c}"] = e
    m[f"ec_{c}"] = e - e.groupby(m.race).transform("mean")


def stats(g):
    out = {"runners": len(g), "races": g.race.nunique()}
    for c in ("cand", "served", "at_0600"):
        ok = g[c].notna()
        out[f"mae_{c}"] = g.loc[ok, f"e_{c}"].abs().mean()
        out[f"mae_centred_{c}"] = g.loc[ok, f"ec_{c}"].abs().mean()
        out[f"spearman_{c}"] = np.log(g.loc[ok, c]).corr(np.log(g.loc[ok, "bsp"]), method="spearman")
    d = g.ec_cand.abs() - g.ec_served.abs()
    out["centred_diff"] = d.mean()
    out["centred_diff_se"] = d.std(ddof=1) / np.sqrt(len(d))
    return pd.Series(out)


pd.set_option("display.width", 250)
print("\n== the candidate (sectionals) and the served blend against the BSP; centred_diff < 0: the candidate closer")
print(pd.concat({"all": stats(m), **{k: stats(g) for k, g in m.groupby("code")}}, axis=1).round(4).to_string())

mv = np.log(m.cand / m.served)
toward = np.sign(mv) == np.sign(np.log(m.bsp / m.served))
print("\n== the candidate's moves from the served price")
print(f"  median |move| {mv.abs().median():.4f}, 90th pct {mv.abs().quantile(0.9):.4f}, largest {mv.abs().max():.3f}")
for lo in (0.0, 0.05, 0.1):
    s = mv.abs() > lo
    print(f"  moves over {lo:.0%}: {s.sum()} runners, toward the BSP {toward[s].mean():.1%}")
print(m.loc[mv.abs().sort_values(ascending=False).index[:12],
            ["track", "race_time", "horse_name", "code", "served", "cand", "bsp"]].round(2).to_string())
