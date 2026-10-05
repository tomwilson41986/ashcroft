"""How many runners does the sales block find, and does the price line up with the market? (read-only)

The sales block (model/blocks/sales.py) finds a runner's horse in Timeform's horse file
(data/external/timeform_sales.csv.gz) by name and foaling year. Before iteration 107 reads
what it is worth, this counts what it finds on the development window (2021 to March 2026,
nothing from the locked holdout): the share of runners known to Timeform's file, the share
known to have sold before racing, by race kind and year; and, where public form is thinnest
(maidens, novices and bumpers, runners with at most two runs), the rank correlation of the
sale price with the Betfair SP inside the race (the market should make the dearer horses
shorter) and the median price by the market's favouritism.
"""

from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.blocks import sales  # noqa: E402

pd.set_option("display.width", 220)
conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
df = pd.read_sql_query(
    """SELECT id AS raceid_row, race_date, race_time, track, horse_name, horse_age, race_type, race_code,
              career_runs, bfsp, number_of_runners
       FROM race_results WHERE race_date >= '2021-01-01' AND race_date < '2026-04-01'""", conn)
df["race_date"] = pd.to_datetime(df["race_date"])
df["raceid"] = df["race_date"].dt.strftime("%Y-%m-%d") + "|" + df["track"] + "|" + df["race_time"].astype(str)
out = sales.build(df)
known = out["sl_known"].eq(1)
sold = out["sl_sold"].eq(1)
rt = out["race_type"].fillna("").str.lower()
kind = np.select([rt.str.contains("bumper|nh flat"), rt.str.contains("maiden|novice") & ~rt.str.contains("hurdle|chase"),
                  rt.str.contains("hurdle|chase"), rt.str.contains("handicap")],
                 ["bumper", "flat maiden/novice", "hurdle/chase", "flat handicap"], "other")
out["kind"] = kind
out["year"] = out["race_date"].dt.year
out["known"], out["sold"] = known, sold
print(f"runners {len(out):,}; in Timeform's file {known.mean():.1%}; sold before racing {sold.mean():.1%}")
print("\nBy race kind")
print(out.groupby("kind").agg(runners=("known", "size"), known=("known", "mean"), sold=("sold", "mean"),
                              median_gbp=("sl_ln_price", lambda s: float(np.exp(s.median())) if s.notna().any() else np.nan))
      .round(3).to_string())
print("\nBy year")
print(out.groupby("year").agg(runners=("known", "size"), known=("known", "mean"), sold=("sold", "mean")).round(3).to_string())

thin = out[out["kind"].isin(["flat maiden/novice", "bumper"]) & (pd.to_numeric(out["career_runs"], errors="coerce") <= 2)]
thin = thin[thin["sl_ln_price"].notna() & (pd.to_numeric(thin["bfsp"], errors="coerce") > 1)].copy()
thin["ln_bsp"] = np.log(pd.to_numeric(thin["bfsp"]))
g = thin.groupby("raceid")
thin["r_price"] = g["sl_ln_price"].rank()
thin["r_bsp"] = g["ln_bsp"].rank()
ok = g["sl_ln_price"].transform("size") >= 3
rho = thin[ok].groupby("raceid").apply(lambda d: d["r_price"].corr(d["r_bsp"], method="spearman")).dropna()
print(f"\nThin form (maidens/novices/bumpers, <= 2 runs, priced): {len(thin):,} runners; races with 3+ priced: "
      f"{rho.size:,}; mean within-race Spearman(price, BSP) {rho.mean():+.3f} (se {rho.std(ddof=1) / np.sqrt(rho.size):.3f})")
thin["fav_rank"] = g["ln_bsp"].rank(method="first")
print(thin.groupby(pd.cut(thin["fav_rank"], [0, 1, 2, 3, 5, 99], labels=["fav", "2nd", "3rd", "4-5th", "6th+"]))
      ["sl_ln_price"].agg(["size", lambda s: float(np.exp(s.median()))]).rename(columns={"<lambda_0>": "median_gbp"})
      .round(0).to_string())
