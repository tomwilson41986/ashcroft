"""The owner (22 Oct): bring the Blandford sectionals through. Before the iteration (iter111): how many of our Flat
and all-weather runners the site's runs (data/external/sectionals.csv.gz) join by day and name, by month and code,
and how many of a day's runners carry an earlier sectional run (what the block can read). Read-only."""
import sqlite3
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from model.blocks.sectionals import _table, TABLE_PATH, norm_name  # noqa: E402
from model.race_shape import race_code  # noqa: E402

conn = sqlite3.connect("file:horse_racing.db?mode=ro", uri=True)
r = pd.read_sql_query("""SELECT race_date, track, horse_name, race_type, surface_type FROM race_results
                         WHERE race_date >= '2024-06-30' AND race_date < '2026-10-22'""", conn)
r["code"] = race_code(r)
r["day"] = pd.to_datetime(r.race_date, errors="coerce").dt.strftime("%Y-%m-%d")
r["name_norm"] = r.horse_name.map(norm_name)
t = _table(TABLE_PATH)
print(f"table: {len(t):,} runs (a name twice on a day dropped), {t.day.min()} .. {t.day.max()}")
r = r.merge(t[["day", "name_norm", "fsp_rel"]].assign(hit=1), on=["day", "name_norm"], how="left")
r["hit"] = r.hit.fillna(0)
r["month"] = r.day.str[:7]
win = r[((r.day >= "2024-06-30") & (r.day <= "2025-06-29")) | (r.day >= "2026-04-18")]
print("\n== our runners joined to a sectional run on the day, inside the site's windows")
print(win.groupby("code").hit.agg(["size", "mean"]).round(3).to_string())
fa = win[win.code.isin(["flat", "aw"])]
print(fa.groupby("month").hit.agg(["size", "mean"]).round(3).to_string())
# what the block reads: a runner with an earlier joined sectional run
r = r.sort_values("day")
r["prior"] = r.groupby("name_norm").hit.cumsum() - r.hit
r["has_prior"] = r.prior > 0
fa = r[r.code.isin(["flat", "aw"])]
print("\n== Flat/AW runners with at least one earlier sectional run (what the block reads), by month")
print(fa.groupby("month").has_prior.mean().round(3).to_string())
