"""Phase 0 data audit, horse racing only (the owner's brief, 18 Nov 2026: "improve, validate and test the racing²
horse racing ... models", Phase 0; greyhounds out of scope). Read-only on horse_racing.db (S3 horseracingresults).

Prints, and writes to out/phase0_data_audit.json:
  - every table: rows, the date column's range
  - race_results: rows and races by year and code (flat turf / all-weather / hurdle / chase / NH flat), date range,
    duplicates (raceid + horse, and date + track + time + horse), every column's null share, key fields' null share
    by year, the raw placing codes (non-runners, fallers, pulled up), declared field size against rows held,
    the BFSP (missing, under 1.01, over 1000, the race's book), rows usable for each target (log BFSP; race-centred
    normalised pounds beaten), the locked holdout's size (counts only; >= 2026-04-01 is not evaluated)
  - the Betfair tables: coverage of the price files and the live record, and how many results rows they match
"""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd

DB = "horse_racing.db"
OUT = "out/phase0_data_audit.json"
HOLDOUT = "2026-04-01"
KEY = ["raceid", "race_date", "race_time", "track", "horse_name", "bfsp", "placing_numerical", "number_of_runners",
       "total_dst_bt", "LB", "dist_furlongs", "prize_money", "race_type", "surface_type", "going", "jockey", "trainer",
       "official_rating", "OR", "age", "weight", "draw"]
report: dict = {}


def say(*a):
    print(*a, flush=True)


con = sqlite3.connect(DB)
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
say(f"== tables in {DB}: {len(tables)} (file {os.path.getsize(DB) / 1e9:.2f} GB)")
tab = []
for t in tables:
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
    n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    dcol = next((c for c in ("race_date", "date", "day", "market_date", "event_date", "settled_date") if c in cols),
                None)
    lo = hi = None
    if dcol and n:
        lo, hi = con.execute(f'SELECT MIN("{dcol}"), MAX("{dcol}") FROM "{t}"').fetchone()
    tab.append({"table": t, "rows": n, "cols": len(cols), "date_col": dcol, "from": lo, "to": hi})
tdf = pd.DataFrame(tab)
say(tdf.to_string(index=False))
report["tables"] = tab

# ---------------------------------------------------------------- race_results
df = pd.read_sql_query("SELECT * FROM race_results", con)
say(f"\n== race_results: {len(df):,} rows, {df.shape[1]} columns")
df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
bad_date = int(df.race_date.isna().sum())
say(f"   unparseable race_date: {bad_date}; range {df.race_date.min().date()} to {df.race_date.max().date()}")
report["race_results"] = {"rows": len(df), "columns": list(df.columns), "unparseable_date": bad_date,
                          "from": str(df.race_date.min().date()), "to": str(df.race_date.max().date())}

try:
    from model.race_shape import race_code, race_key
    df["code"] = race_code(df).astype(str).to_numpy()
    df["_race"] = race_key(df).to_numpy()
except Exception as e:  # noqa: BLE001 - the audit reports what it could not compute
    say(f"   race_code/race_key failed: {e!r}")
    df["code"] = "unknown"
    df["_race"] = df["raceid"] if "raceid" in df.columns else np.arange(len(df))
df["year"] = df.race_date.dt.year

by = df.groupby(["year", "code"]).agg(rows=("horse_name", "size"), races=("_race", "nunique")).reset_index()
piv = by.pivot(index="year", columns="code", values="rows").fillna(0).astype(int)
piv["all"] = piv.sum(axis=1)
say("\n-- rows by year and code\n" + piv.to_string())
pivr = by.pivot(index="year", columns="code", values="races").fillna(0).astype(int)
pivr["all"] = pivr.sum(axis=1)
say("\n-- races by year and code\n" + pivr.to_string())
report["rows_by_year_code"] = piv.reset_index().to_dict(orient="records")
report["races_by_year_code"] = pivr.reset_index().to_dict(orient="records")

# duplicates
hn = df["horse_name"].fillna("").astype(str).str.strip().str.lower()
dups = {}
if "raceid" in df.columns:
    dups["raceid_horse"] = int(pd.DataFrame({"r": df["raceid"], "h": hn}).duplicated().sum())
dups["date_track_time_horse"] = int(pd.DataFrame({"d": df.race_date, "t": df.get("track"), "tm": df.get("race_time"),
                                                  "h": hn}).duplicated().sum())
dups["exact_rows"] = int(df.drop(columns=["code", "_race", "year"]).duplicated().sum())
dups["blank_horse_name"] = int((hn == "").sum())
if "raceid" in df.columns:
    multi = df.groupby("raceid").agg(d=("race_date", "nunique"), t=("track", "nunique"))
    dups["raceid_with_several_dates_or_tracks"] = int(((multi.d > 1) | (multi.t > 1)).sum())
say("\n-- duplicates\n" + json.dumps(dups, indent=1))
report["duplicates"] = dups

# nulls: every column, then the key fields by year
blank = df.replace({"": np.nan, "-": np.nan, "None": np.nan})
nul = blank.isna().mean().sort_values(ascending=False)
say("\n-- null share, every column (blank, '-' and 'None' counted as null)")
say(nul.round(4).to_string())
report["null_share"] = nul.round(5).to_dict()
keys = [c for c in KEY if c in df.columns]
say(f"\n-- key fields held: {keys}; missing from the table: {[c for c in KEY if c not in df.columns]}")
ky = blank[keys].isna().groupby(df["year"]).mean().round(4)
say("\n-- key fields' null share by year\n" + ky.to_string())
report["key_null_by_year"] = ky.reset_index().to_dict(orient="records")

# placing codes
plc = [c for c in df.columns if "plac" in c.lower() or c.lower() in ("pos", "position", "fin_pos")]
for c in plc:
    v = df[c].astype(str)
    nonnum = v[pd.to_numeric(v, errors="coerce").isna()]
    say(f"\n-- {c}: non-numeric values {len(nonnum):,}; top codes")
    say(nonnum.value_counts().head(25).to_string())
    report[f"placing_codes_{c}"] = nonnum.value_counts().head(40).to_dict()
pos = pd.to_numeric(df.get("placing_numerical"), errors="coerce")
report["placing_numerical_le0_or_null"] = int(((pos <= 0) | pos.isna()).sum())

# field size: declared against rows held
held = df.groupby("_race")["horse_name"].transform("size")
dec = pd.to_numeric(df.get("number_of_runners"), errors="coerce")
fs = pd.DataFrame({"race": df["_race"], "held": held, "declared": dec}).drop_duplicates("race")
fs["diff"] = fs.declared - fs.held
fsum = {"races": len(fs), "declared_null": int(fs.declared.isna().sum()), "equal": int((fs["diff"] == 0).sum()),
        "declared_gt_held": int((fs["diff"] > 0).sum()), "declared_lt_held": int((fs["diff"] < 0).sum()),
        "one_runner_races_held": int((fs.held == 1).sum())}
say("\n-- field size, declared against rows held (per race)\n" + json.dumps(fsum, indent=1))
report["field_size"] = fsum

# the BFSP
b = pd.to_numeric(df.get("bfsp"), errors="coerce")
bsum = {"null": int(b.isna().sum()), "le_1": int((b <= 1.0).sum()), "lt_1_01": int(((b > 1) & (b < 1.01)).sum()),
        "gt_1000": int((b > 1000).sum()), "valid": int(((b > 1) & (b <= 1000)).sum())}
ok = (b > 1) & (b <= 1000)
book = (1 / b[ok]).groupby(df.loc[ok, "_race"]).sum()
full = df.loc[ok].groupby("_race").size().reindex(book.index) == held.groupby(df["_race"]).first().reindex(book.index)
bsum["race_book_quantiles_full_races"] = {str(q): round(float(book[full].quantile(q)), 4)
                                          for q in (0.01, 0.05, 0.5, 0.95, 0.99)}
bsum["races_book_lt_0_9_or_gt_1_2_full"] = int(((book[full] < 0.9) | (book[full] > 1.2)).sum())
bsum["valid_by_year"] = ok.groupby(df["year"]).mean().round(4).to_dict()
say("\n-- BFSP\n" + json.dumps(bsum, indent=1, default=str))
report["bfsp"] = bsum

# rows usable per target
lb = pd.to_numeric(df["LB"], errors="coerce") if "LB" in df.columns else pd.Series(np.nan, index=df.index)
fin = pos > 0
nbl_ok = fin & ((pos == 1) | lb.notna())
prize = pd.to_numeric(df.get("prize_money"), errors="coerce")
tgt = {"rows": len(df),
       "bfsp_target_usable": int(ok.sum()),
       "nbl_target_usable_finisher_with_margin": int(nbl_ok.sum()),
       "nbl_and_prize_usable": int((nbl_ok & (prize > 0)).sum()),
       "finishers": int(fin.sum()), "finishers_missing_margin": int((fin & (pos > 1) & lb.isna()).sum()),
       "prize_null_or_0": int(((prize <= 0) | prize.isna()).sum())}
tgt["usable_by_year"] = pd.DataFrame({"bfsp": ok, "nbl": nbl_ok, "nbl_prize": nbl_ok & (prize > 0)}).groupby(
    df["year"]).mean().round(4).reset_index().to_dict(orient="records")
say("\n-- rows usable for each target\n" + json.dumps(tgt, indent=1, default=str))
report["targets"] = tgt

# the locked holdout: counts only
ho = df.race_date >= pd.Timestamp(HOLDOUT)
report["holdout_counts"] = {"rows": int(ho.sum()), "races": int(df.loc[ho, "_race"].nunique()),
                            "rows_from_2026_10_01": int((df.race_date >= "2026-10-01").sum())}
say("\n-- locked holdout (>= 2026-04-01; counted, not evaluated)\n" + json.dumps(report["holdout_counts"]))

# last days held (the HRB pause)
last = df.groupby(df.race_date.dt.date).size().tail(12)
say("\n-- the last 12 days held\n" + last.to_string())
report["last_days"] = {str(k): int(v) for k, v in last.items()}

# ---------------------------------------------------------------- Betfair coverage
say("\n== Betfair tables")
bf = {}
for t in ("betfair_prices", "betfair_prices_files", "betfair_live_markets", "betfair_live_marks", "live_orders",
          "betfair_odds", "bets"):
    if t not in tables:
        bf[t] = "no data found (table absent)"
        continue
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
    bf[t] = {"columns": cols, "rows": con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]}
    for c in ("adjustment_factor", "reduction_factor", "removal_utc", "bsp", "ltp", "traded_volume"):
        if c in cols:
            bf[t][f"{c}_nonnull"] = con.execute(f'SELECT COUNT("{c}") FROM "{t}"').fetchone()[0]
say(json.dumps(bf, indent=1, default=str))
report["betfair"] = bf

# match of the price files to results, by year (when the table carries the date and a name)
if "betfair_prices" in tables:
    cols = [r[1] for r in con.execute('PRAGMA table_info("betfair_prices")')]
    dc = next((c for c in ("race_date", "event_date", "date") if c in cols), None)
    if dc:
        yr = pd.read_sql_query(f'SELECT substr("{dc}",1,4) AS year, COUNT(*) AS rows FROM betfair_prices '
                               f'GROUP BY 1 ORDER BY 1', con)
        say("\n-- betfair_prices rows by year\n" + yr.to_string(index=False))
        report["betfair_prices_by_year"] = yr.to_dict(orient="records")
con.close()

os.makedirs("out", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(report, f, indent=1, default=str)
say(f"\nwritten {OUT}")
