"""Each horse's record before the matrix's start (2021-01-01), measured as the engine measures a run, for the career-before block.

Every feature is built from 2021-01-01, in training and at 06:00 alike, so a horse's runs
before then do not exist for the model (research query career_truncation: 8% of the
development window's runners, priced +0.052 worse within age and race code). A longer
history does not fit a runner's memory (iteration 82). This writes, per horse, from
race_results before 2021-01-01, the pieces model/blocks/career_before.py adds to the
horse's runs since:

    pw_runs, pw_jumps_runs        runs, and runs over hurdles, fences or in bumpers
    pw_<m>_sum, pw_<m>_n          the sum and count of each measure the career windows
                                  average, from model.form_windows.run_measures itself:
                                  won, placed, NFP (1 the winner .. 0 last), the market's
                                  view, won less the market's chance
    pw_nfp_l1..3, pw_mkt_l1..3    the last three runs' NFP and market view, last first
    pw_or_max_flat, _jumps        the highest official rating carried on the Flat (turf
                                  and all-weather) and over jumps
    pw_first_run, pw_last_run

keyed by the name as the engine keys a horse (stripped, lower case). Then, for the horses
that ran on or after 2021-01-01, how many have a record before it, and how many of those
are namesakes by age (the record starts before the horse could have run: its first run
earlier than a year after its birth year), overall and on the development window
(2025-09-27 to 2026-03-31; nothing later is read). Written to
out/pre_window_careers/pre_window_careers_2021-01-01.csv.gz. Read-only.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from model.form_windows import run_measures
from model.race_shape import race_code

CUTOFF = "2021-01-01"
WINDOW = ("2025-09-27", "2026-03-31")
OUT = Path("out/pre_window_careers")
MEASURES = ("win", "plc", "nfp", "mkt", "ae")
LAST = 3
JUMPS = ("hurdle", "chase", "nhflat")


def horse_key(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect("horse_racing.db")
    df = pd.read_sql_query(
        "SELECT race_date, track, race_time, horse_name, placing_numerical, number_of_runners, bfsp, "
        "official_rating, race_type, surface_type, dist_furlongs FROM race_results WHERE race_date < ?",
        conn, params=(CUTOFF,))
    later = pd.read_sql_query(
        "SELECT race_date, horse_name, horse_age FROM race_results WHERE race_date >= ? AND race_date <= ?",
        conn, params=(CUTOFF, WINDOW[1]))
    conn.close()
    df["key"] = horse_key(df["horse_name"])
    df = df[~df["key"].isin(["", "nan", "none"])].reset_index(drop=True)
    print(f"{len(df):,} runs before {CUTOFF}, {df['race_date'].min()} to {df['race_date'].max()}, "
          f"{df['key'].nunique():,} horses")

    m = run_measures(df)
    jumps = race_code(df).isin(JUMPS).to_numpy()
    orr = pd.to_numeric(df["official_rating"], errors="coerce")
    orr = orr.where(orr > 0).to_numpy(dtype=float)
    f = pd.DataFrame({"key": df["key"], "date": df["race_date"].astype(str), "time": df["race_time"].astype(str),
                      "jumps": jumps.astype(float), "or_flat": np.where(jumps, np.nan, orr),
                      "or_jumps": np.where(jumps, orr, np.nan), **{k: m[k] for k in MEASURES}})
    f = f.sort_values(["key", "date", "time"], kind="mergesort").reset_index(drop=True)
    g = f.groupby("key", sort=True)
    out = pd.DataFrame({"pw_runs": g.size(), "pw_jumps_runs": g["jumps"].sum()})
    for k in MEASURES:
        out[f"pw_{k}_sum"] = g[k].sum()
        out[f"pw_{k}_n"] = g[k].count()
    back = g.cumcount(ascending=False) + 1                   # 1 = the horse's last run before the cut-off
    for j in range(1, LAST + 1):
        last = f[back == j].set_index("key")
        out[f"pw_nfp_l{j}"] = last["nfp"]
        out[f"pw_mkt_l{j}"] = last["mkt"]
    out["pw_or_max_flat"] = g["or_flat"].max()
    out["pw_or_max_jumps"] = g["or_jumps"].max()
    out["pw_first_run"] = g["date"].min()
    out["pw_last_run"] = g["date"].max()
    out = out.reset_index().rename(columns={"key": "horse_key"})
    path = OUT / f"pre_window_careers_{CUTOFF}.csv.gz"
    out.to_csv(path, index=False, float_format="%.6g")
    print(f"wrote {path}: {len(out):,} horses, {path.stat().st_size / 1e6:.1f} MB")
    print(out.describe().T.to_string())

    # the horses that ran on or after the cut-off: a record before it, and namesakes by age
    later["key"] = horse_key(later["horse_name"])
    later["age"] = pd.to_numeric(later["horse_age"], errors="coerce")
    later["year"] = pd.to_datetime(later["race_date"], errors="coerce").dt.year
    later = later[~later["key"].isin(["", "nan", "none"])]
    first = pd.to_datetime(later["key"].map(out.set_index("horse_key")["pw_first_run"]), errors="coerce")
    matched = first.notna()
    namesake = matched & later["age"].notna() & (first.dt.year < later["year"] - later["age"] + 1)
    for label, rows in (("all runs since the cut-off", slice(None)),
                        ("the development window", (later["race_date"] >= WINDOW[0]).to_numpy())):
        sub = later[rows]
        mt, ns = matched[rows], namesake[rows]
        print(f"{label}: {len(sub):,} runs, {sub['key'].nunique():,} horses; with a record before {CUTOFF}: "
              f"{mt.mean():.1%} of runs ({sub.loc[mt, 'key'].nunique():,} horses); namesakes by age: "
              f"{int(ns.sum()):,} runs ({sub.loc[ns, 'key'].nunique():,} horses)")
    by_year = later.assign(m=matched & ~namesake).groupby("year")["m"].mean()
    print("share of runs with a (non-namesake) record before the cut-off, by year:",
          {int(k): round(float(v), 3) for k, v in by_year.items()})
    ex = later[namesake].drop_duplicates("key").head(12)
    if len(ex):
        print("namesake examples (name, first run since, age then, record's first run):")
        for _, r in ex.iterrows():
            print(f"  {r['horse_name']}: {r['race_date']}, aged {r['age']:.0f}; "
                  f"{out.set_index('horse_key').at[r['key'], 'pw_first_run']}")


if __name__ == "__main__":
    main()
