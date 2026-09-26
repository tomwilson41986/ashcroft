"""Each horse's record before the matrix's start (2021-01-01), for a block that gives the model the careers it cannot see.

Every feature is built from 2021-01-01, in training and at 06:00 alike, so a horse's runs
before then do not exist for the model (research query career_truncation: 8% of the
development window's runners, priced +0.052 worse within age and race code). A longer
history does not fit a runner's memory (iteration 82). This writes the one thing the
missing years would add, per horse, from race_results before 2021-01-01: runs, wins,
places, the mean normalised finishing position, how the market rated it, its ratings,
the share of its runs over jumps, and its first and last run. Every row the model is
trained or scored on is on or after 2021-01-01, so nothing here is later than any of them.
Written to out/pre_window_careers/pre_window_careers_2021-01-01.csv.gz. Read-only.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = "2021-01-01"
OUT = Path("out/pre_window_careers")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect("horse_racing.db")
    df = pd.read_sql_query(
        "SELECT race_date, track, race_time, horse_name, placing_numerical, number_of_runners, bfsp, "
        "official_rating, race_type FROM race_results WHERE race_date < ?", conn, params=(CUTOFF,))
    conn.close()
    print(f"{len(df):,} runs before {CUTOFF}, {df['race_date'].min()} to {df['race_date'].max()}, "
          f"{df['horse_name'].nunique():,} horses")
    for c in ("placing_numerical", "number_of_runners", "bfsp", "official_rating"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    race = df["race_date"].astype(str) + "|" + df["track"].astype(str) + "|" + df["race_time"].astype(str)
    runners = df.groupby(race)["horse_name"].transform("size").astype(float)
    n = df["number_of_runners"].where(df["number_of_runners"] > 1, runners)
    pos = df["placing_numerical"]
    df["nfp"] = np.where((n > 1) & pos.notna(), (pos - 1) / (n - 1), np.nan)
    df["won"] = (pos == 1).astype(float)
    df["placed"] = (pos <= 3).astype(float)
    inv = 1.0 / df["bfsp"].where(df["bfsp"] > 1)
    book = inv.groupby(race).transform("sum")
    df["mkt"] = np.log(inv / book * runners)                 # 0 at the field's average price, > 0 when backed
    df["jumps"] = df["race_type"].astype(str).str.contains("Hurdle|Chase|NH Flat|Bumper", case=False).astype(float)
    df = df.sort_values(["horse_name", "race_date", "race_time"])
    g = df.groupby("horse_name", sort=True)
    out = pd.DataFrame({
        "pw_runs": g.size(),
        "pw_wins": g["won"].sum(),
        "pw_places": g["placed"].sum(),
        "pw_nfp": g["nfp"].mean(),
        "pw_nfp_last3": g["nfp"].apply(lambda s: s.tail(3).mean()),
        "pw_mkt": g["mkt"].mean(),
        "pw_or_max": g["official_rating"].max(),
        "pw_or_last": g["official_rating"].last(),
        "pw_jumps_share": g["jumps"].mean(),
        "pw_first_run": g["race_date"].min(),
        "pw_last_run": g["race_date"].max(),
    }).reset_index()
    out["pw_or_max"] = out["pw_or_max"].where(out["pw_or_max"] > 0)
    out["pw_or_last"] = out["pw_or_last"].where(out["pw_or_last"] > 0)
    path = OUT / f"pre_window_careers_{CUTOFF}.csv.gz"
    out.to_csv(path, index=False, float_format="%.6g")
    print(f"wrote {path}: {len(out):,} horses, {path.stat().st_size / 1e6:.1f} MB")
    print(out.describe(include="all").T.to_string()[:3000])
    base = out["horse_name"].str.replace(r" \([A-Z]{2,3}\)$", "", regex=True)
    print(f"horses whose names repeat with and without a country suffix: {int(base.duplicated().sum()):,}")


if __name__ == "__main__":
    main()
