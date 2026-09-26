"""Each horse's record before the matrix's start (2021-01-01), measured as the engine measures a run, for the career-before block.

Every feature is built from 2021-01-01, in training and at 06:00 alike, so a horse's runs
before then do not exist for the model (research query career_truncation: 8% of the
development window's runners, priced +0.052 worse within age and race code). A longer
history does not fit a runner's memory (iteration 82). This writes, per horse, from
race_results before 2021-01-01 with a usable price (the runs a block sees on the matrix),
the pieces model/blocks/career_before.py adds to the horse's runs since, built by that
block's own pre_window_table:

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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from model.form_windows import run_measures  # noqa: E402
from model.freshness_features import _col  # noqa: E402
from model.race_shape import race_code  # noqa: E402

CUTOFF = "2021-01-01"
MEASURES = ("win", "plc", "nfp", "mkt", "ae")
STITCHED = ("nfp", "mkt")
LAST = 3
JUMPS = ("hurdle", "chase", "nhflat")
_BLANK = ("", "nan", "none")


def horse_key(names: pd.Series) -> pd.Series:
    """A horse as the engine keys it (form_windows, freshness): the name stripped, in lower case."""
    return names.fillna("").astype(str).str.strip().str.lower()


def _num(df: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(_col(df, name), errors="coerce").to_numpy(dtype=float)


# A verbatim copy of model/blocks/career_before.pre_window_table: the block lands with the
# table this writes (a push under model/ starts the research loop, which needs the table).

def pre_window_table(history: pd.DataFrame, cutoff: str = CUTOFF) -> pd.DataFrame:
    """TABLE's rows from a history: each horse's runs before `cutoff`, one row per horse.

    The runs a block sees on the matrix (a usable price: blocks.matrix_rows), each
    measured by run_measures on its own race as the engine measures the runs since:
    runs and runs over jumps; the sum and count of each measure's known values; the
    last LAST runs' NFP and market view, last first (unknown where a run's is); the
    highest official rating carried on the Flat and over jumps; the first and last
    run. research/queries/done/pre_window_careers.py runs this on race_results."""
    from model.blocks import matrix_rows
    date = pd.to_datetime(history["race_date"], errors="coerce")
    d = history[((date < pd.Timestamp(cutoff)).to_numpy()) & matrix_rows(history)]
    key = horse_key(d["horse_name"])
    d, key = d[~key.isin(_BLANK)], key[~key.isin(_BLANK)]
    m = run_measures(d)
    jumps = np.isin(race_code(d).to_numpy(), JUMPS)
    orr = _num(d, "official_rating")
    orr = np.where(orr > 0, orr, np.nan)
    f = pd.DataFrame({"key": key.to_numpy(), "date": pd.to_datetime(d["race_date"]).to_numpy(),
                      "time": d["race_time"].astype(str).to_numpy(), "jumps": jumps.astype(float),
                      "or_flat": np.where(jumps, np.nan, orr), "or_jumps": np.where(jumps, orr, np.nan),
                      **{k: m[k] for k in MEASURES}})
    f = f.sort_values(["key", "date", "time"], kind="mergesort").reset_index(drop=True)
    g = f.groupby("key", sort=True)
    out = pd.DataFrame({"pw_runs": g.size(), "pw_jumps_runs": g["jumps"].sum()})
    for k in MEASURES:
        out[f"pw_{k}_sum"] = g[k].sum()
        out[f"pw_{k}_n"] = g[k].count()
    back = g.cumcount(ascending=False) + 1                 # 1 = the horse's last run before the cut-off
    for j in range(1, LAST + 1):
        last = f[back == j].set_index("key")
        for k in STITCHED:
            out[f"pw_{k}_l{j}"] = last[k]
    out["pw_or_max_flat"] = g["or_flat"].max()
    out["pw_or_max_jumps"] = g["or_jumps"].max()
    out["pw_first_run"] = g["date"].min().dt.strftime("%Y-%m-%d")
    out["pw_last_run"] = g["date"].max().dt.strftime("%Y-%m-%d")
    return out.rename_axis("horse_key").reset_index()


WINDOW = ("2025-09-27", "2026-03-31")
OUT = Path("out/pre_window_careers")


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
    print(f"{len(df):,} runs before {CUTOFF}, {df['race_date'].min()} to {df['race_date'].max()}, "
          f"{horse_key(df['horse_name']).nunique():,} names")
    out = pre_window_table(df)
    path = OUT / f"pre_window_careers_{CUTOFF}.csv.gz"
    out.to_csv(path, index=False, float_format="%.9g")
    print(f"wrote {path}: {len(out):,} horses, {int(out['pw_runs'].sum()):,} runs with a usable price, "
          f"{path.stat().st_size / 1e6:.1f} MB")
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
