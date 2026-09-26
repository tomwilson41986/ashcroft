"""Careers cut at 2021: how many runners the matrix's start date shortens, and what it costs the price.

Every feature is built from race_results rows on or after the matrix's start (2021-01-01, in training
and at 06:00 alike), so a horse's career, a sire's progeny and a yard's record before 2021 do not
exist for the model: a nine-year-old chaser whose first run was in 2019 reaches it with its 2021
runs only. The database is complete from 2010 (research query db_years). This counts, for the
development window's runners (the 958's out-of-sample forecasts, iteration 68's cw arm), the runs
each horse had before 2021, and compares the forecast's error for horses whose record is cut with
the rest, within age and race code, so the comparison is not only one of older horses against
younger. Read-only; nothing after 2026-03-31 is read.
"""

from __future__ import annotations

import io
import os
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RUN, ARTIFACT, ARM = 36233732671, "research-iter68-served-944-connection-windows-84", "cw"
START = "2021-01-01"


def fetch_predictions() -> pd.DataFrame:
    import requests
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs/{RUN}/artifacts",
                     headers=auth, params={"name": ARTIFACT}, timeout=60)
    arts = r.json().get("artifacts", []) if r.ok else []
    if not arts:
        raise SystemExit(f"artifact {ARTIFACT} of run {RUN} not found ({r.status_code})")
    z = requests.get(arts[0]["archive_download_url"], headers=auth, timeout=600)
    z.raise_for_status()
    dest = Path("candidates/it68")
    zipfile.ZipFile(io.BytesIO(z.content)).extractall(dest)
    return pd.read_csv(dest / f"oos_{ARM}.csv", dtype={"race_time": str})


def main() -> None:
    o = fetch_predictions()
    o = o[o["race_date"] < "2026-04-01"]
    conn = sqlite3.connect("horse_racing.db")
    pre = pd.read_sql_query(
        "SELECT horse_name, COUNT(*) AS runs_before, MIN(race_date) AS first_run FROM race_results "
        "WHERE race_date < ? GROUP BY horse_name", conn, params=(START,))
    ages = pd.read_sql_query(
        "SELECT race_date, track, race_time, horse_name, horse_age, trainer FROM race_results "
        "WHERE race_date >= '2025-09-27' AND race_date < '2026-04-01'", conn)
    tr_pre = pd.read_sql_query(
        "SELECT trainer, COUNT(*) AS trainer_runs_before FROM race_results WHERE race_date < ? GROUP BY trainer",
        conn, params=(START,))
    tr_all = pd.read_sql_query(
        "SELECT trainer, COUNT(*) AS trainer_runs_since FROM race_results WHERE race_date >= ? "
        "AND race_date < '2025-09-27' GROUP BY trainer", conn, params=(START,))
    conn.close()
    key = ["race_date", "track", "race_time", "horse_name"]
    d = o.merge(ages, on=key, how="left").merge(pre, on="horse_name", how="left")
    d = d.merge(tr_pre, on="trainer", how="left").merge(tr_all, on="trainer", how="left")
    d["runs_before"] = d["runs_before"].fillna(0)
    d["cut"] = d["runs_before"] > 0
    d["err"] = (np.log(d["predicted_bfsp"]) - np.log(d["bfsp"])).abs()
    d["age"] = pd.to_numeric(d["horse_age"], errors="coerce").clip(upper=11)
    print(f"{len(d):,} development-window runners, {d['race_date'].min()} to {d['race_date'].max()}")
    print(f"horses with runs before {START} (their record is cut): {d['cut'].mean():.1%} of runners; "
          f"their mean runs before {START}: {d.loc[d['cut'], 'runs_before'].mean():.1f}")
    share = d["trainer_runs_before"].fillna(0) / (d["trainer_runs_before"].fillna(0)
                                                 + d["trainer_runs_since"].fillna(0)).replace(0, np.nan)
    print(f"the yard's record before {START} as a share of all its runs before the window: "
          f"median {share.median():.1%} (runner-weighted)")

    print("\nBy race code: share cut, error cut vs not")
    for code, g in d.groupby("race_code"):
        a, b = g[g["cut"]], g[~g["cut"]]
        print(f"  {str(code):14s} n={len(g):6,}  cut {g['cut'].mean():6.1%}  err cut {a['err'].mean():.4f} (n={len(a):,})"
              f"  not {b['err'].mean():.4f} (n={len(b):,})")

    print("\nWithin age and race code (cells with 100+ of each): error of cut minus not, runner-weighted")
    rows = []
    for (code, age), g in d.groupby(["race_code", "age"]):
        a, b = g[g["cut"]], g[~g["cut"]]
        if len(a) >= 100 and len(b) >= 100:
            rows.append((code, age, len(a), len(b), a["err"].mean() - b["err"].mean(),
                         a["bfsp"].median(), b["bfsp"].median()))
    t = pd.DataFrame(rows, columns=["code", "age", "n_cut", "n_not", "err_diff", "bsp_med_cut", "bsp_med_not"])
    print(t.round(4).to_string(index=False))
    if len(t):
        w = t["n_cut"] + t["n_not"]
        print(f"  weighted mean difference {np.average(t['err_diff'], weights=w):+.4f} over {int(w.sum()):,} runners")

    print("\nBy how much of the career is cut (runs before 2021, horses with any)")
    d["cut_band"] = pd.cut(d["runs_before"], [-1, 0, 3, 8, 15, 1000], labels=["0", "1-3", "4-8", "9-15", "16+"])
    print(d.groupby("cut_band", observed=True).agg(n=("err", "size"), err=("err", "mean"),
                                                   age=("age", "mean"), bsp_med=("bfsp", "median")).round(4)
          .to_string())


if __name__ == "__main__":
    main()
