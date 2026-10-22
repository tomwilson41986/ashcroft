#!/usr/bin/env python3
"""Timeform's race-level pace readings as one table, a row a UK/IE Flat and all-weather run (the owner, 22 Oct: the
Timeform database as a resource for the model).

The yearly Timeform archives (tomwilson41986/racing-speed-figures, data/raw/timeform_<year>.7z, 2015 - 15 Feb 2026)
carry three race-level fields the backend's API drops before our nightly sync (blandford_sync.py) can read them:

    ipHintsOverallPace       how the race was run, as Timeform read it afterwards: Very Weak, Weak, Even, Strong,
                             Very Strong, Extreme (the pace's strength) or Uncontested, Weakly Contested, Contested,
                             Strongly Contested (how it was contested); from 2016, every run from 2017
    paceMapsIdealPosition    Timeform's pre-race pace map: five probabilities over the ideal early position for the
                             race (front .. rear); 2017 - 2023 on most runs, 2024 - 2025 on two thirds
    winnerSectional, distanceSectional, finishingTime
                             the winner's closing sectional: the race's finishing speed as a % of its average speed

Each is a reading about the race, not the horse: for a horse's later runs it says what kind of races it has met
(model/blocks/tf_pace.py). The table holds, per run: day, the normalised horse name, the pace strength (0 Very Weak
.. 5 Extreme), the contest (0 Uncontested .. 3 Strongly Contested), the race's finishing speed %, the pace map's
expected ideal position (0 front .. 4 rear) and its front share, the horse's finishing position as a share of the
field (0 the winner) and the field.

Usage (the archives extracted first, e.g. with py7zr):
    python scripts/build_timeform_pace.py --csv-dir /path/with/timeform_<year>.csv --out data/external/timeform_pace.csv.gz
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from betfair_prices import normalise_horse  # noqa: E402

STRENGTH = {"very weak": 0, "weak": 1, "even": 2, "strong": 3, "very strong": 4, "extreme": 5}
CONTEST = {"uncontested": 0, "weakly contested": 1, "contested": 2, "strongly contested": 3}
COLS = ["meetingDate", "countryCode", "raceType", "horseName", "positionOfficial", "numberOfRunners",
        "ipHintsOverallPace", "paceMapsIdealPosition", "finishingTime", "winnerSectional", "distanceSectional",
        "distanceFurlongs"]


def ideal(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """The pace map's expected ideal position (0 front .. 4 rear) and its front share."""
    exp, front = np.full(len(s), np.nan), np.full(len(s), np.nan)
    for i, v in enumerate(s.to_numpy()):
        if not isinstance(v, str):
            continue
        try:
            p = np.array([float(x) for x in v.split(",")], float)
        except ValueError:
            continue
        if len(p) == 5 and p.sum() > 0:
            p = p / p.sum()
            exp[i] = float((p * np.arange(5)).sum())
            front[i] = float(p[0])
    return exp, front


def build(csv_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(csv_dir, "**", "timeform_*.csv"), recursive=True)):
        d = pd.read_csv(f, usecols=lambda c: c in COLS, low_memory=False)
        d = d[d["countryCode"].isin(["GBR", "IRE"])]
        if "raceType" in d.columns:
            d = d[d["raceType"].fillna("Flat").eq("Flat")]
        frames.append(d)
        print(f"{os.path.basename(f)}: {len(d):,} UK/IE runs")
    d = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"day": pd.to_datetime(d["meetingDate"], errors="coerce").dt.strftime("%Y-%m-%d"),
                        "name_norm": d["horseName"].map(normalise_horse)})
    pace = d["ipHintsOverallPace"].fillna("").astype(str).str.strip().str.lower()
    out["pace"] = pace.map(STRENGTH)
    out["contest"] = pace.map(CONTEST)
    fin = pd.to_numeric(d["finishingTime"], errors="coerce")
    ws = pd.to_numeric(d["winnerSectional"], errors="coerce")
    ds = pd.to_numeric(d["distanceSectional"], errors="coerce")
    dist = pd.to_numeric(d["distanceFurlongs"], errors="coerce")
    ok = (fin > 0) & (ws > 0) & (ds > 0) & (dist > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["race_fsp"] = np.where(ok, 100.0 * (ds / ws) / (dist / fin), np.nan)
    out["ideal_exp"], out["ideal_front"] = ideal(d["paceMapsIdealPosition"])
    pos = pd.to_numeric(d["positionOfficial"], errors="coerce")
    n = pd.to_numeric(d["numberOfRunners"], errors="coerce")
    out["pos_share"] = np.where((pos > 0) & (n > 1), (pos - 1) / (n - 1), np.nan)
    out["n_field"] = n
    has = out[["pace", "contest", "race_fsp", "ideal_exp"]].notna().any(axis=1)
    out = out[has & out["day"].notna() & (out["name_norm"].str.len() > 0)]
    out = out.drop_duplicates(["day", "name_norm"], keep=False)        # a name twice on a day: neither
    for c in ("race_fsp", "ideal_exp", "ideal_front", "pos_share"):
        out[c] = out[c].round(4)
    return out.sort_values(["day", "name_norm"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", default="data/external/timeform_pace.csv.gz")
    a = ap.parse_args(argv)
    t = build(a.csv_dir)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_csv(a.out, index=False, compression="gzip")
    print(f"wrote {a.out}: {len(t):,} runs {t.day.min()} .. {t.day.max()}, {os.path.getsize(a.out) / 1e6:.1f} MB")
    print("present:", {c: f"{t[c].notna().mean():.0%}" for c in ("pace", "contest", "race_fsp", "ideal_exp")})
    print("pace strength:", t.pace.value_counts(dropna=False).sort_index().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
