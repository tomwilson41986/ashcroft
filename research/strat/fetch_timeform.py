"""Pull Timeform run-level data (APIData_Table2) from the Railway backend into monthly parquet files (read-only).

One GET per country per month, no paging (the route returns the whole window when no limit is given), a pause
between requests, and a month already on disk is never fetched again -- except the current month, which grows.

    python research/strat/fetch_timeform.py --out <dir> --from 2014-01 --countries GBR,IRE
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://horseracesbackend-production.up.railway.app/api/APIData_Table2"
KEEP = ["meetingDate", "courseName", "countryCode", "scheduledTimeOfRaceLocal", "raceNumber", "raceType",
        "raceSurfaceName", "distance", "going", "numberOfRunners", "raceClass", "raceTitle", "prizeFund",
        "horseCode", "horseName", "horseAge", "horseGender", "sireName", "damName", "jockeyFullName",
        "trainerFullName", "positionOfficial", "Win", "betfairWinSP", "betfairPlaceSP", "ispDecimal", "performanceRating",
        "preRaceAdjustedRating", "preRaceMasterRating", "timefigure", "distanceBeaten", "performanceSymbol",
        "draw", "officialRating"]


def month_frame(country: str, start: pd.Timestamp) -> pd.DataFrame:
    end = (start + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    r = requests.get(URL, params={"countryCode": country, "startDate": start.strftime("%Y-%m-%d"),
                                  "endDate": end, "order": "asc"}, timeout=600)
    r.raise_for_status()
    rows = r.json().get("data", [])
    df = pd.DataFrame(rows)
    for c in KEEP:
        if c not in df:
            df[c] = None
    return df[KEEP]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="start", default="2014-01")
    ap.add_argument("--to", dest="end", default=pd.Timestamp.today().strftime("%Y-%m"))
    ap.add_argument("--countries", default="GBR,IRE")
    ap.add_argument("--pause", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.today().strftime("%Y-%m")
    for m in pd.period_range(a.start, a.end, freq="M"):
        for c in a.countries.split(","):
            f = out / f"{c}_{m}.parquet"
            if f.exists() and str(m) != now:
                continue
            for attempt in range(4):
                try:
                    t0 = time.time()
                    df = month_frame(c, m.to_timestamp())
                    df.astype(str).replace({"None": None, "nan": None}).to_parquet(f, index=False)
                    print(f"{c} {m}: {len(df)} rows in {time.time() - t0:.1f}s", flush=True)
                    break
                except Exception as e:  # network / server busy: back off and try again
                    print(f"{c} {m}: attempt {attempt + 1} failed: {e}", flush=True)
                    time.sleep(10 * 2 ** attempt)
            time.sleep(a.pause)


if __name__ == "__main__":
    main()
