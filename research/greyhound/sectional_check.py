"""Does GreyhoundAPI hold GB sectionals GBGB does not? (the owner, 17 Nov 2026: "Check GB. Are there more
sectionals?").

GBGB's results (sources/gbgb.py, the model's source) and GreyhoundAPI's GB results (sources/greyhoundapi.py,
``runs_GB_<year>.parquet``) are matched runner by runner on the local day, the canonical track
(greyhound/data.py ``canonical_track``), the off time to the minute and the trap (the same key as
greyhound/pit/irish.py ``dog_map``). For each year: matched runners, sectional held by both, by GBGB only, by the API
only, by neither; where both hold one, the share equal to 0.01 s and the mean absolute difference; the same for the
run time. Read-only, S3 only; writes out/sectional_check/{report.json, report.md}.

    python -m research.greyhound.sectional_check --years 2014-2026
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def keyed_api(api: pd.DataFrame) -> pd.DataFrame:
    from greyhound.data import canonical_track
    a = api.copy()
    s = a.start_local.astype(str)
    a["day"], a["hhmm"] = s.str[:10], s.str[11:16]
    a["trk"] = canonical_track(a.track.astype(str)).str.lower()
    a["trap"] = pd.to_numeric(a.trap, errors="coerce")
    a["api_sec"] = pd.to_numeric(a.sectional_s, errors="coerce")
    a["api_time"] = pd.to_numeric(a.run_time_s, errors="coerce")
    return a[["day", "hhmm", "trk", "trap", "api_sec", "api_time"]]


def keyed_gbgb(runs: pd.DataFrame) -> pd.DataFrame:
    from greyhound.data import canonical_track
    g = runs.copy()
    g["day"] = g.race_date.astype(str).str[:10]
    g["hhmm"] = g.race_time.astype(str).str[:5]
    g["trk"] = canonical_track(g.track.astype(str)).str.lower()
    g["trap"] = pd.to_numeric(g.trap, errors="coerce")
    g["gb_sec"] = pd.to_numeric(g.sectional, errors="coerce")
    g["gb_time"] = pd.to_numeric(g.run_time, errors="coerce")
    return g[["day", "hhmm", "trk", "trap", "gb_sec", "gb_time"]]


def compare(a: pd.DataFrame, g: pd.DataFrame) -> list[dict]:
    k = ["day", "hhmm", "trk", "trap"]
    a = a.drop_duplicates(k)
    g = g.drop_duplicates(k)
    m = g.merge(a, on=k, how="outer", indicator=True)
    m["year"] = m.day.str[:4]
    rows = []
    for y, x in m.groupby("year"):
        both = x[x._merge == "both"]
        gs, asx = both.gb_sec.gt(0), both.api_sec.gt(0)
        b = both[gs & asx]
        gt, at = both.gb_time.gt(0), both.api_time.gt(0)
        bt = both[gt & at]
        rows.append({
            "year": y, "gbgb_runners": int((x._merge != "right_only").sum()), "api_runners": int((x._merge != "left_only").sum()),
            "matched": int(len(both)), "match_share_of_gbgb": round(len(both) / max(int((x._merge != "right_only").sum()), 1), 4),
            "sec_both": int((gs & asx).sum()), "sec_gbgb_only": int((gs & ~asx).sum()),
            "sec_api_only": int((~gs & asx).sum()), "sec_neither": int((~gs & ~asx).sum()),
            "sec_gbgb_share": round(float(gs.mean()), 4) if len(both) else None,
            "sec_api_share": round(float(asx.mean()), 4) if len(both) else None,
            "sec_equal_share": round(float(((b.gb_sec - b.api_sec).abs() <= 0.01).mean()), 4) if len(b) else None,
            "sec_mean_abs_diff": round(float((b.gb_sec - b.api_sec).abs().mean()), 4) if len(b) else None,
            "time_equal_share": round(float(((bt.gb_time - bt.api_time).abs() <= 0.01).mean()), 4) if len(bt) else None,
            "time_api_only": int((~gt & at).sum()),
            "api_unmatched_with_sec": int(((x._merge == "right_only") & x.api_sec.gt(0)).sum()),
        })
    return rows


def report_md(rows: list[dict]) -> str:
    L = ["# GB sectionals: GBGB against GreyhoundAPI", "",
         "Runners matched on the local day, canonical track, off time to the minute and trap.", "",
         "| Year | GBGB | API | Matched | Both | GBGB only | API only | Neither | GBGB % | API % | Equal (0.01 s) | Mean abs diff | API only, unmatched, with a sectional |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['year']} | {r['gbgb_runners']:,} | {r['api_runners']:,} | {r['matched']:,} | {r['sec_both']:,} | "
                 f"{r['sec_gbgb_only']:,} | {r['sec_api_only']:,} | {r['sec_neither']:,} | {r['sec_gbgb_share']} | "
                 f"{r['sec_api_share']} | {r['sec_equal_share']} | {r['sec_mean_abs_diff']} | {r['api_unmatched_with_sec']:,} |")
    tot = {k: sum(r[k] for r in rows) for k in ("matched", "sec_both", "sec_gbgb_only", "sec_api_only", "sec_neither",
                                                 "api_unmatched_with_sec")}
    L += ["", f"All years: {tot['matched']:,} matched; sectional held by both {tot['sec_both']:,}, GBGB only "
          f"{tot['sec_gbgb_only']:,}, the API only {tot['sec_api_only']:,}, neither {tot['sec_neither']:,}; API runners "
          f"not matched to GBGB with a sectional {tot['api_unmatched_with_sec']:,}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from greyhound.data import load_runs
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", default="2014-2026")
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="out/sectional_check")
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.years.split("-"))
    store = Store(root=a.store)
    rows = []
    for y in range(lo, hi + 1):
        api = store.get_parquet(f"greyhoundapi/runs_GB_{y}.parquet")
        if api is None or not len(api):
            log.info("%d: no data found (API)", y)
            continue
        g = load_runs(store, years=[y])
        if g is None or not len(g):
            log.info("%d: no data found (GBGB)", y)
            continue
        rows += compare(keyed_api(api), keyed_gbgb(g))
        log.info("%d done", y)
    rows = [r for r in rows if re.fullmatch(r"\d{4}", str(r["year"])) and lo <= int(r["year"]) <= hi]
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "report.json").write_text(json.dumps(rows, indent=1))
    md = report_md(rows) if rows else "no data found\n"
    (o / "report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
