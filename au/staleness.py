"""The staleness check of the Australian early lay (reports/preregistration_au_early_lay.md, addendum committed in
4bc6101 before this ran).

For each Australian WIN runner: its last traded price before the T-60 mark and how many minutes before the mark that
price last changed (BASIC files carry a price when it changes, so this is the longest it can have stood). Read from the
bundle kept in S3 (au/historic.py --keep-bundle), joined to the pre-registered rule (au/lay_test.py), scored by age.

    python -m au.staleness --bundle-key sources/betfair_historic/bundles/horse_racing_basic_2024.tar
"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)
MARK_MIN = 60


def _ages(item: tuple[str, bytes]) -> list[dict] | None:
    """One stream file -> each AU WIN runner's last price before T-60 and its age in minutes at the mark."""
    import re
    name, raw = item
    try:
        text = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
        if not re.search(rb'"marketType"\s*:\s*"WIN"', text) or not re.search(rb'"countryCode"\s*:\s*"AU"', text):
            return []
        state: dict[str, dict] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            msg = json.loads(line)
            pt = msg.get("pt", 0) / 1000.0
            for mc in msg.get("mc") or []:
                mid = mc.get("id")
                st = state.setdefault(mid, {"defn": {}, "hist": {}, "inplay": None})
                d = mc.get("marketDefinition")
                if d:
                    st["defn"] = d
                    if d.get("inPlay") and st["inplay"] is None:
                        st["inplay"] = pt
                for rc in mc.get("rc") or []:
                    if "ltp" in rc:
                        st["hist"].setdefault(int(rc["id"]), []).append((pt, float(rc["ltp"])))
        rows = []
        for mid, st in state.items():
            d = st["defn"]
            if d.get("marketType") != "WIN" or d.get("countryCode") != "AU" or not d.get("marketTime"):
                continue
            off = datetime.fromisoformat(d["marketTime"].replace("Z", "+00:00")).timestamp()
            mark = off - MARK_MIN * 60
            for sid, hist in st["hist"].items():
                before = [(t, p) for t, p in hist if t < mark and (st["inplay"] is None or t < st["inplay"])]
                if not before:
                    continue
                t, p = before[-1]
                rows.append({"market_id": mid, "selection_id": sid, "ltp_t_60_seen": p,
                             "age_t_60_min": round((mark - t) / 60.0, 2),
                             "changes_last_15min": sum(1 for tt, _ in before if mark - tt <= 900)})
        return rows
    except Exception:
        return None


def stream(path: str) -> tuple[pd.DataFrame, dict]:
    rep = {"files": 0, "unreadable": 0, "rows": 0}
    frames, batch = [], []

    def members():
        try:
            with tarfile.open(path, mode="r|*") as t:
                for m in t:
                    if m.isfile() and m.name.endswith(".bz2") and "PaxHeaders" not in m.name:
                        yield m.name, t.extractfile(m).read()
        except (tarfile.ReadError, EOFError) as exc:
            rep["truncated"] = str(exc)

    with ProcessPoolExecutor() as pool:
        def flush():
            for rows in pool.map(_ages, batch, chunksize=64):
                if rows is None:
                    rep["unreadable"] += 1
                elif rows:
                    frames.append(pd.DataFrame(rows))
                    rep["rows"] += len(rows)
        for item in members():
            batch.append(item)
            rep["files"] += 1
            if len(batch) >= 4000:
                flush()
                batch = []
        if batch:
            flush()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["market_id", "selection_id"], keep="last")
    return df, rep


def score_by_age(j: pd.DataFrame, ages: pd.DataFrame) -> dict:
    from au.lay_test import PRIMARY, bets, score
    a = ages.copy()
    a["market_id"] = a["market_id"].astype(str).str.replace(r"^1\.", "", regex=True).astype("int64")
    a["selection_id"] = a["selection_id"].astype("int64")
    p = PRIMARY
    s = bets(j, p["mark"], p["model_filter"], p["lo"], p["hi"]).merge(a, on=["market_id", "selection_id"], how="left")
    rep = {"primary_bets": int(len(s)), "with_age": int(s["age_t_60_min"].notna().sum()),
           "price_agrees_with_table": round(float((s["ltp_t_60_seen"] - s["L"]).abs().lt(1e-9).mean()), 4),
           "age_quantiles_min": {str(q): round(float(s["age_t_60_min"].quantile(q)), 1)
                                 for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
    s = s[(s["ltp_t_60_seen"] - s["L"]).abs() < 1e-9]           # the age is of the very price laid
    rep["scored_bets_price_agrees"] = int(len(s))
    cuts = {"within_5min (the test)": s["age_t_60_min"] <= 5, "within_15min": s["age_t_60_min"] <= 15,
            "15_to_60min": (s["age_t_60_min"] > 15) & (s["age_t_60_min"] <= 60), "over_60min": s["age_t_60_min"] > 60}
    rep["by_age"] = {k: score(s[m], "green") for k, m in cuts.items()}
    fresh = rep["by_age"]["within_5min (the test)"]
    rep["reading"] = ("survives on fresh prices: gate B worth running"
                      if fresh.get("bets", 0) and fresh["ci90_pct"][0] > 0
                      else "the margin was in stale prices: drop the strategy without gate B")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle-key", default="sources/betfair_historic/bundles/horse_racing_basic_2024.tar")
    ap.add_argument("--dest", default="bundle.tar")
    ap.add_argument("--out-key", default="au/staleness_report.json")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from sources.common import Store
    from au.lay_test import joined
    store = Store()
    store.s3.download_file(store.bucket, a.bundle_key, a.dest)
    ages, srep = stream(a.dest)
    store.put_parquet("betfair_historic/horse_racing_win_ages_au.parquet", ages)
    rep = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "preregistration": "reports/preregistration_au_early_lay.md addendum (commit 4bc6101)", "stream": srep,
           **score_by_age(joined(store), ages)}
    store.put(a.out_key, json.dumps(rep, indent=1, default=str).encode())
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
