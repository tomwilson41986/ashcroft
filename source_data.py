"""Fetch and build the historic data for the prospective models beyond UK/IE horse racing (see sources/__init__.py).

    python source_data.py --source gbgb --fetch --build --max-minutes 300
    python source_data.py --source football,tennis --fetch --build
    python source_data.py --source all --report            # what the store holds
    python source_data.py --source gbgb --fetch --root /tmp/sources --from 2026-09-01   # a local dry run

Raw files and tables go to s3://$CAPTURE_BUCKET/sources/<source>/ (or ``--root``); nothing here touches
horse_racing.db. Run by .github/workflows/source-data.yml.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from sources import OTHER_SOURCES, SOURCES
from sources.common import Budget, Store

log = logging.getLogger("source_data")


def report(store: Store, names) -> dict:
    out = {}
    for name in names:
        files = store.listing(f"{name}/")
        raw = {k: v for k, v in files.items() if "/raw/" in k}
        out[name] = {"raw_files": len(raw), "raw_mb": round(sum(raw.values()) / 1e6, 1),
                     "tables": {k: round(v / 1e6, 2) for k, v in files.items() if k.endswith(".parquet")}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="all",
                    help=f"comma-separated, of {','.join(SOURCES + OTHER_SOURCES)}, or all ({','.join(SOURCES)})")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--from", dest="date_from", default=None, help="first day (gbgb, puntingform) or year")
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--max-minutes", type=float, default=None, help="the whole run's budget, shared in order")
    ap.add_argument("--root", default=None, help="a local folder in place of S3")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    names = SOURCES if a.source == "all" else tuple(x.strip() for x in a.source.split(",") if x.strip())
    bad = [n for n in names if n not in SOURCES + OTHER_SOURCES]
    if bad:
        ap.error(f"unknown source(s): {bad}")
    store = Store(root=a.root)
    budget = Budget(a.max_minutes)
    d_from = date.fromisoformat(a.date_from) if a.date_from and len(a.date_from) == 10 else None
    d_to = date.fromisoformat(a.date_to) if a.date_to else None
    year_from = int(a.date_from[:4]) if a.date_from else None
    results, failed = {}, 0
    for name in names:
        mod = __import__(f"sources.{name}", fromlist=["fetch", "build"])
        res = {}
        try:
            if a.fetch:
                left = None if budget.end is None else max(0.0, (budget.end - budget.clock()) / 60.0)
                kw = {"max_minutes": left}
                if name in ("gbgb", "puntingform"):
                    if d_from:
                        kw["date_from"] = d_from
                    if d_to:
                        kw["date_to"] = d_to
                elif name == "football" and year_from:
                    kw["first_season"] = year_from
                elif name in ("tennis", "tennis_odds") and year_from:
                    kw["first_year"] = year_from
                res["fetch"] = mod.fetch(store, **kw)
            if a.build:
                res["build"] = mod.build(store)
        except Exception as exc:                          # one source failing must not stop the others
            log.exception("%s failed", name)
            res["error"] = str(exc)[:300]
            failed += 1
        results[name] = res
    if a.report:
        results["_store"] = report(store, names)
    print(json.dumps(results, indent=1, default=str))
    return 1 if failed == len(names) else 0


if __name__ == "__main__":
    sys.exit(main())
