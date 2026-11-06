"""python -m greyhound.pit: build the metrics tables, or write the registry."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

from greyhound import pit


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="the greyhound metrics database")
    ap.add_argument("--build", default=None, help="tables, comma-separated, or 'all'")
    ap.add_argument("--first-year", type=int, default=2014)
    ap.add_argument("--last", default=None, help="the last day (default yesterday)")
    ap.add_argument("--root", default=None, help="a local store (default S3)")
    ap.add_argument("--out", default="out/pit")
    ap.add_argument("--registry", action="store_true", help="write reports/greyhound_metrics_registry.md")
    a = ap.parse_args(argv)
    if a.registry:
        Path("reports/greyhound_metrics_registry.md").write_text(pit.registry_markdown())
        print(f"{len(pit.registry())} columns")
    if not a.build:
        return 0
    from greyhound.track import runs_through
    from sources.common import Store
    store = Store(root=a.root)
    names = list(pit.TABLES) if a.build == "all" else [n.strip() for n in a.build.split(",")]
    last = date.fromisoformat(a.last) if a.last else date.today() - timedelta(days=1)
    runs = runs_through(store, last, first_year=a.first_year)
    from greyhound.pit import core
    ctx = {"fetch_lag": core.raw_fetch_lag(store)}
    if "ids" in names:
        from greyhound.pit.ids import load_runners
        ctx["runners"] = load_runners(store, f"{a.first_year}-{last.year}")
    if "market" in names:
        from greyhound.model import load_prices
        ctx["prices"] = load_prices(store, f"{a.first_year}-{last.year}")
    if "weather" in names:
        from greyhound.pit.weather import load_context
        ctx.update(load_context(store))
    metas = pit.build_tables(names, runs, store, ctx)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pit_meta.json").write_text(json.dumps(metas, indent=1, default=str))
    md = [f"## Greyhound metrics tables, built to {last} from {a.first_year}", "",
          f"GBGB runs read: {len(runs):,} ({runs.race_date.min()} to {runs.race_date.max()}). Days read from GBGB within "
          f"{core.FRESH_DAYS} days of racing (trainer at the time known): "
          f"{int((ctx['fetch_lag'] <= core.FRESH_DAYS).sum())} of {len(ctx['fetch_lag'])}.", "",
          "| Table | Rows | From | To | Seconds | Columns (non-null share) |", "|---|---|---|---|---|---|"]
    for n, m in metas.items():
        nn = ", ".join(f"{c} {v:.0%}" for c, v in list(m["non_null"].items())[:6])
        md.append(f"| {n} | {m['rows']:,} | {m['from']} | {m['to']} | {m['seconds']} | {nn} |")
    md += ["", "Notes:", "```", json.dumps({n: m["notes"] for n, m in metas.items()}, indent=1, default=str)[:6000], "```"]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
