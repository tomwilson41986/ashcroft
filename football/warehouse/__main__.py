"""The warehouse's command line (see football/warehouse/__init__.py).

    python -m football.warehouse ingest --zip 2026.zip --source owner_drive_2026_11_11 [--root <local folder>]
    python -m football.warehouse ingest-queue          # every bundle in football/warehouse/bundles.json, each once
    python -m football.warehouse build [--force] [--part 2026-01 ...] [--layer silver|gold|all]
    python -m football.warehouse check
    python -m football.warehouse dictionary --out docs/football_data_dictionary.md
    python -m football.warehouse catalog --out warehouse.sql
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _store(root):
    from sources.common import Store
    return Store(root=root) if root else Store()


def _parts(values):
    if not values:
        return None
    return [(int(v[:4]), int(v[5:7])) for v in values]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The racing² football data warehouse")
    ap.add_argument("--root", default=None, help="a local folder in place of the capture bucket")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest", help="a bundle's market files into bronze")
    i.add_argument("--zip", "--bundle", dest="bundle", required=True, help="a zip or tar of Historic Data files")
    i.add_argument("--source", required=True, help="the bundle's name, kept on every row")
    sub.add_parser("ingest-queue", help="every bundle listed in football/warehouse/bundles.json, each once")
    b = sub.add_parser("build", help="silver and gold for the partitions that changed")
    b.add_argument("--force", action="store_true")
    b.add_argument("--part", action="append", help="YYYY-MM (repeatable); default: every partition")
    b.add_argument("--layer", choices=["silver", "gold", "all"], default="all")
    b.add_argument("--workers", type=int, default=None)
    k = sub.add_parser("check", help="the data-quality report; exit 1 on a failed check")
    k.add_argument("--markdown", default=None, help="also write the report as a markdown table here")
    d = sub.add_parser("dictionary", help="the data dictionary, generated from the schema")
    d.add_argument("--out", default="docs/football_data_dictionary.md")
    c = sub.add_parser("catalog", help="the DuckDB views and macros as a script for the DuckDB shell")
    c.add_argument("--out", default="warehouse.sql")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = _store(a.root)
    if a.cmd == "ingest":
        from football.warehouse import bronze
        out = bronze.ingest_bundle(store, a.bundle, a.source)
    elif a.cmd == "ingest-queue":
        from football.warehouse import bronze
        out = bronze.ingest_queue(store)
    elif a.cmd == "build":
        out = {}
        if a.layer in ("silver", "all"):
            from football.warehouse import silver
            out["silver"] = silver.build(store, _parts(a.part), a.force, a.workers)
        if a.layer in ("gold", "all"):
            from football.warehouse import gold
            out["gold"] = gold.build(store, _parts(a.part), a.force)
    elif a.cmd == "check":
        from football.warehouse import quality
        out = quality.report(store)
        print(json.dumps(out, indent=1, default=str))
        if a.markdown:
            from pathlib import Path
            Path(a.markdown).write_text(quality.markdown(out))
        return 0 if out.get("ok") else 1
    elif a.cmd == "catalog":
        from football.warehouse import catalog
        out = {"written": catalog.write_sql(store, a.out)}
    else:
        from football.warehouse import dictionary
        out = {"written": dictionary.write(a.out)}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
