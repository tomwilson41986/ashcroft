"""Silver: bronze's raw files read into the five typed tables (football.warehouse.schema).

A month of bronze is parsed in worker processes (football.warehouse.parse) and written as one partition of
``silver/market``, ``silver/runner``, ``silver/definition``, ``silver/spell`` and ``silver/tick``. Idempotent: a
partition is rebuilt only when its bronze fingerprint or the parser's version changed (``silver/_manifest.json``),
or with ``force``.

A market that appears in more than one bronze file (Betfair files a rescheduled match under each day) is read from
its longest file only; the others are listed in the manifest as superseded, never silently dropped. Each build ends
with the partition's reconciliation: files in, markets out, files that could not be read, ticks written against the
ticks the market rows count, keys unique.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc

from football.warehouse import bronze
from football.warehouse.parse import PARSER_VERSION, parse_files
from football.warehouse.schema import BRONZE_RAW, SILVER, SILVER_MARKET, SILVER_TICK
from football.warehouse.store_io import (part_name, read_manifest, read_part, utc_now, write_manifest, write_part)

log = logging.getLogger(__name__)

MANIFEST = "silver"
CHUNK = 400                      # files a worker task


def superseded(index: pa.Table | None) -> dict[str, str]:
    """{plan path: the path that supersedes it} for every bronze file whose market is held in a longer file (ties:
    the later path)."""
    if index is None or index.num_rows == 0:
        return {}
    best: dict[str, tuple[int, str]] = {}
    rows = list(zip(index.column("market_id").to_pylist(), index.column("size").to_pylist(),
                    index.column("plan_path").to_pylist()))
    for mid, size, path in rows:
        cur = best.get(mid)
        if cur is None or (size, path) > cur:
            best[mid] = (size, path)
    return {path: best[mid][1] for mid, _size, path in rows if best[mid][1] != path}


def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def build_part(store, year: int, month: int, drop: dict[str, str], workers: int = 4, pool=None) -> dict:
    """One partition from its bronze files: the manifest entry (rows a table, failures, checks)."""
    raw = read_part(store, BRONZE_RAW.name, year, month, columns=["plan_path", "sha256", "data"])
    if raw is None:
        return {"error": "no bronze partition"}
    items = [(p, s, d) for p, s, d in zip(raw.column("plan_path").to_pylist(), raw.column("sha256").to_pylist(),
                                          raw.column("data").to_pylist()) if p not in drop]
    files_in, n_drop = raw.num_rows, raw.num_rows - len(items)
    del raw
    chunks = list(_chunks(items, CHUNK))
    if pool is not None and len(chunks) > 1:
        results = list(pool.map(parse_files, chunks))
    else:
        results = [parse_files(c) for c in chunks]
    del items, chunks
    failed = [f for _t, fails, _n in results for f in fails]
    read = sum(n for _t, _f, n in results)
    entry = {"files_in": files_in, "superseded": n_drop, "files_read": read, "files_failed": len(failed),
             "failed": failed[:50], "parser_version": PARSER_VERSION, "rows": {}, "bytes": {}, "checks": {}}
    tables = {}
    for spec in SILVER:
        t = pa.concat_tables([tabs[spec.name] for tabs, _f, _n in results]) if results else spec.schema.empty_table()
        tables[spec.name] = t
    del results
    for spec in SILVER:
        wrote = write_part(store, spec, year, month, tables[spec.name])
        entry["rows"][spec.name], entry["bytes"][spec.name] = wrote["rows"], wrote["bytes"]
    entry["checks"] = checks(tables)
    entry["built_utc"] = utc_now()
    return entry


def checks(tables: dict[str, pa.Table]) -> dict:
    """The partition's reconciliation; every value named ``*_ok`` must be true."""
    market, tick = tables[SILVER_MARKET.name], tables[SILVER_TICK.name]
    out = {}
    for spec in SILVER:
        t = tables[spec.name]
        if t.num_rows == 0:
            out[f"{spec.name}:key_unique_ok"] = True
            continue
        n_keys = t.group_by(list(spec.key)).aggregate([]).num_rows
        out[f"{spec.name}:key_unique_ok"] = bool(n_keys == t.num_rows)
    ticks_counted = (pc.sum(market.column("n_ticks")).as_py() or 0) if market.num_rows else 0
    out["ticks_match_market_counts_ok"] = bool(ticks_counted == tick.num_rows)
    if tick.num_rows:
        ltp = tick.column("ltp")
        out["ltp_min"], out["ltp_max"] = pc.min(ltp).as_py(), pc.max(ltp).as_py()
        out["ticks_outside_1.01_1000"] = int(pc.sum(pc.or_(pc.less(ltp, 1.01), pc.greater(ltp, 1000.0))).as_py())
        out["ticks_in_play"] = int(pc.sum(tick.column("in_play")).as_py())
    if market.num_rows:
        mt = market.column("market_time")
        out["market_time_min"], out["market_time_max"] = str(pc.min(mt).as_py()), str(pc.max(mt).as_py())
        out["markets_never_in_play"] = int(pc.sum(pc.is_null(market.column("in_play_pt"))).as_py())
        out["markets_without_trade"] = int(pc.sum(pc.equal(market.column("n_ticks"), 0)).as_py())
        out["ltp_invalid_left_out"] = int(pc.sum(market.column("n_ltp_invalid")).as_py() or 0)
        out["markets_by_type"] = {r["market_type"]: r["market_type_count"] for r in
                                  market.group_by("market_type").aggregate([("market_type", "count")]).to_pylist()}
    return out


def build(store, parts: list[tuple[int, int]] | None = None, force: bool = False, workers: int | None = None) -> dict:
    """Silver for every bronze partition that changed (or ``parts``, or all with ``force``); the report."""
    bman = read_manifest(store, BRONZE_RAW.name)
    sman = read_manifest(store, MANIFEST)
    todo = []
    for y, m in (parts if parts is not None else bronze.partitions(store)):
        name = part_name(y, m)
        fp = (bman["parts"].get(name) or {}).get("fingerprint")
        done = sman["parts"].get(name) or {}
        if force or done.get("bronze_fingerprint") != fp or done.get("parser_version") != PARSER_VERSION:
            todo.append((y, m, fp))
    report = {"built": [], "skipped": len((parts if parts is not None else bronze.partitions(store))) - len(todo),
              "failed_checks": []}
    if not todo:
        return report
    drop = superseded(bronze.index(store))
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) if workers > 1 else None
    try:
        for y, m, fp in todo:
            name = part_name(y, m)
            entry = build_part(store, y, m, drop, workers, pool)
            entry["bronze_fingerprint"] = fp
            entry["superseded_files"] = sorted(p for p in drop if _in_part(p, y, m))[:200]
            sman["parts"][name] = entry
            write_manifest(store, MANIFEST, sman)
            bad = [k for k, v in (entry.get("checks") or {}).items() if k.endswith("_ok") and not v]
            if bad or entry.get("error"):
                report["failed_checks"].append({"part": name, "checks": bad, "error": entry.get("error")})
            report["built"].append({"part": name, "files_in": entry.get("files_in"),
                                    "files_failed": entry.get("files_failed"), "rows": entry.get("rows")})
            log.info("silver %s: %s files, %s failed, %s ticks", name, entry.get("files_in"),
                     entry.get("files_failed"), (entry.get("rows") or {}).get(SILVER_TICK.name))
    finally:
        if pool is not None:
            pool.shutdown()
    return report


def _in_part(plan_path: str, year: int, month: int) -> bool:
    ident = bronze.plan_path(plan_path)
    return bool(ident and ident["year"] == year and ident["month"] == month)
