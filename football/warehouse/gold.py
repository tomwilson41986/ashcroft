"""Gold: the tables analysis and models read, built from silver by small modules.

A gold module is a file ``football/warehouse/gold_<name>.py`` (or any module listed in ``MODULES``) with:

    SPECS    the TableSpec of each table it writes (football.warehouse.schema.TableSpec)
    INPUTS   the names of the tables it reads ('silver/tick', 'gold/event', ...)
    VERSION  an integer; raise it and the module's tables are rebuilt
    SCOPE    'part' or 'all'
    COLUMNS  optional {input name: [columns]}: read only these
    build(inputs: dict[str, pyarrow.Table]) -> dict[str, pyarrow.Table]

``build`` is a pure function of Arrow tables: no store, no clock, no network. With SCOPE 'part' it is called once a
(year, month) partition with that partition of every input, and its outputs are written to the same partition; with
'all' it is called once with every partition of its inputs together (event-grain tables only: never the ticks) and
its outputs are single files. The orchestrator here does the reading, the writing, the order (a module runs after
the modules whose tables it reads) and the manifests: an output is rebuilt only when the fingerprint of its inputs
or the module's VERSION changed.

``sql`` runs DuckDB over Arrow tables in memory (times in UTC): the tool for set-based steps such as the as-of
join of prices to checkpoints.
"""

from __future__ import annotations

import importlib
import logging

import pyarrow as pa

from football.warehouse import bronze
from football.warehouse.silver import MANIFEST as SILVER_MANIFEST
from football.warehouse.store_io import (fingerprint, part_name, read_manifest, read_part, read_single, read_table,
                                         utc_now, write_manifest, write_part, write_single)

log = logging.getLogger(__name__)

#: in dependency order is not required: ``order()`` sorts them
MODULES = ["football.warehouse.gold_event", "football.warehouse.gold_checkpoint", "football.warehouse.gold_implied",
           "football.warehouse.gold_path", "football.warehouse.gold_goal", "football.warehouse.gold_inplay",
           "football.warehouse.gold_move", "football.warehouse.gold_team", "football.warehouse.gold_rating",
           "football.warehouse.club"]


def sql(query: str, **tables: pa.Table) -> pa.Table:
    """``query`` over the named Arrow tables, in DuckDB, as an Arrow table. The session's time zone is UTC."""
    import duckdb
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        for name, t in tables.items():
            con.register(name, t)
        res = con.execute(query)
        out = res.to_arrow_table() if hasattr(res, "to_arrow_table") else res.fetch_arrow_table()
    finally:
        con.close()
    return out


def load(names: list[str] | None = None) -> list:
    """The gold modules that exist, imported (a module in MODULES not yet written is skipped)."""
    out = []
    for name in names or MODULES:
        try:
            out.append(importlib.import_module(name))
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    return out


def order(modules: list) -> list:
    """The modules sorted so that each runs after the ones whose tables it reads."""
    made = {s.name: m for m in modules for s in m.SPECS}
    done, out = set(), []

    def visit(m, stack=()):
        if m.__name__ in done:
            return
        if m.__name__ in stack:
            raise ValueError(f"gold modules depend on each other in a circle: {stack + (m.__name__,)}")
        for inp in m.INPUTS:
            if inp in made and made[inp] is not m:
                visit(made[inp], stack + (m.__name__,))
        done.add(m.__name__)
        out.append(m)
    for m in modules:
        visit(m)
    return out


def _input_fp(store, table: str, name: str | None, cache: dict) -> str | None:
    """The fingerprint of one input table's partition (``name`` = 'YYYY-MM'), or of the whole table (None)."""
    key = "silver" if table.startswith("silver/") else table
    if key not in cache:
        cache[key] = read_manifest(store, SILVER_MANIFEST if key == "silver" else table)
    parts = cache[key].get("parts", {})
    if name is None:
        if "all" in parts:
            return str(parts["all"].get("fingerprint"))
        return fingerprint(f"{k}:{_one(v)}" for k, v in parts.items())
    if name in parts:
        return _one(parts[name])
    return str(parts["all"].get("fingerprint")) if "all" in parts else None


def _one(entry: dict) -> str:
    return str(entry.get("fingerprint") or f"{entry.get('bronze_fingerprint')}:{entry.get('parser_version')}")


def want(store, m, name: str | None, cache: dict) -> str | None:
    """The fingerprint a module's output must carry to be current: its VERSION and the fingerprints of its inputs
    for the partition ``name`` ('YYYY-MM'; None: the whole tables, a SCOPE 'all' module). None while an input is
    not built."""
    fps = [_input_fp(store, t, name, cache) for t in m.INPUTS]
    if any(f is None for f in fps):
        return None
    return fingerprint([f"v{m.VERSION}"] + [f"{t}:{f}" for t, f in zip(m.INPUTS, fps)])


def stale(store, modules: list[str] | None = None) -> list[str]:
    """Every gold output that is not built from its current inputs ('gold/event 2026-03', 'gold/team'): what the
    next ``build`` would write over, or add to, a table already built. A table never built, or whose inputs are
    not built yet, is not stale: it is missing."""
    out = []
    parts = bronze.partitions(store)
    for m in order(load(modules or MODULES)):
        cache: dict = {}
        mans = {s.name: read_manifest(store, s.name) for s in m.SPECS}
        for name in ([part_name(y, mo) for y, mo in parts] if m.SCOPE == "part" else [None]):
            fp = want(store, m, name, cache)
            if fp is None:
                continue
            for s in m.SPECS:
                held = mans[s.name]["parts"]
                if held and (held.get(name or "all") or {}).get("fingerprint") != fp:
                    out.append(f"{s.name} {name}" if name else s.name)
    return out


def build(store, parts: list[tuple[int, int]] | None = None, force: bool = False, modules: list[str] | None = None,
          only: list[str] | None = None) -> dict:
    """Every gold module, in order, for the partitions whose inputs changed; the report. ``only``: module names
    (their last part, e.g. 'gold_event') to run, the rest left as they are."""
    names = modules or MODULES
    if only:                                     # only the modules asked are imported: the rest are left as built
        names = [n for n in names if n.rsplit(".", 1)[-1] in only]
    mods = order(load(names))
    all_parts = bronze.partitions(store)
    todo_parts = parts if parts is not None else all_parts
    report = {"modules": {}, "failed": []}
    for m in mods:
        short = m.__name__.rsplit(".", 1)[-1]
        cache: dict = {}
        rep = {"built": [], "skipped": 0, "rows": {}}
        columns = getattr(m, "COLUMNS", {})
        try:
            if m.SCOPE == "part":
                for y, mo in todo_parts:
                    name = part_name(y, mo)
                    fp = want(store, m, name, cache)
                    if fp is None:
                        continue
                    mans = {s.name: read_manifest(store, s.name) for s in m.SPECS}
                    if not force and all((mans[s.name]["parts"].get(name) or {}).get("fingerprint") == fp
                                         for s in m.SPECS):
                        rep["skipped"] += 1
                        continue
                    inputs = {}
                    for t in m.INPUTS:
                        got = read_part(store, t, y, mo, columns.get(t))
                        if got is None:                 # an 'all' table read by a 'part' module
                            got = read_single(store, t, columns.get(t))
                        inputs[t] = got
                    if any(v is None for v in inputs.values()):
                        continue
                    out = m.build(inputs)
                    for s in m.SPECS:
                        wrote = write_part(store, s, y, mo, out[s.name])
                        mans[s.name]["parts"][name] = {"rows": wrote["rows"], "bytes": wrote["bytes"],
                                                       "fingerprint": fp, "version": m.VERSION,
                                                       "built_utc": utc_now()}
                        write_manifest(store, s.name, mans[s.name])
                        rep["rows"][s.name] = rep["rows"].get(s.name, 0) + wrote["rows"]
                    rep["built"].append(name)
                    log.info("%s %s: %s", short, name, {s.name: mans[s.name]["parts"][name]["rows"] for s in m.SPECS})
            else:
                fp = want(store, m, None, cache)
                mans = {s.name: read_manifest(store, s.name) for s in m.SPECS}
                if fp is None:
                    pass                                # an input is not built yet: nothing to build from
                elif not force and all((mans[s.name]["parts"].get("all") or {}).get("fingerprint") == fp
                                       for s in m.SPECS):
                    rep["skipped"] += 1
                else:
                    inputs = {}
                    for t in m.INPUTS:
                        got = read_table(store, t, columns.get(t), with_partition=True)
                        inputs[t] = got if got is not None else read_single(store, t, columns.get(t))
                    if all(v is not None for v in inputs.values()):
                        out = m.build(inputs)
                        for s in m.SPECS:
                            wrote = write_single(store, s, out[s.name])
                            mans[s.name]["parts"] = {"all": {"rows": wrote["rows"], "bytes": wrote["bytes"],
                                                             "fingerprint": fp, "version": m.VERSION,
                                                             "built_utc": utc_now()}}
                            write_manifest(store, s.name, mans[s.name])
                            rep["rows"][s.name] = wrote["rows"]
                        rep["built"].append("all")
                        log.info("%s: %s", short, rep["rows"])
        except Exception as exc:                          # noqa: BLE001 - one module's failure is reported, not hidden
            log.exception("%s failed", short)
            report["failed"].append({"module": short, "error": f"{type(exc).__name__}: {exc}"[:500]})
        report["modules"][short] = rep
    return report
