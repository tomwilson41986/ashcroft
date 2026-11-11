"""DuckDB over the warehouse in place: every table a view, and a few macros for the questions asked most.

    from football.warehouse import catalog
    con = catalog.connect()                      # the capture bucket (needs AWS credentials in the environment)
    con = catalog.connect(Store(root="wh"))      # a local copy
    con.sql("SELECT result, count(*) FROM event GROUP BY 1").show()

    python -m football.warehouse catalog --out warehouse.sql     # the same views as a script for the DuckDB shell

Views are named after the table's last part (``tick``, ``event``, ``checkpoint``, ...; ``bronze_index`` for
bronze/index). Macros:

- ``price_asof(market_id, selection_id, handicap, ts)``: the last traded price at or before a moment;
- ``match_ticks(event_id)``: a match's match-odds prices in time order with each runner's role;
- ``latest_rating()``: each team's latest market-implied rating (once gold/team_rating is built).
"""

from __future__ import annotations

from football.warehouse import ROOT

MACROS = [
    """CREATE OR REPLACE MACRO price_asof(mid, sid, hc, ts) AS (
           SELECT ltp FROM tick WHERE market_id = mid AND selection_id = sid AND handicap = hc AND pt <= ts
           ORDER BY pt DESC, seq DESC LIMIT 1)""",
    """CREATE OR REPLACE MACRO match_ticks(eid) AS TABLE
           SELECT t.pt, t.t_off_s, t.t_inplay_s, t.in_play, t.status, r.role, r.runner_name, t.ltp
           FROM tick t JOIN runner r USING (market_id, selection_id, handicap)
           WHERE t.market_id = (SELECT mo_market_id FROM event WHERE event_id = eid) ORDER BY t.pt, t.seq""",
]
RATING_MACRO = """CREATE OR REPLACE MACRO latest_rating() AS TABLE
           SELECT * FROM team_rating QUALIFY row_number() OVER (PARTITION BY team_id ORDER BY kickoff DESC) = 1"""


def tables(store) -> dict[str, tuple[str, bool]]:
    """{view name: (table name, partitioned)} for every table the store holds."""
    out: dict[str, tuple[str, bool]] = {}
    for key in store.listing(f"{ROOT}/"):
        k = key.replace("\\", "/")
        if not k.endswith(".parquet"):
            continue
        rel = k[len(ROOT) + 1:]
        parts = rel.split("/")
        partitioned = any(p.startswith("year=") for p in parts)
        table = "/".join(p for p in parts[:-1] if "=" not in p)
        if table == "bronze/raw":
            continue                                        # the raw bytes: not a table to scan by accident
        view = table.split("/")[-1] if not table.startswith("bronze/") else table.replace("/", "_")
        out[view] = (table, partitioned)
    return out


def statements(store) -> list[str]:
    out = ["SET TimeZone='UTC'"]
    found = tables(store)
    for view, (table, partitioned) in sorted(found.items()):
        glob = store.where(f"{ROOT}/{table}/" + ("*/*/*.parquet" if partitioned else "*.parquet"))
        hive = ", hive_partitioning = true" if partitioned else ""
        out.append(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{glob}'{hive})")
    if {"tick", "runner", "event"} <= set(found):
        out += MACROS
    if "team_rating" in found:
        out.append(RATING_MACRO)
    return out


def connect(store=None, database: str = ":memory:"):
    """A DuckDB connection with the warehouse's views and macros. ``store``: sources.common.Store (default: the
    capture bucket, read through DuckDB's httpfs with the environment's AWS credentials)."""
    import duckdb
    if store is None:
        from sources.common import Store
        store = Store()
    con = duckdb.connect(database)
    if store.root is None:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        con.execute("CREATE OR REPLACE SECRET warehouse (TYPE s3, PROVIDER credential_chain)")
    for sql in statements(store):
        con.execute(sql)
    return con


def write_sql(store, path: str) -> str:
    from pathlib import Path
    head = ["-- racing² football data warehouse: views and macros for the DuckDB shell (duckdb -init <this file>)"]
    if store.root is None:
        head += ["INSTALL httpfs;", "LOAD httpfs;",
                 "CREATE OR REPLACE SECRET warehouse (TYPE s3, PROVIDER credential_chain);"]
    Path(path).write_text("\n".join(head + [s + ";" for s in statements(store)]) + "\n")
    return path
