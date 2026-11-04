"""Upsert the Cricsheet tables on their keys, so a rerun of the same data adds zero rows.

SQLite for local runs and tests; PostgreSQL (Aurora) by URL once it exists, with the driver ``psycopg`` installed
there (not pinned here: no Aurora yet). The URL comes from the caller or ``CRICKET_DB_URL``, never from code.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = Path(__file__).with_name("schema.sql")
KEYS = {
    "matches": ["match_id"],
    "innings": ["match_id", "innings"],
    "deliveries": ["match_id", "innings", "over", "ball", "repeat"],
    "players": ["match_id", "team", "player"],
    "venues": ["venue", "city"],
    "people_register": ["identifier"],
    "people_names": ["identifier", "name"],
}


def connect(url: str | None = None):
    url = url or os.getenv("CRICKET_DB_URL")
    if not url:
        raise SystemExit("no database URL: pass --db or set CRICKET_DB_URL")
    if url.startswith("sqlite:///"):
        con = sqlite3.connect(url[len("sqlite:///"):])
        con.execute("PRAGMA journal_mode=WAL")
    elif url.startswith(("postgresql://", "postgres://")):
        import psycopg  # installed on the Aurora host only
        con = psycopg.connect(url)
    else:
        raise SystemExit(f"unsupported database URL scheme: {url.split('://')[0]}")
    _create(con)
    return con


def _is_sqlite(con) -> bool:
    return isinstance(con, sqlite3.Connection)


def _create(con) -> None:
    cur = con.cursor()
    for stmt in [s.strip() for s in re.sub(r"--[^\n]*", "", SCHEMA.read_text()).split(";") if s.strip()]:
        cur.execute(stmt)
    con.commit()


def _q(name: str) -> str:
    return f'"{name}"'


def count(con, table: str) -> int:
    cur = con.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def upsert(con, table: str, df: pd.DataFrame, batch: int = 50_000) -> int:
    """Insert or update each row on the table's key; returns the rows the table gained."""
    if df.empty:
        return 0
    keys = KEYS[table]
    cur = con.cursor()
    cur.execute(f"SELECT * FROM {table} LIMIT 0")
    cols = [c for c in (d[0] for d in cur.description) if c in df.columns]
    ph = "?" if _is_sqlite(con) else "%s"
    upd = [c for c in cols if c not in keys]
    sql = (f"INSERT INTO {table} ({', '.join(map(_q, cols))}) VALUES ({', '.join([ph] * len(cols))}) "
           f"ON CONFLICT ({', '.join(map(_q, keys))}) DO "
           + (f"UPDATE SET {', '.join(f'{_q(c)}=excluded.{_q(c)}' for c in upd)}" if upd else "NOTHING"))
    before = count(con, table)
    for start in range(0, len(df), batch):   # converted a batch at a time: 12M deliveries as objects will not fit
        sub = df[cols].iloc[start:start + batch]
        sub = sub.astype(object).where(sub.notna(), None)
        cur.executemany(sql, [tuple(_py(v) for v in r) for r in sub.itertuples(index=False)])
    con.commit()
    return count(con, table) - before


def _py(v):
    """numpy scalars to plain Python for the DB-API drivers."""
    return v.item() if hasattr(v, "item") else v


def upsert_tables(con, tables: dict[str, pd.DataFrame]) -> tuple[dict[str, int], dict[str, int]]:
    added = {name: upsert(con, name, tables[name]) for name in KEYS if name in tables}
    held = {name: count(con, name) for name in KEYS}
    return added, held
