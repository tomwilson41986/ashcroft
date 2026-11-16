"""Partitioned parquet through the Store, and the manifests that make every build idempotent.

A table lives at ``football/warehouse/<layer>/<table>/year=YYYY/month=MM/part-0.parquet`` (hive-style, one file a
partition, rows sorted by the table's key) with ``_manifest.json`` beside it: per partition the rows, the bytes, the
fingerprint of what it was built from and the code version that built it. A build compares fingerprints and skips a
partition that is up to date. Tables without a time partition (dimensions) are a single ``part-0.parquet``.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from football.warehouse import ROOT
from football.warehouse.schema import TableSpec, conform

PART_RE = re.compile(r"year=(\d{4})/month=(\d{2})/part-(\d+)\.parquet$")


def part_key(table: str, year: int, month: int, part: int = 0) -> str:
    return f"{ROOT}/{table}/year={int(year):04d}/month={int(month):02d}/part-{part}.parquet"


def single_key(table: str) -> str:
    return f"{ROOT}/{table}/part-0.parquet"


def manifest_key(table: str) -> str:
    return f"{ROOT}/{table}/_manifest.json"


def parquet_bytes(spec: TableSpec, table: pa.Table) -> tuple[bytes, int]:
    """(the parquet file, rows) for ``table`` under the spec's contract: conformed, sorted, zstd-compressed."""
    t = conform(spec, table)
    if spec.sort and t.num_rows:
        t = t.sort_by([(c, "ascending") for c in spec.sort])
    raw = [c for c in spec.names if c in spec.uncompressed]
    buf = io.BytesIO()
    pq.write_table(
        t, buf, row_group_size=spec.row_group, write_statistics=True,
        compression={c: ("NONE" if c in raw else "ZSTD") for c in spec.names},
        compression_level={c: 6 for c in spec.names if c not in raw} or None,
        use_dictionary=[c for c in spec.names if c not in raw],
    )
    return buf.getvalue(), t.num_rows


def write_part(store, spec: TableSpec, year: int, month: int, table: pa.Table) -> dict:
    data, rows = parquet_bytes(spec, table)
    key = part_key(spec.name, year, month)
    store.put(key, data)
    return {"key": key, "rows": rows, "bytes": len(data)}


def write_single(store, spec: TableSpec, table: pa.Table) -> dict:
    data, rows = parquet_bytes(spec, table)
    key = single_key(spec.name)
    store.put(key, data)
    return {"key": key, "rows": rows, "bytes": len(data)}


def read_key(store, key: str, columns: list[str] | None = None) -> pa.Table | None:
    raw = store.get(key)
    return None if raw is None else pq.read_table(io.BytesIO(raw), columns=columns)


def read_part(store, table: str, year: int, month: int, columns: list[str] | None = None) -> pa.Table | None:
    return read_key(store, part_key(table, year, month), columns)


def read_single(store, table: str, columns: list[str] | None = None) -> pa.Table | None:
    return read_key(store, single_key(table), columns)


def list_parts(store, table: str) -> list[tuple[int, int]]:
    """The (year, month) partitions a table holds, in order."""
    out = set()
    for key in store.listing(f"{ROOT}/{table}/"):
        m = PART_RE.search(key.replace("\\", "/"))
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return sorted(out)


def read_table(store, table: str, columns: list[str] | None = None, parts: list[tuple[int, int]] | None = None,
               with_partition: bool = False) -> pa.Table | None:
    """Every partition of a table (or ``parts``) as one Arrow table; ``with_partition`` adds year and month."""
    got = []
    for y, m in (parts if parts is not None else list_parts(store, table)):
        t = read_part(store, table, y, m, columns)
        if t is None:
            continue
        if with_partition:
            t = t.append_column("year", pa.array([y] * t.num_rows, pa.int16()))
            t = t.append_column("month", pa.array([m] * t.num_rows, pa.int8()))
        got.append(t)
    return pa.concat_tables(got) if got else None


def read_manifest(store, table: str) -> dict:
    raw = store.get(manifest_key(table))
    return json.loads(raw) if raw else {"table": table, "parts": {}}


def write_manifest(store, table: str, manifest: dict) -> None:
    manifest["table"] = table
    manifest["updated_utc"] = utc_now()
    store.put(manifest_key(table), (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())


def part_name(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fingerprint(lines) -> str:
    """One hash for a set of text lines, whatever their order."""
    h = hashlib.sha256()
    for line in sorted(lines):
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()


def dataset_glob(store, table: str, partitioned: bool = True) -> str:
    """Where DuckDB reads the table from: a local glob, or the S3 one."""
    tail = "*/*/*.parquet" if partitioned else "*.parquet"
    return store.where(f"{ROOT}/{table}/{tail}")
