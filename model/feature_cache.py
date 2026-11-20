"""
Cache the built feature matrix so an experiment costs minutes, not hours.

A walk-forward evaluation does two things. It builds ~500 features over the
whole history, which on the real database takes hours, and it then fits one
gradient-boosted model per fold, which takes minutes. The feature build is
identical every time the feature code is unchanged, so paying for it on every
experiment is pure waste: it is why a single evaluation ran for five and a half
hours and was killed before it finished.

This caches the frame at the boundary between the two: everything the metrics
engine and the context builder produce, filtered to rows with a usable price,
written once as Parquet and reloaded on the next run.

The only thing that makes a cache dangerous is serving a stale one, and the way
that happens here is a change to the feature code that the cache does not
notice. So the key is a hash of the feature-producing source files themselves,
alongside a fingerprint of the database. Edit a feature and the key changes and
the cache rebuilds; edit something the feature build never imports and it does
not. The file list is walked from the import graph rather than hand-maintained,
because a hand-maintained list is a promise someone eventually forgets to keep,
and the failure is silent: features built without the new module, served as if
they had it.

What this does NOT make safe: the features must be a pure function of the
history, which is what lag safety already requires. A feature that depended on
the fold boundaries could not be cached across experiments, and would be a bug
in its own right.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where a feature build starts. Everything these reach by import is hashed.
#:
#: `train_bfsp.py` used to be the entry point, which meant every edit to the
#: trainer -- the objective, a CLI flag, a log line -- threw away a feature
#: matrix those edits could not change. The feature list and the pre-race
#: context columns moved to `model/bfsp_features.py` so the hash tracks the
#: features. The trainer is deliberately NOT here.
ENTRY_POINTS = ("model/bfsp_features.py", "model/custom_metrics.py")

#: Bumped when the meaning of a cached matrix changes for a reason the import
#: hash cannot see. v2: the categorical encoding takes a fixed vocabulary.
CACHE_VERSION = 2


def _module_to_path(name: str) -> Path | None:
    rel = Path(name.replace(".", "/"))
    for cand in (REPO_ROOT / rel.with_suffix(".py"), REPO_ROOT / rel / "__init__.py"):
        if cand.exists():
            return cand
    return None


def feature_source_files(entry_points=ENTRY_POINTS) -> list[Path]:
    """Every repo file a feature build reaches, found by walking imports.

    Derived rather than hand-listed. A hand-maintained list is a promise someone
    has to keep: add `from model.new_block import ...` to the metrics engine and
    a stale list silently keeps serving features built without it. Walking the
    import graph cannot drift."""
    seen: set[Path] = set()
    queue = [REPO_ROOT / e for e in entry_points]
    while queue:
        path = queue.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                        # pragma: no cover - not our files
            continue
        for node in ast.walk(tree):                # module-level and in-function alike
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                target = _module_to_path(name)
                if target is not None:
                    queue.append(target)
    return sorted(seen)


def feature_code_hash(entry_points=ENTRY_POINTS) -> str:
    """Hash the feature-producing sources, in a fixed order."""
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}\n".encode())
    for path in feature_source_files(entry_points):
        h.update(str(path.relative_to(REPO_ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def database_fingerprint(db_path: str) -> str:
    """Identify the data, not the file.

    Size and modification time change every time the database is re-downloaded
    from S3, which would invalidate the cache on a machine that had done nothing
    but fetch the same data again. Row count and date range do not."""
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            n, lo, hi = conn.execute(
                "SELECT COUNT(*), MIN(race_date), MAX(race_date) FROM race_results"
            ).fetchone()
        return f"{n}_{lo}_{hi}"
    except Exception as exc:                       # a missing or odd database
        log.warning("Could not fingerprint %s (%s); falling back to size", db_path, exc)
        return f"size_{os.path.getsize(db_path)}" if os.path.exists(db_path) else "absent"


def cache_key(db_path: str, start_date: str | None = None, extra: str = "") -> str:
    parts = [feature_code_hash(), database_fingerprint(db_path), str(start_date), extra]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def _paths(cache_dir: str, key: str) -> tuple[Path, Path]:
    d = Path(cache_dir)
    return d / f"features_{key}.parquet", d / f"features_{key}.json"


def load(cache_dir: str, key: str, columns: list[str] | None = None,
         float32=()) -> tuple[pd.DataFrame, dict] | None:
    """The cached frame and its metadata, or None on a miss. `columns` reads just
    those (Parquet is columnar, so the dates alone cost a second, not a minute); `float32` names float64 columns
    to hold in single precision from the start (see _read_parquet)."""
    data, meta = _paths(cache_dir, key)
    if not (data.exists() and meta.exists()):
        return None
    info = json.loads(meta.read_text())
    df = _read_parquet(data, columns=columns, float32=float32)
    log.info("Feature cache hit: %s rows x %s columns from %s",
             f"{len(df):,}", f"{len(df.columns):,}", data)
    return df, info


def release_memory() -> None:
    """Hand freed memory back to the system: Python's collector, Arrow's pool, then the C allocator's arenas (which
    keep a freed 8 GB frame mapped otherwise, where the next allocation cannot use it and the runner swaps)."""
    import gc
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:  # noqa: BLE001 - best effort
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - not glibc
        pass


def _read_parquet(path: Path, columns: list[str] | None = None, float32=()) -> pd.DataFrame:
    """The Parquet file as ``pd.read_parquet`` returns it, read a row group at a time.

    ``pd.read_parquet`` builds the whole Arrow table and then the frame: two copies of the matrix at once. On the
    history from 2018 (1.07m rows) that is where a fit's memory peaked, 16 GB of RAM full and 9 GB in swap ten
    minutes into every job (run 44473514515, fit st fold 5), and where a runner with a slow disk stopped answering:
    the fits lost with no log 47-48 minutes in (20 Nov 2026). Here each floating-point column is filled into its
    final array from one row group at a time, so the second copy is one row group, a hundredth of the frame. Every
    other column (names, dates, integers: a few dozen) is read by pandas as before. The frame is the same, column
    for column, value for value (tests/test_feature_cache.py).

    ``float32`` names float64 columns to hold in single precision from the start: the caller's promise that nothing
    reads them in double precision before it would cast them itself (evaluate_oos.py --float32)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    have = list(schema.names)
    if columns is None:
        names = have
    else:
        absent = [c for c in columns if c not in have]
        if absent:
            raise KeyError(f"{path}: no columns {absent[:5]}")
        names = list(columns)
    n = pf.metadata.num_rows
    single = set(float32 or ())
    floats: dict[str, "np.ndarray"] = {}
    for name in names:
        t = schema.field(name).type
        if pa.types.is_floating(t):
            dtype = np.float32 if (name in single and pa.types.is_float64(t)) else t.to_pandas_dtype()
            floats[name] = np.empty(n, dtype=dtype)
    others = [c for c in names if c not in floats]
    if floats:
        at = 0
        cols = list(floats)
        for i in range(pf.num_row_groups):
            tb = pf.read_row_group(i, columns=cols)
            m = tb.num_rows
            for name in cols:
                floats[name][at:at + m] = tb.column(name).to_numpy()     # a null is NaN, as pandas reads it
            at += m
            del tb
        if at != n:
            raise ValueError(f"{path}: {at} rows read, {n} in the file's metadata")
    rest = pd.read_parquet(path, columns=others) if others else None
    df = pd.DataFrame({c: (floats[c] if c in floats else rest[c].array) for c in names}, copy=False)
    del floats, rest
    release_memory()
    return df


def column_fingerprints(df: pd.DataFrame) -> dict[str, str]:
    """A content hash per column, in row order.

    Two builds from the same code and the same data should agree on every one.
    Stored with the cache so a second build can be compared against the first
    column by column: a feature whose hash moves between identical builds is
    not a pure function of the history, which is a bug in its own right (the
    live path would then serve it with noise)."""
    out = {}
    for c in df.columns:
        try:
            h = pd.util.hash_pandas_object(df[c], index=False).to_numpy()
            out[c] = hashlib.sha256(h.tobytes()).hexdigest()[:16]
        except TypeError:                          # unhashable cells (lists)
            out[c] = hashlib.sha256(df[c].astype(str).str.cat().encode()).hexdigest()[:16]
    return out


def _write_parquet(df: pd.DataFrame, path: Path, rows_per_chunk: int = 100_000) -> None:
    """The frame to Parquet in row chunks. ``df.to_parquet`` converts the whole frame to an Arrow table first, a
    second copy of a matrix that already fills most of a runner's memory (the 1.05m-row history from 2018: the
    build's peak is the save, and it is where a runner's shutdown signal came twice on 22 Oct). A chunk at a time,
    the second copy is a hundredth of the frame; the file and its contents are the same."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.Schema.from_pandas(df.iloc[:1], preserve_index=False)
    with pq.ParquetWriter(path, schema) as w:
        for start in range(0, len(df), rows_per_chunk):
            w.write_table(pa.Table.from_pandas(df.iloc[start:start + rows_per_chunk], schema=schema,
                                               preserve_index=False))


def save(df: pd.DataFrame, cache_dir: str, key: str, info: dict | None = None) -> Path:
    data, meta = _paths(cache_dir, key)
    data.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(df, data)
    payload = dict(info or {})
    payload.update({
        "key": key, "rows": int(len(df)), "columns": int(len(df.columns)),
        "feature_code_hash": feature_code_hash(),
        "built_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "bytes": data.stat().st_size,
        "column_fingerprints": column_fingerprints(df),
    })
    meta.write_text(json.dumps(payload, indent=2))
    log.info("Feature cache written: %s (%.1f MB)", data, data.stat().st_size / 1e6)
    return data


def build_or_load(builder, db_path: str, start_date: str | None = None,
                  cache_dir: str | None = None, refresh: bool = False,
                  extra: str = "", float32=()) -> tuple[pd.DataFrame, dict]:
    """Return the feature frame, from cache when the key matches.

    `builder` is called with no arguments and must return the finished frame.
    With `cache_dir` unset this is exactly the uncached path, so a caller can
    offer the cache as an option without branching."""
    if not cache_dir:
        return builder(), {"cached": False, "reason": "no cache dir"}
    key = cache_key(db_path, start_date, extra)
    if not refresh:
        hit = load(cache_dir, key, float32=float32)
        if hit is not None:
            df, info = hit
            info["cached"] = True
            return df, info
        log.info("Feature cache miss for key %s; building", key)
    else:
        log.info("Feature cache refresh forced for key %s", key)
    df = builder()
    save(df, cache_dir, key, {"db": os.path.basename(db_path), "start_date": start_date,
                              "db_fingerprint": database_fingerprint(db_path)})
    return df, {"cached": False, "key": key}
