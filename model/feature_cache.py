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
ENTRY_POINTS = ("train_bfsp.py", "model/custom_metrics.py")

CACHE_VERSION = 1


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


def load(cache_dir: str, key: str) -> tuple[pd.DataFrame, dict] | None:
    data, meta = _paths(cache_dir, key)
    if not (data.exists() and meta.exists()):
        return None
    info = json.loads(meta.read_text())
    df = pd.read_parquet(data)
    log.info("Feature cache hit: %s rows x %s columns from %s",
             f"{len(df):,}", f"{len(df.columns):,}", data)
    return df, info


def save(df: pd.DataFrame, cache_dir: str, key: str, info: dict | None = None) -> Path:
    data, meta = _paths(cache_dir, key)
    data.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data, index=False)
    payload = dict(info or {})
    payload.update({
        "key": key, "rows": int(len(df)), "columns": int(len(df.columns)),
        "feature_code_hash": feature_code_hash(),
        "built_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "bytes": data.stat().st_size,
    })
    meta.write_text(json.dumps(payload, indent=2))
    log.info("Feature cache written: %s (%.1f MB)", data, data.stat().st_size / 1e6)
    return data


def build_or_load(builder, db_path: str, start_date: str | None = None,
                  cache_dir: str | None = None, refresh: bool = False,
                  extra: str = "") -> tuple[pd.DataFrame, dict]:
    """Return the feature frame, from cache when the key matches.

    `builder` is called with no arguments and must return the finished frame.
    With `cache_dir` unset this is exactly the uncached path, so a caller can
    offer the cache as an option without branching."""
    if not cache_dir:
        return builder(), {"cached": False, "reason": "no cache dir"}
    key = cache_key(db_path, start_date, extra)
    if not refresh:
        hit = load(cache_dir, key)
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
