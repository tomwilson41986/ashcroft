"""The feature cache has exactly one way to be dangerous: serving a stale matrix.

These tests are about that. A cache that is merely slow costs time; a cache that
hands back features built by code you have since changed costs you a wrong
answer that looks right.
"""

import sqlite3
import textwrap

import pandas as pd
import pytest

from model import feature_cache as fc


def _db(path, rows=3, last="2026-01-02"):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE race_results (race_date TEXT, horse_name TEXT)")
    conn.executemany("INSERT INTO race_results VALUES (?, ?)",
                     [("2025-01-01", f"h{i}") for i in range(rows - 1)] + [(last, "last")])
    conn.commit(); conn.close()
    return str(path)


def _tree(root, files):
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body))


# --- the source list ------------------------------------------------------

def test_the_closure_follows_imports_including_inside_functions(tmp_path, monkeypatch):
    _tree(tmp_path, {
        "entry.py": "from model.a import thing\n",
        "model/__init__.py": "",
        "model/a.py": "import model.b\n",
        "model/b.py": "def f():\n    from model.c import deep\n",   # imported lazily
        "model/c.py": "deep = 1\n",
        "model/unrelated.py": "x = 1\n",
    })
    monkeypatch.setattr(fc, "REPO_ROOT", tmp_path)
    found = {p.relative_to(tmp_path).as_posix() for p in fc.feature_source_files(("entry.py",))}
    assert found == {"entry.py", "model/a.py", "model/b.py", "model/c.py"}
    assert "model/unrelated.py" not in found      # not reachable, not hashed


def test_a_new_feature_module_cannot_escape_the_hash(tmp_path, monkeypatch):
    """The failure a hand-maintained list produces, and this one must not."""
    _tree(tmp_path, {"entry.py": "from model.a import thing\n",
                     "model/__init__.py": "", "model/a.py": "thing = 1\n"})
    monkeypatch.setattr(fc, "REPO_ROOT", tmp_path)
    entry = ("entry.py",)
    before = fc.feature_code_hash(entry)

    (tmp_path / "model" / "new_block.py").write_text("feature = 2\n")
    assert fc.feature_code_hash(entry) == before, "an unimported file must not invalidate"

    (tmp_path / "model" / "a.py").write_text("from model.new_block import feature\nthing = 1\n")
    assert fc.feature_code_hash(entry) != before, "importing it must invalidate"

    after = fc.feature_code_hash(entry)
    (tmp_path / "model" / "new_block.py").write_text("feature = 3\n")
    assert fc.feature_code_hash(entry) != after, "editing a reached file must invalidate"


def test_the_real_closure_covers_the_feature_modules():
    rel = {p.relative_to(fc.REPO_ROOT).as_posix() for p in fc.feature_source_files()}
    for must in ("model/custom_metrics.py", "model/pace_metrics.py", "model/draw_metrics.py",
                 "model/lagsafe.py", "train_bfsp.py"):
        assert must in rel, f"{must} is not in the hashed set"
    assert fc.feature_code_hash() == fc.feature_code_hash()      # deterministic


# --- the database fingerprint --------------------------------------------

def test_the_fingerprint_tracks_the_data_not_the_file(tmp_path):
    """Re-downloading the same database from S3 must not invalidate the cache."""
    import os
    import time

    a, b = _db(tmp_path / "a.db"), _db(tmp_path / "b.db")
    os.utime(b, (time.time() + 10_000, time.time() + 10_000))
    assert os.path.getmtime(a) != os.path.getmtime(b)
    assert fc.database_fingerprint(a) == fc.database_fingerprint(b)

    grown = _db(tmp_path / "c.db", rows=4)
    assert fc.database_fingerprint(grown) != fc.database_fingerprint(a)
    later = _db(tmp_path / "d.db", last="2026-06-01")
    assert fc.database_fingerprint(later) != fc.database_fingerprint(a)


def test_a_missing_database_does_not_raise(tmp_path):
    assert fc.database_fingerprint(str(tmp_path / "nope.db")) == "absent"


# --- build_or_load --------------------------------------------------------

def test_the_builder_runs_once_and_then_never(tmp_path):
    db = _db(tmp_path / "h.db")
    calls = []

    def builder():
        calls.append(1)
        return pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})

    first, info = fc.build_or_load(builder, db, cache_dir=str(tmp_path / "cache"))
    assert len(calls) == 1 and info["cached"] is False

    second, info2 = fc.build_or_load(builder, db, cache_dir=str(tmp_path / "cache"))
    assert len(calls) == 1, "a cache hit must not rebuild"
    assert info2["cached"] is True
    pd.testing.assert_frame_equal(first, second)
    assert info2["rows"] == 2 and info2["feature_code_hash"] == fc.feature_code_hash()


def test_refresh_rebuilds_and_a_changed_database_misses(tmp_path):
    db = _db(tmp_path / "h.db")
    calls = []

    def builder():
        calls.append(1)
        return pd.DataFrame({"a": [1.0]})

    fc.build_or_load(builder, db, cache_dir=str(tmp_path / "c"))
    fc.build_or_load(builder, db, cache_dir=str(tmp_path / "c"), refresh=True)
    assert len(calls) == 2

    bigger = _db(tmp_path / "h2.db", rows=9)
    fc.build_or_load(builder, bigger, cache_dir=str(tmp_path / "c"))
    assert len(calls) == 3, "more data must not reuse the old matrix"


def test_no_cache_dir_is_the_uncached_path(tmp_path):
    calls = []
    db = _db(tmp_path / "h.db")
    for _ in range(2):
        fc.build_or_load(lambda: (calls.append(1), pd.DataFrame({"a": [1]}))[1], db)
    assert len(calls) == 2, "with no cache dir every call must rebuild"
    assert not list(tmp_path.rglob("features_*")), "and nothing may be written"


def test_start_date_is_part_of_the_key(tmp_path):
    db = _db(tmp_path / "h.db")
    assert fc.cache_key(db, "2021-01-01") != fc.cache_key(db, "2023-06-01")
