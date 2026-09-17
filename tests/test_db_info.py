"""The odds audit is plumbing: the verdict must come from the real database.

These tests check that the script runs, groups by year, and reports the
quantities it claims to -- not that any particular number is right. The actual
finding about `race_results.odds` is produced in CI against the S3 database,
because a synthetic frame can be made to say whatever the author expects."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _db(path, spread: float, n_races: int = 80) -> None:
    """A database whose `odds` sits `spread` log-units away from the BSP."""
    rng = np.random.default_rng(0)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE race_results (race_date TEXT, race_time TEXT, "
                 "track TEXT, horse_name TEXT, odds REAL, bfsp REAL)")
    rows = []
    for r in range(n_races):
        for i in range(8):
            bsp = float(np.exp(rng.normal(1.5, 0.6)))
            rows.append((f"2025-0{r % 9 + 1}-01", f"{r % 12 + 1}.00", "T", f"h{i}",
                         bsp * float(np.exp(rng.normal(0.0, spread))), bsp))
    conn.executemany("INSERT INTO race_results VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def _run(path, *args):
    out = subprocess.run([sys.executable, str(ROOT / "scripts/db_info.py"), str(path), *args],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_basic_line_still_prints(tmp_path):
    p = tmp_path / "t.db"
    _db(p, 0.05)
    assert "Rows: 640" in _run(p)


def test_odds_check_separates_a_closing_price_from_an_early_one(tmp_path):
    close, early = tmp_path / "c.db", tmp_path / "e.db"
    _db(close, 0.04)          # two closing prices: tight
    _db(early, 0.35)          # a morning price: wide
    c, e = _run(close, "--odds-check"), _run(early, "--odds-check")
    assert "looks like a closing price" in c
    assert "NOT a closing price" in e
    for text in (c, e):
        assert "fav agree" in text and "2025" in text
        assert "median |log ratio|" in text


def test_odds_check_is_opt_in(tmp_path):
    p = tmp_path / "t.db"
    _db(p, 0.05)
    assert "odds vs Betfair SP" not in _run(p)
