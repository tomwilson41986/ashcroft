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
    _db(early, 0.35)          # something much wider
    c, e = _run(close, "--odds-check"), _run(early, "--odds-check")
    assert "looks like a closing price" in c
    assert "NOT the same quantity as BSP" in e
    for text in (c, e):
        assert "fav agree" in text and "2025" in text
        assert "median |log ratio|" in text


def test_odds_check_recognises_a_units_difference(tmp_path):
    """A column holding decimal-minus-one is not an early price.

    Fractional odds stored as a number -- 4/1 as 4.0 rather than 5.0 -- is the
    same price in another form, and mistaking it for an early one would invent
    a closing-line measurement the database cannot support. The band table
    separates the two: a units gap tracks log((BSP-1)/BSP) and so shrinks as
    the price lengthens, where an overround gap stays flat."""
    import sqlite3

    import numpy as np

    path = tmp_path / "u.db"
    rng = np.random.default_rng(0)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE race_results (race_date TEXT, race_time TEXT, "
                 "track TEXT, horse_name TEXT, odds REAL, bfsp REAL)")
    rows = []
    for r in range(200):
        for i in range(8):
            bsp = float(np.exp(rng.normal(1.6, 0.9)))
            if bsp <= 1.05:
                bsp = 1.06
            rows.append((f"2025-0{r % 9 + 1}-01", f"{r % 12 + 1}.00", "T", f"h{i}",
                         bsp - 1.0, bsp))          # decimal minus one
    conn.executemany("INSERT INTO race_results VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    out = _run(path, "--odds-check")
    assert "NOT the same quantity as BSP" in out
    # Every runner is exactly BSP-1, so the direct test must see it.
    assert "within 5% of (BSP - 1): 1.000" in out
    # and the band table's "gap explained" should sit at ~100% in every band
    band = [l for l in out.splitlines() if l.strip().startswith(("1-2", "2-3", "8-15"))]
    assert band, out
    for line in band:
        pct = int(line.split()[-1].rstrip("%"))
        assert 90 <= pct <= 110, line


def test_odds_check_is_opt_in(tmp_path):
    p = tmp_path / "t.db"
    _db(p, 0.05)
    assert "odds vs Betfair SP" not in _run(p)
