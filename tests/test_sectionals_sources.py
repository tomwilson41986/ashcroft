"""model/blocks/sectionals.py's readings from its two real sources (the site's table and the backend's runs in the
database), which the block tests' synthetic source does not exercise: iteration 111's sectionals arms failed at
attach on a shape error here (22 Oct)."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from model.blocks import sectionals as sc


@pytest.fixture
def sources(tmp_path, monkeypatch):
    site = pd.DataFrame({"day": ["2026-05-01", "2026-05-01"], "name_norm": ["alpha", "beta"], "track": "X", "src": "season",
                         "n_field": 8, "pos": [1, 2], "fsp": [102.0, 99.0], "fsp_rel": [1.5, -1.5], "fsp_vs_pos": [0.1, -0.1],
                         "early_pct": [0.0, 0.5], "late_rel": [-0.2, 0.2], "sl_rel": [0.1, -0.1]})
    csv = tmp_path / "sectionals.csv.gz"
    site.to_csv(csv, index=False, compression="gzip")
    db = tmp_path / "db.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sectional_runs (day TEXT, name_norm TEXT, src TEXT, n_field REAL, pos REAL, fsp REAL, "
                     "fsp_rel REAL, fsp_vs_pos REAL, early_pct REAL, late_rel REAL, sl_rel REAL)")
        conn.executemany("INSERT INTO sectional_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
            ("2026-05-01", "beta", "atr", 8, 2, 98.0, -2.5, -0.2, 0.6, 0.3, None),      # the site has beta: the site wins
            ("2025-09-01", "gamma", "atr", 10, 3, 101.0, 0.5, 0.0, 0.3, -0.1, None),    # only the backend has gamma
            ("2025-09-01", "delta", "rtv", 10, 1, 103.0, 2.0, 0.2, 0.1, None, 0.3),    # Racing TV before At The Races
            ("2025-09-01", "delta", "atr", 10, 1, 103.0, 1.0, 0.1, 0.2, -0.3, None)])
    monkeypatch.setattr(sc, "TABLE_PATH", str(csv))
    monkeypatch.setattr(sc, "DB_PATH", str(db))
    monkeypatch.setattr(sc, "SOURCE", "table")
    sc._table.cache_clear()
    sc._db_runs.cache_clear()
    return site


def test_readings_take_the_site_first_then_the_backend(sources):
    df = pd.DataFrame({"race_date": ["2026-05-01", "2026-05-01", "2025-09-01", "2025-09-01", "2025-09-01"],
                       "horse_name": ["Alpha", "Beta (IRE)", "Gamma", "Delta", "Nobody"]})
    r = sc.readings(df)
    assert list(r.columns) == sc.READINGS and len(r) == 5
    assert r.loc[0, "fsp_rel"] == 1.5 and r.loc[1, "fsp_rel"] == -1.5          # the site's rows, beta over the backend's
    assert r.loc[2, "fsp_rel"] == 0.5 and r.loc[2, "late_rel"] == -0.1          # the backend's gamma
    assert r.loc[3, "fsp_rel"] == 2.0 and r.loc[3, "sl_rel"] == 0.3             # delta: Racing TV's row
    assert np.isnan(r.loc[4]).all()


def test_readings_without_a_database_or_table(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "TABLE_PATH", str(tmp_path / "none.csv.gz"))
    monkeypatch.setattr(sc, "DB_PATH", str(tmp_path / "none.sqlite"))
    monkeypatch.setattr(sc, "SOURCE", "table")
    sc._table.cache_clear()
    sc._db_runs.cache_clear()
    r = sc.readings(pd.DataFrame({"race_date": ["2026-05-01"], "horse_name": ["Alpha"]}))
    assert r.shape == (1, len(sc.READINGS)) and np.isnan(r.to_numpy(dtype=float)).all()
