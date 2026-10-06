"""The two blocks from horseracebase's system builder, by hand (made-up races and prices; code paths only)."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from model.blocks import hrb_extras, inrunning


def _runs(rows):
    df = pd.DataFrame(rows)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["raceid"] = df["race_date"].dt.strftime("%Y-%m-%d") + "|" + df["track"] + "|" + df["race_time"]
    return df


@pytest.fixture
def price_db(tmp_path, monkeypatch):
    db = tmp_path / "prices.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE betfair_prices (market_type TEXT, race_date TEXT, horse_norm TEXT, bsp REAL, "
                     "ipmin REAL, ipmax REAL)")
        conn.executemany("INSERT INTO betfair_prices VALUES (?, ?, ?, ?, ?, ?)", [
            ("win", "2026-01-05", "alpha", 5.0, 1.5, 20.0),       # beaten, traded at 1.5
            ("win", "2026-01-20", "alpha", 4.0, 1.01, 6.0),       # won
            ("place", "2026-01-20", "alpha", 1.8, 1.1, 3.0),      # the place market is not read
            ("win", "2026-01-05", "beta", 8.0, 3.0, 30.0),        # a name twice on one day: neither joins
            ("win", "2026-01-05", "beta", 9.0, 2.0, 40.0),
        ])
    monkeypatch.setattr(inrunning, "DB_PATH", str(db))
    monkeypatch.setattr(inrunning, "SOURCE", "db")
    inrunning._price_table.cache_clear()
    yield db
    inrunning._price_table.cache_clear()


def test_inrunning_reads_earlier_runs_only(price_db):
    df = _runs([
        {"race_date": "2026-01-05", "race_time": "2.00", "track": "York", "horse_name": "Alpha (IRE)",
         "placing_numerical": 3, "bfsp": 5.1},
        {"race_date": "2026-01-20", "race_time": "3.00", "track": "York", "horse_name": "Alpha (IRE)",
         "placing_numerical": 1, "bfsp": 4.1},
        {"race_date": "2026-02-01", "race_time": "4.00", "track": "York", "horse_name": "Alpha (IRE)",
         "placing_numerical": np.nan, "bfsp": np.nan},             # today's card
        {"race_date": "2026-01-05", "race_time": "2.00", "track": "York", "horse_name": "Beta",
         "placing_numerical": 2, "bfsp": 8.0},
        {"race_date": "2026-01-20", "race_time": "3.00", "track": "York", "horse_name": "Beta",
         "placing_numerical": 4, "bfsp": 9.0},
    ])
    out = inrunning.build(df)
    first, second, card = out.iloc[0], out.iloc[1], out.iloc[2]
    assert np.isnan(first["ir_low_l1"]) and first["ir_runs"] == 0
    assert second["ir_low_l1"] == pytest.approx(np.log(1.5)) and second["ir_runs"] == 1
    assert second["ir_lowr_l1"] == pytest.approx(np.log(1.5 / 5.0))
    assert second["ir_highr_l1"] == pytest.approx(np.log(20.0 / 5.0))
    assert second["ir_short_car"] == pytest.approx(1.0)              # its one beaten run traded at 2.0 or less
    assert card["ir_low_l1"] == pytest.approx(np.log(1.01)) and card["ir_runs"] == 2
    assert card["ir_low_m3"] == pytest.approx((np.log(1.5) + np.log(1.01)) / 2)
    assert card["ir_short_car"] == pytest.approx(1.0)                # the win is not a beaten run
    assert out.iloc[4]["ir_runs"] == 0 and np.isnan(out.iloc[4]["ir_low_l1"])   # Beta's first day joined nothing


def test_inrunning_without_a_table_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(inrunning, "DB_PATH", str(tmp_path / "missing.db"))
    monkeypatch.setattr(inrunning, "SOURCE", "db")
    inrunning._price_table.cache_clear()
    df = _runs([{"race_date": "2026-01-05", "race_time": "2.00", "track": "York", "horse_name": "Alpha",
                 "placing_numerical": 1, "bfsp": 3.0}])
    out = inrunning.build(df)
    assert out[inrunning.FEATURES[:-1]].isna().all().all()
    inrunning._price_table.cache_clear()


def test_hrb_extras_by_hand():
    rows = []
    # race 1 (1 Jan 2025, Ascot 6f): A wins at 8.0 in a field of 4, B second, one pulled up
    for h, pos, bsp in (("A", 1, 8.0), ("B", 2, 3.0), ("C", 3, 4.0), ("D", np.nan, 20.0)):
        rows.append({"race_date": "2025-01-01", "race_time": "2.00", "track": "Ascot", "horse_name": h,
                     "jockey_name": "J Smith" if h == "A" else "P Jones", "trainer": "Mrs K Smith" if h == "A"
                     else "T Brown", "number_of_runners": 4, "placing_numerical": pos, "bfsp": bsp,
                     "dist_furlongs": 6.0})
    # race 2 (1 Feb 2025, Kempton 7f): B second again of 3
    for h, pos, bsp in (("E", 1, 2.0), ("B", 2, 5.0), ("C", 3, 6.0)):
        rows.append({"race_date": "2025-02-01", "race_time": "3.00", "track": "Kempton", "horse_name": h,
                     "jockey_name": "P Jones", "trainer": "T Brown", "number_of_runners": 3,
                     "placing_numerical": pos, "bfsp": bsp, "dist_furlongs": 7.0})
    # race 3 (2 Jan 2026, Ascot 6f, today's card): A and B back at the course and trip a year on
    for h in ("A", "B"):
        rows.append({"race_date": "2026-01-02", "race_time": "2.30", "track": "Ascot", "horse_name": h,
                     "jockey_name": "J Smith" if h == "A" else "P Jones", "trainer": "Mrs K Smith" if h == "A"
                     else "T Brown", "number_of_runners": 2, "placing_numerical": np.nan, "bfsp": np.nan,
                     "dist_furlongs": 6.0})
    out = hrb_extras.build(_runs(rows)).set_index(["race_date", "horse_name"])
    a, b = out.loc[(pd.Timestamp("2026-01-02"), "A")], out.loc[(pd.Timestamp("2026-01-02"), "B")]
    assert a["hx_same_surname"] == 1 and b["hx_same_surname"] == 0
    assert b["hx_second_car"] == pytest.approx(1.0) and a["hx_second_car"] == pytest.approx(0.0)
    assert b["hx_lr_winner_lbsp"] == pytest.approx(np.log(2.0))          # its last race, Kempton, won at 2.0
    assert a["hx_lr_winner_lbsp"] == pytest.approx(np.log(8.0))          # its own win at 8.0
    assert a["hx_lr_nonfin"] == pytest.approx(0.25) and b["hx_lr_nonfin"] == pytest.approx(0.0)
    assert a["hx_maxfield_won"] == 4 and np.isnan(b["hx_maxfield_won"])
    assert a["hx_lastyear_nfp"] == pytest.approx(1.0)                    # won the race a year (366 days) ago
    assert b["hx_lastyear_nfp"] == pytest.approx(2 / 3)                  # second of 4: (4 - 2)/(4 - 1)
    first = out.loc[(pd.Timestamp("2025-01-01"), "A")]
    assert np.isnan(first["hx_second_car"]) and np.isnan(first["hx_lastyear_nfp"])


@pytest.mark.parametrize("name,expected", [("Mrs J Harrington", "harrington"), ("R P O'Brien (3)", "obrien"),
                                           ("", ""), (None, "")])
def test_surnames(name, expected):
    assert hrb_extras.surname(name) == expected
