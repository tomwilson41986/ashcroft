"""Tests for Betfair price-file parsing/matching and market/perf features."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

import betfair_prices as bp
from model.market_features import MARKET_FEATURES, add_market_features, compute_run_descriptors
from model.perf_figures import add_perf_figure_features, parse_beaten_lengths, performance_figure_lbs

SAMPLE_CSV = """EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_ID,SELECTION_NAME,WIN_LOSE,BSP,PPWAP,MORNINGWAP,PPMAX,PPMIN,IPMAX,IPMIN,MORNINGTRADEDVOL,PPTRADEDVOL,IPTRADEDVOL
1001,GB / Kemp 12th Mar,2m Hcap,12-03-2026 14:30,501,Casts Tasha,1,4.2,4.6,6.0,7.0,4.0,4.5,1.01,1200.5,25000,8000
1001,GB / Kemp 12th Mar,2m Hcap,12-03-2026 14:30,502,Chloes Court,0,8.0,7.5,7.0,9.0,6.8,20,2.5,300,9000,3000
1001,GB / Kemp 12th Mar,2m Hcap,12-03-2026 14:30,503,Unknown Runner,0,30,28,25,34,24,120,30,50,800,100
1002,IRE / Dund 12th Mar,1m Mdn,12-03-2026 18:00,601,Some Horse,0,3.5,3.4,3.0,3.8,3.2,4.0,1.2,700,15000,5000
"""


def test_parse_file_and_naming(tmp_path):
    p = tmp_path / "dwbfpricesukwin12032026.csv"
    p.write_text(SAMPLE_CSV)
    assert bp.parse_file_name(p.name) == ("uk", "win", pd.Timestamp("2026-03-12").date())
    assert bp.file_name("ire", "place", pd.Timestamp("2026-03-12").date()) == "dwbfpricesireplace12032026.csv"
    df = bp.parse_file(p)
    assert len(df) == 4 and df["market_type"].iloc[0] == "win" and df["country"].iloc[0] == "uk"
    assert df["race_date"].iloc[0] == "2026-03-12" and df["race_time"].iloc[0] == "14:30"
    assert df["horse_norm"].iloc[0] == "caststasha" and df["bsp"].iloc[0] == 4.2


def test_normalisers():
    assert bp.normalise_horse("1. Casts Tasha (IRE)") == "caststasha"
    assert bp.db_time_to_24h("2.30") == "14:30" and bp.db_time_to_24h("12.15.") == "12:15" and bp.db_time_to_24h("1.00.") == "13:00"
    assert bp.course_from_hint("GB / Kemp 12th Mar") == "kempton"
    assert bp.course_from_hint("IRE / Dund 12th Mar", {"dundalk": "Dundalk"}) == "Dundalk"
    assert bp.course_from_hint("GB / Zzzz 1st Jan", {"kempton": "Kempton"}) is None


def test_load_and_match_to_results(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE race_results (id INTEGER PRIMARY KEY, race_date TEXT, race_time TEXT, track TEXT,
                    horse_name TEXT, placing_numerical INTEGER, number_of_runners INTEGER, trainer TEXT, jockey_name TEXT)""")
    conn.executemany("INSERT INTO race_results VALUES (?,?,?,?,?,?,?,?,?)", [
        (1, "2026-03-12", "2.30", "Kempton", "Casts Tasha (IRE)", 1, 3, "T1", "J1"),
        (2, "2026-03-12", "2.30", "Kempton", "Chloes Court", 2, 3, "T2", "J2"),
        (3, "2026-03-12", "6.00", "Dundalk", "Some Horse", 3, 5, "T1", "J3"),
    ])
    conn.commit(); conn.close()
    p = tmp_path / "dwbfpricesukwin12032026.csv"; p.write_text(SAMPLE_CSV)
    assert bp.load_files([p], str(db))["rows"] == 4
    stats = bp.match_to_results(str(db))
    assert stats["matched"] == 3 and stats["total"] == 4
    rep = bp.coverage_report(str(db))
    assert rep["rows_loaded"].sum() == 4


def test_market_features_are_lag_safe():
    rr = pd.DataFrame({
        "id": [1, 2, 3, 4], "race_date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"]),
        "race_time": ["2.30"] * 4, "track": ["Kempton", "Kempton", "Kempton", "Kempton"],
        "horse_name": ["A", "B", "A", "B"], "placing_numerical": [1, 2, 2, 1], "number_of_runners": [2, 2, 2, 2],
        "trainer": ["T", "T", "T", "T"], "jockey_name": ["J", "K", "J", "K"],
    })
    mk = pd.DataFrame({"race_results_id": [1, 2, 3, 4], "bfp_bsp": [2.0, 3.0, 2.5, 2.2], "bfp_ppwap": [2.1, 2.9, 2.5, 2.3],
                       "bfp_morningwap": [3.0, 2.5, 2.4, 2.6], "bfp_ppmax": [3.2, 3.1, 2.7, 2.8], "bfp_ppmin": [1.9, 2.4, 2.3, 2.1],
                       "bfp_ipmax": [10, 12, 8, 9], "bfp_ipmin": [1.01, 3.0, 1.2, 1.01], "bfp_morning_vol": [100, 50, 80, 90],
                       "bfp_pp_vol": [1000, 500, 800, 900], "bfp_ip_vol": [300, 200, 100, 400], "bfp_place_bsp": [1.3, 1.5, 1.4, 1.3]})
    out = add_market_features(rr, market=mk).sort_values(["horse_name", "race_date"]).reset_index(drop=True)
    a = out[out["horse_name"] == "A"]
    assert np.isnan(a["LR_mkt_steam"].iloc[0])                      # first run: no history
    assert abs(a["LR_mkt_steam"].iloc[1] - np.log(3.0 / 2.0)) < 1e-9  # second run sees first run's move
    assert a["LR_mkt_ip_low_ratio"].iloc[1] == pytest.approx(1.01 / 2.0)
    assert a["horse_mkt_runs"].iloc[1] == 1
    assert set(MARKET_FEATURES).issubset(out.columns)
    desc = compute_run_descriptors(rr.merge(mk, left_on="id", right_on="race_results_id"))
    assert desc["mkt_rank"].tolist() == [1, 2, 2, 1]


def test_perf_figures_and_parsing():
    assert parse_beaten_lengths("nk") == 0.3 and parse_beaten_lengths("2 1/2") == 2.5 and parse_beaten_lengths("1¾") == 1.75
    assert np.isnan(parse_beaten_lengths(""))
    df = pd.DataFrame({"race_date": ["2025-01-01"] * 3, "race_time": ["1.00"] * 3, "track": ["Ascot"] * 3,
                       "horse_name": ["a", "b", "c"], "official_rating": [80, 75, 70], "total_dst_bt": ["0", "1.5", "nk"],
                       "placing_numerical": [1, 2, 3], "dist_furlongs": [8, 8, 8], "median_or": [75] * 3})
    perf = performance_figure_lbs(df)
    assert perf.tolist() == pytest.approx([83.0, 72.0, 69.4])
    hist = pd.concat([df.assign(race_date="2025-03-01"), df], ignore_index=True)
    out = add_perf_figure_features(hist).sort_values(["horse_name", "race_date"])
    a = out[out["horse_name"] == "a"]
    assert np.isnan(a["horse_perf_lbs_ewm"].iloc[0]) and a["horse_perf_lbs_ewm"].iloc[1] == pytest.approx(83.0)
