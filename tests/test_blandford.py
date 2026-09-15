"""Tests for the Blandford / Timeform feed parser, matcher and features."""

import json
import sqlite3

import numpy as np
import pandas as pd

import blandford_sync as bs
from model.blandford_features import BLANDFORD_FEATURES, add_blandford_features, load_blandford_frame

ROWS = [
    {"meetingDate": "2026-03-12 00:00:00", "courseName": "CHELMSFORD CITY", "raceNumber": 8, "raceTitle": "HCAP", "distance": "8.0",
     "raceClass": "5.0", "raceType": "Flat", "raceSurfaceName": "All Weather", "going": "Standard", "numberOfRunners": 3,
     "horseName": "JUST A GAMBLER", "horseCode": 633837, "countryCode": "GBR", "positionOfficial": 1, "distanceBeaten": "0.0",
     "distanceCumulative": "0.1", "finishingTime": "100.1", "sectionalFinishingTime": "100.1", "leaderSectional": "35.25",
     "winnerSectional": "35.0", "distanceSectional": "3.0", "timefigure": 62, "performanceRating": 71, "preRaceAdjustedRating": 82,
     "preRaceMasterRating": 68, "ispDecimal": "21.0", "betfairWinSP": "20.0", "betfairPlaceSP": "4.1", "ipMin": "1.01", "ipMax": "24.0",
     "bSPAdvantage": "28.9", "xWR": "0.33", "xWM": "0.2", "Percent_RB2": "1.0", "draw": 4, "horseAge": 5, "horseGender": "g",
     "jockeyFullName": "J One", "trainerFullName": "T One", "ownerFullName": "O", "sireName": "S", "damName": "D", "damsireName": "DS",
     "foalingDate": "2021-03-01T00:00:00Z", "scheduledTimeOfRaceLocal": "2026-03-12T20:30:00Z", "Win": 1},
]
ROWS.append({**ROWS[0], "horseName": "TEWKESBURY", "horseCode": 639322, "positionOfficial": 2, "distanceBeaten": "0.1",
             "performanceRating": 78, "preRaceMasterRating": 69, "betfairWinSP": "7.5", "ipMin": "2.6", "Win": 0})
ROWS.append({**ROWS[0], "horseName": "FRENCH HORSE", "horseCode": 1, "countryCode": "FRA", "courseName": "CHANTILLY"})


def test_parse_rows_types_and_time():
    df = bs.parse_rows(ROWS, "f.json")
    assert len(df) == 3 and df["race_time"].iloc[0] == "20:30" and df["meeting_date"].iloc[0] == "2026-03-12"
    assert df["pre_race_master_rating"].iloc[0] == 68 and df["horse_norm"].iloc[0] == "justagambler"
    assert df["winner_sectional"].dtype.kind == "f"


def test_load_match_and_features(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE race_results (id INTEGER PRIMARY KEY, race_date TEXT, race_time TEXT, track TEXT, horse_name TEXT, official_rating INTEGER)")
    conn.executemany("INSERT INTO race_results VALUES (?,?,?,?,?,?)", [
        (1, "2026-03-12", "8.30", "Chelmsford", "Just A Gambler (IRE)", 70),
        (2, "2026-03-12", "8.30", "Chelmsford", "Tewkesbury", 72),
        (3, "2026-04-01", "2.30", "Kempton", "Just A Gambler (IRE)", 71),
    ])
    conn.commit(); conn.close()
    p = tmp_path / "apidata_2026-03-12_2026-03-12.json"; p.write_text(json.dumps(ROWS))
    assert bs.load_files([p], str(db))["rows"] == 2          # French row dropped
    st = bs.match_to_results(str(db))
    assert st["matched"] == 2 and st["total"] == 2
    rep = bs.coverage_report(str(db)); assert rep["match_pct"].iloc[0] == 100.0
    rr = pd.read_sql_query("SELECT * FROM race_results", sqlite3.connect(db))
    f = add_blandford_features(rr, feed=load_blandford_frame(str(db))).sort_values(["horse_name", "race_date"])
    assert set(BLANDFORD_FEATURES) <= set(f.columns)
    jag = f[f["horse_name"].str.startswith("Just A")].sort_values("race_date")
    assert jag["tf_master_pre"].iloc[0] == 68 and np.isnan(jag["LR_tf_perf"].iloc[0])
    assert jag["LR_tf_perf"].iloc[1] == 71 and jag["tf_runs"].iloc[1] == 1     # lag-safe
    assert abs(jag["LR_race_fsp_pct"].iloc[1] - 100 * (3.0 / 35.0) / (8.0 / 100.1)) < 1e-9
    assert jag["tf_master_vs_or"].iloc[0] == 68 - 70
    assert f.loc[f["race_date"] == "2026-03-12", "tf_master_rank"].tolist() == [2.0, 1.0] or \
           sorted(f.loc[f["race_date"] == "2026-03-12", "tf_master_rank"].tolist()) == [1.0, 2.0]


def test_course_key_normalisation():
    assert bs.course_key("KEMPTON PARK") == bs.course_key("Kempton")
    assert bs.course_key("CHELMSFORD CITY") == bs.course_key("Chelmsford")
    assert bs.course_key("Newmarket (July)") == bs.course_key("NEWMARKET")
