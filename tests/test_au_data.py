"""The Australian BSP targets (au/data.py): parsing Betfair's price-file names and the build's accounting.

The rows below are format fixtures in Betfair's price-file layout, written for the parser; they are not data and
no figure from them is used anywhere.
"""

from datetime import date, datetime

import pandas as pd

from au import config
from au.data import (KEEP, anz_check, apply_track_names, build_targets, learn_track_names, cloth_and_name, meeting_date, parse_event_name, parse_hint,
                     read_csv_bytes)


def test_the_meeting_name_gives_track_country_and_day():
    assert parse_hint("WAGGA (AUS) 28TH FEB") == {"track": "Wagga", "country": "AUS", "day": 28, "month": 2,
                                                  "abbrev": False}
    short = parse_hint("AUS / Flem (AUS) 2nd Nov")
    assert (short["track"], short["abbrev"], short["day"], short["month"]) == ("Flem", True, 2, 11)
    assert parse_hint("WARWICK FARM (AUS) 1ST APR")["track"] == "Warwick Farm"
    assert parse_hint("TRENTHAM (NZL) 3RD JAN")["country"] == "NZL"
    assert parse_hint("") is None and parse_hint("GB / Kemp 12th Mar") is None


def test_the_market_name_gives_race_distance_type_and_gait():
    assert parse_event_name("R3 1200m Hcap") == {"race_no": 3, "distance_m": 1200, "race_type": "Hcap", "harness": False}
    assert parse_event_name("R1 1750m Pace M")["harness"] is True
    assert parse_event_name("R2 2300m Trot S")["harness"] is True
    assert parse_event_name("R4 1400m Grp1")["race_type"] == "Grp1"


def test_the_meeting_date_takes_its_year_across_new_year():
    assert meeting_date(28, 2, datetime(2025, 2, 28, 4, 10)) == date(2025, 2, 28)
    assert meeting_date(1, 1, datetime(2025, 12, 31, 23, 50)) == date(2026, 1, 1)   # NZ's first race of the year
    assert meeting_date(31, 12, datetime(2026, 1, 1, 0, 30)) == date(2025, 12, 31)
    assert meeting_date(30, 2, datetime(2025, 2, 1)) is None


def test_cloth_number_is_split_from_the_name():
    assert cloth_and_name("3. Mahmoa") == (3.0, "Mahmoa")
    c, n = cloth_and_name("Mahmoa")
    assert n == "Mahmoa" and c != c                                                  # NaN


def test_race_family_is_descriptive():
    assert [config.race_family(x) for x in ("Grp1", "Listed", "CL3", "Mdn", "Hcap", "BM64", "3yo", "Qlty")] == [
        "group", "listed", "class", "maiden", "handicap", "handicap", "age", "other"]


def _raw(rows):
    return pd.DataFrame([dict(zip(KEEP + ["FILE"], r)) for r in rows])


def test_the_build_keeps_au_thoroughbreds_and_accounts_for_every_row():
    f = "dwbfpricesauswin28022025.csv"
    raw = _raw([
        ["1", "WAGGA (AUS) 28TH FEB", "R1 1000m Mdn", "28-02-2025 03:15", "11", "1. Alpha", "1", "4.2", "100", "10", f],
        ["1", "WAGGA (AUS) 28TH FEB", "R1 1000m Mdn", "28-02-2025 03:15", "12", "2. Beta", "0", "3.1", "90", "5", f],
        ["1", "WAGGA (AUS) 28TH FEB", "R1 1000m Mdn", "28-02-2025 03:15", "12", "2. Beta", "0", "3.1", "90", "5", f],
        ["2", "MENANGLE (AUS) 28TH FEB", "R1 1609m Pace M", "28-02-2025 05:00", "21", "1. Gamma", "1", "2.0", "9", "1", f],
        ["3", "TRENTHAM (NZL) 28TH FEB", "R1 1200m Hcap", "28-02-2025 02:00", "31", "1. Delta", "1", "5.0", "9", "1", f],
        ["4", "", "R1 1200m Hcap", "28-02-2025 02:00", "41", "1. Eps", "1", "5.0", "9", "1", f],
    ])
    t, rep = build_targets(raw)
    assert rep["rows_in"] == 6 and rep["rows_out"] == 2
    assert rep["dropped"]["harness (Pace/Trot)"] == 1
    assert rep["dropped"]["not Australian (country in the meeting name)"] == 1
    assert rep["dropped"]["meeting name unreadable (no '(AUS)')"] == 1
    assert rep["dropped"]["duplicate (market, selection): first kept"] == 1
    assert rep["rows_in_minus_out_equals_dropped"] is True
    assert rep["markets_winners"] == {"one": 1, "dead_heat": 0, "none": 0}
    r = t.iloc[0]
    assert (r.track, r.meeting_date, r.race_no, r.distance_m, r.race_family, r.horse, r.cloth) == (
        "Wagga", "2025-02-28", 1, 1000, "maiden", "Alpha", 1.0)
    assert set(t.columns) >= {"market_id", "selection_id", "bsp", "won", "off_uk", "horse_norm"}


def test_an_empty_or_odd_file_is_handled():
    assert read_csv_bytes(b"").empty
    assert read_csv_bytes(b"A,B\n1,2\n") is None


def test_the_anz_check_matches_on_market_and_selection():
    raw = _raw([["1", "WAGGA (AUS) 28TH FEB", "R1 1000m Mdn", "28-02-2025 03:15", "11", "1. Alpha", "1", "4.2", "1",
                 "1", "f.csv"],
                ["1", "WAGGA (AUS) 28TH FEB", "R1 1000m Mdn", "28-02-2025 03:15", "12", "2. Beta", "0", "3.1", "1",
                 "1", "f.csv"]])
    t, _ = build_targets(raw)
    anz = pd.DataFrame({"LOCAL_MEETING_DATE": ["2025-02-28", "28/02/2025"], "SCHEDULED_RACE_TIME": ["14:15:00.000"] * 2,
                        "TRACK": ["Wagga"] * 2, "STATE_CODE": ["NSW"] * 2, "RACE_NO": ["1"] * 2,
                        "WIN_MARKET_ID": ["1", "1"], "DISTANCE": ["1000"] * 2, "SELECTION_ID": ["11", "12"],
                        "WIN_RESULT": ["WINNER", "LOSER"], "WIN_BSP": ["4.2", "3.1"]})
    rep = anz_check(t, anz)
    assert rep["matched_rows"] == 2 and rep["targets_found_in_anz"] == 1.0
    assert rep["bsp_equal_within_0.01"] == 1.0 and rep["result_agrees"] == 1.0
    assert rep["track_agrees"] == 1.0 and rep["race_no_agrees"] == 1.0 and rep["distance_agrees"] == 1.0
    assert rep["clock_offset_hours_top"] == {"11.0": 2}


def test_short_track_names_are_learned_from_the_anz_files_and_applied():
    raw = _raw([["7", "AUS / Flem (AUS) 2nd Nov", "R1 1200m Hcap", "02-11-2025 04:00", "1", "1. A", "1", "3", "1", "1",
                 "f.csv"],
                ["8", "AUS / Zzzz (AUS) 2nd Nov", "R1 1200m Hcap", "02-11-2025 04:30", "1", "1. A", "1", "3", "1", "1",
                 "f.csv"]])
    t, _ = build_targets(raw)
    anz = pd.DataFrame({"WIN_MARKET_ID": ["7"], "TRACK": ["Flemington"]})
    names, rep = learn_track_names(t, anz)
    assert names == {"Flem": "Flemington"} and rep["short_names_learned"] == 1
    t2, rep2 = apply_track_names(t, names)
    assert list(t2["track"]) == ["Flemington", "Zzzz"]
    assert rep2["rows_short"] == 2 and rep2["rows_mapped"] == 1 and rep2["short_names_left"] == {"Zzzz": 1}
