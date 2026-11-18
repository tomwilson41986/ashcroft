"""The Betfair card fallback (card_betfair.py): the HRB card's form from Betfair's recorded catalogue, and the
fallback in predict_bfsp_today.fetch_racecard_from_hrb when horseracebase.com gives no card."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

import card_betfair as cb


def test_race_name_and_distance():
    assert cb.expand_race_name("2m4f Hcap Chs") == "2m4f Handicap Chase"
    assert cb.expand_race_name("2m INHF") == "2m NH Flat"
    assert cb.expand_race_name("2m3f Mdn Hrd") == "2m3f Maiden Hurdle"
    assert cb.expand_race_name("6f Nov Stks") == "6f Novices Stakes"
    assert cb.distance_from_name("2m4f Hcap Chs") == "2m4f"
    assert cb.distance_from_name("5f Nursery") == "5f"
    assert cb.distance_from_name("1m Hcap") == "1m"
    assert cb.distance_from_name("Hcap") is None
    from model.card_enrich import parse_distance_furlongs, race_signature, race_type_by_rule
    assert parse_distance_furlongs(cb.distance_from_name("2m4f Hcap Chs")) == 20
    assert race_type_by_rule(race_signature(cb.expand_race_name("2m4f Hcap Chs"))) == "Handicap Chase"
    assert race_type_by_rule(race_signature(cb.expand_race_name("2m1f Nov Hrd"))) == "Novices Hurdle"


def test_headgear_time_and_forecast():
    assert cb.headgear_from_wearing("cheekpieces and tongue strap") == "CkPcTT"     # HRB's form, codes joined
    assert cb.headgear_from_wearing("hood and blinkers") == "H+B"
    assert cb.headgear_from_wearing("blinkers and eyeshields") == "BlnkEye"
    assert cb.headgear_from_wearing("Blinkers") == "Blnk"
    assert cb.headgear_from_wearing(None) is None
    assert cb.headgear_from_wearing(np.nan) is None
    assert cb.card_time("2026-11-17T13:50:00.000Z") == "1:50"        # GMT in November
    assert cb.card_time("2026-11-17T11:50:00.000Z") == "11:50"
    assert cb.card_time("2026-07-01T12:20:00Z") == "1:20"            # BST: UTC + 1
    assert cb.forecast_odds(9, 2) == pytest.approx(5.5)
    assert np.isnan(cb.forecast_odds(None, 2))


def _names(rows):
    return cb.NameIndex(pd.DataFrame(rows, columns=["name", "last", "n"]))


def test_names_as_the_results_table_writes_them():
    jk = _names([("Skelton, Harry", "2026-11-10", 300), ("McGill, Mr O", "2026-11-01", 20),
                 ("de Boinville, Nico", "2026-11-12", 400), ("Wadge, P", "2026-11-11", 50),
                 ("Skelton, H", "2024-01-01", 3)])
    assert jk.match("Harry Skelton") == "Skelton, Harry"             # the recent of two candidates
    assert jk.match("Mr O. McGill") == "McGill, Mr O"
    assert jk.match("Nico de Boinville") == "de Boinville, Nico"
    assert jk.match("Paddy Wadge") == "Wadge, P"
    assert jk.match("Nobody Known") is None
    assert jk.match("Jonjo O'Neill Jr.") is None
    assert _names([("ONeill, Jonjo (Jr)", "2026-11-10", 100)]).match("Jonjo O'Neill Jr.") == "ONeill, Jonjo (Jr)"
    tr = cb.NameIndex(pd.DataFrame(
        [("Russell, Lucinda / Scudamore, M", "2026-11-15", 90), ("Skelton, Daniel", "2026-11-16", 900),
         ("ONeill, Jonjo and AJ", "2026-11-10", 100), ("McCain Jnr, D", "2026-11-10", 300),
         ("Thomson, A M", "2026-11-01", 50), ("Hammond, Micky", "2026-11-01", 50), ("Easterby, M W", "2026-11-01", 50),
         ("Easterby, T D", "2026-11-01", 50), ("Prescott, Sir Mark", "2026-11-01", 50),
         ("Mullins, W P", "2026-11-01", 5), ("Mullins, J W", "2026-11-01", 5)], columns=["name", "last", "n"]),
        unique_since="2025-11-17")
    assert tr.match("Lucinda Russell & Michael Scudamore") == "Russell, Lucinda / Scudamore, M"
    assert tr.match("Dan Skelton") == "Skelton, Daniel"
    assert tr.match("Jonjo & A.J. O'Neill") == "ONeill, Jonjo and AJ"          # one surname, two licence holders
    assert tr.match("Donald McCain") == "McCain Jnr, D"
    assert tr.match("Sandy Thomson") == "Thomson, A M"                          # the only Thomson: the surname alone
    assert tr.match("Micky Hammond & Becky Smith") == "Hammond, Micky"          # the table names the first
    assert tr.match("Michael & David Easterby") == "Easterby, M W"
    assert tr.match("Sir Mark Prescott Bt") == "Prescott, Sir Mark"
    assert tr.match("Seamus Mullins") is None                                   # two Mullins: no guess


def test_course_distance():
    d = pd.DataFrame({"track": ["Limerick", "Limerick", "Limerick"],
                      "race_type": ["Maiden Hurdle", "Handicap Hurdle", "Handicap"],
                      "race_distance": ["2m2½f", "2m3½f", "2m3f"], "n": [30, 5, 40]})
    assert cb.course_distance("Limerick", True, "2m3f", d) == "2m2½f"           # nearest jumps trip, the commoner
    assert cb.course_distance("Limerick", True, "3m", d) == "3m"                # nothing within half a furlong
    assert cb.course_distance("Elsewhere", True, "2m3f", d) == "2m3f"


def test_track_names():
    tracks = pd.DataFrame({"track": ["Bangor", "Newmarket (Rowley)", "Newmarket (July)", "Carlisle"],
                           "last": ["2026-11-01", "2026-10-30", "2026-08-01", "2026-11-17"]})
    assert cb.track_name("Bangor-on-Dee", tracks) == "Bangor"
    assert cb.track_name("Newmarket", tracks) == "Newmarket (Rowley)"
    assert cb.track_name("Carlisle", tracks) == "Carlisle"
    assert cb.track_name("Nowhere", tracks) == "Nowhere"


def _catalogue():
    base = {"market_type": "WIN", "country": "GB", "event_id": "1", "event_name": "Carlisle 17th Nov",
            "sort_priority": 1, "form": "1-", "recorded_utc": "2026-11-17T01:00:00Z", "each_way_divisor": None}
    rows = [
        dict(base, market_id="1.1", market_name="2m4f Hcap Chs", market_start_utc="2026-11-17T13:50:00.000Z",
             venue="Carlisle", race_type="Chase", selection_id="11", runner_name="Alpha", cloth_number=1,
             stall_draw=None, jockey="Harry Skelton", trainer="Dan Skelton", age=6, weight_value=158,
             official_rating=120, days_since_last_run=21, wearing="cheekpieces", forecast_num=5, forecast_den=2),
        dict(base, market_id="1.1", market_name="2m4f Hcap Chs", market_start_utc="2026-11-17T13:50:00.000Z",
             venue="Carlisle", race_type="Chase", selection_id="12", runner_name="Beta", cloth_number=2,
             stall_draw=None, jockey="Unknown Rider", trainer="Dan Skelton", age=7, weight_value=150,
             official_rating=110, days_since_last_run=40, wearing=None, forecast_num=3, forecast_den=1),
        dict(base, market_id="1.1", market_name="2m4f Hcap Chs", market_start_utc="2026-11-17T13:50:00.000Z",
             venue="Carlisle", race_type="Chase", selection_id="13", runner_name="Gone", cloth_number=3,
             stall_draw=None, jockey="Harry Skelton", trainer="Dan Skelton", age=8, weight_value=148,
             official_rating=105, days_since_last_run=10, wearing=None, forecast_num=8, forecast_den=1),
        dict(base, market_id="1.2", market_name="6f Hcap", market_start_utc="2026-11-17T16:00:00.000Z",
             venue="Southwell", race_type="Flat", selection_id="21", runner_name="Gamma", cloth_number=4,
             stall_draw=7, jockey="Harry Skelton", trainer="Dan Skelton", age=3, weight_value=130,
             official_rating=70, days_since_last_run=14, wearing=None, forecast_num=4, forecast_den=1),
        dict(base, market_id="1.3", market_name="2m Hrd", market_start_utc="2026-11-18T13:00:00.000Z",
             venue="Carlisle", race_type="Hurdle", selection_id="31", runner_name="Tomorrow", cloth_number=1,
             stall_draw=None, jockey="Harry Skelton", trainer="Dan Skelton", age=5, weight_value=160,
             official_rating=0, days_since_last_run=30, wearing=None, forecast_num=1, forecast_den=1),
        dict(base, market_id="1.4", market_name="To Be Placed", market_type="PLACE",
             market_start_utc="2026-11-17T13:50:00.000Z", venue="Carlisle", race_type="Chase", selection_id="11",
             runner_name="Alpha", cloth_number=1, stall_draw=None, jockey="Harry Skelton", trainer="Dan Skelton",
             age=6, weight_value=158, official_rating=120, days_since_last_run=21, wearing=None, forecast_num=5,
             forecast_den=2),
    ]
    return pd.DataFrame(rows)


def test_card_from_catalogue():
    lookups = {
        "jockeys": _names([("Skelton, Harry", "2026-11-10", 300)]),
        "trainers": _names([("Skelton, Daniel", "2026-11-16", 900)]),
        "tracks": pd.DataFrame({"track": ["Carlisle", "Southwell"], "last": ["2026-11-17", "2026-11-16"]}),
        "going": pd.DataFrame({"track": ["Carlisle", "Carlisle", "Southwell"],
                               "race_type": ["Handicap Chase", "Handicap", "Handicap"],
                               "race_date": ["2026-11-10", "2026-08-01", "2026-11-13"],
                               "race_time": ["1:50", "2:00", "6:00"],
                               "going_description": ["Good To Soft", "Good", "Standard"]}),
    }
    card = cb.card_from_catalogue(_catalogue(), date(2026, 11, 17), lookups, removed={("1.1", "13")})
    assert sorted(card.horse_name) == ["Alpha", "Beta", "Gamma"]          # removed, tomorrow and PLACE left out
    a = card.set_index("horse_name")
    assert a.loc["Alpha", "race_time"] == "1:50" and a.loc["Alpha", "track"] == "Carlisle"
    assert a.loc["Alpha", "jockey_name"] == "Skelton, Harry"
    assert a.loc["Beta", "jockey_name"] == "Unknown Rider"                # no match: Betfair's own name
    assert a.loc["Alpha", "trainer"] == "Skelton, Daniel"
    assert a.loc["Alpha", "race_name"] == "2m4f Handicap Chase" and a.loc["Alpha", "race_distance"] == "2m4f"
    assert a.loc["Alpha", "number_of_runners"] == 2
    assert a.loc["Alpha", "stall"] == 1 and a.loc["Gamma", "stall"] == 7  # the saddlecloth over jumps, the draw on the flat
    assert a.loc["Alpha", "odds"] == pytest.approx(3.5)
    assert a.loc["Alpha", "headgear"] == "CkPc"
    assert a.loc["Alpha", "pounds"] == 158
    assert a.loc["Alpha", "going_description"] == "Good To Soft"         # the chase's, not the summer flat's
    assert a.loc["Gamma", "going_description"] == "Standard"
    assert (card.card_source == "betfair").all()


def test_going_left_empty_when_nothing_recent():
    going = pd.DataFrame({"track": ["Carlisle"], "race_type": ["Handicap Chase"], "race_date": ["2026-10-01"],
                          "race_time": ["1:50"], "going_description": ["Soft"]})
    assert cb.going_estimate("Carlisle", True, date(2026, 11, 17), going) == (None, "none recent")
    assert cb.going_estimate("Carlisle", False, date(2026, 11, 17), going) == (None, "none")


def test_removed_runners_from_the_latest_book():
    books = pd.DataFrame({"polled_utc": ["2026-11-17T01:00Z", "2026-11-17T02:00Z", "2026-11-17T02:00Z"],
                          "market_id": ["1.1", "1.1", "1.1"], "selection_id": ["13", "13", "12"],
                          "runner_status": ["ACTIVE", "REMOVED", "ACTIVE"]})
    assert cb.removed_runners(books) == {("1.1", "13")}


def test_fetch_falls_back_when_the_login_fails(monkeypatch):
    import daily_predictions
    import predict_bfsp_today as p
    card = pd.DataFrame({"horse_name": ["Alpha"], "card_source": ["betfair"]})
    monkeypatch.setattr(daily_predictions, "create_session", lambda: object())
    monkeypatch.setattr(daily_predictions, "login", lambda s: None)
    monkeypatch.setattr(cb, "fetch_card", lambda day, db_path=None: card)
    monkeypatch.delenv("CARD_FALLBACK", raising=False)
    monkeypatch.delenv("CARD_SOURCE", raising=False)
    assert p.fetch_racecard_from_hrb(date(2026, 11, 17)).empty          # off by default until a price check passes
    monkeypatch.setenv("CARD_FALLBACK", "on")
    assert p.fetch_racecard_from_hrb(date(2026, 11, 17)).equals(card)


def test_hrb_card_is_kept_when_it_comes(monkeypatch):
    import daily_predictions
    import predict_bfsp_today as p
    hrb = pd.DataFrame({"horse_name": ["Alpha"]})
    monkeypatch.setattr(daily_predictions, "create_session", lambda: object())
    monkeypatch.setattr(daily_predictions, "login", lambda s: "u1")
    monkeypatch.setattr(daily_predictions, "get_todays_runners", lambda s, u, d: hrb)
    monkeypatch.setattr(cb, "fetch_card", lambda day, db_path=None: pytest.fail("the fallback ran"))
    monkeypatch.delenv("CARD_SOURCE", raising=False)
    assert p.fetch_racecard_from_hrb(date(2026, 11, 17)).equals(hrb)
    monkeypatch.setenv("CARD_SOURCE", "betfair")                          # the check run reads Betfair's card
    monkeypatch.setattr(cb, "fetch_card", lambda day, db_path=None: pd.DataFrame({"horse_name": ["B"]}))
    assert list(p.fetch_racecard_from_hrb(date(2026, 11, 17)).horse_name) == ["B"]


def test_class_and_prize_from_the_course_s_like_races():
    races = pd.DataFrame({
        "track": ["Wolverhampton"] * 5 + ["Kempton"],
        "race_date": ["2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"],
        "race_time": ["6:00"] * 6,
        "race_type": ["Handicap", "Handicap", "Handicap", "Maiden", "Handicap Hurdle", "Handicap"],
        "race_class": ["Class 6", "Class 6", "Class 4", "Class 5", "Class 3", "Class 2"],
        "prize_money": [3000, 3200, 6000, 4000, 9000, 20000],
        "top_or": [60, 62, 85, 0, 130, 100],
    })
    cls, prize, src = cb.class_prize_estimate("Wolverhampton", False, True, 61, races, k=2)
    assert cls == "Class 6" and prize == 3100 and src.startswith("course")          # the two nearest in OR
    cls, prize, src = cb.class_prize_estimate("Wolverhampton", False, False, np.nan, races, k=3)
    assert cls == "Class 5" and prize == 4000                                       # the unrated race
    cls, prize, src = cb.class_prize_estimate("Southwell", False, True, 98, races, k=1)
    assert cls == "Class 2" and src.startswith("all courses")                      # nothing at the course
    assert cb.class_prize_estimate("Southwell", True, False, 100, races)[2] == "none"


def test_horse_names_take_the_results_table_spelling():
    horses = pd.DataFrame({"horse_name": ["Dyker Heights (IRE)", "Lady Luck (GB)", "Lady Luck (IRE)", "Alpha"],
                           "last": ["2026-11-01", "2024-05-01", "2026-10-01", "2026-09-01"], "age": [5, 7, 4, 6]})
    out = cb.history_names(pd.Series(["Dyker Heights", "Lady Luck", "Alpha", "Brand New"]), horses)
    assert list(out) == ["Dyker Heights (IRE)", "Lady Luck (IRE)", "Alpha", "Brand New"]   # the recent namesake
