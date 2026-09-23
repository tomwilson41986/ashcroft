"""The live card gets the fields the model was trained with, filled only from the
card's own text and earlier rows, never overwriting what the card has (QA C1)."""

import numpy as np
import pandas as pd

from model.card_enrich import CARD_FILLED, enrich_card, parse_distance_furlongs, race_signature, race_type_by_rule


def _history():
    rows = []
    # Kempton runs both an all-weather and a turf track; direction differs by trip
    for i, (going, surface, dist, direction) in enumerate([
        ("Standard", "Polytrack", 8.0, "Right Handed"), ("Standard To Slow", "Polytrack", 6.0, "Right Handed"),
        ("Good", "Flat", 8.0, "Right Handed"), ("Good To Firm", "Flat", 5.0, "Straight")] * 3):
        rows.append({"race_date": f"2025-0{1 + i % 9}-1{i % 9}", "race_time": "2.30", "track": "Kempton",
                     "going_description": going, "surface_type": surface, "dist_furlongs": dist,
                     "track_direction": direction, "race_name": "Anything Handicap", "race_type": "Handicap",
                     "horse_name": f"filler{i}", "jockey_name": "Other Jockey", "jockeys_claim": "0"})
    # a horse with three runs; sex recorded as Filly; sire and damsire
    for k, d in enumerate(["2025-03-01", "2025-05-01", "2025-07-01"]):
        rows.append({"race_date": d, "race_time": "3.00", "track": "Kempton", "going_description": "Standard",
                     "surface_type": "Polytrack", "dist_furlongs": 8.0, "track_direction": "Right Handed",
                     "race_name": "Some Novice Stakes", "race_type": "Novices",
                     "horse_name": "Lady Luck (IRE)", "horse_sex": "Filly", "stallion": "Kodiac",
                     "dam_stallion": "Galileo (IRE)", "jockey_name": "Apprentice Ann",
                     "jockeys_claim": ["7", "5", "3"][k]})
    # the learned table must beat the rule for a signature history has seen
    for i in range(12):
        rows.append({"race_date": "2025-08-01", "race_time": f"{i + 1}.00", "track": "Ascot",
                     "race_name": f"Sponsor {i} Hunters Chase", "race_type": "Hunters Chase",
                     "horse_name": f"h{i}", "going_description": "Good"})
    return pd.DataFrame(rows)


def _card():
    base = {"race_date": "2026-09-24", "track": "Kempton", "race_class": "Class 4", "prize_money": "4,187"}
    return pd.DataFrame([
        {**base, "race_time": "6:30", "going_description": "Standard", "race_distance": "1m",
         "race_name": "Racing TV Fillies Novice Stakes", "horse_name": "Lady Luck (IRE)", "horse_age": 5,
         "jockey_name": "Apprentice Ann", "official_rating": None, "stall": 3},
        {**base, "race_time": "6:30", "going_description": "Standard", "race_distance": "1m",
         "race_name": "Racing TV Fillies Novice Stakes", "horse_name": "Newcomer", "horse_age": 2,
         "jockey_name": "Unknown Rider (5)", "official_rating": 80, "stall": 1},
        {**base, "race_time": "2:10", "going_description": "Good To Firm", "race_distance": "5f",
         "race_name": "Sprint Handicap", "horse_name": "Speedy", "horse_age": 4,
         "jockey_name": "Other Jockey", "official_rating": 70, "stall": 2, "race_type": "Kept As Is"},
    ])


def test_every_field_is_present_after_the_fill():
    out = enrich_card(_card(), _history())
    assert set(CARD_FILLED) <= set(out.columns)


def test_header_fields():
    out = enrich_card(_card(), _history())
    assert out["dist_furlongs"].tolist() == [8.0, 8.0, 5.0]
    assert out["prize_money"].tolist() == [4187.0] * 3
    assert out.loc[0, "race_type"] == "Novices"
    assert out.loc[2, "race_type"] == "Kept As Is"                 # the card's own value survives


def test_surface_and_direction_follow_the_going_class_and_the_trip():
    out = enrich_card(_card(), _history())
    assert out.loc[0, "surface_type"] == "Polytrack"                 # all-weather going at a dual course
    assert out.loc[2, "surface_type"] == "Flat"                      # turf going, same course
    assert out.loc[2, "track_direction"] == "Straight"               # 5f on the turf is the straight
    assert out.loc[0, "track_direction"] == "Right Handed"


def test_horse_fields_come_from_its_own_history():
    out = enrich_card(_card(), _history())
    lady = out.loc[0]
    assert lady["stallion"] == "Kodiac" and lady["dam_stallion"] == "Galileo (IRE)"
    assert lady["horse_sex"] == "Mare"                               # a filly of 5 is a mare
    assert lady["career_runs"] == 3                                  # earlier runs, not counting this one
    new = out.loc[1]
    assert new["career_runs"] == 0 and pd.isna(new["stallion"]) and pd.isna(new["horse_sex"])


def test_claims_ratings_and_the_race_or_spread():
    out = enrich_card(_card(), _history())
    assert out.loc[0, "jockeys_claim"] == 3                          # the latest claim on record
    assert out.loc[1, "jockeys_claim"] == 5                          # printed on the card, then stripped
    assert out.loc[1, "jockey_name"] == "Unknown Rider"
    assert out.loc[2, "jockeys_claim"] == 0
    assert out.loc[0, "official_rating"] == 0                        # unrated is 0 in the table
    assert out.loc[0, "max_or_in_race"] == 80 and out.loc[0, "median_or"] == 40
    lone = enrich_card(_card().iloc[[0]], _history())                # an all-unrated race: 0, as the table has it
    assert lone.loc[0, "median_or"] == 0


def test_a_learned_signature_beats_the_rule_and_the_rule_covers_the_rest():
    h = _history()
    card = _card().iloc[[2]].drop(columns="race_type").assign(race_name="Local Hunters Chase")
    assert enrich_card(card, h)["race_type"].iloc[0] == "Hunters Chase"
    assert race_type_by_rule(race_signature("Big Sprat Novices Handicap Chase (GBB Race)")) == "Handicap Novices Chase"
    assert race_type_by_rule(race_signature("Kingdom Of Bahrain July Stakes (Group 2)")) == "Non-Handicap"


def test_nothing_the_card_has_is_overwritten():
    card = _card().assign(stallion="Card Sire", career_runs=9, surface_type="Card Surface")
    out = enrich_card(card, _history())
    assert (out["stallion"] == "Card Sire").all() and (out["career_runs"] == 9).all()
    assert (out["surface_type"] == "Card Surface").all()


def test_distance_text_forms():
    assert parse_distance_furlongs("2m½f") == 16.5
    assert parse_distance_furlongs("2m3.5f") == 19.5
    assert parse_distance_furlongs("1m2f110y") == 10.5
    assert np.isnan(parse_distance_furlongs("")) and np.isnan(parse_distance_furlongs(None))


def test_namesakes_bred_in_different_countries_stay_apart():
    namesake = pd.DataFrame([{"race_date": "2025-09-01", "race_time": "4.00", "track": "Ascot",
                              "going_description": "Good", "horse_name": "Lady Luck (GB)", "horse_sex": "Gelding",
                              "stallion": "Frankel", "dam_stallion": "Dansili", "jockey_name": "Other Jockey",
                              "jockeys_claim": "0"}])
    h = pd.concat([_history(), namesake], ignore_index=True)
    card = _card().iloc[[0, 0]].reset_index(drop=True)
    card.loc[1, "horse_name"] = "Lady Luck (GB)"
    out = enrich_card(card, h)
    assert out.loc[0, "stallion"] == "Kodiac" and out.loc[0, "career_runs"] == 3
    assert out.loc[1, "stallion"] == "Frankel" and out.loc[1, "career_runs"] == 1
    # a card that prints the name without its suffix still finds the horse
    bare = enrich_card(_card().iloc[[0]].assign(horse_name="Lady Luck"), _history())
    assert bare.loc[0, "stallion"] == "Kodiac" and bare.loc[0, "career_runs"] == 3


def test_career_runs_continue_the_table_s_count_past_the_history_window():
    h = _history()
    lady = h.horse_name == "Lady Luck (IRE)"
    h.loc[lady, "career_runs"] = [9, 10, 11]                       # runs before 2025 are outside this history
    out = enrich_card(_card(), h)
    assert out.loc[0, "career_runs"] == 12                           # 11 before her last run, plus that run
    assert out.loc[1, "career_runs"] == 0                            # a debutant
