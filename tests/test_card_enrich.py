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


def test_a_median_ending_in_half_is_missing_as_the_table_stores_it():
    card = _card().iloc[[0, 1]].reset_index(drop=True)          # one race: unrated (0) and 80 -> median 40
    four = pd.concat([card, card.assign(horse_name=["Third", "Fourth"], official_rating=[71, 60])], ignore_index=True)
    out = enrich_card(four, _history())                          # 0, 60, 71, 80 -> median 65.5
    assert out["median_or"].isna().all()
    assert enrich_card(card, _history())["median_or"].tolist() == [40.0, 40.0]


def test_the_card_csv_keeps_the_dams_sire_under_the_tables_name():
    from daily_predictions import parse_racecard_csv
    csv_text = "racedate,racetime,track,horse_name,stallion,damstallion,official_rating\n" \
               "2026-09-24,2.30,Kempton,Newcomer,Kodiac,Galileo (IRE),\n"
    card = parse_racecard_csv(csv_text)
    assert card.loc[0, "dam_stallion"] == "Galileo (IRE)" and card.loc[0, "stallion"] == "Kodiac"


def _with_pedigree_history():
    """History with the sire's and the dam's other runners, and debutant males of two on the all-weather."""
    base = {"track": "Kempton", "going_description": "Standard", "surface_type": "Polytrack", "race_type": "Novices",
            "jockey_name": "Other Jockey", "jockeys_claim": "0"}
    extra = [{**base, "race_date": "2025-06-01", "race_time": "5.00", "horse_name": "Older Sibling",
              "horse_sex": "Gelding", "stallion": "Harry Angel (IRE)", "dam": "Twist n Shake",
              "dam_stallion": "Dansili", "horse_age": 3, "career_runs": 4}]
    extra += [{**base, "race_date": "2025-06-02", "race_time": f"{i + 1}.00", "horse_name": f"debut colt {i}",
               "horse_sex": "Colt", "stallion": "Kodiac", "horse_age": 2, "career_runs": 0} for i in range(3)]
    extra += [{**base, "race_date": "2025-06-03", "race_time": "1.00", "horse_name": "debut gelding",
               "horse_sex": "Gelding", "stallion": "Kodiac", "horse_age": 2, "career_runs": 0}]
    return pd.concat([_history(), pd.DataFrame(extra)], ignore_index=True)


def test_a_debutant_takes_its_pedigree_and_sex_from_the_card_tooltip():
    card = _card()                                   # row 1, 'Newcomer', two, on the all-weather: no history
    card["card_stallion"] = [None, "Harry Angel", None]          # printed without the suffix history has
    card["card_dam"] = [None, "Twist n Shake", None]
    card["card_sex"] = ["Male", "Male", None]
    out = enrich_card(card, _with_pedigree_history())
    new = out.loc[1]
    assert new["stallion"] == "Harry Angel (IRE)"                # as history spells it
    assert new["dam_stallion"] == "Dansili"                      # from the dam's other offspring
    assert new["horse_sex"] == "Colt"                            # history's male debutants of two here: 3 colts, 1 gelding
    # a horse history knows keeps history's pedigree and sex; the tooltip's 'Male' cannot make Lady Luck a colt
    assert out.loc[0, "stallion"] == "Kodiac" and out.loc[0, "horse_sex"] == "Mare"


def test_the_dam_comes_from_the_horse_s_rows_else_the_tooltip_as_history_spells_her():
    card = _card()
    card["card_dam"] = [None, "twist n shake", None]               # the tooltip's own spelling
    hist = _with_pedigree_history()
    hist.loc[hist["horse_name"] == "Lady Luck (IRE)", "dam"] = "Lucky Lady"
    out = enrich_card(card, hist)
    assert out.loc[0, "dam"] == "Lucky Lady"                      # history's, for a horse it knows
    assert out.loc[1, "dam"] == "Twist n Shake"                   # the tooltip's, as history spells her
    assert pd.isna(out.loc[2, "dam"])                             # neither: missing, not guessed
    kept = card.assign(dam=["Card Dam", None, None])
    assert enrich_card(kept, hist).loc[0, "dam"] == "Card Dam"    # the card's own is never overwritten


def test_the_tooltip_fills_only_what_it_can():
    card = _card().iloc[[1, 1]].reset_index(drop=True)
    card["horse_age"] = [2, 6]
    card["card_stallion"] = ["First Crop Sire (GB)", "First Crop Sire (GB)"]   # no progeny in history yet
    card["card_dam"] = ["Maiden Dam", "Maiden Dam"]                            # a first foal
    card["card_sex"] = ["Female", "Female"]
    out = enrich_card(card, _with_pedigree_history())
    assert out["stallion"].tolist() == ["First Crop Sire (GB)"] * 2           # the card's own name
    assert out["dam_stallion"].isna().all()                                     # nothing to find it from
    assert out["horse_sex"].tolist() == ["Filly", "Mare"]


def test_the_card_tooltip_is_read_from_the_horse_cell():
    from daily_predictions import parse_horse_title
    assert parse_horse_title("Bay, Male, Stallion - Harry Angel (IRE), Dam - Twist n Shake") == {
        "card_stallion": "Harry Angel (IRE)", "card_dam": "Twist n Shake", "card_sex": "Male"}
    assert parse_horse_title("Bay/Brown, Female, Stallion - Kodiac, Dam - Rue De Russie (IRE)")["card_dam"] \
        == "Rue De Russie (IRE)"
    assert parse_horse_title("Name of horse - click for full record in new screen") == {}
    assert parse_horse_title(None) == {}


def test_the_html_card_carries_the_tooltip_into_the_runner():
    from datetime import date

    from daily_predictions import scrape_racecard_html

    html = """<html><body><span>1.20 Curragh(2 runners)</span><span>Class 1, 6f , Good, 2yo, Win: £10,000</span>
    <span>Some Maiden</span>
    <table><tr><td>No.</td><td>Form</td><td>Days</td><td>Horse</td><td>Age</td><td>Weight</td><td>Headgear</td>
    <td>Jockey</td><td>Trainer</td><td>Stall</td><td>OR</td><td>Odds</td></tr>
    <tr><td>6</td><td>none</td><td>-</td><td title="Bay, Male, Stallion - Harry Angel (IRE), Dam - Twist n Shake">
    <a href="horses.php?id=415718">Shake It Out</a></td><td>2</td><td>9-7</td><td></td><td>Whelan, R P</td>
    <td>Cotter, Kieran P</td><td>10</td><td>0</td><td>50/1</td></tr>
    <tr><td>7</td><td>2</td><td>17</td><td title="Name of horse - click for full record in new screen">
    <a href="horses.php?id=1">Known Horse</a></td><td>2</td><td>9-7</td><td></td><td>Coen, Ben M</td>
    <td>Murphy, Daniel</td><td>8</td><td>0</td><td>10/1</td></tr></table></body></html>"""

    class _Resp:
        text = html

        def raise_for_status(self):
            pass

    class _Session:
        def get(self, url):
            return _Resp()

    df = scrape_racecard_html(_Session(), date(2026, 9, 26)).set_index("horse_name")
    assert df.loc["Shake It Out", "card_stallion"] == "Harry Angel (IRE)"
    assert df.loc["Shake It Out", "card_dam"] == "Twist n Shake" and df.loc["Shake It Out", "card_sex"] == "Male"
    assert pd.isna(df.loc["Known Horse", "card_stallion"])


def test_a_first_foal_takes_its_damsire_from_the_dam_s_own_races():
    h = _with_pedigree_history()
    raced = [{"race_date": "2019-06-01", "race_time": "2.00", "track": "Kempton", "going_description": "Standard",
              "horse_name": "Racing Dam (IRE)", "horse_sex": "Filly", "stallion": "Sea The Stars (IRE)",
              "jockey_name": "Other Jockey", "jockeys_claim": "0"},
             {"race_date": "2019-06-01", "race_time": "3.00", "track": "Kempton", "going_description": "Standard",
              "horse_name": "Namesake", "horse_sex": "Gelding", "stallion": "Wrong Sire",
              "jockey_name": "Other Jockey", "jockeys_claim": "0"}]
    card = _card().iloc[[1, 1]].reset_index(drop=True)
    card["horse_name"] = ["First Foal", "Other First Foal"]
    card["card_stallion"] = ["Kodiac", "Kodiac"]
    card["card_dam"] = ["Racing Dam", "Namesake"]           # the card may print the dam without her suffix
    card["card_sex"] = ["Male", "Male"]
    out = enrich_card(card, pd.concat([h, pd.DataFrame(raced)], ignore_index=True))
    assert out.loc[0, "dam_stallion"] == "Sea The Stars (IRE)"       # her own sire
    assert pd.isna(out.loc[1, "dam_stallion"])                        # a gelding of that name is not a dam
