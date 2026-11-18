"""sources/greyhound_prices.py: the menu hints of the older price files (2014-2022) read to the track, the variant
market and the country (samples copied from the files, research/greyhound/join_diag.py run 44337531017)."""
import pandas as pd

from sources.greyhound_prices import market_label, parse


def test_older_menu_hints():
    cases = {
        "BAGS cards / BVue (AvB) 31st Dec": ("BVue", "AvB", "GB"),
        "Forecast / Kinsl (F/C) 31stDec": ("Kinsl", "Forecast", "GB"),
        "BAGS cards / Hove Without Favs": ("Hove", "Without Favs", "GB"),
        "Afternoon Cards / Romfd  (F/C) 2nd Jan": ("Romfd", "F/C", "GB"),
        "SIS / Harl 31st Dec": ("Harl", "", "GB"),
        "Cann (Aus) 31st Dec": ("Cann", "", "AUS"),
        "AUS / Albion Park (AUS) 3rd Oct": ("Albion Park", "", "AUS"),
        "Sky Sports Racing / Kinsley 31st Dec": ("Kinsley", "", "GB"),
        "Romford 3rd Oct": ("Romford", "", "GB"),
    }
    for hint, want in cases.items():
        assert market_label(hint) == want, hint


def test_parse_keeps_variants_out_of_the_win_market():
    csv = ("EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_ID,SELECTION_NAME,WIN_LOSE,BSP\n"
           "1,SIS / Harl 31st Dec,A5 480m,31-12-2018 14:52,11,1. Fast Dog,1,3.5\n"
           "2,Forecast / Harl (F/C) 31st Dec,A5 480m,31-12-2018 14:52,12,1 - 3,0,9.0\n"
           "3,Cann (Aus) 31st Dec,R1,31-12-2018 09:00,13,2. Aussie Dog,0,5.0\n")
    d = parse(csv.encode(), "dwbfgreyhoundwin31122018.csv")
    assert list(d.market) == ["win", "win_variant", "win"]
    assert list(d.country) == ["GB", "GB", "AUS"] and d.track.iloc[0] == "Harlow" and d.trap.iloc[0] == 1


def test_the_learnt_aliases_and_the_newer_variant_tags():
    from sources.greyhound_prices import TRACK_ALIASES
    assert TRACK_ALIASES["Romfd"] == "Romford" and TRACK_ALIASES["Sittb"] == "Central Park"
    assert "Shelb" not in TRACK_ALIASES                                   # Irish: left unmapped
    for hint in ("Romfd (FC) 3rd Jan", "Wdon (OvE) 3rd Jan", "Wdon (Dist) 3rd Jan", "Henl (AvB0 3rd Jan",
                 "Trap Challenge 3rd Jan"):
        assert market_label(hint)[1] != "", hint
    csv = ("EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_ID,SELECTION_NAME,WIN_LOSE,BSP\n"
           "1,BAGS cards / Romfd 2nd Jan,A5 400m,02-01-2016 12:00,11,3. Fast Dog,1,3.5\n")
    assert parse(csv.encode(), "dwbfgreyhoundwin02012016.csv").track.iloc[0] == "Romford"


def test_a_forced_year_is_rebuilt_though_every_file_was_read(tmp_path):
    import json

    from sources.common import Store
    from sources.greyhound_prices import _force_years, build
    raw = tmp_path / "betfair_prices_raw"
    raw.mkdir()
    (raw / "dwbfgreyhoundwin02012016.csv").write_text(
        "EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_ID,SELECTION_NAME,WIN_LOSE,BSP\n"
        "1,BAGS cards / Romfd 2nd Jan,A5 400m,02-01-2016 12:00,11,3. Fast Dog,1,3.5\n")
    store = Store(root=tmp_path / "sources")
    assert build(store)[2016]["runners"] == 1
    assert build(store)[2016].get("unchanged") is True
    assert build(store, force_years={2016})[2016]["runners"] == 1
    assert _force_years({"GREYHOUND_PRICES_REBUILD": "2014-2016"}) == {2014, 2015, 2016}
    assert json.loads(store.get("greyhound_prices/built_2016.json")) == ["dwbfgreyhoundwin02012016.csv"]
