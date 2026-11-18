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
    assert list(d.country) == ["GB", "GB", "AUS"] and d.track.iloc[0] == "Harl" and d.trap.iloc[0] == 1
