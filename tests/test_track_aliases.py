"""One course under two names: Betfair's "Bangor-on-Dee" is the card's "Bangor". Until 11 Nov 2026 the two normalised
differently, so no Bangor runner was matched to its market and none was traded."""
from datetime import datetime, timezone

import pandas as pd

from betfair_prices import normalise_track
from trading.exchange import Market
from trading.matching import attach_ids


def test_bangor_is_one_track():
    assert normalise_track("Bangor-on-Dee") == normalise_track("Bangor") == "bangor"
    assert normalise_track("Bangor-On-Dee") == "bangor"
    assert normalise_track("Newmarket (July)") == "newmarket" and normalise_track("Ayr") == "ayr"


def test_a_bangor_card_runner_finds_its_betfair_market():
    pred = pd.DataFrame({"track": ["Bangor", "Ayr"], "race_time": ["12:30", "1:10"],
                         "horse_name": ["Some Horse (IRE)", "Other Horse"], "predicted_bfsp": [4.0, 6.0]})
    markets = [Market("1.1", "Bangor-on-Dee", "GB", datetime(2026, 11, 11, 12, 30, tzinfo=timezone.utc),
                      runners={11: "Some Horse"}),
               Market("1.2", "Ayr", "GB", datetime(2026, 11, 11, 13, 10, tzinfo=timezone.utc), runners={22: "Other Horse"})]
    out = attach_ids(pred, markets)
    assert list(out.market_id) == ["1.1", "1.2"] and list(out.selection_id.astype(int)) == [11, 22]


def test_a_course_under_an_unknown_second_name_is_still_matched_by_time_and_horse():
    pred = pd.DataFrame({"track": ["Newname", "Ayr", "Ayr"], "race_time": ["2:00", "1:10", "3:00"],
                         "horse_name": ["Some Horse", "Other Horse", "Not On Betfair"], "predicted_bfsp": [4.0, 6.0, 9.0]})
    markets = [Market("1.1", "Newname Park", "GB", datetime(2026, 11, 11, 14, 0, tzinfo=timezone.utc),
                      runners={11: "Some Horse"}),
               Market("1.2", "Ayr", "GB", datetime(2026, 11, 11, 13, 10, tzinfo=timezone.utc), runners={22: "Other Horse"})]
    out = attach_ids(pred, markets)
    assert list(out.market_id[:2]) == ["1.1", "1.2"] and pd.isna(out.market_id[2])
    assert int(out.selection_id[0]) == 11 and len(out) == 3


def test_a_name_on_two_markets_is_not_guessed():
    pred = pd.DataFrame({"track": ["Elsewhere"], "race_time": ["2:00"], "horse_name": ["Twin"], "predicted_bfsp": [4.0]})
    markets = [Market("1.1", "A", "GB", datetime(2026, 11, 11, 14, 0, tzinfo=timezone.utc), runners={1: "Twin"}),
               Market("1.2", "B", "GB", datetime(2026, 11, 11, 14, 0, tzinfo=timezone.utc), runners={2: "Twin"})]
    assert pd.isna(attach_ids(pred, markets).market_id[0])
