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
