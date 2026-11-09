"""research/greyhound/commission_check.py: the rate read off the settled markets' money, both ways."""
import pandas as pd

from research.greyhound import commission_check as cc


def test_the_rate_is_read_off_the_settled_markets():
    cleared = pd.DataFrame({"market": ["1", "2", "3"], "settled": ["2026-11-01", "2026-11-02", "2026-11-03"],
                            "profit": [10.0, 20.0, -5.0], "commission": [0.5, 1.0, 0.0]})
    rates = pd.DataFrame({"market": ["a", "b"], "country": ["GB", "GB"], "type": ["WIN", "WIN"],
                          "base_rate": [5.0, 5.0], "discount_allowed": [True, True]})
    rep = cc.summarise({"discountRate": 0.0, "firstName": "x"}, rates, cleared)
    assert "firstName" not in rep["account"]                                   # nothing personal is reported
    s = rep["settled_markets"]
    assert s["markets"] == 3 and s["markets_won"] == 2 and s["rate_on_profit"] == 0.05
    assert s["rate_on_profit_plus_commission"] == round(1.5 / 31.5, 5)
    assert rep["market_base_rates"][0]["base_rates"] == {5.0: 2}
    assert "no data found" in cc.summarise({}, pd.DataFrame(), pd.DataFrame())["settled_markets"]
