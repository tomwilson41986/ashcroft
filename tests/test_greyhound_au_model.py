"""research/greyhound/au_model.py: the join to Betfair's Historic Data and the value rule (planted rows only)."""
import numpy as np
import pandas as pd

from research.greyhound import au_model as am


def test_join_on_name_venue_and_minute():
    api = pd.DataFrame({"dog_name": ["Fast Lad", "Slow Lad"], "track": ["Wentworth Park"] * 2,
                        "start_utc": ["2026-03-01T09:10:00Z"] * 2, "position": [1, 2]})
    bf = pd.DataFrame({"market_id": ["1.2", "1.2"], "venue": ["Wentworth Park"] * 2,
                       "market_time": ["2026-03-01T09:10:00.000Z"] * 2, "runner_name": ["1. Fast Lad", "2. Slow Lad"],
                       "runner_status": ["WINNER", "LOSER"], "won": [1, 0], "bsp": [2.5, 3.1], "ltp_close": [2.4, 3.2]})
    m, notes = am.join_prices(api, bf)
    assert list(m.bf_bsp) == [2.5, 3.1] and notes["matched"] == 2 and notes["result_agrees"] == 1.0


def test_no_historic_markets_is_said():
    api = pd.DataFrame({"dog_name": ["A"], "track": ["X"], "start_utc": ["2026-03-01T09:10:00Z"], "position": [1]})
    m, notes = am.join_prices(api, pd.DataFrame())
    assert "no data found" in notes["note"] and m.bf_bsp.isna().all()


def test_value_rule_counts_only_value_bets():
    p = np.array([0.5, 0.1]); price = np.array([3.0, 5.0]); won = np.array([1, 0]); day = np.array(["d1", "d1"])
    v = am.value_rule(p, price, won, day, 0.0, 0.05)
    assert v["bets"] == 1 and np.isclose(v["roi%"], 100 * 2.0 * 0.95)


def test_the_join_carries_each_markets_runner_count():
    """A race is scored only when every Betfair runner is matched (17 Nov 2026: races with a runner the API lacked,
    kept only when the winner was matched, made the 0.2 edge's return)."""
    api = pd.DataFrame({"dog_name": ["Fast Lad", "Slow Lad"], "track": ["Wentworth Park"] * 2,
                        "start_utc": ["2026-03-01T09:10:00Z"] * 2, "position": [1, 2]})
    bf = pd.DataFrame({"market_id": ["1.2"] * 3, "venue": ["Wentworth Park"] * 3,
                       "market_time": ["2026-03-01T09:10:00.000Z"] * 3,
                       "runner_name": ["1. Fast Lad", "2. Slow Lad", "3. Missing Lad"],
                       "runner_status": ["WINNER", "LOSER", "LOSER"], "won": [1, 0, 0], "bsp": [2.5, 3.1, 4.0],
                       "ltp_close": [2.4, 3.2, 4.1]})
    m, _ = am.join_prices(api, bf)
    assert (m.bf_runners == 3).all()                 # two matched of three: the race is not scored
