"""greyhound/au_paper.py: the card rows, the match to Betfair, the decision book and the rule (planted rows only)."""
from datetime import date

import numpy as np
import pandas as pd

from greyhound import au_paper as ap


def _race(rid, utc, status="scheduled", region="AU"):
    return {"race_id": rid, "region": region, "status": status, "scheduled_start": {"utc": utc, "local": utc},
            "track": {"name": "Sandown Park", "region": region}, "distance_m": 515, "class_number": 5, "grade": "5",
            "runners": [{"dog_id": 1, "dog_name": "Fast Lad", "trap": 1, "runner_status": "runner"},
                        {"dog_id": 2, "dog_name": "Slow Lad", "trap": 2, "runner_status": "runner"},
                        {"dog_id": 3, "dog_name": "Gone Lad", "trap": 3, "runner_status": "scratched"}]}


def test_card_rows_keep_the_days_australian_runners_only():
    races = [_race(1, "2026-11-18T03:10:00Z"), _race(2, "2026-11-19T03:10:00Z"),
             _race(3, "2026-11-18T03:30:00Z", region="GB"), _race(4, "2026-11-18T03:40:00Z", status="complete")]
    c = ap.card_rows(races, date(2026, 11, 18))
    assert set(c.race_id) == {1} and list(c.dog_name) == ["Fast Lad", "Slow Lad"] and (c.field_size == 2).all()
    assert c.position.isna().all()


def _books():
    rows = []
    for t, near in (("2026-11-18T03:08:00Z", (2.0, 3.0)), ("2026-11-18T03:09:50Z", (2.2, 2.6)),
                    ("2026-11-18T03:10:30Z", (9.0, 9.0))):                      # the last is after the off
        for sid, p in zip((11, 12), near):
            rows.append({"polled_utc": t, "source": "live", "market_id": "1.5", "selection_id": sid,
                         "runner_status": "ACTIVE", "sp_near": p, "sp_far": np.nan, "back1": p - 0.1})
    return pd.DataFrame(rows)


def _markets():
    return pd.DataFrame({"market_id": ["1.5", "1.5"], "market_start_utc": ["2026-11-18T03:10:00Z"] * 2,
                         "venue": ["Sandown Park"] * 2, "selection_id": [11, 12],
                         "runner_name": ["1. Fast Lad", "2. Slow Lad"]})


def test_the_decision_is_the_last_book_before_the_off():
    d = ap.decision_prices(_books(), _markets())
    assert list(d.sort_values("selection_id").dec_price) == [2.2, 2.6] and (d.price_kind == "sp_near").all()


def test_match_and_the_rule():
    prices = pd.DataFrame({"race_id": [1, 1], "dog_id": [1, 2], "dog_name": ["Fast Lad", "Slow Lad"],
                           "track": ["Sandown Park"] * 2, "start_utc": ["2026-11-18T03:10:00Z"] * 2, "trap": [1, 2],
                           "p_model": [0.8, 0.2]})
    m = ap.match_runners(prices, _markets())
    assert list(m.sort_values("selection_id").selection_id) == [11, 12]
    f = m.merge(ap.decision_prices(_books(), _markets()), on=["market_id", "selection_id"])
    out = ap.decide(f, [1.0, 0.0]).set_index("selection_id")   # the blend all model: the favourite is value
    assert bool(out.bet[11]) and not bool(out.bet[12])
    p = out.p_blend[11]                                    # softmax of logit(0.8), logit(0.2): 4 / 4.25
    assert np.isclose(p, 4 / 4.25) and np.isclose(out.ev[11], p * 1.2 * 0.95 - (1 - p))


def test_no_betfair_order_call_in_the_module():
    src = open(ap.__file__).read()
    for call in ("placeOrders", "place_order", "BetfairData", "login(", "cancelOrders"):
        assert call not in src, call
