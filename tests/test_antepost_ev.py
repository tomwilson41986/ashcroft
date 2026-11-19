"""The ante-post EV and entry rule (antepost/ev.py). The frames are format fixtures from a seeded generator, written
for the test; no figure from them is used anywhere."""

import numpy as np
import pandas as pd

from antepost import ev
from tests.test_antepost_price import _panel


def _books():
    return pd.DataFrame({"status": ["OPEN"] * 4 + ["SUSPENDED"], "runner_status": ["ACTIVE"] * 5,
                         "back1": [4.0, 4.0, 8.0, 30.0, 4.0], "lay1": [4.4, 4.4, 8.8, 1000.0, 9.0]})


def test_the_spread_reads_open_two_sided_books_only():
    s = ev.spread_by_band(_books())
    assert s == {"b1": 0.1, "b2": 0.1}                       # the 1000 lay and the suspended book are left out


def test_the_fill_is_the_price_less_half_the_spread():
    f = ev.fill_price(np.log(np.array([4.4, 8.8])), {"b1": 0.1, "b2": 0.2})
    assert np.allclose(f, [4.4 / 1.05, 8.8 / 1.1])


def test_commission_comes_off_wins_only():
    assert np.allclose(ev.net([0.5, -0.2, 0.0]), [0.49, -0.2, 0.0])


def test_each_horse_is_backed_once_at_its_first_qualifying_mark():
    d = pd.DataFrame({"market_id": ["m"] * 3, "selection_id": [1, 1, 2], "mark": ["d7", "d30", "d3"],
                      "ev": [0.5, 0.2, 0.01], "ret": [0.1, 0.3, -1.0]})
    b = ev.bets(d, 0.1)
    assert list(b["mark"]) == ["d30"] and list(b["selection_id"]) == [1]


def test_the_rule_runs_end_to_end_and_reads_as_registered():
    rep = ev.run(_panel(), _books().assign(back1=[4.0, 4.0, 8.0, 30.0, 4.0]))
    assert rep["theta"] in ev.THRESHOLDS and set(rep["valid"]) == {str(t) for t in ev.THRESHOLDS}
    assert rep["n_trials"] == ev.PRIOR_TRIALS + len(ev.THRESHOLDS)
    r = ev.reading(rep)
    assert r["verdict"] in ("PASS", "FAIL")
    if rep["test"]["bets"]:
        assert {"roi", "interval_90", "dsr"} <= set(rep["test"])
    assert ev.reading({"skipped": "x"})["verdict"] == "FAIL"
