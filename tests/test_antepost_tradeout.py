"""The ante-post trade-out rule (antepost/tradeout.py). The frames are format fixtures written for the test; no
figure from them is used anywhere."""

import numpy as np
import pandas as pd

from antepost import tradeout as to
from tests.test_antepost_ev import _books
from tests.test_antepost_price import _panel

SPREADS = {"b1": 0.1, "b2": 0.2}                 # 3-5: half 0.05; 5-10: half 0.10


def _entry(**kw):
    base = {"market_id": "m", "selection_id": 1, "mark": "d30", "y": "ran", "p_ran": 0.8, "log_ltp": np.log(8.0),
            "bsp": 5.0}
    base.update(kw)
    return pd.DataFrame([base])


def test_the_exit_is_the_first_mark_the_target_is_reached():
    path = {("m", 1): [(5, 8.0), (6, 7.5), (7, 6.0), (9, 4.0)]}      # d30 8.0, d21 7.5, d14 6.0, d3 4.0
    t = to.trades(_entry(), path, 0.20, SPREADS).iloc[0]
    assert t["exit"] == "target" and t["exit_mark"] == "d14" and t["exit_price"] == 6.0
    a_fill, l_fill = 8.0 / 1.10, 6.0 * 1.10
    assert np.isclose(t["ret"], (a_fill / l_fill - 1) * 0.98)


def test_no_target_means_the_last_mark_and_a_loss_when_it_drifts():
    path = {("m", 1): [(5, 8.0), (7, 9.0), (9, 9.5)]}
    t = to.trades(_entry(), path, 0.20, SPREADS).iloc[0]
    assert t["exit"] == "last" and t["exit_mark"] == "d3"
    assert np.isclose(t["ret"], 8.0 / 1.10 / (9.5 * 1.10) - 1)          # a loss: no commission
    held = to.trades(_entry(), path, None, SPREADS).iloc[0]
    assert held["exit"] == "last" and held["exit_mark"] == "d3"


def test_removed_horses_are_void_and_unexited_ones_settle():
    path = {("m", 1): [(5, 8.0), (7, 6.0)]}
    assert to.trades(_entry(y="void"), path, 0.2, SPREADS).iloc[0]["ret"] == 0.0
    lone = {("m", 1): [(5, 8.0)]}
    lost = to.trades(_entry(y="lost"), lone, 0.2, SPREADS).iloc[0]
    assert lost["exit"] == "none" and lost["ret"] == -1.0
    ran = to.trades(_entry(), lone, 0.2, SPREADS).iloc[0]
    assert np.isclose(ran["ret"], (8.0 / 1.10 / 5.0 - 1) * 0.98)


def test_entries_take_the_first_qualifying_mark_and_only_entry_marks():
    d = pd.DataFrame({"market_id": "m", "selection_id": [1, 1, 2, 3], "mark": ["d14", "d45", "d30", "d7"],
                      "has_price": 1, "p_ran": [0.9, 0.6, 0.4, 0.9]})
    e = to.entries(d, 0.5)
    assert list(zip(e["selection_id"], e["mark"])) == [(1, "d45")]


def test_the_rule_runs_end_to_end_and_reads_as_registered():
    rep = to.run(_panel(), _books())
    assert rep["n_trials"] == to.PRIOR_TRIALS + 12 and len(rep["valid"]) == 12
    assert to.reading(rep)["verdict"] in ("PASS", "FAIL")
    assert to.reading({"skipped": "x"})["verdict"] == "FAIL"
