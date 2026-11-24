"""greyhound/pit/bends.py: the comment parser on GBGB's own codes, the contract, later results move nothing on or
before their day, a card row gets what its run had. Planted data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.pit import bends
from tests.test_greyhound_league import rewrite_results, synth
from tests.test_greyhound_pit import cardify_last_day

KEY = ["race_id", "dog_id"]
COLS = list(bends.COLUMNS)


@pytest.mark.parametrize("comment,led,tr", [
    ("Led2To4,Crd1", [(0.4, 0.8)], (1.0, 0.0, 0.0)),
    ("LdFr1/2", [(0.5, 1.0)], (0.0, 0.0, 0.0)),
    ("LedTo2", [(0.0, 0.4)], (0.0, 0.0, 0.0)),
    ("QAway,AlwaysLed", [(0.0, 1.0)], (0.0, 0.0, 0.0)),
    ("Crd1&2&3&4", [], (1.0, 2.0, 1.0)),
    ("BadlyBaulked1", [], (2.0, 0.0, 0.0)),
    ("SnLdToBBlkW1", [(0.2, 0.2)], (2.0, 0.0, 0.0)),
    ("RailsToMiddle,ChlFr4ToLed5", [(0.8, 1.0)], (0.0, 0.0, 0.0)),
    ("SlightlyCrd2", [], (0.0, 0.5, 0.0)),
])
def test_parse_gbgb_codes(comment, led, tr):
    p = bends.parse(comment, 4)
    assert p["led"] == led
    assert (p["tr_early"], p["tr_mid"], p["tr_late"]) == tr


def test_a_challenge_and_easing():
    assert bends.parse("ChlFr4ToLed5", 4)["chl"] == 1.0 and bends.parse("Esd1/2", 4)["eased"] == 1.0


def test_bends_by_trip():
    assert [bends.bends_for(d) for d in (270, 480, 640, 925)] == [2, 4, 6, 8]


def test_run_feed_positions():
    r = pd.DataFrame({"race_id": [1] * 3, "dog_id": [1, 2, 3], "distance_m": 480, "position": [1, 2, 3],
                      "sectional": [3.9, 3.8, 4.0], "comment": ["ChlFr4ToLed5", "LedTo4", "Crd1"]})
    f = bends.run_feed(r).set_index("dog_id")
    assert f.loc[2, "pos_first"] == 1.0 and f.loc[1, "pos_first"] >= 2.0     # dog 2 led at the first bend
    assert f.loc[2, "led"] == 1 and f.loc[2, "lost_lead"] == 1 and f.loc[1, "led_late"] == 1
    assert f.loc[3, "tr_early"] == 1.0


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = bends.build(runs)
    return runs, tab, notes


def test_the_contract(built):
    runs, tab, notes = built
    assert list(tab.columns) == KEY + COLS and all(c.startswith(bends.PREFIX) for c in COLS)
    assert len(tab) == len(runs) and not tab.duplicated(KEY).any()
    assert notes["runs_parsed"] > 0


@pytest.mark.parametrize("cut", ["last", "middle"])
def test_later_results_move_nothing_on_or_before_their_day(built, cut):
    runs, tab, _ = built
    days = sorted(runs.race_date.unique())
    since = days[-1] if cut == "last" else days[len(days) // 2]
    tab2, _ = bends.build(rewrite_results(runs, since))
    keep = (runs.race_date <= since).to_numpy()
    a = runs.loc[keep, KEY].merge(tab, on=KEY, how="left")[COLS].to_numpy(float)
    b = runs.loc[keep, KEY].merge(tab2, on=KEY, how="left")[COLS].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True, atol=1e-5)


def test_a_card_row_gets_what_its_run_had(built):
    runs, tab, _ = built
    card_tab, _ = bends.build(cardify_last_day(runs))
    last = (runs.race_date == runs.race_date.max()).to_numpy()
    x = runs.loc[last, KEY].merge(tab, on=KEY)[COLS].to_numpy(float)
    y = runs.loc[last, KEY].merge(card_tab, on=KEY)[COLS].to_numpy(float)
    assert np.allclose(x, y, equal_nan=True, atol=1e-5)
