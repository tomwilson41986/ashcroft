"""greyhound/pit/speed.py: the speed figures. The last day's results rewritten move nothing on that day or before; a
middle day's rewritten move nothing before it; a card row gets what its run had; a long beaten distance is capped;
one row a (race, dog) in the input order; a planted fast dog has the higher figure. Planted data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.pit import speed
from tests.test_greyhound_league import rewrite_results, synth
from tests.test_greyhound_pit import cardify_last_day

COLS = list(speed.COLUMNS)
KEY = ["race_id", "dog_id"]


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = speed.build(runs)
    return runs, tab, notes


def _on(runs, tab, mask):
    return runs.loc[mask, KEY].merge(tab, on=KEY, how="left")[COLS].to_numpy(float)


def test_the_contract(built):
    runs, tab, notes = built
    assert list(tab.columns) == KEY + COLS and all(c.startswith(speed.PREFIX) for c in COLS)
    assert len(tab) == len(runs) and not tab.duplicated(KEY).any()
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    late = (runs.race_date >= "2025-03-01").to_numpy()
    for c in ("sf_mean3", "sf_ewm", "sf_vs", "sf_time_mean3", "sf_sec_ewm"):
        assert tab.loc[late, c].notna().mean() > 0.9, c
    assert notes["runs_figured"] > 0 and notes["k_total"] > 0


@pytest.mark.parametrize("cut", ["last", "middle"])
def test_later_results_move_nothing_on_or_before_their_day(built, cut):
    runs, tab, _ = built
    days = sorted(runs.race_date.unique())
    since = days[-1] if cut == "last" else days[len(days) // 2]
    tab2, _ = speed.build(rewrite_results(runs, since))
    keep = (runs.race_date <= since).to_numpy()
    assert np.allclose(_on(runs, tab, keep), _on(runs, tab2, keep), equal_nan=True, atol=1e-5)


def test_a_card_row_gets_what_its_run_had(built):
    runs, tab, _ = built
    card_tab, _ = speed.build(cardify_last_day(runs))
    last = (runs.race_date == runs.race_date.max()).to_numpy()
    x = runs.loc[last, KEY].merge(tab, on=KEY)[COLS].to_numpy(float)
    y = runs.loc[last, KEY].merge(card_tab, on=KEY)[COLS].to_numpy(float)
    assert np.allclose(x, y, equal_nan=True, atol=1e-5)


def test_a_long_beaten_distance_is_capped():
    r = pd.DataFrame({"race_id": [1, 1, 1], "position": [1, 2, 3], "beaten_lengths": [0.0, 2.0, 18.0],
                      "adjusted_time": [29.0, np.nan, np.nan]})
    t = speed.capped_times(r, np.ones(3, bool), cap=6.0)
    assert np.allclose(t, [29.0, 29.16, 29.0 + 0.08 * 6.0])
    r.loc[1, "beaten_lengths"] = np.nan                       # a distance in front missing: no time behind it
    t = speed.capped_times(r, np.ones(3, bool), cap=6.0)
    assert np.isfinite(t[0]) and np.isnan(t[1]) and np.isnan(t[2])


def test_the_planted_strong_league_has_the_higher_figures(built):
    runs, tab, _ = built
    m = runs.merge(tab, on=KEY)
    late = m[m.race_date >= "2025-03-01"]
    s = late.groupby(["track", "race_class"]).sf_ewm.mean()
    assert s[("Hove", "A1")] > s[("Hove", "A5")]
