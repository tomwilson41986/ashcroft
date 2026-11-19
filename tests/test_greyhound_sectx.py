"""greyhound/pit/sectx.py: sectionals and times taken apart. Later results move nothing on or before their day; a
card row gets what its run had; the second sectional and the Finishing Speed % read a planted strong finisher.
Planted data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.pit import sectx
from tests.test_greyhound_league import rewrite_results, synth
from tests.test_greyhound_pit import cardify_last_day

COLS = list(sectx.COLUMNS)
KEY = ["race_id", "dog_id"]


@pytest.fixture(scope="module")
def built():
    runs = synth()
    rng = np.random.default_rng(3)
    runs = runs.assign(sectional=4.5 + rng.normal(0, 0.05, len(runs)))
    tab, notes = sectx.build(runs)
    return runs, tab, notes


def _on(runs, tab, mask):
    return runs.loc[mask, KEY].merge(tab, on=KEY, how="left")[COLS].to_numpy(float)


def test_the_contract(built):
    runs, tab, notes = built
    assert list(tab.columns) == KEY + COLS and all(c.startswith(sectx.PREFIX) for c in COLS)
    assert len(tab) == len(runs) and not tab.duplicated(KEY).any()
    late = (runs.race_date >= "2025-03-01").to_numpy()
    for c in ("sc_s1_ewm", "sc_s2_ewm", "sc_fsp_ewm", "sc_fsp_vs", "sc_race_pace", "sc_s1_rp_ewm"):
        assert tab.loc[late, c].notna().mean() > 0.8, c
    assert notes["runs_with_both_sections"] > 0 and 95 < notes["fsp_median"] < 105


@pytest.mark.parametrize("cut", ["last", "middle"])
def test_later_results_move_nothing_on_or_before_their_day(built, cut):
    runs, tab, _ = built
    days = sorted(runs.race_date.unique())
    since = days[-1] if cut == "last" else days[len(days) // 2]
    tab2, _ = sectx.build(rewrite_results(runs, since))
    keep = (runs.race_date <= since).to_numpy()
    assert np.allclose(_on(runs, tab, keep), _on(runs, tab2, keep), equal_nan=True, atol=1e-4)


def test_a_card_row_gets_what_its_run_had(built):
    runs, tab, _ = built
    card_tab, _ = sectx.build(cardify_last_day(runs))
    last = (runs.race_date == runs.race_date.max()).to_numpy()
    x = runs.loc[last, KEY].merge(tab, on=KEY)[COLS].to_numpy(float)
    y = runs.loc[last, KEY].merge(card_tab, on=KEY)[COLS].to_numpy(float)
    assert np.allclose(x, y, equal_nan=True, atol=1e-4)


def test_a_planted_closer_has_the_higher_fsp_and_second_sectional(built):
    runs, _, _ = built
    r = runs.copy()
    closer = r.dog_id == r.dog_id.value_counts().index[0]
    r.loc[closer, "sectional"] = r.loc[closer, "sectional"] + 0.20      # away slowly, the same time: home faster
    f = sectx.run_figures(r)["out"]
    late = (r.race_date >= "2025-01-01").to_numpy()
    c = closer.to_numpy() & late
    o = ~closer.to_numpy() & late
    assert np.nanmean(f["fsp"][c]) > np.nanmean(f["fsp"][o]) + 0.5      # 0.2 s of a ~25 s run home: +0.78
    assert np.nanmean(f["s2"][c]) > np.nanmean(f["s2"][o]) and np.nanmean(f["s1"][c]) < np.nanmean(f["s1"][o])
    assert np.nanmean(f["close"][c]) > np.nanmean(f["close"][o])
