"""greyhound/pit/going.py and greyhound/pit/origin.py: the contract, later results move nothing on or before their
day, a card row gets what its run had; the origin codes; a troubled run is left out of the allowance. Planted data
only (tests/test_greyhound_league.synth)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.pit import going, origin
from tests.test_greyhound_league import rewrite_results, synth
from tests.test_greyhound_pit import cardify_last_day

KEY = ["race_id", "dog_id"]


def _with_origin(runs):
    r = runs.copy()
    d = pd.to_numeric(r.dog_id)
    r["origin"] = np.where(d % 3 == 0, "Br", np.where(d % 3 == 1, "Ir", None))
    return r


@pytest.fixture(scope="module", params=["going", "origin"])
def built(request):
    mod = going if request.param == "going" else origin
    runs = _with_origin(synth())
    tab, notes = mod.build(runs)
    return mod, runs, tab, notes


def _on(runs, tab, mask, cols):
    return runs.loc[mask, KEY].merge(tab, on=KEY, how="left")[cols].to_numpy(float)


def test_the_contract(built):
    mod, runs, tab, notes = built
    cols = list(mod.COLUMNS)
    assert list(tab.columns) == KEY + cols and all(c.startswith(mod.PREFIX) for c in cols)
    assert len(tab) == len(runs) and not tab.duplicated(KEY).any()
    late = (runs.race_date >= "2025-03-01").to_numpy()
    first = cols[0]
    assert tab.loc[late, first].notna().mean() > 0.5, first


@pytest.mark.parametrize("cut", ["last", "middle"])
def test_later_results_move_nothing_on_or_before_their_day(built, cut):
    mod, runs, tab, _ = built
    days = sorted(runs.race_date.unique())
    since = days[-1] if cut == "last" else days[len(days) // 2]
    tab2, _ = mod.build(rewrite_results(runs, since))
    keep = (runs.race_date <= since).to_numpy()
    cols = list(mod.COLUMNS)
    assert np.allclose(_on(runs, tab, keep, cols), _on(runs, tab2, keep, cols), equal_nan=True, atol=1e-5)


def test_a_card_row_gets_what_its_run_had(built):
    mod, runs, tab, _ = built
    card_tab, _ = mod.build(cardify_last_day(runs))
    last = (runs.race_date == runs.race_date.max()).to_numpy()
    cols = list(mod.COLUMNS)
    x = runs.loc[last, KEY].merge(tab, on=KEY)[cols].to_numpy(float)
    y = runs.loc[last, KEY].merge(card_tab, on=KEY)[cols].to_numpy(float)
    assert np.allclose(x, y, equal_nan=True, atol=1e-5)


def test_origin_codes():
    s = pd.Series(["Ir", "Br", "Aus", None, " ir ", "GB"])
    assert origin.code(s).tolist() == ["ir", "br", "other", None, "ir", "br"]


def test_origin_values_and_new_imports():
    runs = _with_origin(synth())
    tab, notes = origin.build(runs)
    m = runs[KEY + ["origin"]].merge(tab, on=KEY)
    assert (m.loc[m.origin == "Ir", "og_ir"] == 1).all() and (m.loc[m.origin == "Br", "og_ir"] == 0).all()
    assert m.loc[m.origin.isna(), "og_ir"].isna().all()
    assert (m.og_ir_new.dropna() <= m.og_ir.dropna().reindex(m.og_ir_new.dropna().index) + 1e-6).all()
    assert notes["irish"] > 0 and notes["british"] > 0


def test_a_troubled_run_is_left_out_of_the_allowance():
    runs = synth()
    tab, notes = going.build(runs)
    r2 = runs.copy()
    r2["comment"] = "Crd1"                                   # every run troubled: no clean runner, no allowance
    tab2, notes2 = going.build(r2)
    assert notes2["clean_runners"] == 0 and notes["clean_runners"] > 0
    assert np.nanmax(np.abs(tab2.gw_var_last.to_numpy(float))) < 1e-6
