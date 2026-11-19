"""greyhound/pit/secfig.py: sectional figures linked across tracks. Later results move nothing on or before their
day; a card row gets what its run had; a track whose sectionals are planted slow is put back on the common scale by
the dogs that move. Planted data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.pit import secfig
from tests.test_greyhound_league import rewrite_results, synth
from tests.test_greyhound_pit import cardify_last_day

COLS = list(secfig.COLUMNS)
KEY = ["race_id", "dog_id"]


@pytest.fixture(scope="module")
def built():
    runs = synth()
    tab, notes = secfig.build(runs)
    return runs, tab, notes


def _on(runs, tab, mask):
    return runs.loc[mask, KEY].merge(tab, on=KEY, how="left")[COLS].to_numpy(float)


def test_the_contract(built):
    runs, tab, notes = built
    assert list(tab.columns) == KEY + COLS and all(c.startswith(secfig.PREFIX) for c in COLS)
    assert len(tab) == len(runs) and not tab.duplicated(KEY).any()
    late = (runs.race_date >= "2025-03-01").to_numpy()
    for c in ("sx_ewm", "sx_vs", "sx_rel_ewm"):
        assert tab.loc[late, c].notna().mean() > 0.8, c
    assert notes["sectionals_figured"] > 0


@pytest.mark.parametrize("cut", ["last", "middle"])
def test_later_results_move_nothing_on_or_before_their_day(built, cut):
    runs, tab, _ = built
    days = sorted(runs.race_date.unique())
    since = days[-1] if cut == "last" else days[len(days) // 2]
    tab2, _ = secfig.build(rewrite_results(runs, since))
    keep = (runs.race_date <= since).to_numpy()
    assert np.allclose(_on(runs, tab, keep), _on(runs, tab2, keep), equal_nan=True, atol=1e-5)


def test_a_card_row_gets_what_its_run_had(built):
    runs, tab, _ = built
    card_tab, _ = secfig.build(cardify_last_day(runs))
    last = (runs.race_date == runs.race_date.max()).to_numpy()
    x = runs.loc[last, KEY].merge(tab, on=KEY)[COLS].to_numpy(float)
    y = runs.loc[last, KEY].merge(card_tab, on=KEY)[COLS].to_numpy(float)
    assert np.allclose(x, y, equal_nan=True, atol=1e-5)


def _two_tracks(seed=0):
    """60 dogs: 30 live at Fast (skill 1.0), 30 at Slow (skill 0.0). Each day each track runs 5 races of 6 of its own
    dogs, one place a race (a random trap) taken by a visitor from the other track (the link). Sectional = 5.0 x exp(-(skill +
    noise) / 100): the higher the skill, the quicker, in percent. Both tracks' own pars hide the gap; only the visitors
    show it."""
    rng = np.random.default_rng(seed)
    skill = np.r_[np.full(30, 1.0), np.zeros(30)] + rng.normal(0, 0.1, 60)
    rows, rid = [], 0
    for d in pd.date_range("2023-01-01", "2025-06-30", freq="2D"):
        for t, (lo, other) in enumerate(((0, 30), (30, 0))):
            home = rng.permutation(np.arange(lo, lo + 30))
            for k in range(5):
                rid += 1
                field = list(rng.permutation(list(home[k * 5:(k + 1) * 5]) + [int(rng.integers(other, other + 30))]))
                perf = skill[field] + rng.normal(0, 0.3, 6)
                order = np.argsort(-perf)
                for j, dg in enumerate(field):
                    rows.append({"race_id": rid, "dog_id": int(dg) + 1, "race_date": d.strftime("%Y-%m-%d"),
                                 "track": ["Fast", "Slow"][t], "distance_m": 480, "trap": j + 1,
                                 "position": int(np.flatnonzero(order == j)[0]) + 1, "race_class": "A3",
                                 "meeting_id": f"{d:%Y%m%d}{t}", "sectional": 5.0 * np.exp(-perf[j] / 100)})
    return pd.DataFrame(rows)


def test_linking_puts_two_tracks_on_one_scale():
    r = _two_tracks()
    f = secfig.figures(r)
    late = (r.race_date >= "2025-01-01").to_numpy()
    fast = (r.dog_id <= 30).to_numpy()
    home = ((r.track == "Fast").to_numpy() == fast)
    gap = np.nanmean(f["L"][late & home & fast]) - np.nanmean(f["L"][late & home & ~fast])
    assert 0.8 < gap < 1.2                       # the planted 1.0 percent, recovered at home by the link (0.34 unlinked)
    rel = np.nanmean(f["rel"][late & ~home & fast]) - np.nanmean(f["rel"][late & ~home & ~fast])
    assert rel > 0.6                             # a visitor from the fast track beats the field it meets
