"""greyhound/pit/cardread.py: the fitted card-reader (review "optics, salience"). On a constructed card where the SP is
set by the printed last position alone, the reader recovers it out of sample; the fits are walk-forward (the first
period has none); the surplus is log ability share less log card share; a day's own results move nothing on that day
or before it; a card row gets what its run had. Constructed data is for these unit tests only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import cardread, core
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day
from tests.test_greyhound_pit import synth as pit_synth

KEY = ["race_id", "dog_id"]
COLS = list(cardread.COLUMNS)


@pytest.fixture(autouse=True)
def weekly(monkeypatch):
    monkeypatch.setattr(cardread, "FREQ", "W")
    monkeypatch.setattr(cardread, "MIN_FIT_RACES", 50)


def card_driven(days=60, seed=1):
    """One track and trip, 48 dogs, eight races of six a night, finishing order at random. The SP is set by the
    printed card only: a dog that won last time is short, one that finished last is long."""
    rng = np.random.default_rng(seed)
    last = {}
    rows, rid = [], 100
    for d in range(days):
        day = (pd.Timestamp("2025-01-06") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        dogs = rng.permutation(np.arange(1, 49))
        for r in range(8):
            rid += 1
            field = dogs[6 * r:6 * r + 6]
            lp = np.array([last.get(int(x), 3.5) for x in field])
            w = np.exp(-0.6 * (lp - 3.5))
            sp = 1.0 + 0.9 * w.sum() / w                                 # a book of about 111%
            pos = rng.permutation(6) + 1
            t = 29.5 + 0.05 * pos + rng.normal(0, 0.02, 6)
            for k, dog in enumerate(field):
                rows.append({"race_date": day, "race_time": f"{18 + r // 2}:{30 * (r % 2):02d}", "meeting_id": d,
                             "track": "Hove", "race_id": rid, "race_class": "A3", "distance_m": 500.0, "trap": k + 1,
                             "dog_id": int(dog), "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d",
                             "sp_decimal": float(sp[k]), "position": int(pos[k]), "beaten_lengths": 1.0,
                             "sectional": 4.5, "run_time": float(t[k]), "adjusted_time": float(t[k]),
                             "weight_kg": 30.0, "comment": "QAw"})
            for k, dog in enumerate(field):
                last[int(dog)] = int(pos[k])
    return clean(pd.DataFrame(rows))


def flat_xt(runs, v=None):
    return {"xtime": runs[KEY].assign(xt_tot=0.0 if v is None else v)}


def cardread_xt(runs):
    from greyhound.pit import xtime
    return xtime.build(runs, core.per_run_figures(runs))[0]


def test_every_column_is_registered_and_keys_are_unique():
    runs = pit_synth()
    tab, notes = cardread.build(runs)
    assert list(tab.columns) == KEY + COLS and all(c.startswith(cardread.PREFIX) for c in COLS)
    assert (tab[COLS].dtypes == np.float32).all() and len(tab) == len(runs) == notes["rows"]
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and not tab.duplicated(KEY).any()
    assert notes["rows_with_a_reading"] > 0.5 and set(notes["card_safe"]) == set(COLS)


def test_the_reader_recovers_a_planted_card_effect_out_of_sample():
    runs = card_driven()
    tab, notes = cardread.build(runs, ctx=flat_xt(runs))
    y = core.debiased_probs(runs.sp_decimal.to_numpy(), runs.race_id.to_numpy())
    week = core.day(runs).dt.to_period("W").astype("int64").to_numpy()
    first = week == week.min()
    assert tab.loc[first, COLS].isna().all().all()                       # nothing fitted before the first week
    later = ~first
    assert tab.cr_share[later].notna().all()
    assert np.corrcoef(tab.cr_share[later], y[later])[0, 1] > 0.9
    fit = [f for f in notes["fits"].values() if f][-1]["beta"]
    assert fit["pos1"] < -0.3 and abs(fit["pos2"]) < 0.2                 # the last position, not the one before
    # the shares are a race's chances; the rank is 1 for the card's pick
    s = tab[later].groupby("race_id").cr_share.sum()
    np.testing.assert_allclose(s, 1.0, atol=1e-5)
    pick = tab[later].loc[tab[later].groupby("race_id").cr_share.idxmax()]
    assert (pick.cr_rank == 1).all()
    # the form group is the largest standout for a dog that won last time
    m = tab[later].assign(lp=runs.groupby("dog_id").position.shift(1)[later])
    won_last = m[(m.lp == 1) & (m.cr_rank == 1)]
    assert len(won_last) and (won_last.cr_for_attr == cardread.GROUPS.index("form")).mean() > 0.9


def test_the_surplus_is_the_ability_share_against_the_card_share():
    runs = card_driven()
    rng = np.random.default_rng(0)
    skill = pd.Series(rng.normal(0, 3, 49))
    tab, notes = cardread.build(runs, ctx=flat_xt(runs, runs.dog_id.map(skill).to_numpy()))
    ok = tab.cr_surplus.notna()
    assert ok.mean() > 0.85
    np.testing.assert_allclose(tab.cr_surplus[ok], np.log(tab.cr_ab_share[ok]) - np.log(tab.cr_share[ok]),
                               atol=1e-4)
    # the SP never read the skill: the ability reader gives it (about) no weight
    assert all(abs(f["beta"]["xt_tot"]) < 0.05 for f in notes["ability_fits"].values() if f)


def test_the_days_own_results_move_nothing():
    runs = pit_synth()
    last = runs.race_date.max()
    a, _ = cardread.build(runs)
    b, _ = cardread.build(perturb_last_day(runs))
    keys = runs.loc[runs.race_date == last, KEY]
    x, y = keys.merge(a, on=KEY), keys.merge(b, on=KEY)
    assert x[COLS].notna().all(axis=1).mean() > 0.8
    pd.testing.assert_frame_equal(x, y)


def test_a_middle_days_results_move_nothing_on_it_or_before():
    runs = pit_synth()
    days = sorted(runs.race_date.unique())
    mid = days[len(days) // 2]
    # the expected-time table is held fixed here: its own constants move with the store's span (optics.py, "What it
    # inherits"); its point in time is tested in tests/test_greyhound_pit.py
    xt = cardread_xt(runs)
    a, _ = cardread.build(runs, ctx={"xtime": xt})
    upto = runs.race_date <= mid
    keys = runs.loc[upto, KEY]
    cut, _ = cardread.build(runs[upto].reset_index(drop=True), ctx={"xtime": xt})
    pd.testing.assert_frame_equal(keys.merge(a, on=KEY), keys.merge(cut, on=KEY))
    r = runs.copy()
    on = r.race_date == mid
    r.loc[on, "position"] = 7 - r.loc[on, "position"]
    r.loc[on, "adjusted_time"] += 0.7
    r.loc[on, "sectional"] += 0.3
    r.loc[on, "sp_decimal"] = 4.0
    r["won"] = (r.position == 1).astype(float)
    b, _ = cardread.build(r, ctx={"xtime": xt})
    pd.testing.assert_frame_equal(keys.merge(a, on=KEY), keys.merge(b, on=KEY))
    after = runs.loc[~upto, KEY]
    x, y = after.merge(a, on=KEY)[COLS].to_numpy(float), after.merge(b, on=KEY)[COLS].to_numpy(float)
    assert (~np.isclose(x, y, equal_nan=True)).any()


def test_a_card_row_gets_the_features_its_run_had():
    runs = pit_synth()
    last = runs.race_date.max()
    card = cardify_last_day(runs)
    a, _ = cardread.build(runs)
    b, _ = cardread.build(card)
    keys = runs.loc[runs.race_date == last, KEY]
    x, y = keys.merge(a, on=KEY), keys.merge(b, on=KEY)
    assert len(y) == len(keys) and y[COLS].notna().all(axis=1).mean() > 0.8
    pd.testing.assert_frame_equal(x, y)
    # a card on a store's first day: no fit, nothing made up
    first = card[card.race_date == card.race_date.min()].assign(is_card=True)
    c, _ = cardread.build(clean(first.drop(columns=["won", "placed2", "t", "field"], errors="ignore")))
    assert c[COLS].isna().all().all()
