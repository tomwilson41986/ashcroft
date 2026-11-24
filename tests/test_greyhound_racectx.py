"""greyhound/pit/racectx.py (review section 5, race context): planted ratings and early pace give the field's spread,
top-two gap, ranks and pace counts; debutants are counted from earlier days; a race's own result never moves its row,
and a card row gets the historic row's values. Constructed frames for the unit tests only."""
import numpy as np
import pandas as pd

from greyhound.pit import core, racectx
from tests.test_greyhound_pit import perturb_last_day, synth


def planted(runs):
    f = core.per_run_figures(runs)
    f["gsr"] = runs.dog_id.to_numpy() / 10.0                    # each dog's level is its id / 10, every run
    d = runs.dog_id.to_numpy()
    f["esr"] = np.where(d % 5 == 0, 3.0, -d / 160.0)             # every fifth dog quick away, the rest slower
    return f


def test_planted_ratings_and_pace_are_recovered(monkeypatch):
    monkeypatch.setattr(racectx, "EP_Q", 0.9)
    runs = synth()
    tab, notes = racectx.build(runs, planted(runs))
    assert (tab.index == runs.index).all()
    late = (runs.race_date > "2025-01-25").to_numpy()
    t, r = tab[late], runs[late]
    rated = t.rx_rating.notna()
    assert np.allclose(t.rx_rating[rated], r.dog_id[rated] / 10.0)
    for rid, g in t[rated].groupby(r.race_id[rated]):
        ids = np.sort(r.dog_id[g.index].to_numpy() / 10.0)[::-1]
        if len(ids) < 2 or g.rx_n_rated.iloc[0] != len(ids):
            continue
        assert np.isclose(g.rx_top2_gap.iloc[0], ids[0] - ids[1])
        assert np.isclose(g.rx_field_sd.iloc[0], np.std(ids, ddof=1))
        best = g.index[np.argmax(r.dog_id[g.index].to_numpy())]
        assert g.rx_rating_rank[best] == 1 and g.rx_gap_to_top[best] == 0
    quick = (r.dog_id % 5 == 0) & t.rx_ep.notna()
    n_quick = quick.groupby(r.race_id).transform("sum")
    has = (t.rx_ep.notna().groupby(r.race_id).transform("all")) & (n_quick > 0)
    assert (t.rx_n_ep_near_top[has] == n_quick[has]).all()
    assert (t.rx_n_ep_strong[has] == n_quick[has]).all()
    assert notes["ep_strong_threshold_lengths"] == 3.0


def test_debutants_are_counted_from_earlier_days():
    runs = synth(days=12)
    tab, _ = racectx.build(runs, planted(runs))
    racing = ~runs.is_trial
    first = runs[racing].groupby("dog_id").race_date.min()
    debut = (runs.race_date <= runs.dog_id.map(first)).to_numpy()   # no earlier race run (trials do not count)
    want = pd.Series(debut, index=runs.index).groupby(runs.race_id).transform("sum")
    assert (tab.rx_n_debut == want).all()
    assert (tab.rx_n_lt3 >= tab.rx_n_debut).all()
    day1 = runs.race_date == runs.race_date.min()
    assert (tab.rx_n_debut[day1] == runs.field[day1]).all() and tab.rx_rating[day1].isna().all()


def test_a_races_own_result_never_moves_its_row_and_the_card_matches():
    runs = synth()
    last = runs.race_date == runs.race_date.max()
    a, _ = racectx.build(runs, core.per_run_figures(runs))
    p = perturb_last_day(runs)
    b, _ = racectx.build(p, core.per_run_figures(p))
    pd.testing.assert_frame_equal(a[last.to_numpy()], b[last.to_numpy()])
    card = runs.copy()
    card.loc[last, "is_card"] = True
    card.loc[last, ["position", "won", "run_time", "adjusted_time", "sectional"]] = np.nan
    c, _ = racectx.build(card, core.per_run_figures(card))
    pd.testing.assert_frame_equal(a[last.to_numpy()], c[last.to_numpy()])
