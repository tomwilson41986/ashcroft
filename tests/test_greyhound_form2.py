"""greyhound/pit/form2.py (the feature review of 17 Nov 2026, form): the damped trend and the within-dog age curve;
point in time and card-safe. Constructed frames only."""
import numpy as np
import pandas as pd

from greyhound.data import clean
from greyhound.pit import core, form2
from tests.greyhound_pit_synth2 import synth


def test_the_trend_is_damped_over_days_off():
    # one dog improving a length a month, run weekly for 20 weeks, then 120 days off
    dn = np.r_[np.arange(20) * 7.0, 19 * 7.0 + 120]
    z = np.r_[np.arange(20) * 7.0 / 30.44, np.nan]
    prm = {"ql": 1e-4, "qs": 1e-7, "r": 0.01, "m0": 0.0, "p0": 1.0, "s0": 1e-2}
    st_d, _ = form2.damped_filter(np.ones(21), dn, z, np.zeros(21, bool), {**prm, "tau": 30.0})
    st_u, _ = form2.damped_filter(np.ones(21), dn, z, np.zeros(21, bool), {**prm, "tau": 1e9})
    assert np.isclose(st_u.ext.iloc[-1], 120 / 30.44, rtol=0.1)                      # undamped: the whole gap
    s_after = st_d.slope.iloc[-1] / 30.44 / np.exp(-120 / 30.0)                    # the slope at the last run
    assert np.isclose(st_d.ext.iloc[-1], s_after * 30 * (1 - np.exp(-4)))           # damped: tau's worth at most
    assert st_d.ext.iloc[-1] < 0.1 * st_u.ext.iloc[-1]
    assert np.isclose(st_d.slope.iloc[-1], s_after * 30.44 * np.exp(-4))


def test_the_within_dog_age_curve_beats_survivor_bias():
    # g(age) rises to 30 months then falls; the weak dogs retire young, so a cross-section of ages is biased
    rng = np.random.default_rng(0)
    true = lambda a: np.where(a < 30, (a - 30) * 0.15, -(a - 30) * 0.2)            # noqa: E731
    dog, age, f = [], [], []
    for d in range(600):
        alpha = rng.normal(0, 3)
        a0 = rng.uniform(15, 25)
        for a in a0 + np.arange(0, 40, 0.7):
            if alpha + true(a) < -4 and a > 30:                                     # the declining weak dogs retire
                break
            dog.append(d), age.append(a), f.append(alpha + true(a) + rng.normal(0, 1))
    dog, age, f = map(np.asarray, (dog, age, f))
    curves = form2.age_curve_fe(dog, age, np.zeros(len(age)), f, np.ones(len(age), bool))
    est = form2.curve_at(curves, form2.KNOTS, np.zeros(len(form2.KNOTS)))
    ok = (form2.KNOTS >= 18) & (form2.KNOTS <= 60)
    assert np.abs(est[ok] - true(form2.KNOTS[ok])).max() < 0.8
    naive = pd.Series(f).groupby(np.round(age / 6) * 6).mean()
    naive = naive - naive.loc[30]
    assert abs(naive.loc[60] - true(60.0)) > 1.5                                    # the cross-section is biased


def test_no_row_reads_its_days_own_results_and_the_card_gets_the_same_values():
    runs, _ = synth(days=30)
    last = runs.race_date == runs.race_date.max()
    a = form2.build(runs, core.per_run_figures(runs), {})[0]
    p = runs.copy()
    p.loc[last, "position"] = 7 - p.loc[last, "position"]
    p.loc[last, ["adjusted_time", "run_time"]] += 0.5
    p.loc[last, "sectional"] += 0.2
    p["won"] = (p.position == 1).astype(float)
    b = form2.build(p, core.per_run_figures(p), {})[0]
    card = runs.copy()
    card.loc[last, ["position", "adjusted_time", "run_time", "sectional", "sp_decimal"]] = np.nan
    card.loc[last, "is_card"] = True
    card = clean(card)
    c = form2.build(card, core.per_run_figures(card), {})[0]
    cols = list(form2.COLUMNS)
    key = runs.loc[last, ["race_id", "dog_id"]]
    x, y, z = (key.merge(t, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
               for t in (a, b, c))
    assert np.allclose(x, y, equal_nan=True) and np.allclose(x, z, equal_nan=True)
    assert set(cols) <= set(a.columns) and len(a) == len(runs) and np.isfinite(x[:, 0]).mean() > 0.9
