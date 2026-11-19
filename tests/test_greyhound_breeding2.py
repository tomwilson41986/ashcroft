"""greyhound/pit/breeding2.py (the feature review of 17 Nov 2026, breeding): dog-first, leave-one-dog-out (and
littermates for the dam), empirical-Bayes progeny means; point in time and card-safe. Constructed frames only."""
import numpy as np
import pandas as pd

from greyhound.data import clean
from greyhound.pit import breeding2 as b2, core
from tests.greyhound_pit_synth2 import synth


def _rows(unit, key, lit, dn):
    return pd.DataFrame({"unit": unit, "key": key, "lit": lit, "dn": np.asarray(dn, float)})


def test_progeny_is_averaged_a_dog_first_and_leaves_the_dog_out():
    # sire A: dog 1 ran 10 times at 1.0, dog 2 once at 5.0; dog 3 (the query) once at 100 (its own: never counted)
    unit = ["1"] * 10 + ["2", "3"]
    ev = b2.unit_events(unit, ["a"] * 12, ["l1"] * 10 + ["l2", "l3"], [1] * 10 + [2, 3], [1.0] * 10 + [5.0, 100.0])
    S = b2.before(ev, "key", ["dS", "dM"], ["a"], [10])[0]
    assert S[1] == 3 and np.isclose(S[0], 106.0)                       # 3 dogs, the sum of their means
    rows = _rows(["3"], ["a"], ["l3"], [10])
    X = b2.before(ev.assign(uu=ev.u), "uu", ["dS", "dM"], rows.unit, rows.dn)[0]
    assert np.isclose((S[0] - X[0]) / (S[1] - X[1]), 3.0)              # (1 + 5) / 2: dog-first, dog 3 out
    # strictly earlier days: on day 2 only dog 1 is known
    assert b2.before(ev, "key", ["dS", "dM"], ["a"], [2])[0][1] == 1


def test_the_dam_leaves_out_the_whole_litter():
    # dam D: litter X (dogs 1, 2) very fast, litter Y (dog 3) slow; dog 1's dam reading is litter Y's alone
    ev = b2.unit_events(["1", "2", "3"], ["d"] * 3, ["d|x", "d|x", "d|y"], [1, 1, 1], [10.0, 10.0, -2.0])
    rows = _rows(["1"], ["d"], ["d|x"], [5])
    S = b2.before(ev, "key", ["dS", "dM"], rows.key, rows.dn)[0]
    X = b2.before(ev, "lit", ["dS", "dM"], rows.lit, rows.dn)[0]
    assert S[1] - X[1] == 1 and np.isclose(S[0] - X[0], -2.0)


def test_fit_k_recovers_the_spread_ratio():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(400), 6)
    x = rng.normal(0, 1.0, 400)[g] + rng.normal(0, 2.0, len(g))         # tau 1, sigma 2: k = 4
    fit = b2.fit_k(pd.Series(x), pd.Series(g.astype(str)))
    assert 3.0 < fit["k"] < 5.3


def test_a_planted_sire_effect_is_recovered_and_dams_with_none_are_shrunk():
    runs, eff = synth()
    tab, notes = b2.build(runs, core.per_run_figures(runs), {})
    late = (runs.race_date >= "2025-02-15").to_numpy()
    s = runs.sire.str.split().str[1].astype(int).to_numpy()
    assert np.corrcoef(tab.bq_sire_gsr[late], eff[s[late]])[0, 1] > 0.9
    assert tab.bq_dam_gsr[late].std() < 0.2 * tab.bq_sire_gsr[late].std()   # no planted dam effect: shrunk hard
    assert set(b2.COLUMNS) <= set(tab.columns) and len(tab) == len(runs)


def test_no_row_reads_its_days_own_results_and_the_card_gets_the_same_values():
    runs, _ = synth(days=30)
    last = runs.race_date == runs.race_date.max()
    a = b2.build(runs, core.per_run_figures(runs), {})[0]
    p = runs.copy()
    p.loc[last, "position"] = 7 - p.loc[last, "position"]
    p.loc[last, ["adjusted_time", "run_time"]] += 0.5
    p.loc[last, "sp_decimal"] = 9.0
    p["won"] = (p.position == 1).astype(float)
    p["sp_p"] = 1.0 / p.sp_decimal
    p["sp_p_norm"] = p.sp_p / p.groupby("race_id").sp_p.transform("sum")
    b = b2.build(p, core.per_run_figures(p), {})[0]
    card = runs.copy()
    card.loc[last, ["position", "adjusted_time", "run_time", "sectional", "sp_decimal"]] = np.nan
    card.loc[last, "is_card"] = True
    card = clean(card)
    c = b2.build(card, core.per_run_figures(card), {})[0]
    cols = list(b2.COLUMNS)
    key = runs.loc[last, ["race_id", "dog_id"]]
    x, y, z = (key.merge(t, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
               for t in (a, b, c))
    assert np.allclose(x, y, equal_nan=True) and np.allclose(x, z, equal_nan=True)


def test_trial_only_progeny_count_from_the_day_after():
    runs, _ = synth(days=30, trial_every=0)
    tr = pd.DataFrame({"race_date": "2025-01-10", "race_time": "", "meeting_id": -1, "track": "Romford",
                       "distance_m": 480.0, "race_id": 900 + np.arange(4), "race_class": "T3", "race_type": "flat",
                       "going": 0.0, "trap": 1, "dog_id": 5001 + np.arange(4), "dog_name": "x",
                       "sire": "Sire 0", "dam": "Dam 99", "born": "Jan-2024", "sex": "d", "season": None,
                       "trainer": "T", "position": 1, "sectional": 4.8, "run_time": 28.9, "adjusted_time": 28.9,
                       "weight_kg": 30.0, "comment": "QAw"})
    a, na = b2.build(runs, core.per_run_figures(runs), {})
    b, nb = b2.build(runs, core.per_run_figures(runs), {"trial_runs": tr})
    assert na["trial_figures"] == 0 and nb["trial_figures"] == 4
    s0 = (runs.sire == "Sire 0").to_numpy()
    on = (runs.race_date == "2025-01-10").to_numpy()
    after = (runs.race_date > "2025-01-10").to_numpy()
    assert (b.bq_sire_fig_dogs[s0 & on] == a.bq_sire_fig_dogs[s0 & on]).all()
    assert (b.bq_sire_fig_dogs[s0 & after] == a.bq_sire_fig_dogs[s0 & after] + 4).all()
    assert np.allclose(b.bq_sire_gsr, a.bq_sire_gsr, equal_nan=True)     # the engine figures have no trials


def test_a_run_with_no_trip_builds():
    """A run with no distance (pandas 3 keeps NaN through astype(str)) is band '?' and does not break the build
    (greyhound-pit.yml 44122130606)."""
    runs, _ = synth(days=30)
    runs = runs.copy()
    runs.loc[runs.index[::7], "distance_m"] = np.nan
    tab, _ = b2.build(runs, core.per_run_figures(runs), {})
    assert len(tab) and tab.filter(like="bq_").notna().any().any()
