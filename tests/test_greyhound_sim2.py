"""greyhound/pit/sim2.py: the race simulator with fitted constants, shrunk inputs, a lane and trial/debut priors
(the feature review of 17 Nov 2026, row "sim block"). On a planted card (each dog an early pace and a run-in; the
bend run with noise of 2 lengths, twice sim.py's hand-set 1; late dogs debut after a trial): the fit moves the bend
noise toward the planted value and beats the hand-set constants on the later pre-test races; a debutant's input comes
from its trial; the lane and the shrinkage do what they say by hand; the day's own results move nothing and a card row
gets what its run had. All data here is planted for the tests; nothing is evaluated on it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, sim2
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S
COLS = list(sim2.COLUMNS)
START = pd.Timestamp("2025-01-01")
EARLY_SD = 2.0


def synth(days=150, seed=5, late_from=90):
    rng = np.random.default_rng(seed)
    pace, run = rng.normal(0, 2.0, 81), rng.normal(0, 1.5, 81)
    rows, rid = [], 5000

    def row(day, track, r, ti, rid_, cls, trap, dog, pos, bend, fin):
        return {"race_date": day, "race_time": f"{14 + r}:{10 * ti:02d}", "meeting_id": 2 * d + ti, "track": track,
                "race_id": rid_, "race_class": cls, "distance_m": 480.0, "trap": int(trap), "dog_id": int(dog),
                "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "d", "sp_decimal": 6.0, "position": int(pos),
                "beaten_lengths": 0.5, "sectional": float(4.5 - bend * L), "run_time": float(29.5 - fin * L),
                "adjusted_time": float(29.5 - fin * L), "weight_kg": 30.0, "comment": "QAw,Mid"}

    for d in range(days):
        day = (START + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        pool = np.arange(1, 81) if d >= late_from else np.arange(1, 61)
        dogs = rng.permutation(pool)
        for ti, track in enumerate(("Hove", "Romford")):
            for r in range(4):
                field = dogs[6 * (4 * ti + r):6 * (4 * ti + r) + 6]
                rid += 1
                bend = pace[field] + rng.normal(0, EARLY_SD, 6)
                fin = bend + run[field] + rng.normal(0, 1.5, 6)
                pos = np.argsort(np.argsort(-fin)) + 1
                for k, dog in enumerate(field):
                    rows.append(row(day, track, r, ti, rid, "A3", k + 1, dog, pos[k], bend[k], fin[k]))
        if late_from - 6 <= d < late_from:                    # the late dogs' trials, in pairs, before they debut
            for t, pair in enumerate(np.arange(61, 81).reshape(-1, 2)):
                rid += 1
                bend = pace[pair] + rng.normal(0, 0.5, 2)
                fin = bend + run[pair]
                pos = np.argsort(np.argsort(-fin)) + 1
                for k, dog in enumerate(pair):
                    rows.append(row(day, "Hove", 0, 0, rid, "T3", 2 * k + 1, dog, pos[k], bend[k], fin[k]))
    return clean(pd.DataFrame(rows)), pace


def _patch(mp):
    mp.setattr(sim2, "MIN_FIT_RACES", 100)
    mp.setattr(sim2, "WARMUP_DAYS", 20)
    mp.setattr(sim2, "FIT_END", "2025-04-01")              # the fit window: 21 Jan to 7 Feb 2025
    mp.setattr(sim2, "FIT_FRAC", 0.25)
    mp.setattr(sim2, "MAX_EVALS", 250)


@pytest.fixture(scope="module")
def built():
    mp = pytest.MonkeyPatch()
    _patch(mp)
    runs, pace = synth()
    tab, notes = sim2.build(runs, core.per_run_figures(runs))
    mp.undo()
    m = runs[["race_id", "dog_id", "race_date", "trap", "sectional", "won", "field", "is_trial"]].merge(
        tab, on=["race_id", "dog_id"])
    return runs, pace, m, tab, notes


def test_a_row_a_run_and_every_column_registered(built):
    runs, _, m, tab, notes = built
    assert list(tab.columns) == ["race_id", "dog_id"] + COLS
    assert all(c.startswith(sim2.PREFIX) for c in COLS) and all(sim2.COLUMNS.values())
    assert len(tab) == len(runs) and not tab.duplicated(["race_id", "dog_id"]).any()
    assert (tab[COLS].dtypes == np.float32).all()
    assert notes["fitted"] and notes["window"] == ["2025-01-21", "2025-02-07"]
    assert set(notes["card_safe"]) == set(COLS)


def test_the_fit_moves_toward_the_planted_noise_and_beats_the_hand_constants(built):
    _, _, m, _, notes = built
    c = notes["constants"]
    assert c["early_sd"] > 1.3 * sim2.HAND["early_sd"]                     # planted 2.0, hand-set 1.0
    assert notes["ll_fit"] > notes["ll_fit_hand"]
    assert notes["val_races"] > 100 and notes["ll_val"] > notes["ll_val_hand"]  # races after the window, pre-test


def test_the_simulation_adds_up_and_finds_the_leader(built):
    _, _, m, _, _ = built
    late = m[(m.race_date >= "2025-03-01") & ~m.is_trial]
    g = late.groupby("race_id")
    assert np.allclose(g.sm2_p_lead.sum(), 1.0, atol=1e-5) and np.allclose(g.sm2_p_win.sum(), 1.0, atol=1e-5)
    assert np.allclose(g.sm2_p_top2.sum(), 2.0, atol=1e-5)
    f = g.field.first()
    assert np.allclose(g.sm2_exp_bend.sum(), f * (f + 1) / 2, atol=1e-4)
    leader = late.loc[g.sectional.idxmin()]
    assert leader.sm2_p_lead.mean() > 1 / 6 + 0.1
    assert late[late.won == 1].sm2_p_win.mean() > 1 / 6 + 0.05
    assert ((late.sm2_p_crowd >= 0) & (late.sm2_p_crowd <= 1)).all() and (late.sm2_lost >= 0).all()


def test_a_debutant_reads_its_trial(built):
    _, pace, m, _, _ = built
    deb = m[(m.sm2_prior == 1) & ~m.is_trial]
    assert len(deb) >= 10 and (deb.sm2_evid == 1).all()
    assert np.corrcoef(deb.sm2_early, pace[deb.dog_id])[0, 1] > 0.5


def _x(**cols):
    base = {"race": 1, "trap": 1.0, "esr_m": np.nan, "esr_n": 0.0, "fsr_m": np.nan, "fsr_n": 0.0, "esr_par": 0.0,
            "fsr_par": 0.0, "sd": 2.0, "path_m": np.nan, "path_n": 0.0, "t_sec": np.nan, "t_run": np.nan,
            "t_path": np.nan}
    n = max(len(v) if hasattr(v, "__len__") else 1 for v in cols.values())
    return pd.DataFrame({k: cols.get(k, [v] * n) for k, v in base.items()})


def test_shrinkage_trial_and_debut_priors_by_hand():
    p = dict(sim2.HAND, k=2.0, debut=-1.0, trial_off=0.5, mis=0.0)
    x = _x(esr_m=[5.0, np.nan, np.nan], esr_n=[1.0, 0.0, 0.0], esr_par=[1.0, 1.0, 1.0],
           t_sec=[9.0, 3.0, np.nan], t_run=[np.nan, 1.0, np.nan])
    f = sim2.features(x, p)
    assert np.isclose(f["early"][0], (5.0 + 2 * 1.0) / 3)               # one day of +5 toward a par of 1, the trial unread
    assert np.isclose(f["early"][1], (3.5 + 2 * (1.0 - 1.0)) / 3)       # the trial (+ offset) and the debut par
    assert np.isclose(f["early"][2], 0.0)                                # the debut prior alone: par + debut
    assert list(f["prior"]) == [0.0, 1.0, 2.0] and list(f["evid"]) == [1.0, 1.0, 0.0]
    assert np.isclose(f["mult"][0], 1 + 0.5 / 2) and np.isclose(f["mult"][2], 1.5)
    # the line: a railer's history makes its box 1-ish, shrunk to the middle (3.5) by K_PATH days
    y = _x(path_m=[1.0, np.nan], path_n=[8.0, 0.0], trap=[6.0, 3.0])
    g = sim2.features(y, dict(p, lane=0.5))
    assert np.isclose(g["box"][0], 2.5 * (8 + 2 * sim2.K_PATH) / (8 + sim2.K_PATH) - 1.5)
    assert np.isclose(g["box"][1], 3.5) and np.isclose(g["lat"][0], 6 + 0.5 * (g["box"][0] - 6))


def test_the_lane_brings_converging_dogs_together():
    """Traps 1 and 4 level at the bend: no collision when each keeps its trap (3 boxes apart, REACH 2); a collision
    when the trap 4 dog is a railer and the lane weight takes it in."""
    p = dict(sim2.HAND, early_sd=0.3, close=1.5, check=2.0)
    z = sim2._draws(2, 400, 9)
    early = np.array([[0.2, 0.0]])
    args = (early, np.zeros((1, 2)), np.full((1, 2), 2.0), np.ones((1, 2)))
    apart = sim2.simulate(*args, np.array([[1.0, 4.0]]), p, z)
    inside = sim2.simulate(*args, np.array([[1.0, 4.0 + 1.0 * (1.0 - 4.0)]]), p, z)
    assert apart["p_crowd"].max() == 0 and apart["lost"].max() == 0
    assert inside["p_crowd"].min() > 0.2 and inside["lost"].min() > 0
    assert np.allclose(apart["p_lead"].sum(), 1) and np.allclose(apart["exp_bend"].sum(), 3)


def test_unfitted_falls_back_to_the_hand_constants(monkeypatch):
    monkeypatch.setattr(sim2, "MIN_FIT_RACES", 10 ** 6)
    runs, _ = synth(days=40, late_from=1000)
    tab, notes = sim2.build(runs, core.per_run_figures(runs))
    assert notes["fitted"] is False and tab[COLS].notna().all().all()


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change, monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(sim2, "MAX_EVALS", 60)
    runs, _ = synth(days=120)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = sim2.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = sim2.build(r2, core.per_run_figures(r2))[0]
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[COLS].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[COLS].to_numpy(float)
    assert len(x) and np.isfinite(x).all() and np.allclose(x, z, atol=1e-5)
