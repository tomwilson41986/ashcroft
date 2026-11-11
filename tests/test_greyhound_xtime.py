"""greyhound/pit/xtime.py: the expected time of each dog in each race. Read from earlier days only; a card row gets
what its run had; the dog's level, a planted trap effect, a planted layoff effect and the price of trouble found; a
meeting run slow on every dog read as the going, not the dogs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.data import clean
from greyhound.pit import core, xtime
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

TRAP_COST = {1: -0.10, 6: 0.10}          # seconds: trap 1 quick, trap 6 slow at Romford
LAYOFF_COST = 0.20                       # seconds slower on the first run after 40 days off
SLOW_NIGHT = 0.30                        # seconds: one meeting runs slow on every dog


def synth(days=120, seed=1):
    rng = np.random.default_rng(seed)
    dogs = np.arange(1, 161)
    skill = rng.normal(0, 0.12, len(dogs) + 1)                 # seconds, below 0 faster
    rest = {int(d): -1 for d in dogs}                           # the day a dog comes back from a break
    last_run = {}
    rows, rid = [], 5000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for ti, (track, std) in enumerate((("Romford", 24.5), ("Hove", 29.8))):
            slow = SLOW_NIGHT if (d == days - 20 and ti == 0) else rng.normal(0, 0.08)   # the going of the night
            for r in range(6):
                rid += 1
                free = [int(x) for x in dogs if rest[int(x)] <= d and last_run.get(int(x)) != d]   # a run a day
                field = rng.choice(free, 6, replace=False)
                crd = rng.random(6) < 0.2
                t = np.empty(6)
                for k, dog in enumerate(field):
                    off = d - last_run.get(int(dog), d)
                    t[k] = (std + skill[dog] + (TRAP_COST.get(k + 1, 0.0) if ti == 0 else 0.0)
                            + (LAYOFF_COST if off >= 40 else 0.0) + (0.25 if crd[k] else 0.0) + slow
                            + rng.normal(0, 0.06))
                    last_run[int(dog)] = d
                    if rng.random() < 0.012:
                        rest[int(dog)] = d + 45
                pos = np.argsort(np.argsort(t)) + 1
                for k, dog in enumerate(field):
                    rows.append({"race_date": day, "race_time": f"{18 + r}:{10 * ti:02d}", "meeting_id": d * 10 + ti,
                                 "track": track, "race_id": rid, "race_class": f"A{1 + dog % 4}",
                                 "distance_m": 400.0 if ti == 0 else 500.0, "trap": k + 1, "dog_id": int(dog),
                                 "dog_name": f"Dog {dog}", "born": "Jan-2023", "sex": "b" if dog % 2 else "d",
                                 "sp_decimal": 6.0, "position": int(pos[k]), "beaten_lengths": 0.5,
                                 "sectional": 4.0 + skill[dog] * 0.2 + rng.normal(0, 0.02),
                                 "run_time": float(t[k]), "adjusted_time": float(t[k]), "weight_kg": 30.0,
                                 "comment": "Crd1" if crd[k] else "QAw"})
    return clean(pd.DataFrame(rows)), skill


@pytest.fixture()
def small(monkeypatch):
    monkeypatch.setattr(xtime, "MIN_FIT", 500)
    monkeypatch.setattr(xtime, "REFIT_MONTHS", 1)
    monkeypatch.setattr(xtime, "MIN_COUNT", 50)
    monkeypatch.setattr(xtime, "K_TRAP", 10.0)
    monkeypatch.setattr(xtime, "LAM", 1.0)


@pytest.fixture()
def built(small):
    runs, skill = synth()
    tab, notes = xtime.build(runs, core.per_run_figures(runs))
    return runs, skill, runs.merge(tab, on=["race_id", "dog_id"]), notes


def test_every_column_is_registered_and_the_table_is_a_row_a_run(built):
    runs, _, m, _ = built
    cols = [c for c in m.columns if c.startswith(xtime.PREFIX)]
    assert set(cols) == set(xtime.COLUMNS) and len(m) == len(runs)


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(small, change):
    runs, _ = synth(days=70)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = xtime.build(runs, core.per_run_figures(runs))[0]
    r2 = change(runs)
    b = xtime.build(r2, core.per_run_figures(r2))[0]
    cols = list(xtime.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.isfinite(x).mean() > 0.9 and np.allclose(x, z, atol=1e-5, equal_nan=True)


def test_the_level_finds_the_dog_and_the_forecast_beats_the_standard(built):
    runs, skill, m, notes = built
    late = m[m.race_date >= "2025-03-15"]
    assert np.corrcoef(late.xt_tot_level, -skill[late.dog_id.to_numpy()])[0, 1] > 0.85
    y = notes["tot"]["by_year"][2025]
    assert y["rmse_within_race"] < 0.8 * y["rmse_within_race_no_dog"]
    # the expected time in seconds is the time it ran, give or take the noise, the trouble and a layoff
    err = (late.xt_tot_time - late.adjusted_time)
    assert abs(err.mean()) < 0.05 and err.std() < 0.16


def test_the_trap_the_layoff_and_the_trouble_are_priced(built):
    _, _, m, notes = built
    late = m[(m.race_date >= "2025-04-01") & (m.track == "Romford")]
    by = late.groupby("trap").xt_trap.mean()                    # lengths, higher faster
    assert by[1] > 0.5 and by[6] < -0.5 and abs(by[3]) < 0.4     # 0.10 s is 1.25 lengths
    hove = m[(m.race_date >= "2025-04-01") & (m.track == "Hove")].groupby("trap").xt_trap.mean()
    assert hove.abs().max() < 0.4
    fx = notes["tot"]["effects_pct"]
    # the first run back after 45 days off is both "36-60 days off" and "run 1 since a layoff": the two together
    back = np.mean([fx[f"off:36-60d {s}"] + fx[f"back:run 1 {s}"] for s in ("dog", "bitch")])
    usual = np.mean([fx[f"off:{o} {s}"] + fx[f"back:run 6 {s}"] for o in ("<5d", "5-7d") for s in ("dog", "bitch")])
    assert back - usual > 0.4                                    # 0.20 s on about 27 s is 0.7%
    assert notes["tot"]["code_cost_pct_top"]["crd1"] > 0.3       # 0.25 s against a clean run


def test_a_slow_night_is_the_going_not_the_dogs(small):
    runs, _ = synth()
    tab, _ = xtime.build(runs, core.per_run_figures(runs))
    m = runs.merge(tab, on=["race_id", "dog_id"])
    night = sorted(runs.race_date.unique())[-20]
    ran = set(m[(m.race_date == night) & (m.track == "Romford")].dog_id)
    nxt = m[(m.race_date > night) & m.dog_id.isin(ran)].sort_values("t").groupby("dog_id").head(1)
    # 0.30 s is 3.75 lengths: the dogs' next surprise shows a small part of it at most
    assert nxt.xt_last_surprise.mean() > -1.2
