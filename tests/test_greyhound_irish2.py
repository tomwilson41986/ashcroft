"""greyhound/pit/irish2.py (review "irish ie_") on planted data: two Irish tracks of different class (the strong
pool's figures read low against its fast winners), dogs that move to one GB track. The link offsets put the two on
one scale (the level ranks the dogs by their true speed better than the raw Irish figure), the walk-forward
transfer predicts the first GB figure of later movers, results on and after a day never move its rows or earlier
ones, a card row gets what its run had, and no API data gives NaN columns. Constructed data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.pit import irish2

KEY = ["race_id", "dog_id"]
COLS = list(irish2.COLUMNS)


def synth(weeks=150, seed=11):
    """A cohort of 12 dogs a week from 4 Jan 2021: six to Shelbourne (skill about +1), six to Limerick (about -1),
    eight weekly Irish races among themselves, then six weekly GB races at Romford (two races of six, mixed).
    Time = 30 - 0.1 x skill + noise."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2021-01-04")
    ie, gb, skill = [], [], {}
    rid, did = 0, 0
    for w in range(weeks):
        coh = []
        for j, (trk, mu) in enumerate((("Shelbourne", 1.0), ("Limerick", -1.0))):
            dogs = list(range(did, did + 6))
            did += 6
            for d in dogs:
                skill[d] = mu + rng.normal(0, 0.6)
            coh += dogs
            for k in range(8):
                rid += 1
                day = start + pd.Timedelta(weeks=w + k, days=j)
                t = np.array([30 - 0.1 * skill[d] + rng.normal(0, 0.04) for d in dogs])
                pos = np.argsort(np.argsort(t)) + 1
                for i, d in enumerate(dogs):
                    ie.append({"race_id": f"ie{rid}", "dog_id": f"api{d}", "dog_name": f"Dog {d}", "track": trk,
                               "distance_m": 525, "start_local": f"{day.date()}T20:00:00+00:00", "trap": i + 1,
                               "position": int(pos[i]), "run_time_s": float(t[i]), "adjusted_time_s": None,
                               "sectional_s": float(3.5 - 0.02 * skill[d] + rng.normal(0, 0.01)), "class_number": 3})
        for k in range(6):
            order = rng.permutation(coh)
            day = start + pd.Timedelta(weeks=w + 10 + k, days=3)
            for h in range(2):
                rid += 1
                dogs = order[6 * h:6 * h + 6]
                t = np.array([29.0 - 0.1 * skill[d] + rng.normal(0, 0.04) for d in dogs])
                pos = np.argsort(np.argsort(t)) + 1
                for i, d in enumerate(dogs):
                    gb.append({"race_id": rid, "dog_id": int(d) + 1, "dog_name": f"Dog {d}",
                               "race_date": str(day.date()), "race_time": f"{19 + h}:00", "track": "Romford",
                               "trap": i + 1, "position": int(pos[i]), "run_time": float(t[i]), "adjusted_time": np.nan,
                               "sectional": float(4.0 - 0.02 * skill[d] + rng.normal(0, 0.01)), "distance_m": 400.0,
                               "is_card": False, "is_trial": False})
    runs = pd.DataFrame(gb).sort_values(["race_date", "race_time", "race_id"]).reset_index(drop=True)
    return runs, pd.DataFrame(ie), {d + 1: s for d, s in skill.items()}


@pytest.fixture(scope="module")
def data():
    runs, ie, skill = synth()
    ctx = {"api_ie": ie, "api_gb": None}
    tab, notes = irish2.build(runs, None, ctx)
    return runs, ie, skill, ctx, tab, notes


def test_the_contract(data):
    runs, _, _, _, tab, notes = data
    assert list(tab.columns) == KEY + COLS and all(c.startswith(irish2.PREFIX) for c in COLS)
    assert all(tab[c].dtype == np.float32 for c in COLS) and len(tab) == len(runs)
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all()
    late = (runs.race_date >= "2022-03-01").to_numpy()
    for c in COLS:      # the field columns: a cohort's first GB race has no runner with a GB figure yet
        assert tab.loc[late, c].notna().mean() > (0.75 if c.endswith("_field") else 0.9), c
    assert tab.loc[runs.race_date < "2022-01-01", "iq_level"].isna().all()   # before the first refit
    assert len(notes["refits"]) == 3 and all(f["fitted"] for f in notes["refits"])


def test_the_offsets_put_the_two_tracks_on_one_scale(data):
    runs, ie, skill, _, tab, _ = data
    m = runs.assign(**{c: tab[c].to_numpy() for c in COLS}, skill=runs.dog_id.map(skill))
    m = m[m.iq_level.notna()].drop_duplicates("dog_id")
    from greyhound.pit import irish
    raw = irish.run_figures(ie.reset_index(drop=True)).assign(dog=lambda d: d.dog_id.str[3:].astype(int) + 1)
    raw = raw.groupby("dog").fig.mean()
    assert np.corrcoef(m.iq_level, m.skill)[0, 1] > 0.85
    assert np.corrcoef(m.iq_level, m.skill)[0, 1] > np.corrcoef(raw.loc[m.dog_id], m.skill)[0, 1] + 0.2
    assert np.corrcoef(m.iq_sec_level, m.skill)[0, 1] > 0.7
    assert (m.iq_level_sd > 0).all()


def test_the_transfer_predicts_later_movers_first_gb_figure(data):
    runs, _, _, _, tab, _ = data
    gf = irish2.gb_figures(runs)
    first = runs.loc[gf.index].assign(fig=gf.fig).sort_values("race_date").drop_duplicates("dog_id")
    first = first[first.race_date >= "2023-01-01"]
    p = tab.loc[first.index, "iq_pred_gb"].to_numpy(float)
    ok = np.isfinite(p) & first.fig.notna().to_numpy()
    assert ok.sum() > 100
    assert np.corrcoef(p[ok], first.fig.to_numpy()[ok])[0, 1] > 0.8



def test_the_gap_to_the_field_follows_the_true_gap(data):
    runs, _, skill, _, tab, _ = data
    m = runs.assign(gap=tab.iq_pred_vs_field.to_numpy(float), s=runs.dog_id.map(skill))
    tot, cnt = m.groupby("race_id").s.transform("sum"), m.groupby("race_id").s.transform("count")
    m["true_gap"] = m.s - (tot - m.s) / (cnt - 1)
    ok = m.gap.notna()
    assert ok.sum() > 1000 and np.corrcoef(m.gap[ok], m.true_gap[ok])[0, 1] > 0.5


def _rewrite_from(runs, ie, day):
    r, i = runs.copy(), ie.copy()
    m = r.race_date >= day
    r.loc[m, "run_time"] = 31.0 - (r.loc[m, "run_time"] - 29.0)
    r.loc[m, "position"] = 7 - r.loc[m, "position"]
    mi = i.start_local.str[:10] >= day
    i.loc[mi, "run_time_s"] = 33.0 - i.loc[mi, "run_time_s"] + 27.0
    i.loc[mi, "position"] = 7 - i.loc[mi, "position"]
    return r, i


def test_results_on_and_after_a_day_move_nothing_on_or_before_it(data):
    runs, ie, _, ctx, tab, _ = data
    day = "2023-06-15"
    assert (runs.race_date == day).any()
    r2, i2 = _rewrite_from(runs, ie, day)
    t2, _ = irish2.build(r2, None, {"api_ie": i2, "api_gb": None})
    upto = (runs.race_date <= day).to_numpy()
    a, b = tab.loc[upto, COLS].to_numpy(float), t2.loc[upto, COLS].to_numpy(float)
    assert np.array_equal(a, b, equal_nan=True)
    later = (runs.race_date > day).to_numpy()
    assert not np.array_equal(tab.loc[later, COLS].to_numpy(float), t2.loc[later, COLS].to_numpy(float), equal_nan=True)


def test_a_card_row_gets_what_its_run_had(data):
    runs, ie, _, ctx, tab, _ = data
    last = runs.race_date.max()
    c = runs.copy()
    m = c.race_date == last
    c.loc[m, ["position", "run_time", "sectional"]] = np.nan
    c["is_card"] = m
    t2, _ = irish2.build(c, None, ctx)
    assert np.allclose(tab.loc[m, COLS].to_numpy(float), t2.loc[m, COLS].to_numpy(float), equal_nan=True, atol=0)


def test_no_api_data_is_not_an_error(data):
    runs = data[0]
    tab, notes = irish2.build(runs, None, {})
    assert "no data found" in notes["note"] and tab[COLS].isna().all().all()
