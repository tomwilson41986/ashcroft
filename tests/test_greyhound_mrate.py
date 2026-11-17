"""greyhound/pit/mrate.py (review "figs fg_mr" and B5): a trap the market overrates does not raise the rating of the
dogs that drew it (it does in figs.market_rating); the noise is fitted larger for longshots when their shares are
noisier; the error memory finds a planted underrated set; a layoff pulls the rating toward the field it last met;
a day's own results move none of its rows; a card row gets what its run gets. The data is planted (a synthetic
card), as a unit test's may be; nothing is evaluated on it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.data import clean
from greyhound.pit import mrate
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

KEY = ["race_id", "dog_id"]


def synth(days=50, seed=0, trap_boost=1.0, noise_long=0.0, noise_fav=0.0, bonus=0.0, overround=1.2):
    """Two tracks, 6 races of 6 a day from 90 dogs, traps at random. The SP prices each dog's skill; at Hove trap 1
    the market's share is ``trap_boost`` times the dog's (no true effect); the log share carries noise of sd
    ``noise_long`` under a 0.12 share and ``noise_fav`` above; dogs with id % 10 == 0 win exp(``bonus``) times
    more than the market says."""
    rng = np.random.default_rng(seed)
    dogs = np.arange(1, 91)
    skill = rng.normal(0, 0.8, 91)
    rows, rid = [], 1000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for ti, track in enumerate(("Romford", "Hove")):
            pool = rng.permutation(dogs)
            for r in range(6):
                rid += 1
                field = pool[r * 6:(r + 1) * 6]
                pt = np.exp(skill[field])
                pt = pt / pt.sum()
                w = pt * np.exp(np.where(field % 10 == 0, bonus, 0.0))
                winner = rng.choice(6, p=w / w.sum())
                pm = pt.copy()
                if track == "Hove":
                    pm[0] *= trap_boost
                pm = pm / pm.sum()
                pm = pm * np.exp(rng.normal(0, 1, 6) * np.where(pt < 0.12, noise_long, noise_fav))
                pm = pm / pm.sum()
                odds = 1.0 / (pm * overround)
                pos = np.empty(6, int)
                rest = [k for k in rng.permutation(6) if k != winner]
                pos[winner] = 1
                pos[rest] = np.arange(2, 7)
                for k, dog in enumerate(field):
                    rows.append({"race_date": day, "race_time": f"{18 + r}:{10 * ti:02d}", "meeting_id": d * 10 + ti,
                                 "track": track, "race_id": rid, "race_class": "A3", "distance_m": 480.0,
                                 "trap": k + 1, "dog_id": int(dog), "dog_name": f"Dog {dog}",
                                 "trainer": f"Trainer {dog % 5}", "sp_decimal": float(odds[k]),
                                 "market_pos": int(np.argsort(np.argsort(odds))[k]) + 1, "position": int(pos[k]),
                                 "beaten_lengths": 1.0, "sectional": 4.8, "run_time": 29.0, "adjusted_time": 29.0,
                                 "weight_kg": 30.0, "comment": "QAw,Rls"})
    return clean(pd.DataFrame(rows))


def rows_of(tab, runs, day):
    return runs.loc[runs.race_date == day, KEY].merge(tab, on=KEY).sort_values(KEY)


def _gap(m, runs):
    """The mean rating of dogs whose last earlier run was from Hove trap 1, less the others' (the last 20 days)."""
    d = runs.assign(h1=((runs.track == "Hove") & (runs.trap == 1)).astype(float))
    prev = d.sort_values("t").groupby("dog_id").h1.shift(1).reindex(d.index)
    late = (runs.race_date >= sorted(runs.race_date.unique())[-20]) & prev.notna()
    return float(np.mean(m[late & (prev == 1)]) - np.mean(m[late & (prev == 0)]))


def test_a_strong_trap_does_not_raise_the_dogs_rating():
    """The market triples Hove trap 1's share with no true effect. Read without the trap mean (figs' observation),
    the dogs that last drew it rate higher; with it, they rate as they do when nothing is planted (same draws)."""
    runs = synth(days=50, trap_boost=3.0)
    tab, notes = mrate.build(runs)
    base, _ = mrate.build(synth(days=50, trap_boost=1.0))
    prm = {**notes["rating"], "tau": np.inf}
    q = mrate.shin_probs(runs.sp_decimal, runs.race_id)
    blind, _, _ = mrate.rating(runs, q, np.zeros(len(runs)), prm)
    with_trap, _, _ = mrate.rating(runs, q, tab.mv_trap.to_numpy(float), prm)
    g_blind, g_trap, g_base = _gap(blind.m, runs), _gap(with_trap.m, runs), _gap(base.mv_mr, runs)
    assert g_blind > g_base + 0.2
    assert abs(g_trap - g_base) < 0.35 * (g_blind - g_base)
    late = runs.race_date >= sorted(runs.race_date.unique())[-20]
    hove1 = late & (runs.track == "Hove") & (runs.trap == 1)
    assert tab.mv_trap[hove1].mean() > 0.5
    # the forecast share puts the trap back: Hove trap 1 is forecast a larger share than the other Hove traps
    hove = late & (runs.track == "Hove")
    assert tab.mv_mr_share[hove1].mean() > tab.mv_mr_share[hove & (runs.trap != 1)].mean()


def test_longshots_noisier_shares_get_a_larger_noise():
    runs = synth(days=60, noise_long=0.8, noise_fav=0.05, seed=2)
    _, notes = mrate.build(runs)
    rb = notes["rating"]["band_r_raw"]
    cnt = notes["rating"]["band_counts"]
    assert cnt[0] >= mrate.MIN_BAND and cnt[3] >= mrate.MIN_BAND
    assert rb[0] > 2 * rb[3]


def test_the_error_memory_finds_a_planted_underrated_set():
    runs = synth(days=60, bonus=1.2, seed=3)
    tab, notes = mrate.build(runs)
    late = runs.race_date >= sorted(runs.race_date.unique())[-15]
    u = runs.dog_id % 10 == 0
    assert tab.mv_err[late & u].mean() > tab.mv_err[late & ~u].mean() + 0.03
    assert notes["error"]["fitted"]


def test_a_layoff_pulls_the_rating_toward_the_field_it_last_met():
    """Dog 1 rates high from a field of 0-rated dogs, then is off 200 days: with tau the rating comes back toward
    the field's mean; with no reversion it does not move."""
    rows = []
    for i, day in enumerate(["2025-01-01", "2025-01-08", "2025-01-15", "2025-08-01"]):
        for k, dog in enumerate((1, 2, 3)):
            rows.append({"race_date": day, "race_id": 10 + i, "dog_id": dog, "is_trial": False, "field": 3,
                         "sp": (1.5, 6.0, 6.0)[k]})
    df = pd.DataFrame(rows)
    q = (1 / df.sp) / (1 / df.sp).groupby(df.race_id).transform("sum")
    te = np.zeros(len(df))
    base = {"r_band": [0.2] * (len(mrate.BANDS) - 1), "q": 0.0, "v0": 1.0}
    a, _, _ = mrate.rating(df, q.to_numpy(), te, {**base, "tau": np.inf})
    b, _, _ = mrate.rating(df, q.to_numpy(), te, {**base, "tau": 60.0})
    last = (df.race_date == "2025-08-01") & (df.dog_id == 1)
    assert a.m[last].iloc[0] > 0.5 and a.pull[last].iloc[0] == 0.0
    assert 0 < b.m[last].iloc[0] < 0.2 * a.m[last].iloc[0] + 0.1 and b.pull[last].iloc[0] < 0
    # no reversion within GAP0 days
    mid = (df.race_date == "2025-01-15") & (df.dog_id == 1)
    assert b.pull[mid].iloc[0] == 0.0


def test_a_days_own_results_move_none_of_its_rows():
    runs = synth(days=30, trap_boost=2.0, bonus=0.8)
    last = runs.race_date.max()
    a, _ = mrate.build(runs)
    b, _ = mrate.build(perturb_last_day(runs))
    cols = list(mrate.COLUMNS)
    x = rows_of(a, runs, last)[[c for c in cols if not c.startswith("mv_bmr")]].to_numpy(float)
    y = rows_of(b, runs, last)[[c for c in cols if not c.startswith("mv_bmr")]].to_numpy(float)
    assert np.isfinite(x[:, 0]).mean() > 0.8 and np.allclose(x, y, equal_nan=True)


def test_a_card_gets_what_its_run_gets():
    runs = synth(days=30, trap_boost=2.0, bonus=0.8)
    last = runs.race_date.max()
    card = cardify_last_day(runs)
    a, _ = mrate.build(runs)
    b, _ = mrate.build(card)
    cols = list(mrate.COLUMNS)
    x = rows_of(a, runs, last)[cols].to_numpy(float)
    y = rows_of(b, card, last)[cols].to_numpy(float)
    assert len(x) == len(y) > 0 and np.allclose(x, y, equal_nan=True)


def test_the_bsp_rating_reads_the_exchange_where_the_race_is_whole():
    runs = synth(days=20)
    q = 1.0 / runs.sp_decimal
    runs = runs.assign(bsp_p_norm=(q / q.groupby(runs.race_id).transform("sum")).where(runs.track == "Hove"))
    tab, notes = mrate.build(runs)
    assert notes["bsp_rating"]["races_on_bsp"] == 20 * 6
    assert tab.mv_bmr.notna().mean() > 0.8


def test_columns_and_prefix():
    runs = synth(days=10)
    tab, _ = mrate.build(runs)
    assert list(tab.columns) == KEY + list(mrate.COLUMNS) and len(tab) == len(runs)
    assert mrate.DEPENDS == ("figs", "market2")
