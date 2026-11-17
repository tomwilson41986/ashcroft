"""greyhound/pit/league2.py (review "league lg_"): league.py's new keyword arguments leave it bit-identical; the
constants are fitted on the early slice only (later results do not move them) and a planted short memory is
found; the daily refit reads a Wednesday's results on the Thursday where the weekly table does not, and never a
race's own day; a card row gets what its run had. Planted data only (tests/test_greyhound_league.py's synth)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound.pit import league, league2
from tests.test_greyhound_league import rewrite_results, synth
from tests.test_greyhound_pit import cardify_last_day

KEY = ["race_id", "dog_id"]
COLS = list(league2.COLUMNS)
CONST = {"half_life": 180.0, "kd": 5.0, "kl": 20.0}


def same(a, b, keys, cols):
    x = keys.merge(a, on=KEY, how="left")[cols].to_numpy()
    y = keys.merge(b, on=KEY, how="left")[cols].to_numpy()
    return bool(np.array_equal(x, y, equal_nan=True))


@pytest.fixture(scope="module")
def runs():
    return synth(days=80)


def test_league_keywords_at_the_constants_are_bit_identical(runs):
    a, _ = league.build(runs)
    b, _ = league.build(runs, None, None, half_life=league.HALF_LIFE, kd=league.KD, kl=league.KL, step=7)
    assert a.equals(b)
    c, _ = league.build(runs, None, None, half_life=90.0)
    assert not a.equals(c)                                       # the keyword is read


def test_the_contract_and_the_prefix(runs):
    tab, notes = league2.build(runs, None, {"league2_constants": CONST})
    assert list(tab.columns) == KEY + COLS and all(c.startswith("lgd_") for c in COLS)
    assert len(COLS) == len(league.COLUMNS) and not any(c.startswith("lg_") for c in COLS)
    assert (tab.race_id.to_numpy() == runs.race_id.to_numpy()).all() and not tab.duplicated(KEY).any()
    late = (runs.race_date >= "2025-02-15").to_numpy()
    assert tab.loc[late, "lgd_str"].notna().mean() > 0.95
    assert notes["constants"]["from_ctx"] and "each day" in notes["refit"]


def test_constants_are_fitted_on_the_slice_only(runs):
    c1 = league2.fit_constants(runs)
    assert c1["fitted"] and c1["slice"] == "first_share" and c1["scoring_mondays"] >= 1
    assert c1["half_life"] in league2.GRID["half_life"] and c1["mse_fitted"] <= c1["mse_default"] + 1e-12
    after = pd.Timestamp(c1["slice_end"]) + pd.Timedelta(days=7)    # the scoring week after the slice's last Monday
    c2 = league2.fit_constants(rewrite_results(runs, after.strftime("%Y-%m-%d")))
    assert {k: c1[k] for k in CONST} == {k: c2[k] for k in CONST}


def _flipping(days=100, flip=50, seed=3):
    """Two leagues of 36 dogs; every dog's skill changes sign on day ``flip`` (form that does not last)."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1.0, 72)
    rows, rid = [], 0
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        sk = skill if d < flip else -skill
        home = np.repeat([0, 1], 36)
        away = rng.random(72) < 0.15
        where = np.where(away, 1 - home, home)
        for j in (0, 1):
            pool = rng.permutation(np.flatnonzero(where == j))
            for r in range(len(pool) // 6):
                rid += 1
                f = pool[6 * r:6 * r + 6]
                pos = np.argsort(np.argsort(-(sk[f] + rng.normal(0, 0.7, 6)))) + 1
                for k, dog in enumerate(f):
                    rows.append({"race_date": day, "race_id": rid, "dog_id": int(dog) + 1, "track": ("Hove", "Romford")[j],
                                 "race_class": "A3", "position": int(pos[k]), "is_card": False})
    return pd.DataFrame(rows)


def test_a_planted_short_memory_is_found():
    """Every dog's form reverses on day 50: scored after it, a short half-life beats a long one, and the slice's
    search picks one of the short ones."""
    r = _flipping()
    p = league2._prepare(r)
    dn = lambda s: int((np.datetime64(s) - np.datetime64("1970-01-01")).astype(int))
    m = league2._mondays(p, dn("2025-04-11"))
    m = m[m >= dn("2025-03-03")]
    short, _ = league2.score(p, m, 7.0, 5.0, 20.0)
    long_, _ = league2.score(p, m, 730.0, 5.0, 20.0)
    assert short < long_
    grid = {v: league2.score(p, m, v, 5.0, 20.0)[0] for v in league2.GRID["half_life"]}
    assert min(grid, key=grid.get) <= 90.0


def test_the_daily_refit_reads_earlier_days_of_the_week_never_its_own(runs):
    wed, thu = "2025-03-12", "2025-03-13"
    assert pd.Timestamp(wed).dayofweek == 2
    ctx = {"league2_constants": CONST}
    a, _ = league2.build(runs, None, ctx)
    b, _ = league2.build(rewrite_results(runs, wed), None, ctx)
    upto = runs.loc[runs.race_date <= wed, KEY]
    assert same(a, b, upto, COLS)                                # the day's own results never enter its rows
    t = runs.loc[runs.race_date == thu, KEY]
    assert not same(a, b, t, ["lgd_str", "lgd_rating_vs", "lgd_sos_prb2"])   # Thursday reads Wednesday
    w1, _ = league.build(runs)
    w2, _ = league.build(rewrite_results(runs, wed))
    assert same(w1, w2, t, list(league.COLUMNS))                 # the weekly table does not


def test_a_card_row_gets_what_its_run_had(runs):
    ctx = {"league2_constants": CONST}
    a, _ = league2.build(runs, None, ctx)
    card = cardify_last_day(runs)
    b, _ = league2.build(card, None, ctx)
    keys = runs.loc[runs.race_date == runs.race_date.max(), KEY]
    x, y = keys.merge(a, on=KEY)[COLS].to_numpy(), keys.merge(b, on=KEY)[COLS].to_numpy()
    assert np.allclose(x, y, atol=1e-6, equal_nan=True) and np.isfinite(y).mean() > 0.9
