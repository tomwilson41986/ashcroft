"""greyhound/pit/market2.py (review B5, market): the gamma-posterior A/E on the de-biased SP finds a planted
underrated dog set; it does not read habitual favourites as beating the market when the SP's bias is Shin's; the
drift on race-normalised shares does not see the bookmakers' overround; the decayed sums match a brute-force reading;
a day's own results move none of its rows; a card row gets what its run gets. The data is planted (a synthetic card),
as a unit test's may be; nothing is evaluated on it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from greyhound.data import clean
from greyhound.pit import core, market, market2
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

KEY = ["race_id", "dog_id"]


def shin_odds(p: np.ndarray, z: float) -> np.ndarray:
    """Decimal odds a Shin bookmaker quotes for true chances ``p`` with insider share ``z``."""
    s = np.sqrt(z * p + (1 - z) * p * p)
    return 1.0 / (s * s.sum())


def synth(days=60, seed=0, bonus=0.0, z=0.0, overround=1.0, fav=False, bsp=False):
    """Two tracks, 6 races of 6 a day from 90 dogs. The SP prices each dog's skill; dogs with id % 10 == 0 win
    exp(``bonus``) times more than the market says. ``z``: a Shin bookmaker; ``overround``: a proportional margin.
    ``fav``: dogs 1-12 are always in, each race with one of them, the strong dog. ``bsp``: a Betfair SP with
    the true shares and no margin."""
    rng = np.random.default_rng(seed)
    dogs = np.arange(1, 91)
    skill = rng.normal(0, 0.8, 91)
    rows, rid = [], 1000
    for d in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        for ti, track in enumerate(("Romford", "Hove")):
            pool = rng.permutation(dogs[12:] if fav else dogs)
            for r in range(6):
                rid += 1
                field = pool[r * 6:(r + 1) * 6].copy()
                if fav:
                    field[0] = 1 + (d * 12 + ti * 6 + r) % 12
                    s = np.r_[np.log(5.0), np.zeros(5)]
                else:
                    s = skill[field]
                pm = np.exp(s) / np.exp(s).sum()
                pt = pm * np.exp(np.where(field % 10 == 0, bonus, 0.0))
                pt = pt / pt.sum()
                winner = rng.choice(6, p=pt)
                odds = shin_odds(pm, z) if z > 0 else 1.0 / (pm * overround)
                order = rng.permutation(6)
                pos = np.empty(6, int)
                pos[winner] = 1
                pos[[k for k in order if k != winner]] = np.arange(2, 7)
                for k, dog in enumerate(field):
                    row = {"race_date": day, "race_time": f"{18 + r}:{10 * ti:02d}", "meeting_id": d * 10 + ti,
                           "track": track, "race_id": rid, "race_class": "A3", "distance_m": 480.0, "trap": k + 1,
                           "dog_id": int(dog), "dog_name": f"Dog {dog}", "trainer": f"Trainer {dog % 5}",
                           "sp_decimal": float(odds[k]), "market_pos": int(np.argsort(np.argsort(odds))[k]) + 1,
                           "position": int(pos[k]), "beaten_lengths": 1.0, "sectional": 4.8, "run_time": 29.0,
                           "adjusted_time": 29.0, "weight_kg": 30.0, "comment": "QAw,Rls"}
                    if bsp:
                        row["bsp"] = float(1.0 / pm[k])
                    rows.append(row)
    df = clean(pd.DataFrame(rows))
    if bsp:
        q = 1.0 / df.bsp
        df["bsp_p_norm"] = q / q.groupby(df.race_id).transform("sum")
    return df



def ctx(runs):
    return {"fetch_lag": pd.Series(0, index=sorted(runs.race_date.unique()))}


def last_rows(tab, runs, day):
    keys = runs.loc[runs.race_date == day, KEY]
    return keys.merge(tab, on=KEY).sort_values(KEY)


def test_decayed_sums_match_a_brute_force_reading():
    rng = np.random.default_rng(1)
    n = 300
    key = rng.integers(-1, 6, n)
    dn = rng.integers(0, 200, n)
    v = rng.normal(size=(n, 2))
    got = market2.decayed_sums(key, dn, v, 40.0)
    lam = np.log(2) / 40.0
    for i in range(n):
        if key[i] < 0:
            assert np.allclose(got[i], 0)
            continue
        m = (key == key[i]) & (dn < dn[i])
        want = (v[m] * np.exp(-lam * (dn[i] - dn[m]))[:, None]).sum(0)
        assert np.allclose(got[i], want)
    flat = market2.decayed_sums(key, dn, v, np.inf)
    i = int(np.argmax(np.where(key >= 0, dn, -1)))
    assert np.allclose(flat[i], v[(key == key[i]) & (dn < dn[i])].sum(0))


def test_the_posterior_shrinks_little_evidence_and_has_a_spread():
    lae, sd = market2.posterior(np.array([0.0, 2.0, 20.0]), np.array([0.0, 1.0, 10.0]), 10.0)
    assert lae[0] == 0.0 and 0 < lae[1] < lae[2] < np.log(2.0)
    assert sd[0] > sd[1] > sd[2]


def test_a_planted_underrated_set_reads_above_the_rest(monkeypatch):
    monkeypatch.setattr(market2, "MIN_FIT_RACES", 100)
    runs = synth(days=80, bonus=1.0, overround=1.2)
    tab, notes = market2.build(runs, None, ctx(runs))
    late = runs.race_date >= sorted(runs.race_date.unique())[-15]
    under = (runs.dog_id % 10 == 0) & late
    rest = (runs.dog_id % 10 != 0) & late
    assert tab.mq_sp_lae[under].mean() > tab.mq_sp_lae[rest].mean() + 0.3
    assert tab.mq_sp_z[under].mean() > 1.0
    assert notes["sp"]["fitted"] and notes["sp"]["ll"] > notes["sp"]["ll_market"]
    # the trainer columns (Trainer 0 trains every underrated dog)
    t0 = (runs.trainer == "Trainer 0") & late
    assert tab.mq_tr_sp_lae[t0].mean() > tab.mq_tr_sp_lae[~(runs.trainer == "Trainer 0") & late].mean() + 0.2


def test_shin_bias_does_not_read_as_beating_the_market():
    """A Shin bookmaker (insider share 0.1) prices a strong dog in every race: the proportional book understates
    its chance, so market.py's A/E reads it above 1; the de-biased A/E reads it near 1."""
    runs = synth(days=60, z=0.1, fav=True, seed=4)
    old, _ = market.build(runs, None, ctx(runs))
    new, _ = market2.build(runs, None, ctx(runs))
    late = (runs.race_date >= sorted(runs.race_date.unique())[-10]) & (runs.dog_id <= 12)
    assert np.log(old.mk_sp_ae_all[late]).mean() > 0.1
    assert abs(new.mq_sp_lae[late].mean()) < 0.5 * np.log(old.mk_sp_ae_all[late]).mean()


def test_drift_does_not_see_the_overround():
    runs = synth(days=20, overround=1.25, bsp=True)
    old, _ = market.build(runs, None, ctx(runs))
    new, _ = market2.build(runs, None, ctx(runs))
    late = runs.race_date >= sorted(runs.race_date.unique())[-5]
    assert old.mk_drift5[late].mean() > 0.2                      # log 1.25 for every dog
    assert np.allclose(new.mq_drift5[late].dropna(), 0.0, atol=1e-9)
    assert new.mq_bsp_lae[late].notna().mean() > 0.9


def test_a_days_own_results_move_none_of_its_rows():
    runs = synth(days=40, bonus=0.7, overround=1.2)
    last = runs.race_date.max()
    a, _ = market2.build(runs, None, ctx(runs))
    b, _ = market2.build(perturb_last_day(runs), None, ctx(runs))
    cols = list(market2.COLUMNS)
    x = last_rows(a, runs, last)[cols].to_numpy(float)
    y = last_rows(b, runs, last)[cols].to_numpy(float)
    assert np.isfinite(x[:, 0]).mean() > 0.9 and np.allclose(x, y, equal_nan=True)
    # a later day's results move nothing on an earlier day (a day past the early slice)
    mid = sorted(runs.race_date.unique())[30]
    r2 = runs.copy()
    later = r2.race_date > mid
    r2.loc[later, "won"] = 1.0 - r2.loc[later, "won"]
    c, _ = market2.build(r2, None, ctx(runs))
    early = runs.race_date <= mid
    assert np.allclose(a.loc[early, cols].to_numpy(float), c.loc[early, cols].to_numpy(float), equal_nan=True)


def test_a_card_gets_what_its_run_gets():
    runs = synth(days=40, bonus=0.7, overround=1.2)
    last = runs.race_date.max()
    card = cardify_last_day(runs)
    a, _ = market2.build(runs, None, ctx(runs))
    b, _ = market2.build(card, None, ctx(card))
    cols = list(market2.COLUMNS)
    x = last_rows(a, runs, last)[cols].to_numpy(float)
    y = last_rows(b, card, last)[cols].to_numpy(float)
    assert len(x) == len(y) > 0 and np.allclose(x, y, equal_nan=True)


def test_columns_and_prefix():
    runs = synth(days=12)
    tab, _ = market2.build(runs, None, ctx(runs))
    assert list(tab.columns) == KEY + list(market2.COLUMNS) and len(tab) == len(runs)
    assert all(c.startswith(market2.PREFIX) for c in market2.COLUMNS)


def test_a_planted_effect_is_fitted_with_less_shrinkage(monkeypatch):
    monkeypatch.setattr(market2, "MIN_FIT_RACES", 100)
    a = {}
    for b in (0.0, 1.5):
        runs = synth(days=80, bonus=b, overround=1.2)
        a[b] = market2.build(runs, None, ctx(runs))[1]["sp"]["a"]
    assert a[1.5] < a[0.0]


def test_the_vectorised_shin_is_core_debiased_probs():
    rng = np.random.default_rng(5)
    n = 600
    race = np.repeat(np.arange(n // 6), 6)
    odds = 1.0 / (rng.dirichlet(np.ones(6), n // 6).ravel() * rng.uniform(0.9, 1.4, n // 6).repeat(6))
    odds[::17] = np.nan                                  # unpriced runners
    odds[:6] = [np.nan] * 5 + [2.0]                      # a race with one price
    a = market2.shin_probs(odds, race)
    b = core.debiased_probs(odds, race, "shin")
    assert np.allclose(a, b, equal_nan=True, atol=1e-9, rtol=0)
