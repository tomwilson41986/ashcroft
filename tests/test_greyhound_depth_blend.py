"""The blend by market depth (research/greyhound/depth_blend.py): planted markets, never the real data."""

import numpy as np
import pandas as pd

from research.greyhound import depth_blend as db


def _market(n_races=6000, seed=0, thin_noise=0.5, deep_noise=0.05):
    """Races of six. The true chance comes from a hidden strength; the SP sees it through noise that is large in a
    thin race and small in a deep one; our price sees it through a middling noise everywhere."""
    rng = np.random.default_rng(seed)
    rows = []
    days = pd.date_range("2026-06-01", periods=120).strftime("%Y-%m-%d")
    for r in range(n_races):
        thin = r % 2 == 0
        s = rng.normal(0, 1, 6)
        true = np.exp(s) / np.exp(s).sum()
        mk = s + rng.normal(0, thin_noise if thin else deep_noise, 6)
        ours = s + rng.normal(0, 0.35, 6)
        q = np.exp(mk) / np.exp(mk).sum()
        p = np.exp(ours) / np.exp(ours).sum()
        w = rng.choice(6, p=true)
        vol = rng.uniform(50, 150) if thin else rng.uniform(1500, 3000)
        for i in range(6):
            rows.append({"race_id": r, "race_date": days[r * 120 // n_races], "race_time": "10:30" if thin else "19:30",
                         "track": "Thin Park" if thin else "Deep Park", "trap": i + 1, "dog_id": r * 10 + i,
                         "won": float(i == w), "p_model": p[i], "bsp": 1.0 / q[i] / 1.0, "pp_vol": vol / 6})
    return pd.DataFrame(rows)


def test_the_thin_band_gives_our_price_the_weight_and_beats_the_sp():
    res = db.run(_market(), "2026-08-01")
    thin, deep = res["realised"]["bands"]["thin"], res["realised"]["bands"]["deep"]
    assert thin["weights"]["model"] > 0.4 > deep["weights"]["model"]
    assert thin["scored"]["beats_the_sp"] and thin["scored"]["diff"] < 0
    assert not deep["scored"]["beats_the_sp"]
    # the expectation (track and part of day) finds the same races before a penny is matched
    e = res["expected"]
    assert e["rank_correlation_with_the_money"] > 0.7
    assert e["bands"]["thin"]["weights"]["model"] > 0.4
    assert e["pooled"]["diff"] < 0


def test_a_market_that_knows_everything_is_not_beaten():
    d = _market(seed=1, thin_noise=0.0, deep_noise=0.0)
    res = db.run(d, "2026-08-01")
    assert abs(res["one_blend"]["weights"]["market"] - 1.0) < 0.1
    assert abs(res["one_blend"]["weights"]["model"]) < 0.1
    assert not res["one_blend"]["scored"]["beats_the_sp"]
    assert not res["realised"]["pooled"]["beats_the_sp"]


def test_nothing_is_fitted_on_a_scored_day():
    d = _market(n_races=3000, seed=2)
    res = db.run(d, "2026-08-01")
    # the scored part's results and money rewritten: the weights and the bands' cuts do not move
    late = d.race_date >= "2026-08-01"
    e = d.copy()
    rng = np.random.default_rng(5)
    e.loc[late, "won"] = 0.0
    first = e[late].groupby("race_id").head(1).index
    e.loc[first, "won"] = 1.0
    e.loc[late, "pp_vol"] = rng.uniform(1, 5000, int(late.sum()))
    res2 = db.run(e, "2026-08-01")
    assert res["one_blend"]["weights"] == res2["one_blend"]["weights"]
    for kind in ("realised", "expected"):
        assert res[kind]["cuts"] == res2[kind]["cuts"]
        for b in db.BANDS:
            assert res[kind]["bands"][b].get("weights") == res2[kind]["bands"][b].get("weights")


def test_a_race_the_sp_does_not_price_whole_is_left_out_and_no_money_means_no_band():
    d = _market(n_races=1200, seed=3)
    d.loc[d.race_id == 7, "bsp"] = np.where(d[d.race_id == 7].trap == 1, np.nan, d[d.race_id == 7].bsp)
    p = db.prepare(d)
    assert 7 not in set(p.race_id)
    d2 = d.drop(columns="pp_vol")
    res = db.run(d2, "2026-08-01")
    assert "realised" not in res and "one_blend" in res


def test_the_weights_recover_a_planted_blend():
    rng = np.random.default_rng(4)
    rows = []
    for r in range(8000):
        a, b = rng.normal(0, 1, 6), rng.normal(0, 1, 6)
        q, p = np.exp(a) / np.exp(a).sum(), np.exp(b) / np.exp(b).sum()
        z = 0.7 * np.log(q) + 0.3 * np.log(p)
        t = np.exp(z) / np.exp(z).sum()
        w = rng.choice(6, p=t)
        for i in range(6):
            rows.append({"race_id": r, "race_date": "2026-06-01", "won": float(i == w), "p_model": p[i], "bsp": 1 / q[i]})
    w = db.fit_weights(db.prepare(pd.DataFrame(rows)))
    assert abs(w["market"] - 0.7) < 0.06 and abs(w["model"] - 0.3) < 0.06
