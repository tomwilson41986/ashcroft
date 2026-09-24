"""Same-race market features: arithmetic on known books."""

import numpy as np
import pandas as pd
import pytest

from model.market_block import ORDER_DELTA, ORDER_GAMMA, add_same_race_market_features
from model.ordering import place_probabilities


def _race(p, place_places=3, sp_scale=None, place_from=None, rid="r1"):
    p = np.asarray(p, float)
    bsp = 1.0 / p
    implied = place_probabilities(p, place_places, ORDER_GAMMA, ORDER_DELTA)
    place = 1.0 / (place_from if place_from is not None else implied)
    sp = bsp * (sp_scale if sp_scale is not None else np.ones(len(p)))
    return pd.DataFrame({"raceid": rid, "bfsp": bsp, "odds": sp, "bfsp_place": place,
                         "bf_plcs_paid": place_places, "horse_name": [f"{rid}{i}" for i in range(len(p))]})


def test_a_place_market_that_agrees_with_the_win_market_scores_zero():
    p = np.array([0.35, 0.25, 0.15, 0.1, 0.08, 0.07])
    out, names = add_same_race_market_features(_race(p, place_places=2))
    assert np.allclose(out["mkt_place_vs_win"], 0.0, atol=1e-6)
    assert np.allclose(np.exp(out["mkt_ln_pi_bsp"]).sum(), 1.0)
    assert out["mkt_bsp_rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert np.allclose(out["mkt_sp_vs_bsp"], 0.0)


def test_disagreements_show_up_with_the_right_sign():
    p = np.array([0.35, 0.25, 0.15, 0.1, 0.08, 0.07])
    implied = place_probabilities(p, 2, ORDER_GAMMA, ORDER_DELTA)
    tilted = implied.copy(); tilted[3] *= 1.3                         # the place market likes runner 3
    sp_scale = np.ones(6); sp_scale[1] = 0.8                           # bookmakers shorter on runner 1
    out, _ = add_same_race_market_features(_race(p, 2, sp_scale=sp_scale, place_from=tilted))
    assert out["mkt_place_vs_win"].idxmax() == 3
    assert out["mkt_sp_vs_bsp"].idxmax() == 1
    assert out["mkt_sp_book"].nunique() == 1                          # race level


def test_an_incomplete_book_gives_nan_for_the_race_not_a_wrong_number():
    d = _race([0.4, 0.3, 0.2, 0.1], place_places=2)
    d.loc[2, "bfsp_place"] = np.nan
    d.loc[1, "odds"] = np.nan
    out, _ = add_same_race_market_features(d)
    assert out["mkt_place_vs_win"].isna().all() and out["mkt_ln_pi_sp"].isna().all()
    assert out["mkt_ln_pi_bsp"].notna().all()
