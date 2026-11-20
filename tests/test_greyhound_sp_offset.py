"""research/greyhound/sp_offset.py: a residual on the BSP learns what the BSP gets wrong, and nothing where it is right."""

import numpy as np
import pandas as pd
import pytest

from research.greyhound import sp_offset as so

SMALL = dict(so.PARAMS, min_child_samples=50, learning_rate=0.1)


def _frame(bias: float, races: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Six-runner races over 2022-2024. The true chance is softmax(ability + bias * x); the BSP knows the ability only.
    With bias 0 the BSP is the truth and the feature x is noise."""
    rng = np.random.default_rng(seed)
    n = races * 6
    race = np.repeat(np.arange(races), 6)
    ability, x = rng.normal(0, 0.6, n), rng.normal(0, 1, n)
    soft = lambda z: (lambda e: e / np.add.reduceat(e, np.arange(0, n, 6)).repeat(6))(np.exp(z))
    p_true, q = soft(ability + bias * x), soft(ability)
    u = rng.random(races).repeat(6)
    cum = np.add.reduceat(p_true, np.arange(0, n, 6))  # 1 a race
    c = p_true.reshape(races, 6).cumsum(1)
    win = (c >= u.reshape(races, 6)).argmax(1)
    won = (np.arange(6)[None, :] == win[:, None]).ravel().astype(float)
    days = pd.date_range("2022-01-01", "2024-12-30", periods=races).strftime("%Y-%m-%d").to_numpy().repeat(6)
    assert np.allclose(cum, 1.0)
    return pd.DataFrame({"race_id": race, "day": days, "won": won, "bsp_p_norm": q, "bsp": 1.0 / q, "x": x,
                         "noise": rng.normal(0, 1, n)})


def _fold(d, **k):
    orig = so.PARAMS
    so.PARAMS = SMALL
    try:
        return so.run_fold(d, ["x", "noise"], "2024-01-01", "2024-12-31", **k)
    finally:
        so.PARAMS = orig


def test_the_residual_finds_what_the_bsp_misses_and_the_placebo_does_not():
    biased = _fold(_frame(0.5))
    assert biased["gain"] > 0.01 and biased["gain_90%"][0] > 0 and biased["top_features"][0][0] == "x"
    placebo = _fold(_frame(0.5), shuffle_seed=7)
    assert placebo["placebo"] and placebo["gain_90%"][0] <= 0 and abs(placebo["gain"]) < biased["gain"] / 3


def test_where_the_bsp_is_right_the_gain_is_nothing():
    fair = _fold(_frame(0.0))
    assert abs(fair["gain"]) < 0.004 and fair["gain_90%"][0] <= 0


def test_the_loss_is_the_winners_and_the_shuffle_stays_within_the_race():
    race = np.array([0, 0, 0, 1, 1, 1])
    won = np.array([1.0, 0, 0, 0, 0, 1.0])
    rc, ll = so.race_ll(np.zeros(6), won, race)
    assert list(rc) == [0, 1] and ll == pytest.approx([np.log(3), np.log(3)])
    sh = so._shuffle_within(np.tile([1.0, 0, 0, 0, 0, 0], 200), np.repeat(np.arange(200), 6), np.random.default_rng(1))
    assert (np.add.reduceat(sh, np.arange(0, 1200, 6)) == 1).all() and 0.05 < sh.reshape(200, 6)[:, 0].mean() < 0.35
    g = so.read_gain(np.array(["a", "a", "b", "c"]), np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.9, 1.0, 0.8, 1.1]))
    assert g["gain"] == pytest.approx(0.05) and g["races"] == 4 and g["days"] == 3


def test_the_feature_placebo_learns_nothing_and_stage_two_bets_on_the_card_only_residual():
    """Each fit race's feature rows dealt at random: the planted bias cannot be learnt, the gain is nothing. With the
    true features the centred residual picks backs that win more than the BSP says and lays that win less."""
    d = _frame(0.5)
    dealt = _fold(d, feature_shuffle_seed=11)
    assert dealt["placebo"] and dealt["gain"] < 0.004 and dealt["gain_90%"][0] <= 0 and "_runners" not in dealt
    assert dealt["gain"] < _fold(d)["gain"] / 3
    deal = so._deal_within(np.repeat(np.arange(50), 6), np.random.default_rng(0))
    assert (deal // 6 == np.repeat(np.arange(50), 6)).all() and sorted(deal) == list(range(300))
    real = _fold(d)
    assert set(real["cuts"]) == {n for n, _, _ in so.STAGE2} and real["cuts"]["back_top2"] > real["cuts"]["back_top10"] > 0
    st = so.stage2([real], 0.02, 10)
    by = {r["rule"]: r for r in st["rules"]}
    assert by["back_top5"]["pooled"]["roi_%"] > 0 and by["lay_bottom5"]["pooled"]["roi_%"] > 0
    assert by["back_top2"]["pooled"]["bets"] < by["back_top10"]["pooled"]["bets"]
    assert all(r["folds_positive"] <= 1 and r["pass"] is False for r in st["rules"])      # one fold: no pass on one
