"""The feature report's pieces, and the estimated-penalty rating fit (review A1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.open_close import football_feature_report as fr


def test_each_margin_removal_sums_to_one_and_keeps_the_order():
    prices = np.array([[1.5, 4.2, 6.5], [2.6, 3.3, 2.7], [1.12, 9.0, 21.0]])
    for name, f in fr.DEVIG.items():
        q = f(prices)
        assert np.allclose(q.sum(1), 1.0, atol=1e-9), name
        assert (np.argsort(q, 1) == np.argsort(1 / prices, 1)).all(), name
    # power and Shin take more of the margin from the outsider than the proportional split does
    assert fr.devig_power(prices)[2, 2] < fr.devig_proportional(prices)[2, 2]
    assert fr.devig_shin(prices)[2, 2] < fr.devig_proportional(prices)[2, 2]


def test_family_names():
    assert fr.family_of("fs_pts_l2_diff") == "short form (fs_)"
    assert fr.family_of("d_pae_mkt_ewm") == "team windows (d_)"
    assert fr.family_of("dis_open") == "dis_open"


def test_permutation_finds_the_family_that_carries_the_move():
    rng = np.random.default_rng(0)
    n = 4000
    te = pd.DataFrame({"x_a": rng.normal(size=n), "d_b": rng.normal(size=n),
                       "league": pd.Categorical(rng.choice(["E0", "E1"], n))})
    y = 0.05 * te["x_a"].to_numpy() + rng.normal(scale=0.01, size=n)

    class M:
        def predict(self, d):
            return 0.05 * d["x_a"].to_numpy()
    out = fr.permutation({"h": M()}, te, {"h": y}, {"xG (x_)": ["x_a"], "team windows (d_)": ["d_b"]},
                         np.array(["top five"] * n), repeats=2)
    assert list(out)[0] == "xG (x_)"
    assert out["xG (x_)"]["skill_lost"] > 0.5 and abs(out["team windows (d_)"]["skill_lost"]) < 1e-9


def _league(noise: float, seed: int = 0):
    """Twenty sides of known rating, each match's margin its rating difference plus noise; and a newcomer with two
    matches, both far from its (average) rating by chance."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(20)]
    true = dict(zip(teams, rng.normal(0, 0.4, 20)))
    rows, d0 = [], pd.Timestamp("2024-01-01")
    for r in range(4):
        for i, h in enumerate(teams):
            for j, a in enumerate(teams):
                if i != j and (i + j + r) % 3 == 0:
                    rows.append((d0 + pd.Timedelta(days=10 * r + i), h, a, 0.3 + true[h] - true[a] + rng.normal(0, noise)))
    rows.append((d0 + pd.Timedelta(days=60), "NEW", "T0", 4.0 + true["T0"] * -1))
    rows.append((d0 + pd.Timedelta(days=67), "T1", "NEW", -4.0 + true["T1"]))
    return pd.DataFrame(rows, columns=["date", "home_team_id", "away_team_id", "margin"]), true


def test_the_first_massey_fit_does_not_shrink_and_the_estimated_penalty_does():
    from football.xg_features import fit_massey
    d, _ = _league(noise=1.6)
    as_of = pd.Timestamp("2024-06-01")
    first, eb = fit_massey(d, as_of), fit_massey(d, as_of, eb=True)
    # the fault: two matches four goals from the average and the newcomer is rated near four goals above it
    assert first["r"]["NEW"] > 3.0
    # the repair: the same two matches leave it far nearer the average, and the penalty is material
    assert eb["r"]["NEW"] < 0.5 * first["r"]["NEW"]
    assert eb["k"] > 1.0


def test_the_estimated_penalty_is_light_when_the_target_is_precise():
    from football.xg_features import fit_massey
    d, true = _league(noise=0.02)
    d = d[d["home_team_id"].ne("NEW") & d["away_team_id"].ne("NEW")]
    eb = fit_massey(d, pd.Timestamp("2024-06-01"), eb=True)
    got = np.array([eb["r"][t] for t in true])
    want = np.array(list(true.values()))
    assert eb["k"] < 0.1
    assert np.corrcoef(got, want)[0, 1] > 0.999
