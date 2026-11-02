"""Team features and their evaluation: mechanics on synthetic matches only (never a reported figure)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from football import feature_eval as fe
from football import team_features as tf


def probs(lh, la, k=12):
    g = np.arange(k + 1)
    j = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    d, t = np.subtract.outer(g, g), np.add.outer(g, g)
    return j[d > 0].sum(), j[d == 0].sum(), j[d < 0].sum(), j[t >= 3].sum()


def synthetic(n_teams=12, seasons=(2013, 2014, 2015, 2016, 2017), seed=0, open_sd=0.05, model_sd=0.15):
    """A pool of sides with true attack and defence; the close prices the truth, the open a noisy copy, the model
    a noisier one. Double round robin each season, one round a week."""
    rng = np.random.default_rng(seed)
    att, dfn = rng.normal(0, 0.25, n_teams), rng.normal(0, 0.25, n_teams)
    rows, day = [], pd.Timestamp("2013-08-03")
    teams = [f"ENG:t{i}" for i in range(n_teams)]
    for s in seasons:
        pairs = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
        rng.shuffle(pairs)
        for r in range(0, len(pairs), n_teams // 2):
            for i, j in pairs[r:r + n_teams // 2]:
                lh, la = np.exp(0.3 + 0.2 + att[i] - dfn[j]), np.exp(0.3 + att[j] - dfn[i])
                ph, pd_, pa, po = probs(lh, la)
                noisy = lambda p, sd: p * np.exp(rng.normal(0, sd))         # noqa: E731
                oh, od, oa = noisy(ph, open_sd), noisy(pd_, open_sd), noisy(pa, open_sd)
                mh, md, ma = noisy(ph, model_sd), noisy(pd_, model_sd), noisy(pa, model_sd)
                z = mh + md + ma
                rows.append({"match_id": f"m{len(rows)}", "status": "FINISHED", "match_date": day.strftime("%Y-%m-%d"),
                             "season": s, "country": "ENG", "competition_id": "ENG1",
                             "home_team_id": teams[i], "away_team_id": teams[j],
                             "ft_home": rng.poisson(lh), "ft_away": rng.poisson(la),
                             "sot_home": rng.poisson(3 * lh), "sot_away": rng.poisson(3 * la),
                             "open_h": 1 / (oh * 1.03), "open_d": 1 / (od * 1.03), "open_a": 1 / (oa * 1.03),
                             "close_h": 1 / (ph * 1.02), "close_d": 1 / (pd_ * 1.02), "close_a": 1 / (pa * 1.02),
                             "open_o25": 1 / (po * 1.03), "open_u25": 1 / ((1 - po) * 1.03),
                             "close_o25": 1 / (po * 1.02), "close_u25": 1 / ((1 - po) * 1.02),
                             "open_1x2_src": "PS", "close_1x2_src": "PSC",
                             "lam_h": lh * np.exp(rng.normal(0, .1)), "lam_a": la * np.exp(rng.normal(0, .1)),
                             "p_h": mh / z, "p_d": md / z, "p_a": ma / z, "p_o25": po})
            day += pd.Timedelta(days=7)
        day += pd.Timedelta(days=56)
    g = pd.DataFrame(rows)
    dc = g[["match_id", "lam_h", "lam_a", "p_h", "p_d", "p_a", "p_o25"]].copy()
    return g.drop(columns=["lam_h", "lam_a", "p_h", "p_d", "p_a", "p_o25"]), dc, dict(zip(teams, att + dfn))


def test_implied_rates_recover_the_poisson_pair():
    ph, pd_, pa, po = probs(1.6, 0.9)
    lh, la = tf.implied_rates([ph], [pa], [po])
    assert lh[0] == pytest.approx(1.6, abs=0.03) and la[0] == pytest.approx(0.9, abs=0.03)


def test_market_ratings_rank_the_true_strengths():
    g, dc, truth = synthetic()
    mt = tf.match_table(g, dc)
    fit = tf.fit_market_ratings(mt, mt["date"].max() + pd.Timedelta(days=1))
    r = {t: fit["att"][t] + fit["def"][t] for t in truth}
    assert np.corrcoef([r[t] for t in truth], list(truth.values()))[0, 1] > 0.95


def test_windows_never_see_the_match_itself():
    g, dc, _ = synthetic(seasons=(2013,))
    mt = tf.match_table(g, dc)
    t = tf.team_rows(mt, tf.market_ratings(mt))
    w = tf.windows(t)
    first = w.groupby("team").head(1)
    assert first["pts_l5"].isna().all() and (first["n_prev"] == 0).all()
    one = t[t["team"] == t["team"].iloc[0]].reset_index(drop=True)
    ww = w[w["team"] == one["team"].iloc[0]].reset_index(drop=True)
    assert ww.loc[5, "pts_l5"] == pytest.approx(one.loc[0:4, "pts"].mean())


def test_a_days_features_do_not_move_when_that_days_results_and_closes_are_rewritten():
    """The lag-safety rewrite test: change the last day's scores, shots and closing prices; every feature of that
    day's matches must be unchanged (the horse and greyhound engines' guard)."""
    g, dc, _ = synthetic(seasons=(2013, 2014))
    last = g["match_date"].max()
    a, _ = tf.build(g, dc, start="2013-07-01")
    g2 = g.copy()
    on = g2["match_date"] == last
    g2.loc[on, ["ft_home", "ft_away"]] = g2.loc[on, ["ft_away", "ft_home"]].to_numpy() + 3
    g2.loc[on, ["sot_home", "sot_away"]] = 0
    g2.loc[on, ["close_h", "close_a"]] = g2.loc[on, ["close_a", "close_h"]].to_numpy()
    b, _ = tf.build(g2, dc, start="2013-07-01")
    feats = [c for c in a.columns if c.startswith("d_") or c in ("r_h", "r_a", "dis_open", "n_prev_min")]
    x = a[a["match_date"] == last].set_index("match_id")[feats].sort_index()
    y = b[b["match_date"] == last].set_index("match_id")[feats].sort_index()
    pd.testing.assert_frame_equal(x, y)


def test_the_outcome_test_finds_a_feature_that_carries_the_truth():
    g, dc, truth = synthetic(n_teams=20, seasons=tuple(range(2013, 2021)), seed=1, open_sd=0.3, model_sd=0.4)
    f, _ = tf.build(g, dc, start="2013-07-01")
    d = fe.prepare(f)
    d["oracle"] = d["home_team_id"].map(truth) - d["away_team_id"].map(truth)    # the open's noise is what it fixes
    res = fe.outcome_test(d, {"oracle": ["oracle"], "noise": ["n_prev_min"]})
    assert res["rows"] > 0
    assert res["gain_vs_open_model"]["oracle"][1] > 0                  # the truth adds, its interval above zero
    lo, hi = res["gain_vs_open_model"]["noise"][1:]
    assert lo < 0.002 and hi > -0.002                                    # a meaningless column does not
