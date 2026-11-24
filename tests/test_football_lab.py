"""The football lab: every block is lag-safe, the cache rebuilds only what changed, the runner runs (synthetic rows)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from football import lab
from football import team_features as tf
from football.blocks import available
from tests.test_football_team_features import synthetic


@pytest.fixture(scope="module")
def matrix():
    g, dc, _ = synthetic(n_teams=20, seasons=(2013, 2014, 2015, 2016, 2017), split=True, seed=3)
    m, _ = tf.build(g, dc, start="2013-07-01")
    rng = np.random.default_rng(0)
    # the columns blocks may read beyond the team features: Betfair, early books, the handicap
    for o in ("h", "d", "a"):
        m[f"bfe_open_{o}"] = m[f"open_{o}"] * 1.01
        m[f"bfe_close_{o}"] = m[f"close_{o}"] * 1.01
        m[f"bk_avg_{o}"] = m[f"open_{o}"] * 0.97
        m[f"bk_max_{o}"] = m[f"open_{o}"] * 1.02
        m[f"bk_b365_{o}"] = m[f"open_{o}"] * 0.96
    m["bk_avg_o"], m["bk_avg_u"] = m["open_o25"] * 0.97, m["open_u25"] * 0.97
    m["bk_max_o"], m["bk_max_u"] = m["open_o25"] * 1.02, m["open_u25"] * 1.02
    m["open_ah_line"] = -np.round((m["molam_h"] - m["molam_a"]) * 4) / 4
    m["open_ah_h"], m["open_ah_a"] = 1.95 + rng.normal(0, 0.03, len(m)), 1.95 + rng.normal(0, 0.03, len(m))
    m["fd_code"], m["open_1x2_src"], m["close_1x2_src"] = "E0", "PS", "PSC"
    # the tactics stage's raw columns (football/tactics.py): shapes and game states, as played
    for sd in ("h", "a"):
        m[f"tq_def_{sd}"] = rng.choice([3.0, 4.0, 5.0], len(m), p=[0.25, 0.6, 0.15])
        m[f"tq_fwd_{sd}"] = rng.choice([1.0, 2.0, 3.0], len(m))
        m[f"tq_mid_{sd}"] = 10 - m[f"tq_def_{sd}"] - m[f"tq_fwd_{sd}"]
    cut = rng.dirichlet([2, 1, 1], len(m)) * 90
    m["gs_min_level"], m["gs_min_lead"], m["gs_min_trail"] = cut[:, 0], cut[:, 1], cut[:, 2]
    for t, ft in (("h", "ft_home"), ("a", "ft_away")):
        parts = rng.multinomial(1, [0.5, 0.25, 0.25], len(m)) * m[ft].fillna(0).to_numpy()[:, None]
        m[f"gs_{t}g_level"], m[f"gs_{t}g_lead"], m[f"gs_{t}g_trail"] = parts[:, 0], parts[:, 1], parts[:, 2]
    # the extra football-data columns (football/lab.py early_books, the gold table's closing handicap)
    m["bk_bfe_ahh"], m["bk_bfe_aha"] = m["open_ah_h"] * 1.02, m["open_ah_a"] * 1.02
    m["bk_avg_ahh"], m["bk_avg_aha"] = m["open_ah_h"] * 0.98, m["open_ah_a"] * 0.98
    m["bfe_open_o25"], m["bfe_open_u25"] = m["open_o25"] * 1.02, m["open_u25"] * 1.02
    m["bk_n_1x2"], m["bk_n_ah"] = rng.integers(10, 50, len(m)).astype(float), rng.integers(5, 30, len(m)).astype(float)
    m["close_ah_line"] = m["open_ah_line"] + rng.choice([-0.25, 0.0, 0.0, 0.25], len(m))
    m["close_ah_h"] = m["open_ah_h"] + rng.normal(0, 0.03, len(m))
    m["close_ah_a"] = m["open_ah_a"] + rng.normal(0, 0.03, len(m))
    # the Sportmonks stage's raw columns (football/sportmonks_stage.py), as played
    from football.sportmonks_stage import COLUMNS as SMX
    for c in SMX:
        m[c] = rng.gamma(2.0, 0.6, len(m)) if "xg_" in c else rng.integers(0, 40, len(m)).astype(float)
    return m


@pytest.mark.parametrize("name", sorted(available()))
def test_every_block_ignores_what_happened_after_its_collection(matrix, name):
    """Rewrite the results and closes of the last collection window's matches: that window's block features must
    not move (the generic lag test; a new block is covered the day it is added)."""
    mod = available()[name]
    m = matrix
    cut = tf.collection_day(m["date"])
    last = cut == cut.max()
    a = mod.build(m).set_index("match_id").loc[m.loc[last, "match_id"], mod.FEATURES].sort_index()
    m2 = m.copy()
    m2.loc[last, ["ft_home", "ft_away"]] = m2.loc[last, ["ft_away", "ft_home"]].to_numpy() + 4
    for c in ["qc_h", "qc_a", "close_h", "close_a", "mclam_h", "mclam_a", "bfe_close_h", "bfe_close_a",
              "close_ah_line", "close_ah_h", "close_ah_a"] + \
            [c for c in m.columns if c.startswith(("tq_", "gs_", "smx_"))]:         # what was played is not known either
        m2.loc[last, c] = m2.loc[last, c].to_numpy()[::-1]
    b = mod.build(m2).set_index("match_id").loc[m.loc[last, "match_id"], mod.FEATURES].sort_index()
    pd.testing.assert_frame_equal(a, b)
    assert all(f.startswith(mod.FEATURES[0].split("_")[0] + "_") for f in mod.FEATURES)   # one prefix a block


def test_the_runner_scores_arms_and_writes_the_ledger(matrix, tmp_path, monkeypatch):
    name = "matrix_test.parquet"
    matrix.to_parquet(tmp_path / name, index=False)
    (tmp_path / "LATEST").write_text(name)
    monkeypatch.setattr(lab, "LEDGER", tmp_path / "ledger.jsonl")
    res = lab.run(str(tmp_path), {"base": "team_all", "plus": "team_all+market_state+rest_days"}, ["ridge"],
                  2015, 2017, 2013, None, None, False, False)
    assert set(res["arms"]) == {"base:ridge", "plus:ridge"}
    r = res["arms"]["plus:ridge"]
    assert r["rows"] > 0 and r["n_features"] > res["arms"]["base:ridge"]["n_features"]
    assert json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])["arms"]
    with pytest.raises(SystemExit):
        lab.run(str(tmp_path), {"x": "no_such_family"}, ["ridge"], 2015, 2017, 2013, None, None, False, False)


def test_a_stage_key_changes_only_with_its_code_or_inputs(tmp_path, monkeypatch):
    c = lab.Cache(str(tmp_path))
    monkeypatch.setattr(c, "inputs", lambda: {"gold": "g1", "silver": "s1", "history": "h1"})
    k_team = c.key("team", [], ["gold"])
    assert c.key("team", [], ["gold"]) == k_team
    monkeypatch.setattr(c, "inputs", lambda: {"gold": "g2", "silver": "s1", "history": "h1"})
    assert c.key("team", [], ["gold"]) != k_team                 # new data: rebuilt
    monkeypatch.setattr(c, "inputs", lambda: {"gold": "g1", "silver": "s2", "history": "h1"})
    assert c.key("team", [], ["gold"]) == k_team                 # an input it does not read: kept
    monkeypatch.setattr(lab, "_sha", lambda paths: "changed")
    assert c.key("team", [], ["gold"]) != k_team                 # its code changed: rebuilt


def test_the_lag_check_catches_a_leaky_block(matrix):
    """A block that reads the match's own result must fail the same check the real blocks pass."""
    class Leaky:
        FEATURES = ["lk_gd"]

        @staticmethod
        def build(m):
            return pd.DataFrame({"match_id": m["match_id"], "lk_gd": m["ft_home"] - m["ft_away"]})
    m = matrix
    cut = tf.collection_day(m["date"])
    last = cut == cut.max()
    a = Leaky.build(m).set_index("match_id").loc[m.loc[last, "match_id"]]
    m2 = m.copy()
    m2.loc[last, ["ft_home", "ft_away"]] = m2.loc[last, ["ft_away", "ft_home"]].to_numpy() + 4
    b = Leaky.build(m2).set_index("match_id").loc[m.loc[last, "match_id"]]
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(a, b)


def test_a_name_that_is_both_a_family_and_a_block_is_refused(matrix):
    with pytest.raises(SystemExit):
        lab.resolve("x", {"x": ["a"]}, {"x": ["b"]}, matrix.columns)


def test_line_up_time_features_are_refused_at_the_early_price(matrix, tmp_path, monkeypatch):
    name = "matrix_test.parquet"
    matrix.to_parquet(tmp_path / name, index=False)
    (tmp_path / "LATEST").write_text(name)
    monkeypatch.setattr(lab, "LEDGER", tmp_path / "ledger.jsonl")
    with pytest.raises(SystemExit, match="team sheets"):
        lab.run(str(tmp_path), {"x": "team_all+players"}, ["ridge"], 2015, 2017, 2013, None, None, False, False)


@pytest.mark.parametrize("name", ["formation", "match_sim", "sm_xg", "fd_lines"])
def test_the_tactics_blocks_give_values_on_the_played_shapes_and_states(matrix, name):
    mod = available()[name]
    f = mod.build(matrix)[mod.FEATURES]
    assert (f.notna().mean() > 0.5).all(), f.notna().mean().to_dict()
    assert (f.std() > 0).all()
