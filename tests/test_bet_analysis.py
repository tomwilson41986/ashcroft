"""Tests for the overlay / rank bet analysis on a synthetic prediction frame."""

import numpy as np
import pandas as pd

from model.bet_analysis import (bet_report, cluster_bootstrap_roi, cumulative_overlays, overlay_table,
                                prepare_bets, rank_disagreement, rank_table)


def _frame(n_races=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = int(rng.integers(5, 12))
        s = rng.normal(0, 1, n)
        p_true = np.exp(s) / np.exp(s).sum()
        winner = rng.choice(n, p=p_true)
        for i in range(n):
            rows.append({"race_date": "2025-01-01", "race_time": f"{r}.00", "track": "T", "horse_name": f"h{i}",
                         "predicted_win_prob_norm": p_true[i] * rng.uniform(0.7, 1.3), "bfsp": 1.02 / p_true[i],
                         "won": i == winner, "placing_numerical": 1 if i == winner else int(rng.integers(2, n + 1))})
    return pd.DataFrame(rows)


def test_prepare_bets_columns_and_returns():
    d = prepare_bets(_frame(), commission=0.05, blend_lambda=0.5)
    assert {"p_model", "p_market", "edge_pct", "model_rank", "market_rank", "ret_back", "ret_lay", "p_blend"} <= set(d.columns)
    assert np.allclose(d.groupby("raceid")["p_model"].sum(), 1) and np.allclose(d.groupby("raceid")["p_blend"].sum(), 1)
    w = d[d["won"]]
    assert np.allclose(w["ret_back"], (w["bsp"] - 1) * 0.95) and np.allclose(d[~d["won"]]["ret_back"], -1)
    assert (d.groupby("raceid")["model_rank"].min() == 1).all()


def test_tables_shape_and_sanity():
    d = prepare_bets(_frame(), blend_lambda=0.5)
    ot = overlay_table(d)
    assert ot["n"].sum() == len(d) and abs(ot["share_pct"].sum() - 100) < 1e-6
    cu = cumulative_overlays(d, n_boot=50)
    assert cu["n_bets"].is_monotonic_decreasing and (cu["roi_ci_lo"] <= cu["roi_back"]).all()
    rt = rank_table(d, "model_rank")
    assert rt["win_rate"].iloc[0] > rt["win_rate"].iloc[-1]   # top pick wins more than the rest
    dis = rank_disagreement(d)
    assert abs(dis[dis["pick"] == "model #1"]["share_pct"].sum() - 100) < 1e-6
    lo, hi = cluster_bootstrap_roi(d, n_boot=100)
    assert lo <= d["ret_back"].mean() <= hi


def test_bet_report_keys():
    rep = bet_report(_frame(100), blend_lambda=0.5)
    for k in ("overlay_tiers", "cumulative_overlays", "model_ranks", "market_ranks", "disagreement", "rank_by_field",
              "concordance_by_field", "blend_cumulative_overlays"):
        assert k in rep and len(rep[k]) > 0
