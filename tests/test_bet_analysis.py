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


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

def _segmented_frame(n_races=400, seed=1):
    """The same synthetic races, labelled with the columns the CSV carries.

    One race type is deliberately rare, so the pooling rule has something to
    pool: a table that silently dropped it would still look tidy."""
    rng = np.random.default_rng(seed)
    d = _frame(n_races, seed=seed)
    codes = rng.choice(["Flat", "National Hunt"], size=n_races, p=[0.6, 0.4])
    types = rng.choice(["Handicap Flat", "Handicap Chase", "Selling Stakes"],
                       size=n_races, p=[0.6, 0.39, 0.01])
    months = rng.choice(["2025-01-01", "2025-02-01", "2025-03-01"], size=n_races)
    idx = d["race_time"].str.split(".").str[0].astype(int).to_numpy()
    d["race_code"] = codes[idx]
    d["race_type"] = types[idx]
    d["race_class"] = "Class 5"
    d["race_date"] = months[idx]
    return d


def test_segment_labels_only_adds_what_the_frame_supports():
    from model.bet_analysis import segment_labels
    full = segment_labels(_segmented_frame(60))
    assert {"code", "race_group", "month", "class_band"} <= set(full.columns)
    bare = segment_labels(_frame(20).drop(columns=["race_date"]))
    assert not {"code", "race_group", "month", "class_band"} & set(bare.columns)


def test_broad_race_type_collapses_free_text():
    from model.bet_analysis import broad_race_type
    assert broad_race_type("Class 4 Handicap Chase") == "Handicap Chase"
    assert broad_race_type("Handicap Hurdle (Div I)") == "Handicap Hurdle"
    assert broad_race_type("Novice Stakes") == "Novices"
    assert broad_race_type("Nursery") == "Handicap Flat"
    assert broad_race_type(np.nan) == "Unknown"


def test_segment_rows_add_back_to_the_whole():
    """Small segments are pooled, never dropped.

    A segment table that loses part of the sample turns a finding into a
    selection effect, and nothing in the output says which rows went."""
    from model.bet_analysis import prepare_bets, rank_by_segment, segment_labels
    d = segment_labels(prepare_bets(_segmented_frame(), blend_lambda=None))
    t = rank_by_segment(d, "race_group", min_n=50)
    assert t["races"].sum() == d["raceid"].nunique()
    assert t["runners"].sum() == len(d)
    assert "(other)" in set(t["segment"])          # the 1% race type survives, pooled
    assert (t["ci_lo"] <= t["roi_back"]).all() and (t["roi_back"] <= t["ci_hi"]).all()


def test_concordance_by_segment_reports_both_sides():
    from model.bet_analysis import concordance_by_segment, prepare_bets, segment_labels
    d = segment_labels(prepare_bets(_segmented_frame(200), blend_lambda=None))
    t = concordance_by_segment(d, "code", min_n=50)
    assert set(t["segment"]) == {"Flat", "National Hunt"}
    assert np.allclose(t["gap"], t["model_concordance"] - t["market_concordance"])
    assert t["runners"].sum() == len(d)


def test_bet_report_carries_the_segment_tables():
    rep = bet_report(_segmented_frame(200), blend_lambda=0.5)
    for k in ("by_race_code", "by_race_type", "by_race_class", "by_month",
              "concordance_by_race_type"):
        assert k in rep and len(rep[k]) > 0
