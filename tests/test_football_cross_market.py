"""Cross-market consistency: labels, derived chances, and the gap's sign."""

from __future__ import annotations

import pandas as pd

from research.open_close import football_cross_market as cm


def _rows(event, t, mo, dc, close_mo=None, close_dc=None):
    close_mo, close_dc = close_mo or mo, close_dc or dc
    out = []
    for (name, sp), p, pc in zip([("Home FC", 1), ("Away FC", 2), ("The Draw", 3)], mo, close_mo):
        out.append({"event_id": event, "market_id": f"{event}.mo", "market_type": "MATCH_ODDS", "runner_name": name,
                    "sort_priority": sp, "won": float(sp == 1), "market_time": t, "ltp_t_60": p, "ltp_t_15": p,
                    "ltp_t_1": p, "ltp_close": pc})
    for (name, sp), p, pc in zip([("Home or Draw", 1), ("Draw or Away", 2), ("Home or Away", 3)], dc, close_dc):
        out.append({"event_id": event, "market_id": f"{event}.dc", "market_type": "DOUBLE_CHANCE", "runner_name": name,
                    "sort_priority": sp, "won": float(name != "Draw or Away"), "market_time": t, "ltp_t_60": p,
                    "ltp_t_15": p, "ltp_t_1": p, "ltp_close": pc})
    return out


def test_labels_cover_correct_score_goals():
    r = pd.Series({"market_type": "CORRECT_SCORE", "runner_name": "2 - 1", "sort_priority": 10})
    assert cm._label(r) == "O"
    r = pd.Series({"market_type": "CORRECT_SCORE", "runner_name": "1 - 1", "sort_priority": 6})
    assert cm._label(r) == "U"
    r = pd.Series({"market_type": "CORRECT_SCORE", "runner_name": "Any Other Draw", "sort_priority": 19})
    assert cm._label(r) == "O"


def test_double_chance_derived_from_match_odds_and_the_gap_sign():
    df = pd.DataFrame(_rows("e1", "2026-09-01T15:00:00Z", [2.0, 4.0, 4.0], [1.25, 2.2, 1.3]))
    w = cm.events(df)
    der, own = cm.derived(w, "ltp_t_60")["DOUBLE_CHANCE:HD"]
    assert abs(der.iloc[0] - 0.75) < 1e-9                     # match odds: 0.5 home + 0.25 draw
    assert own.iloc[0] < der.iloc[0] + 1                      # own chance normalised from the DC prices
    assert len(w) == 1 and w["mid|DOUBLE_CHANCE"].iloc[0] == "e1.dc"


def test_double_chance_chances_sum_to_two_and_a_fair_book_shows_no_gap():
    df = pd.DataFrame(_rows("e1", "2026-09-01T15:00:00Z", [2.0, 4.0, 4.0], [1 / 0.75, 1 / 0.5, 1 / 0.75]))
    w = cm.events(df)
    for k in ("DOUBLE_CHANCE:HD", "DOUBLE_CHANCE:DA", "DOUBLE_CHANCE:HA"):
        der, own = cm.derived(w, "ltp_t_60")[k]
        assert abs(der.iloc[0] - own.iloc[0]) < 1e-9


def test_run_end_to_end_on_a_small_book():
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        h = rng.uniform(1.6, 4.0)
        rows += _rows(f"e{i}", f"2026-09-{1 + i // 4:02d}T15:00:00Z", [h, 4.0, 3.4], [1.3, 1.9, 1.35])
    out = cm.run(pd.DataFrame(rows))
    assert out["events"] == 80 and out["rule_scan"]["rules_tried"] > 0
