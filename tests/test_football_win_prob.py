"""The win-probability test: complete books only, renormalised, scored on the result."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.open_close import football_win_prob as wp


def rows(p, won, event="e1", cp=60, fresh=True):
    return pd.DataFrame({"event_id": event, "cp_min": cp, "role": ["home", "draw", "away"][:len(p)], "p_cp": p,
                         "won": won, "cp_fresh": fresh, "schedule_fixed": True, "day": "2026-01-01"})


def test_only_complete_books_with_one_winner_are_kept_and_chances_sum_to_one():
    d = pd.concat([rows([0.5, 0.3, 0.3], [1, 0, 0]),                      # overround: renormalised
                   rows([0.5, 0.3], [1, 0], event="e2"),                     # a missing outcome: out
                   rows([0.5, 0.3, 0.2], [np.nan] * 3, event="e3"),          # no result: out
                   rows([0.5, 0.3, 0.2], [1, 0, 0], event="e4", fresh=False)])  # stale book: out
    got = wp.complete_rows(d)
    assert set(got["event_id"]) == {"e1"}
    assert got["p_mkt"].sum() == pytest.approx(1.0) and got["p_mkt"].iloc[0] == pytest.approx(0.5 / 1.1)


def test_log_loss_reads_the_winner_s_chance_and_the_paired_change_favours_the_sharper_forecast():
    d = pd.concat([rows([0.5, 0.3, 0.2], [1, 0, 0], event=f"e{i}") for i in range(10)], ignore_index=True)
    d["p_mkt"] = d["p_cp"]
    d["f_sharp"] = np.tile([0.6, 0.25, 0.15], 10)
    b, a = wp.scores(d, "p_mkt"), wp.scores(d, "f_sharp")
    assert b["ll"].iloc[0] == pytest.approx(-np.log(0.5))
    r = wp.paired(b, a)
    assert r["log_loss_saved"] == pytest.approx(np.log(0.6) - np.log(0.5), abs=1e-6) and r["lo95"] > 0
