"""predict_bfsp_today.results_freshness (the review of 20 Nov 2026, C3): the results' lag behind the card is logged."""
import logging

import pandas as pd

import predict_bfsp_today as P


def test_a_gap_past_two_days_is_a_warning_with_the_runners_in_it(caplog):
    hist = pd.Series(pd.to_datetime(["2026-11-14", "2026-11-16"]))
    card = pd.DataFrame({"horse_name": ["a", "b", "c"], "days_since_lr": [3, 10, None]})
    with caplog.at_level(logging.INFO, logger=P.log.name):
        out = P.results_freshness(hist, pd.Timestamp("2026-11-21").date(), card)
    assert out["gap_days"] == 5 and out["runners_in_gap"] == 1 and out["last_result_day"] == "2026-11-16"
    assert any(r.levelno == logging.WARNING and "RESULTS FRESHNESS" in r.message for r in caplog.records)


def test_yesterdays_results_are_not_a_warning(caplog):
    with caplog.at_level(logging.INFO, logger=P.log.name):
        out = P.results_freshness(pd.Series(pd.to_datetime(["2026-11-20"])), pd.Timestamp("2026-11-21").date())
    assert out["gap_days"] == 1 and out["runners_in_gap"] is None
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
