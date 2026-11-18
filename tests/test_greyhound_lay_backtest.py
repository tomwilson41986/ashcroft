"""The early lay on a model's out-of-sample prices (research/greyhound/lay_backtest.py): the three fixed rules scored
walk-forward with the limit study's code, each with its interval and deflated Sharpe, and the CLI's report. A
constructed panel for the unit test only."""

from __future__ import annotations

import json
from datetime import date

from research.greyhound import lay_backtest as lb, limit_study as ls
from tests.test_greyhound_limit_study import synthetic_panel


def test_the_three_rules_are_scored(monkeypatch, tmp_path):
    p = synthetic_panel(n_days=34, races=8)
    monkeypatch.setattr(ls, "load", lambda *a, **k: p)
    r = lb.run(None, "s3:x", date(2026, 10, 6), date(2026, 11, 8), "retimed")
    assert set(r["rules"]) == {"live_rule", "spclose", "traded_ref"} and r["days"] == 34 - ls.MIN_TRAIN_DAYS
    lr = r["rules"]["live_rule"]
    assert lr["orders"] > 0 and "roi_90%" in lr and "dsr" in lr and r["grid_rules"] > 3
    assert set(r["reading"]) == {"live_rule_interval_above_zero", "live_rule_dsr_passed", "pass"}
    monkeypatch.setattr("sources.common.Store", lambda *a, **k: None)
    assert lb.main(["--pred-dir", "s3:x", "--out", str(tmp_path)]) == 0
    assert "s3:x" in json.loads((tmp_path / "report.json").read_text())
