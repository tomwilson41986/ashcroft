"""research/greyhound/day_replay.py: the arms and the report (no S3, no Betfair)."""

from research.greyhound import day_replay as dr


def test_arms_are_the_ledgers_rules():
    from research.greyhound.limit_study import EXITS, FILLS, GATES, MARGINS
    for _, rule, mp in dr.arms():
        assert rule["fill"] in FILLS and rule["exit"] in EXITS and rule["gate"] in GATES
        assert rule["param"] in MARGINS
        assert mp in (None, 3.0)


def test_report_lists_each_arm_and_a_missing_one():
    out = {"day": "2026-11-15", "prices": "p", "books": {"live": {
        "a": {"races": 2, "lays_placed": 5, "matched": 3, "laid_dogs_won": 1, "liability": 300.0, "pnl": -4.5,
              "roi%": -1.5, "low": -6.0, "high": 1.0, "last_race": "21:40 Hove", "source": "s"},
        "b": {"note": "no data found"}}}}
    md = dr.report_md(out)
    assert "| a | 2 | 5 | 3 | 1 | 300.00 | -4.50 | -1.5 |" in md
    assert "| b | no data found |" in md
