"""The trading watchdog's rules (scripts/session_watchdog.py; the owner, 3 Nov 2026).

The case that made it: on 3 Nov the session made at 05:17 UTC ended in "failure" at 06:09 UTC (the UK runner's
shutdown signal) and nothing restarted it until 07:16.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import session_watchdog as w  # noqa: E402


def run(i, made, status="completed", conclusion=None, updated=None, event="schedule", actor="tomwilson41986"):
    return {"id": i, "created_at": made, "updated_at": updated or made, "status": status, "conclusion": conclusion,
            "event": event, "triggering_actor": {"login": actor}}


def at(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


SETTLE = run(1, "2026-11-02T22:29:43Z", conclusion="success")
DIED = run(2, "2026-11-03T05:17:14Z", conclusion="failure", updated="2026-11-03T06:09:49Z")


def test_the_3_nov_drop_is_restarted():
    d = w.decide([DIED, SETTLE], at("2026-11-03T06:20"), stop="0")
    assert d.action == "restart" and "failure" in d.reason and d.key == "dropped:2"


def test_a_running_session_is_left_alone():
    live = run(3, "2026-11-03T05:17:14Z", status="in_progress")
    assert w.decide([live, SETTLE], at("2026-11-03T09:00"), stop="0").action == "ok"


def test_yesterdays_settle_is_not_todays_session():
    # before 05:45 UTC nothing is due; after it, no session today is a missed schedule
    assert w.decide([SETTLE], at("2026-11-03T05:30"), stop="0").action == "ok"
    d = w.decide([SETTLE], at("2026-11-03T05:50"), stop="0")
    assert d.action == "restart" and d.key == "missing:2026-11-03"


def test_the_owners_cancel_and_a_session_that_ended_itself_are_not_restarted():
    cancelled = run(4, "2026-11-03T05:17:14Z", conclusion="cancelled")
    ended = run(5, "2026-11-03T05:17:14Z", conclusion="success")
    assert w.decide([cancelled], at("2026-11-03T08:00"), stop="0").action == "ok"
    assert w.decide([ended], at("2026-11-03T21:40"), stop="0").action == "ok"


def test_the_kill_switch_set_or_unreadable_never_restarts():
    assert w.decide([DIED], at("2026-11-03T06:20"), stop="1").action == "alert"
    assert w.decide([DIED], at("2026-11-03T06:20"), stop="unknown").action == "alert"


def test_no_restart_late_in_the_day():
    late = run(6, "2026-11-03T05:17:14Z", conclusion="failure", updated="2026-11-03T20:35:00Z")
    assert w.decide([late], at("2026-11-03T20:40"), stop="0").action == "alert"


def test_at_most_three_restarts_a_day():
    bot = [run(10 + k, f"2026-11-03T0{6 + k}:20:00Z", conclusion="failure", event="workflow_dispatch",
               actor=w.BOT) for k in range(3)]
    assert w.decide(bot + [DIED], at("2026-11-03T09:30"), stop="0").action == "alert"
    assert w.decide(bot[:2] + [DIED], at("2026-11-03T09:30"), stop="0").action == "restart"


def test_a_session_stuck_waiting_for_the_runner_is_reported_not_restarted():
    stuck = run(7, "2026-11-03T05:17:14Z", status="queued")
    assert w.decide([stuck], at("2026-11-03T05:30"), stop="0").action == "ok"
    d = w.decide([stuck], at("2026-11-03T05:40"), stop="0")
    assert d.action == "alert" and d.key == "queued:7"


def test_in_summer_time_the_uk_day_is_used():
    # 1 Jul: the 05:15 UTC schedule is 06:15 UK; a run made at 23:30 UTC the day before is the previous UK day's settle
    prev = run(8, "2026-06-30T22:30:00Z", conclusion="success")
    died = run(9, "2026-07-01T05:15:00Z", conclusion="failure", updated="2026-07-01T06:00:00Z")
    assert w.decide([died, prev], at("2026-07-01T06:10"), stop="0").action == "restart"


def test_the_watchdog_runs_off_the_uk_runner_and_never_holds_the_live_key():
    spec = yaml.safe_load((ROOT / ".github" / "workflows" / "trading-watchdog.yml").read_text())
    job = spec["jobs"]["watch"]
    assert job["runs-on"] == "ubuntu-latest"
    text = (ROOT / ".github" / "workflows" / "trading-watchdog.yml").read_text()
    assert "BETFAIR" not in text and "TRADING_LIVE" not in text
