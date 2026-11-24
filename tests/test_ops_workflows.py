"""The workflow fixes of 24 Nov 2026 (audit OPS-4, OPS-6, OPS-9): toy clocks and the workflow files only."""

import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _load(name):
    return yaml.safe_load((WF / name).read_text())


def _on(spec):
    return spec.get("on", spec.get(True)) or {}


# ---------------------------------------------------------------- OPS-4: the settle's day

def _choose(event, schedule="", day="", at=None, mode=""):
    """Run live-trade.yml's 'Choose the session' script with the clock fixed at `at` (UTC); its outputs."""
    step = next(s for s in _load("live-trade.yml")["jobs"]["live"]["steps"] if s.get("name") == "Choose the session")
    script = step["run"]
    if at is not None:
        epoch = int(datetime.fromisoformat(at).replace(tzinfo=timezone.utc).timestamp())
        assert "$(date +%s)" in script
        script = script.replace("$(date +%s)", str(epoch))
    fd, name = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    out = Path(name)
    env = {**os.environ, "EVENT": event, "SCHEDULE": schedule, "MODE": mode, "DAY": day, "TRADE_UNTIL": "",
           "LIVE": "yes", "GITHUB_OUTPUT": str(out)}
    subprocess.run(["bash", "-c", script], env=env, check=True, capture_output=True)
    got = dict(ln.split("=", 1) for ln in out.read_text().splitlines() if "=" in ln)
    out.unlink()
    return got


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash and GNU date")
@pytest.mark.parametrize("at,day", [
    ("2026-06-10T22:25", "2026-06-10"),     # summer: on time (23:25 UK)
    ("2026-06-10T23:40", "2026-06-10"),     # summer: after UK midnight; the wall clock would say the 11th
    ("2026-06-11T05:06", "2026-06-10"),     # summer: held behind the price archive until its end
    ("2026-11-10T22:25", "2026-11-10"),     # winter: on time
    ("2026-11-11T00:05", "2026-11-10"),     # winter: after midnight
    ("2026-11-11T05:00", "2026-11-10"),     # winter: held behind the archive
])
def test_the_scheduled_settle_settles_the_racing_day_whenever_it_starts(at, day):
    got = _choose("schedule", "25 22 * * *", at=at)
    assert got["mode"] == "settle" and got["day"] == day


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash and GNU date")
def test_the_trade_schedule_and_a_manual_settle_are_unchanged():
    assert _choose("schedule", "15 5 * * *", at="2026-11-11T05:15") == {
        "trade_until": "", "until": "", "mode": "trade", "day": "", "go": "true"}
    got = _choose("workflow_dispatch", day="2026-11-09", mode="settle")
    assert got["mode"] == "settle" and got["day"] == "2026-11-09"
    assert _choose("workflow_dispatch", mode="settle")["day"] == ""      # the owner's default: today


# ---------------------------------------------------------------- OPS-6: a failed predictions run is told

def test_a_failed_or_timed_out_predictions_run_tells_the_owner():
    spec = _load("predict.yml")
    alert = spec["jobs"]["alert"]
    assert alert["needs"] == "predict" and alert["runs-on"] == "ubuntu-latest"
    assert "needs.predict.result == 'failure'" in alert["if"] and "needs.predict.result == 'cancelled'" in alert["if"]
    assert alert["permissions"] == {"contents": "read", "issues": "write"}
    run = "\n".join(str(s.get("run", "")) for s in alert["steps"])
    assert "session_watchdog.py" in run and "--tell" in run and "--key" in run
    assert "secrets." not in yaml.safe_dump(alert)                       # the job's own token only
    assert [c["cron"] for c in _on(spec)["schedule"]] == ["0 3 * * *"]   # the predictions job itself unchanged
    assert spec["jobs"]["predict"]["timeout-minutes"] == 150


# ---------------------------------------------------------------- OPS-9: the old order path never runs by itself

def test_no_scheduled_workflow_runs_the_old_order_path():
    for f in sorted(WF.glob("*.yml")):
        spec = yaml.safe_load(f.read_text())
        if "schedule" not in _on(spec):
            continue
        text = f.read_text()
        for path in ("pipeline.execute", "pipeline/execute", "pipeline.settle"):
            assert path not in text, f"{f.name} is scheduled and runs {path}"


def test_the_old_order_path_is_manual_and_dry_by_default():
    ex = _load("execute.yml")
    assert "schedule" not in _on(ex) and "schedule" not in _on(_load("settle.yml"))
    assert _on(ex)["workflow_dispatch"]["inputs"]["dry_run"]["default"] is True
    env = ex["jobs"]["execute"]["steps"][-1]["env"]
    assert env["DRY_RUN"] == "${{ format('{0}', inputs.dry_run) == 'false' && 'false' || 'true' }}"
