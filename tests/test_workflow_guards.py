"""The nightly ingestion threw away its own work for six months, silently.

`daily-results.yml` scraped a night of results, failed on the step that emails a
report about them, and because the S3 upload that followed carried no condition,
GitHub's default skipped it. The runner was destroyed and the night went with it.
Every night from the day the workflow was written. `horse_racing.db` stayed at
2026-03-22 while the 06:00 predictions kept pricing cards on six-month-old form,
and the failure-alert step needed the same missing credentials, so nobody was told.

Two shapes of bug, both tested here:

1. A reporting step that can fail the job before the step that persists the data.
2. `VAR: ${{ secrets.MISSING }}`, which sets the variable to an EMPTY STRING
   rather than leaving it unset -- so every `os.getenv(name, default)` downstream
   returns "" and the default never applies. That made `int("")` raise in
   results_report.py, and it defeated config.py's SMTP_USERNAME fallback.

These are cheap assertions about YAML and about two env readers. They are here
because the failure mode was silence: nothing crashed loudly enough to notice.
"""

import re
import os
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"


def _steps(workflow: str, job: str) -> list[dict]:
    spec = yaml.safe_load((WORKFLOWS / workflow).read_text())
    return spec["jobs"][job]["steps"]


def _step(workflow: str, job: str, name_contains: str) -> dict:
    for s in _steps(workflow, job):
        if name_contains.lower() in str(s.get("name", "")).lower():
            return s
    raise AssertionError(f"no step matching {name_contains!r} in {workflow}:{job}")


# ---------------------------------------------------------------------------
# The night's data must outlive the email about it
# ---------------------------------------------------------------------------

def test_the_results_email_cannot_fail_the_job():
    email = _step("daily-results.yml", "collect-results", "Send results email")
    assert email.get("continue-on-error") is True, (
        "a notification failing must never cost the day's data -- this is the "
        "exact step that froze horse_racing.db at 2026-03-22"
    )


def test_the_database_upload_survives_a_failed_report():
    upload = _step("daily-results.yml", "collect-results", "Upload updated database")
    cond = str(upload.get("if", ""))
    assert cond, "the upload step needs a condition, or any earlier failure skips it"
    assert "always()" in cond, cond


def test_the_upload_is_keyed_on_the_scrape_not_on_always_alone():
    """A half-written database must not overwrite a good remote copy."""
    upload = _step("daily-results.yml", "collect-results", "Upload updated database")
    cond = str(upload.get("if", ""))
    assert "steps.scrape" in cond and "success" in cond, (
        f"upload should run when the scrape succeeded, whatever the reporting "
        f"did -- and not when the scrape itself failed; got {cond!r}"
    )
    scrape = _step("daily-results.yml", "collect-results", "Scrape results from HRB")
    assert scrape.get("id") == "scrape", "the condition above references steps.scrape"


# ---------------------------------------------------------------------------
# A workflow must pass the secret names the code actually reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workflow,job", [
    ("daily-results.yml", "collect-results"),
    ("settle.yml", "settle"),
    ("paper-trade.yml", "paper"),
])
def test_no_live_workflow_passes_the_unset_smtp_aliases(workflow, job):
    """`SMTP_USER`/`SMTP_PASS` are not set; passing them blanks the fallback."""
    for step in _steps(workflow, job):
        env = step.get("env") or {}
        for alias in ("SMTP_USER", "SMTP_PASS"):
            assert alias not in env, (
                f"{workflow} step {step.get('name')!r} passes {alias}, which is "
                f"unset -- the empty string it becomes shadows the "
                f"SMTP_USERNAME/SMTP_PASSWORD the code falls back to"
            )


def test_the_paper_trading_workflow_only_paper_trades():
    """It runs the paper trader and nothing that can send an order (TRADING.md)."""
    runs = "\n".join(str(s.get("run", "")) for s in _steps("paper-trade.yml", "paper"))
    assert "auto_trade.py" in runs
    for live in ("pipeline.execute", "pipeline/execute", "placeOrders", "TRADING_MODE", "--live"):
        assert live not in runs, f"paper-trade.yml runs {live!r}"
    env = {k for s in _steps("paper-trade.yml", "paper") for k in (s.get("env") or {})}
    assert not {"TRADING_MODE", "LIVE_ACK"} & env


# ---------------------------------------------------------------------------
# Empty is absent
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_smtp_env(monkeypatch):
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS",
              "SMTP_USERNAME", "SMTP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_config_treats_an_empty_variable_as_missing(clean_smtp_env):
    """This is what an unset GitHub secret actually looks like to Python."""
    import importlib

    from ultra_betting import config

    clean_smtp_env.setenv("SMTP_PORT", "")          # the unset-secret shape
    clean_smtp_env.setenv("SMTP_USER", "")
    clean_smtp_env.setenv("SMTP_USERNAME", "real@example.com")

    reloaded = importlib.reload(config)             # module scope: it must not raise
    assert reloaded.SMTP_PORT == 587
    assert reloaded.SMTP_USER == "real@example.com", (
        "an empty SMTP_USER must not shadow SMTP_USERNAME")

    for k in ("SMTP_PORT", "SMTP_USER", "SMTP_USERNAME"):
        clean_smtp_env.delenv(k, raising=False)
    importlib.reload(config)                        # leave the module as we found it


def test_results_report_does_not_die_on_an_empty_port(clean_smtp_env):
    """`int("")` used to raise two lines before the credentials check."""
    import results_report

    clean_smtp_env.setenv("SMTP_PORT", "")
    clean_smtp_env.setenv("SMTP_HOST", "")
    # No username or password, so it must report that and return False --
    # rather than raising ValueError on the port and never getting there.
    assert results_report.send_email("x@example.com", "s", "t", "<p>h</p>") is False


def test_only_the_trader_runs_on_the_live_key():
    """The owner's live application key (6 Oct) is for betting: Betfair permits no read-only use of live data, so
    the market records beside the trader, and every other workflow, keep the delayed key (BETFAIR_APP_KEY)."""
    step = _step("live-trade.yml", "live", "Trade or settle")
    assert step["env"]["BETFAIR_LIVE_APP_KEY"] == "${{ secrets.BETFAIR_LIVE_APP_KEY }}"
    assert step["env"]["BETFAIR_APP_KEY"] == "${{ secrets.BETFAIR_APP_KEY }}"
    lines = [ln.strip() for ln in step["run"].splitlines()]
    assert 'TRADER_KEY="${BETFAIR_LIVE_APP_KEY:-$BETFAIR_APP_KEY}"' in lines
    traders = [ln for ln in lines if "auto_trade.py" in ln and not ln.startswith("#")]
    assert len(traders) == 2 and all(ln.startswith('BETFAIR_APP_KEY="$TRADER_KEY" python auto_trade.py') for ln in traders)
    recorders = [ln for ln in lines if "betfair_recorder.py" in ln and not ln.startswith("#")]
    assert recorders and not any("TRADER_KEY" in ln or "LIVE_APP_KEY" in ln for ln in recorders)
    assert not any("echo" in ln and "$TRADER_KEY" in ln for ln in lines)       # the key is never printed
    # the workflows that place bets, and nothing else, name the live key; in each, only a trader's own process
    # takes it (the horse trader, the greyhound trader: greyhound-trade.yml, 6 Oct; the football trader, and only
    # with FOOTBALL_LIVE=yes: football-trade.yml and beside the horse trader, 7 Oct), never a record beside it
    # paper-models.yml too, from 2 Nov 2026 (the owner: "use the live key, just don't execute bets yet"): its greyhound
    # paper trader and greyhound book recorder only, read-only, checked in the paper-models test below
    named = {p.name for p in WORKFLOWS.glob("*.yml") if "BETFAIR_LIVE_APP_KEY" in p.read_text()}
    assert named <= {"live-trade.yml", "greyhound-trade.yml", "football-trade.yml", "paper-models.yml"}, named
    traders = ("auto_trade.py", "greyhound.live --trade", "football.live --trade", "greyhound.book_record")
    for name in named:
        text = (WORKFLOWS / name).read_text()
        for ln in text.splitlines():
            ln = ln.strip()
            if ln.startswith("#"):
                continue
            if "python" in ln and any(k in ln for k in ("LIVE_APP_KEY", "TRADER_KEY", "FBKEY", "GHKEY")):
                assert any(t in ln for t in traders), (name, ln)
            if "FBKEY=" in ln and ("LIVE_APP_KEY" in ln or "TRADER_KEY" in ln):
                assert '"$FOOTBALL_LIVE" = yes' in ln, (name, ln)           # the live key only when switched on
            if "betfair_recorder.py" in ln or "tote_recorder.py" in ln:
                assert not any(k in ln for k in ("LIVE_APP_KEY", "TRADER_KEY", "FBKEY", "GHKEY")), (name, ln)
        # on paper: read-only, the delayed key (paper-models.yml gives the football trader the delayed key by default,
        # with no FBKEY at all: its own test below holds that)
        if "football.live --trade" in text and name != "paper-models.yml":
            assert 'FBKEY="$BETFAIR_APP_KEY"' in text, name
            assert not re.search(r"echo[^;]*\$\{?(BETFAIR_LIVE_APP_KEY|TRADER_KEY)\b", ln), (name, ln)


def test_the_paper_models_run_apart_from_the_live_trader_and_cannot_bet():
    """The owner, 2 Nov 2026: the greyhound and football paper traders run in paper-models.yml, never in the live
    trader's job or on its runner, with no route to a real order; the delayed key for everything but the greyhound
    paper trader and the greyhound book recorder (the owner, 2 Nov 2026: "use the live key, just don't execute bets
    yet"), which take it only through GHKEY."""
    import yaml as _yaml
    live = (WORKFLOWS / "live-trade.yml").read_text()
    for t in ("greyhound.live --trade", "football.live --trade", "GREYHOUND_LIVE", "FOOTBALL_LIVE"):
        assert t not in live, f"live-trade.yml still names {t!r}"
    text = (WORKFLOWS / "paper-models.yml").read_text()
    wf = _yaml.safe_load(text)
    job = wf["jobs"]["paper"]
    assert "BETFAIR_RUNNER" not in str(job["runs-on"]), job["runs-on"]
    steps = job["steps"]
    assert any("Not on the live trader's runner" == s.get("name") for s in steps)
    runs = "\n".join(str(s.get("run", "")) for s in steps)
    # the live key: only in the trading step's env, only through GHKEY, only for the two greyhound processes
    for s in steps:
        if s.get("name") != "Trade on paper (read-only)":
            assert "LIVE_APP_KEY" not in str(s.get("env", {})) + str(s.get("run", "")), s.get("name")
    for ln in (x.strip() for x in runs.splitlines()):
        if ln.startswith("#") or not ln:
            continue
        if "LIVE_APP_KEY" in ln:
            assert ln == 'GHKEY="${BETFAIR_LIVE_APP_KEY:-$BETFAIR_APP_KEY}"', ln
        if "GHKEY" in ln and "python" in ln:
            assert ln.startswith('BETFAIR_APP_KEY="$GHKEY" python -m greyhound.'), ln
            assert "greyhound.live --trade" in ln or "greyhound.book_record" in ln, ln
        if "football" in ln or "betfair_recorder.py" in ln:
            assert "GHKEY" not in ln and "LIVE_APP_KEY" not in ln, ln
        assert not re.search(r"echo[^;]*\$\{?(BETFAIR_LIVE_APP_KEY|GHKEY)\b", ln), ln
    assert "--live" not in runs and "greyhound.live --trade" in runs and "football.live --trade" in runs
    env = next(s["env"] for s in steps if s.get("name") == "Trade on paper (read-only)")
    assert env["GREYHOUND_LIVE"] == "no" and env["FOOTBALL_LIVE"] == "no"
    assert env["BETFAIR_APP_KEY"] == "${{ secrets.BETFAIR_APP_KEY }}"
    # no other workflow starts either trader
    for p in WORKFLOWS.glob("*.yml"):
        if p.name != "paper-models.yml":
            body = p.read_text()
            assert "greyhound.live --trade" not in body and "football.live --trade" not in body, p.name


def test_the_live_horse_jobs_carry_nothing_from_another_sport():
    """The owner, 2 Nov 2026: nothing in the live UK/IE horse system changes for another sport or country; other
    models reuse its features and setup from their own modules and jobs (paper-models.yml)."""
    for name in ("live-trade.yml", "live-record.yml"):
        text = (WORKFLOWS / name).read_text()
        for other in ("betfair_census", "census-racing", "parse_odds", "parse_racing", "greyhound.live --trade",
                      "football.live --trade", "paper-models", "--event-type 1 "):
            assert other not in text, f"{name} names {other!r}"
    import betfair_recorder
    assert not hasattr(betfair_recorder, "census_racing_countries")
    assert not hasattr(betfair_recorder, "SOCCER")
