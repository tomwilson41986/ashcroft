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
