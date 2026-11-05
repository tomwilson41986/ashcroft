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
    assert named <= {"live-trade.yml", "greyhound-trade.yml", "football-trade.yml", "paper-models.yml",
                     "morning-models.yml"}, named
    # the greyhound stream recorder too (the owner, 5 Nov 2026, the audit C1): read-only, no order subscription
    traders = ("auto_trade.py", "greyhound.live --trade", "football.live --trade", "greyhound.book_record",
               "greyhound.stream_record",
               "greyhound.early_lay --trade", "greyhound.quote_mid --trade", "morning.live")
    for name in named:
        text = (WORKFLOWS / name).read_text()
        for ln in text.splitlines():
            ln = ln.strip()
            if ln.startswith("#"):
                continue
            if "python" in ln and any(k in ln for k in ("LIVE_APP_KEY", "TRADER_KEY", "FBKEY", "GHKEY", "MKEY")):
                assert any(t in ln for t in traders), (name, ln)
            if "FBKEY=" in ln and ("LIVE_APP_KEY" in ln or "TRADER_KEY" in ln):
                # the football trader reads on the live key (the owner, 4 Nov 2026: "use live key"); it bets only with
                # FOOTBALL_LIVE=yes, held by test_the_football_trader_bets_small_and_only_when_switched_on
                assert name == "football-trade.yml" and "LIVEFLAG" not in ln, (name, ln)
            if "MKEY=" in ln and ("LIVE_APP_KEY" in ln or "TRADER_KEY" in ln):
                assert '"$MORNING_LIVE" = yes' in ln, (name, ln)
            if "betfair_recorder.py" in ln or "tote_recorder.py" in ln:
                assert not any(k in ln for k in ("LIVE_APP_KEY", "TRADER_KEY", "FBKEY", "GHKEY")), (name, ln)
        # on paper: read-only, the delayed key (paper-models.yml gives the football trader the delayed key by default,
        # with no FBKEY at all: its own test below holds that)
        if "football.live --trade" in text and name != "paper-models.yml":
            assert 'FBKEY="$BETFAIR_APP_KEY"' in text, name
            assert not re.search(r"echo[^;]*\$\{?(BETFAIR_LIVE_APP_KEY|TRADER_KEY)\b", ln), (name, ln)


def test_the_paper_models_run_apart_from_the_live_trader_and_cannot_bet():
    """The owner, 2 Nov 2026: the greyhound paper trader runs in paper-models.yml, never in the live trader's job or
    on its runner, with no route to a real order; the delayed key for everything but the greyhound paper trader and
    the greyhound book recorder (the owner, 2 Nov 2026: "use the live key, just don't execute bets yet"), which take it
    only through GHKEY. One exception, the owner, 3 Nov 2026 ("lets go live now, i will monitor and stop manually if
    any concerns"; "use early lay"): the greyhound early lay (greyhound/early_lay.py), given --live, sending orders
    only with GREYHOUND_EARLY_LAY=live; and the owner, 4 Nov 2026 ("go live with live trading today"): the greyhound
    quotes around the mid (greyhound/quote_mid.py), given --live, sending orders only with GREYHOUND_QUOTE_MID=live.
    No other process here is given --live. The football trader left this job on 3 Nov 2026 for
    football-trade.yml (test_the_football_trader_bets_small_and_only_when_switched_on)."""
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
    trade = "Trade (the greyhound early lay live, the quotes and the rest on paper)"
    for s in steps:
        if s.get("name") != trade:
            assert "LIVE_APP_KEY" not in str(s.get("env", {})) + str(s.get("run", "")), s.get("name")
    for ln in (x.strip() for x in runs.splitlines()):
        if ln.startswith("#") or not ln:
            continue
        if "LIVE_APP_KEY" in ln:
            assert ln == 'GHKEY="${BETFAIR_LIVE_APP_KEY:-$BETFAIR_APP_KEY}"', ln
        if "GHKEY" in ln and "python" in ln:
            assert ln.startswith('BETFAIR_APP_KEY="$GHKEY" python -m greyhound.'), ln
            assert any(t in ln for t in ("greyhound.live --trade", "greyhound.book_record", "greyhound.stream_record",
                                         "greyhound.early_lay --trade", "greyhound.quote_mid --trade")), ln
        if "--live" in ln:                          # the early lay and the quotes alone, on the live key
            assert ln.startswith(('BETFAIR_APP_KEY="$GHKEY" python -m greyhound.early_lay --trade --live ',
                                  'BETFAIR_APP_KEY="$GHKEY" python -m greyhound.quote_mid --trade --live ')), ln
        if "football" in ln or "betfair_recorder.py" in ln:
            assert "GHKEY" not in ln and "LIVE_APP_KEY" not in ln, ln
        assert not re.search(r"echo[^;]*\$\{?(BETFAIR_LIVE_APP_KEY|GHKEY)\b", ln), ln
    assert "greyhound.live --trade" in runs and "football.live --trade" not in runs
    assert [ln for ln in runs.splitlines() if "--live" in ln and not ln.strip().startswith("#")] == \
        [ln for ln in runs.splitlines() if "greyhound.early_lay --trade --live" in ln
         or "greyhound.quote_mid --trade --live" in ln]
    assert "greyhound.live --trade --live" not in runs and "football.live --trade --live" not in runs
    # the early lay's paper books (the owner, 4 Nov 2026: "run this on paper. Keep running 10% too"): never --live
    paper = [ln for ln in runs.splitlines() if "--paper-variants" in ln]
    assert len(paper) == 1 and "--live" not in paper[0] and "GREYHOUND_EARLY_LAY" not in paper[0], paper
    env = next(s["env"] for s in steps if s.get("name") == trade)
    assert env["GREYHOUND_LIVE"] == "no" and "FOOTBALL_LIVE" not in env and "API_FOOTBALL_KEY" not in env
    # no football here at all (the owner, 3 Nov 2026: football apart from greyhounds and racing, football-trade.yml)
    code = "\n".join(ln for ln in runs.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    for t in ("football", "--event-type 1", "api_football"):
        assert t not in code, f"paper-models.yml still runs {t!r}"
    assert not any("football" in s.get("name", "").lower() for s in steps)
    # paper unless switched on, and live only with a daily loss limit (the owner, 5 Nov 2026, the audit)
    assert env["GREYHOUND_EARLY_LAY"] == "${{ inputs.early_lay || vars.GREYHOUND_EARLY_LAY || 'paper' }}"
    assert wf[True]["workflow_dispatch"]["inputs"]["early_lay"]["default"] == "paper"
    assert env["GREYHOUND_EARLY_LAY_DAILY_LOSS"] == "${{ vars.GREYHOUND_EARLY_LAY_DAILY_LOSS }}"
    # on paper unless switched on (4 Nov 2026: losing on the corrected simulator)
    assert env["GREYHOUND_QUOTE_MID"] == "${{ inputs.quote_mid || vars.GREYHOUND_QUOTE_MID || 'paper' }}"
    assert env["GREYHOUND_QUOTE_MID_DAILY_LOSS"] == "${{ vars.GREYHOUND_QUOTE_MID_DAILY_LOSS }}"
    assert wf[True]["workflow_dispatch"]["inputs"]["quote_mid"]["default"] == "paper"
    assert env["BETFAIR_APP_KEY"] == "${{ secrets.BETFAIR_APP_KEY }}"
    # no other workflow starts the greyhound trader; the football trader only football-trade.yml
    for p in WORKFLOWS.glob("*.yml"):
        body = p.read_text()
        if p.name != "paper-models.yml":
            assert "greyhound.live --trade" not in body, p.name
        if p.name != "football-trade.yml":
            assert "football.live --trade" not in body, p.name


def test_the_football_trader_bets_small_and_only_when_switched_on():
    """The owner, 3 Nov 2026: every football match on Betfair recorded and paper-traded from the live feed; 4 Nov 2026:
    the trader's process reads on the live key, under Betfair's limits. --live (real bets) only with FOOTBALL_LIVE=yes; its own
    runner, never the live horse trader's or the paper models'; the real stakes capped at GBP2 a bet and GBP20 a day."""
    import json as _json

    import yaml as _yaml
    text = (WORKFLOWS / "football-trade.yml").read_text()
    job = _yaml.safe_load(text)["jobs"]["trade"]
    assert "BETFAIR_RUNNER" not in str(job["runs-on"]) and "PAPER_MODELS_RUNNER" not in str(job["runs-on"])
    guard = next(s for s in job["steps"] if s.get("name") == "Not on another job's runner")["run"]
    assert '"$FOOTBALL_RUNNER_LABEL" = "$LIVE_RUNNER_LABEL"' in guard and '"$FOOTBALL_RUNNER_LABEL" = "$PAPER_RUNNER_LABEL"' in guard
    step = next(s for s in job["steps"] if s.get("name", "").startswith("Football (the trader"))
    assert step["env"]["FOOTBALL_LIVE"] == "${{ vars.FOOTBALL_LIVE || 'no' }}"
    assert step["env"]["BETFAIR_APP_KEY"] == "${{ secrets.BETFAIR_APP_KEY }}"
    lines = [ln.strip() for ln in step["run"].splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert lines[:3] == ["set +e", 'FBKEY="$BETFAIR_APP_KEY"', 'LIVEFLAG=""']
    switch = [ln for ln in lines if "BETFAIR_LIVE_APP_KEY" in ln]
    # the owner, 4 Nov 2026 ("use live key", "do not pass betfair thresholds"): the trader's process reads on the live
    # key, through the feed's throttle; --live (real bets) still only with FOOTBALL_LIVE=yes
    assert switch == ['if [ -n "$BETFAIR_LIVE_APP_KEY" ]; then FBKEY="$BETFAIR_LIVE_APP_KEY"; fi',
                      'if [ "$FOOTBALL_LIVE" = yes ] && [ -n "$BETFAIR_LIVE_APP_KEY" ]; then LIVEFLAG="--live"; fi']
    assert sum('LIVEFLAG="--live"' in ln for ln in lines) == 1
    # the key the trader may hold reaches the trader alone; the records and the collector read on the delayed key
    keyed = [ln for ln in lines if "FBKEY" in ln and "python" in ln]
    assert len(keyed) == 1 and keyed[0].startswith('BETFAIR_APP_KEY="$FBKEY" ')
    assert "football.live --trade --all-games" in keyed[0] and keyed[0].endswith("$LIVEFLAG")
    others = [ln for ln in lines if "python" in ln and ln is not keyed[0]]
    assert others and not any(k in ln for ln in others for k in ("FBKEY", "LIVE_APP_KEY", "LIVEFLAG", "--live"))
    assert sum("betfair_recorder.py --record" in ln for ln in others) == 3       # the day, the evening, the collection
    assert any("sources.api_football_lineups --collect" in ln for ln in others)
    assert all("--event-type 1" in ln for ln in others if "betfair_recorder.py" in ln)
    tables = next(s for s in job["steps"] if s.get("name") == "Football line-up tables")
    assert "api_football_lineups --build" in tables["run"] and "LIVE" not in str(tables.get("env"))
    for name in ("live-trade.yml", "live-record.yml", "paper-models.yml"):
        assert "football-trade" not in (WORKFLOWS / name).read_text() or name == "paper-models.yml"
    cfg = _json.loads((Path(__file__).resolve().parents[1] / "football" / "live_config.json").read_text())
    lim = {**cfg["limits"], **cfg["live_limits"]}
    assert lim["level_stake"] <= 2.0 and lim["max_match_stake"] <= 2.0 and lim["max_daily_turnover"] <= 20.0


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


def test_the_runner_setup_registers_only_its_own_label():
    """scripts/setup_runner.sh: a new runner on the UK server carries only its own label (no 'self-hosted', which the
    live horse trader's runner label may be), for the two jobs that need one, from a checked download."""
    text = (Path(__file__).resolve().parents[1] / "scripts" / "setup_runner.sh").read_text()
    assert "--no-default-labels" in text and '--labels "$LABEL"' in text
    assert "football|paper-models|data|morning) ;;" in text
    assert "sha256sum -c -" in text and 'read -r -s -p "Registration token' in text
    for name, label in (("football-trade.yml", "'football'"), ("paper-models.yml", "'paper-models'"),
                        ("other-data.yml", "'data'"), ("morning-models.yml", "'morning'")):
        assert label in (WORKFLOWS / name).read_text()


def test_the_horse_racing_runner_runs_horse_racing_alone():
    """The owner, 3 Nov 2026: nothing but horse racing on the horse racing runner (vars.BETFAIR_RUNNER). Every workflow
    there runs no greyhound, football, tennis or other-sport code; the price archive there leaves out the greyhound
    files, which the greyhound runner archives; the other sports' fetches run on the data runner."""
    import yaml as _yaml
    on_horse = []
    for p in WORKFLOWS.glob("*.yml"):
        wf = _yaml.safe_load(p.read_text())
        for job in (wf.get("jobs") or {}).values():
            if "BETFAIR_RUNNER" in str(job.get("runs-on", "")):
                on_horse.append((p.name, job))
    assert {n for n, _ in on_horse} >= {"live-trade.yml", "live-record.yml", "betfair-prices.yml"}
    banned = ("4339", "greyhound", "tennis", "betfair_historic", "football", "--event-type 1", "api_football",
              "census-sports", "source_data.py")
    for name, job in on_horse:
        for step in job.get("steps", []):
            code = "\n".join(ln for ln in str(step.get("run", "")).splitlines()
                              if ln.strip() and not ln.strip().startswith("#"))
            code = code.replace("--skip-market-prefix greyhound", "")      # the one mention: leaving them out
            for t in banned:
                assert t not in code.lower(), f"{name}: {step.get('name')!r} runs {t!r} on the horse racing runner"
    prices = (WORKFLOWS / "betfair-prices.yml").read_text()
    assert 'betfair_prices.py --archive "${ARGS[@]}" --skip-market-prefix greyhound' in prices
    # where the moved work runs now: never the horse racing runner
    dogs = _yaml.safe_load((WORKFLOWS / "greyhound-prices.yml").read_text())["jobs"]["archive"]
    assert "PAPER_MODELS_RUNNER" in str(dogs["runs-on"]) and "BETFAIR_RUNNER" not in str(dogs["runs-on"])
    assert "betfair_prices.py --archive --market-prefix greyhound" in str(dogs["steps"])
    data = _yaml.safe_load((WORKFLOWS / "other-data.yml").read_text())["jobs"]["fetch"]
    assert "DATA_RUNNER" in str(data["runs-on"]) and "BETFAIR_RUNNER" not in str(data["runs-on"])
    runs = str(data["steps"])
    assert "tennis_odds" in runs and "betfair_historic.py --fetch" in runs and "LIVE_APP_KEY" not in runs
    paper = (WORKFLOWS / "paper-models.yml").read_text()
    assert "betfair_recorder.py --record --final --until 21:30 --event-type 4339 --tag greyhound" in paper
    for wf_name, job_name in (("greyhound-prices.yml", "archive"), ("other-data.yml", "fetch")):
        guard = _yaml.safe_load((WORKFLOWS / wf_name).read_text())["jobs"][job_name]["steps"][0]
        assert guard["name"] == "Not on the horse racing runner" and '"$LIVE_RUNNER_LABEL"' in guard["run"]


def test_the_runner_fix_keeps_upgrades_off_the_trader_and_is_dispatched_by_hand():
    """The owner, 4 Nov 2026 ("fix the update issue"): needrestart never restarts the runner's service, and the
    upgrade timer runs after the trading day; a dispatch only, never a schedule, and it changes nothing without sudo."""
    import yaml as _yaml
    wf = _yaml.safe_load((WORKFLOWS / "runner-fix-upgrades.yml").read_text())
    assert set(wf.get("on", wf.get(True))) == {"workflow_dispatch"}
    job = wf["jobs"]["fix"]
    assert job["runs-on"] == "${{ vars.BETFAIR_RUNNER }}"
    run = job["steps"][0]["run"]
    assert "sudo -n true" in run and run.index("sudo -n true") < run.index("sudo tee")
    assert "$nrconf{override_rc}{qr(^actions\\.runner\\.)} = 0;" in run
    assert "OnCalendar=*-*-* 21:45" in run and "OnCalendar=\\n" in run
    assert "reboot" not in run.split("== unattended-upgrades")[0].lower().replace("a reboot would", "")


def test_netlify_builds_nothing():
    """The owner, 4 Nov 2026: "block any netlify builds or connection, not needed". The site is still linked on
    Netlify's side, so both places Netlify reads its config from (the repository root and the frontend base
    directory) skip every context, and nothing else in the repository points at Netlify."""
    import tomllib
    for p in (ROOT / "netlify.toml", ROOT / "frontend" / "netlify.toml"):
        cfg = tomllib.loads(p.read_text())
        assert cfg["build"]["ignore"] == "exit 0", p
        for ctx in ("production", "deploy-preview", "branch-deploy"):
            assert cfg["context"][ctx]["ignore"] == "exit 0", (p, ctx)
        assert "functions" not in cfg and "redirects" not in cfg, p
    assert not (ROOT / "frontend" / "netlify").exists()
    assert not (ROOT / "frontend" / "_redirects").exists()


def test_the_morning_models_bet_only_when_switched_on_and_never_beside_the_live_trader():
    """The owner, 4 Nov 2026: the steamer and quick-return models live, WIN at 10:00 UK, half stakes. The live key and
    --live only with MORNING_LIVE=yes, to morning.live alone; the trade on its own runner, never the live horse
    trader's, the paper models' or football's; the selection on GitHub's runner with no Betfair key at all."""
    import yaml as _yaml
    text = (WORKFLOWS / "morning-models.yml").read_text()
    wf = _yaml.safe_load(text)
    sel, trade = wf["jobs"]["select"], wf["jobs"]["trade"]
    assert sel["runs-on"] == "ubuntu-latest" and "BETFAIR" not in str(sel["steps"])
    assert trade["runs-on"] == "${{ vars.MORNING_RUNNER || 'morning' }}"
    guard = next(s for s in trade["steps"] if s.get("name") == "Not on another job's runner")["run"]
    for lab in ("$LIVE_RUNNER_LABEL", "$PAPER_RUNNER_LABEL", "$FOOTBALL_RUNNER_LABEL"):
        assert f'"$MORNING_RUNNER_LABEL" = "{lab}"' in guard
    step = next(s for s in trade["steps"] if s.get("name", "").startswith("Trade or settle"))
    assert step["env"]["MORNING_LIVE"] == "${{ vars.MORNING_LIVE || 'no' }}"
    lines = [ln.strip() for ln in step["run"].splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert 'MKEY="$BETFAIR_APP_KEY"' in lines and 'LIVEFLAG=""' in lines
    assert [ln for ln in lines if "BETFAIR_LIVE_APP_KEY" in ln] == [
        'if [ "$MORNING_LIVE" = yes ] && [ -n "$BETFAIR_LIVE_APP_KEY" ]; then MKEY="$BETFAIR_LIVE_APP_KEY"; '
        'LIVEFLAG="--live"; fi']
    py = [ln for ln in lines if "python" in ln]
    assert py == ['BETFAIR_APP_KEY="$MKEY" python -m morning.live --"$STEP" ${DAY:+--date "$DAY"} $LIVEFLAG']
    assert not re.search(r"echo[^;]*\$\{?(BETFAIR_LIVE_APP_KEY|MKEY)\b", step["run"])
    for name in ("live-trade.yml", "live-record.yml", "paper-models.yml", "football-trade.yml"):
        assert "morning.live" not in (WORKFLOWS / name).read_text()
    import morning.live as ML
    assert ML.TO_WIN == 50.0 and ML.MAX_STAKE == 100.0 and ML.DAY_LIMIT == 2000.0
    assert not any(p.startswith(ML.LIVE_TRADER_PREFIX) for p in ML.MODELS.values())
