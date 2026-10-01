"""The workflows that keep the market data (the owner's ask, 30 Sep 2026) must never get in the way of the trading or
of the nightly results.

One UK runner serves the live trader and the data jobs, and one job (daily-results.yml) writes the database. So: the
morning recording run gives way when the trader is live; the price files are fetched after the last race and through
the night, never past 06:15 UTC, and that job never touches the database; the nightly loads cannot fail the night;
the trader records its books on the UK runner only.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _spec(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _on(spec: dict) -> dict:
    return spec.get("on", spec.get(True))            # YAML 1.1 reads a bare `on:` key as True


def _step(spec: dict, job: str, contains: str) -> dict:
    for s in spec["jobs"][job]["steps"]:
        if contains.lower() in str(s.get("name", "")).lower():
            return s
    raise AssertionError(f"no step matching {contains!r}")


def test_the_morning_recording_gives_way_to_the_live_trader():
    spec = _spec("live-record.yml")
    crons = [c["cron"] for c in _on(spec)["schedule"]]
    assert crons == ["55 6 * * *", "10 10 * * *"]
    gate = _step(spec, "record", "Choose the run")["run"]
    assert "'55 6 * * *'" in gate and '"$LIVE" = yes' in gate and "GO=false" in gate
    for s in spec["jobs"]["record"]["steps"][1:]:
        assert "steps.run.outputs.go == 'true'" in str(s.get("if", "")), s.get("name", s.get("uses"))
    assert spec["jobs"]["record"]["runs-on"] == "${{ vars.BETFAIR_RUNNER }}"


def test_the_recorder_never_reaches_the_order_api():
    import betfair_recorder
    text = Path(betfair_recorder.__file__).read_text()
    assert "placeOrders" not in text and "cancelOrders" not in text
    assert "LiveExchange" not in text


def test_the_price_file_job_runs_only_at_night_and_never_writes_the_database():
    spec = _spec("betfair-prices.yml")
    crons = [c["cron"] for c in _on(spec)["schedule"]]
    assert crons and all(c.split()[1] in ("22", "23") for c in crons), crons      # after the 21:45 UTC settlement
    assert spec["jobs"]["archive"]["runs-on"] == "${{ vars.BETFAIR_RUNNER }}"
    assert spec["jobs"]["archive"]["timeout-minutes"] <= 480                       # 22:30 UTC + 8 h, before the trader
    step = _step(spec, "archive", "Archive the price files")
    assert "betfair_prices.py --archive" in step["run"] and '--max-minutes "$MAX"' in step["run"]
    text = (WORKFLOWS / "betfair-prices.yml").read_text()
    body = text.split("\non:", 1)[1]
    for writer in ("backup_to_s3", "--upload-db", "fetch_data.py", "--load", "--pull-s3", "setup-python",
                   "horse_racing.db"):
        assert writer not in body, writer
    groups = {_spec(w)["concurrency"]["group"] for w in ("live-record.yml", "live-trade.yml")}
    assert spec["concurrency"]["group"] not in groups


def _budget(start: str, want: str = "") -> str:
    """The archive step's shell, at a chosen start time (UTC), with the fetch itself replaced by an echo."""
    if not shutil.which("bash") or subprocess.run(["date", "-u", "-d", "@0"], capture_output=True).returncode:
        pytest.skip("needs bash and GNU date")
    run = _step(_spec("betfair-prices.yml"), "archive", "Archive the price files")["run"]
    run = run.replace("now=$(date -u +%s)", 'now=$(date -u -d "$START" +%s)')
    run = run.replace("python betfair_prices.py", "echo python betfair_prices.py")
    env = {"PATH": "/usr/bin:/bin", "START": start, "WANT": want, "DATE_FROM": "", "DATE_TO": "", "MARKET": ""}
    out = subprocess.run(["bash", "-e", "-c", run], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_the_price_files_take_the_night_and_stop_before_the_morning_trading():
    assert _budget("2026-10-01 22:30").endswith("--max-minutes 450")       # 22:30 to 06:00 UTC at the most
    assert _budget("2026-10-02 01:00").endswith("--max-minutes 315")       # a late start still stops by 06:15
    assert _budget("2026-10-02 06:10").endswith("--max-minutes 5")
    assert _budget("2026-10-01 10:05").endswith("--max-minutes 15")        # by day, between the runner's jobs
    assert _budget("2026-10-01 21:35").endswith("--max-minutes 5")         # never into the 21:45 settlement
    assert _budget("2026-10-01 22:30", want="60").endswith("--max-minutes 60")
    assert _budget("2026-10-01 22:30", want="900").endswith("--max-minutes 450")


def test_the_recorder_fetches_the_uk_and_irish_price_files_only_after_the_last_race():
    spec = _spec("live-record.yml")
    names = [s.get("name", "") for s in spec["jobs"]["record"]["steps"]]
    step = _step(spec, "record", "price files")
    assert names.index(step["name"]) > names.index("Record the day's markets to the last race")
    assert step.get("continue-on-error") is True and "!cancelled()" in step["if"]
    assert step["run"].startswith("timeout ") and "--archive" in step["run"] and "--max-minutes" in step["run"]
    assert "--market ukwin,ukplace,irewin,ireplace" in step["run"]
    assert "--push-s3" not in (WORKFLOWS / "live-record.yml").read_text()    # no stale copy over a rewritten file


def test_the_nightly_market_loads_cannot_fail_the_night():
    spec = _spec("daily-results.yml")
    names = [s.get("name", "") for s in spec["jobs"]["collect-results"]["steps"]]
    for contains in ("Load Betfair's price files", "Load the live Betfair record"):
        step = _step(spec, "collect-results", contains)
        assert step.get("continue-on-error") is True
        assert step["run"].startswith("timeout ")
        assert names.index(step["name"]) < names.index("Upload updated database to S3")
    prices = _step(spec, "collect-results", "Load Betfair's price files")["run"]
    assert "--max-files" in prices and "--load-from 2018-01-01" in prices      # a long backfill loads over nights


def test_the_trader_records_its_books_on_the_uk_runner():
    spec = _spec("live-trade.yml")
    env = _step(spec, "live", "Trade or settle")["env"]
    assert env["BETFAIR_RECORD"] == "1"
