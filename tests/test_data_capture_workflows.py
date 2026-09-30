"""The workflows that keep the market data (the owner's ask, 30 Sep 2026) must never get in the way of the trading or
of the nightly results.

One UK runner serves the live trader and the data jobs, and one job (daily-results.yml) writes the database. So: the
morning recording run gives way when the trader is live; the price-file job has no schedule of its own and never
touches the database; the nightly loads cannot fail the night; the trader records its books on the UK runner only.
"""

from pathlib import Path

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


def test_the_price_file_job_has_no_schedule_and_never_writes_the_database():
    spec = _spec("betfair-prices.yml")
    assert "schedule" not in _on(spec)
    text = (WORKFLOWS / "betfair-prices.yml").read_text()
    body = text.split("\non:", 1)[1]
    for writer in ("backup_to_s3", "--upload-db", "fetch_data.py", "--load", "setup-python"):
        assert writer not in body, writer
    assert spec["concurrency"]["group"] != _spec("live-record.yml")["concurrency"]["group"]


def test_the_nightly_market_loads_cannot_fail_the_night():
    spec = _spec("daily-results.yml")
    names = [s.get("name", "") for s in spec["jobs"]["collect-results"]["steps"]]
    for contains in ("Load Betfair's price files", "Load the live Betfair record"):
        step = _step(spec, "collect-results", contains)
        assert step.get("continue-on-error") is True
        assert step["run"].startswith("timeout ")
        assert names.index(step["name"]) < names.index("Upload updated database to S3")


def test_the_trader_records_its_books_on_the_uk_runner():
    spec = _spec("live-trade.yml")
    env = _step(spec, "live", "Trade or settle")["env"]
    assert env["BETFAIR_RECORD"] == "1"
