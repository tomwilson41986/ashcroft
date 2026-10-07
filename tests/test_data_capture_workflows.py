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


def test_on_a_trading_day_the_trade_job_records_the_markets_beside_the_trader():
    run = _step(_spec("live-trade.yml"), "live", "Trade or settle")["run"]
    rec = run.index("betfair_recorder.py --record --final --until 21:30")
    assert run[rec:].split("\n", 1)[0].rstrip().endswith("&")           # in the background
    assert rec < run.index("python auto_trade.py \"${ARGS[@]}\" --trade-until") < run.index('wait "$REC"')
    assert "exit $CODE" in run                                            # the job is the trader's verdict


def test_the_trade_job_records_tomorrows_win_markets_the_evening_before_beside_the_trader():
    """The owner's question of 4 Oct (betting at 7pm the evening before): read-only, to its own files, in the
    background, and never in the way of the trader or of its verdict."""
    run = _step(_spec("live-trade.yml"), "live", "Trade or settle")["run"]
    flat = run.replace("\\\n", " ")
    line = next(x for x in flat.splitlines() if "--tag evening" in x)
    assert "betfair_recorder.py --record" in line and '--date "$TOMORROW"' in line
    assert "--evening 17:00-21:30" in line and "--market-types WIN" in line and "--final" not in line
    assert line.rstrip().endswith("&")                                     # in the background
    assert '|| TOMORROW=""' in flat                                       # a failed date cannot stop the trading
    trader = flat.index('python auto_trade.py "${ARGS[@]}" --trade-until')
    assert flat.index("--tag evening") < trader < flat.index('wait "$EVE"') < flat.index("exit $CODE")


def test_tomorrows_markets_are_recorded_the_evening_before_apart_and_bounded():
    spec = _spec("live-record.yml")
    names = [s.get("name", "") for s in spec["jobs"]["record"]["steps"]]
    step = _step(spec, "record", "the evening before")
    assert names.index(step["name"]) > names.index("Market types on tomorrow's racing (census, read-only)")
    assert step.get("continue-on-error") is True and "!cancelled()" in step["if"]
    run = " ".join(step["run"].split())
    assert run.startswith("timeout ") and "--record" in run and "-d tomorrow" in run
    assert "--tag evening" in run and "--for-minutes" in run and "--market-types WIN" in run
    assert "--final" not in run                     # tomorrow's markets are unsettled; the day's own run settles them


# ---------------------------------------------------------------------------------------------------------------
# The trader comes first: the recorder gives way to a Live trading run waiting for the runner (from 7 Oct 2026)
# ---------------------------------------------------------------------------------------------------------------

def test_the_recorder_gives_way_to_a_trader_waiting_for_the_runner():
    """7 Oct: the recorder's 10:10 UTC run took the one UK runner when the owner cancelled the trader to restart it,
    and held it; both restarts waited behind it. It now looks for a waiting trader before anything slow and every
    minute while it records, and once it has given way nothing else of the run goes on."""
    spec = _spec("live-record.yml")
    assert spec["permissions"] == {"contents": "read", "actions": "write"}
    steps = spec["jobs"]["record"]["steps"]
    names = [s.get("name", s.get("uses", "")) for s in steps]
    first = _step(spec, "record", "Give way to a Live trading run")
    assert names.index(first["name"]) == names.index("actions/checkout@v4") + 1      # before Python and the installs
    assert "python3 scripts/give_way.py --check" in first["run"]                     # the system's Python 3
    assert first["env"]["GH_TOKEN"] == "${{ github.token }}"
    for s in steps[names.index(first["name"]) + 1:]:
        assert "env.GAVE_WAY != '1'" in str(s.get("if", "")), s.get("name")
    rec = _step(spec, "record", "Record the day's markets to the last race")
    assert rec["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert 'give_way.py --watch --pids "$REC,$OTHER,$DOGS"' in rec["run"]
    # the trader's own workflow is untouched by the recorder: the recorder only reads it
    assert "give_way" not in (WORKFLOWS / "live-trade.yml").read_text().split("\njobs:", 1)[1]


def _stub_python(tmp_path: Path, name: str) -> dict:
    """A stand-in for python: logs its arguments; give_way.py answers as told; the records sleep a moment."""
    if not shutil.which("bash"):
        pytest.skip("needs bash")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / name
    stub.write_text("#!/bin/bash\n"
                    'echo "$*" >> "$LOG"\n'
                    'case "$*" in\n'
                    '  *give_way.py\\ --check*) exit "${CHECK_RC:-3}" ;;\n'
                    '  *give_way.py\\ --watch*) exit "${WATCH_RC:-3}" ;;\n'
                    '  *give_way.py\\ --requeue*) exit "${REQUEUE_RC:-0}" ;;\n'
                    '  *betfair_recorder.py*--tag*) sleep 0.2; exit 0 ;;\n'
                    '  *betfair_recorder.py*) sleep 0.2; exit "${MAIN_RC:-0}" ;;\n'
                    "esac\n"
                    "exit 99\n")
    stub.chmod(0o755)
    return {"PATH": f"{bin_dir}:/usr/bin:/bin", "LOG": str(tmp_path / "calls.log"),
            "GITHUB_ENV": str(tmp_path / "github_env"), "UNTIL": "21:30"}


def _run_step(tmp_path: Path, contains: str, env: dict, **answers) -> tuple[int, str, str]:
    run = _step(_spec("live-record.yml"), "record", contains)["run"]
    for f in (env["LOG"], env["GITHUB_ENV"]):
        Path(f).write_text("")
    out = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", run], cwd=tmp_path,
                         env={**env, **{k: str(v) for k, v in answers.items()}}, capture_output=True, text=True,
                         timeout=60)
    return out.returncode, Path(env["LOG"]).read_text(), Path(env["GITHUB_ENV"]).read_text()


def test_the_record_step_stops_and_queues_again_when_the_trader_waits(tmp_path):
    env = _stub_python(tmp_path, "python")
    # a trader waits: the run gives way, queues again to the same time, and the step ends clean
    rc, calls, genv = _run_step(tmp_path, "Record the day's markets", env, WATCH_RC=0)
    assert rc == 0 and "GAVE_WAY=1" in genv
    watch = next(line for line in calls.splitlines() if "--watch" in line)
    pids = watch.split("--pids ", 1)[1].split()[0].split(",")
    assert len(pids) == 3 and all(p.isdigit() for p in pids)
    assert "scripts/give_way.py --requeue --until 21:30" in calls
    # a failed requeue is a warning, not a failed step
    rc, calls, genv = _run_step(tmp_path, "Record the day's markets", env, WATCH_RC=0, REQUEUE_RC=1)
    assert rc == 0 and "GAVE_WAY=1" in genv
    # the records ran to their end: no requeue, and the step is the main record's verdict, as before
    for watch_rc in (3, 1):                                     # ended by themselves, or the watch itself failed
        rc, calls, genv = _run_step(tmp_path, "Record the day's markets", env, WATCH_RC=watch_rc)
        assert rc == 0 and "GAVE_WAY" not in genv and "--requeue" not in calls
        rc, calls, genv = _run_step(tmp_path, "Record the day's markets", env, WATCH_RC=watch_rc, MAIN_RC=2)
        assert rc == 2 and "GAVE_WAY" not in genv
    assert calls.count("betfair_recorder.py --record --final --until 21:30") == 3


def test_the_first_look_gives_way_before_anything_slow(tmp_path):
    env = _stub_python(tmp_path, "python3")
    rc, calls, genv = _run_step(tmp_path, "Give way to a Live trading run", env, CHECK_RC=0)
    assert rc == 0 and "GAVE_WAY=1" in genv and "scripts/give_way.py --requeue --until 21:30" in calls
    rc, calls, genv = _run_step(tmp_path, "Give way to a Live trading run", env, CHECK_RC=0, REQUEUE_RC=1)
    assert rc == 0 and "GAVE_WAY=1" in genv
    for check_rc in (3, 1):                                     # nothing waits, or GitHub could not be read
        rc, calls, genv = _run_step(tmp_path, "Give way to a Live trading run", env, CHECK_RC=check_rc)
        assert rc == 0 and genv == "" and "--requeue" not in calls

