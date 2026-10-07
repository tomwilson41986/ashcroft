"""The trader comes first on the UK runner (scripts/give_way.py): the market recorder gives way to a Live trading run
waiting for the runner, stops its records cleanly and queues again behind it.

On 7 Oct 2026 the recorder took the one UK runner when the owner cancelled the trader to restart it, and held it:
both restarts sat queued for 50 minutes. A fake GitHub API and real child processes stand in here; nothing reaches
GitHub, Betfair or S3.
"""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("give_way", ROOT / "scripts" / "give_way.py")
gw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gw)


def api(runs, jobs=None, calls=None):
    """GitHub's answers: the Live trading runs, newest first, and each run's jobs."""
    jobs = jobs or {}

    def get(path):
        if calls is not None:
            calls.append(path)
        if path.startswith("actions/workflows/live-trade.yml/runs"):
            return {"total_count": len(runs), "workflow_runs": runs}
        assert path.startswith("actions/runs/") and path.endswith("/jobs"), path
        return {"total_count": 1, "jobs": jobs.get(int(path.split("/")[2]), [])}
    return get


#: the watch reads each process's state from /proc (an ended child not yet reaped is a zombie, not a record)
needs_proc = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="reads process states from /proc")


def _sleeper(seconds=60.0, ignore_term=False):
    code = "import time\n"
    if ignore_term:
        code += "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    code += f"time.sleep({seconds})\n"
    return subprocess.Popen([sys.executable, "-c", code])


def test_a_trader_waiting_for_the_runner_is_found_and_one_on_it_is_not():
    runs = [{"id": 30, "status": "queued"}, {"id": 20, "status": "in_progress"}, {"id": 10, "status": "completed"}]
    jobs = {30: [{"status": "queued", "runner_name": None}], 20: [{"status": "in_progress", "runner_name": "ashcroft"}]}
    calls = []
    assert gw.waiting_runs(api(runs, jobs, calls)) == [30]
    assert not any("/runs/10/" in c for c in calls)                       # a finished run is not looked into
    # the job says so, not the run: a run in progress with a job waiting for a runner waits
    assert gw.waiting_runs(api([{"id": 40, "status": "in_progress"}], {40: [{"status": "queued"}]})) == [40]
    # a run just created, no job yet, waits too
    assert gw.waiting_runs(api([{"id": 50, "status": "queued"}])) == [50]
    # but not one held by its own concurrency group (another Live trading run under way) or awaiting an approval
    assert gw.waiting_runs(api([{"id": 60, "status": "pending"}, {"id": 61, "status": "waiting"}])) == []
    # the trader on the runner, settled runs, or none at all: nothing waits
    assert gw.waiting_runs(api([{"id": 20, "status": "in_progress"}], jobs)) == []
    assert gw.waiting_runs(api([{"id": 10, "status": "completed"}])) == []
    assert gw.waiting_runs(api([])) == []


def test_a_failed_read_never_gives_way():
    def broken(path):
        raise OSError("GitHub unreachable")
    said = []
    assert gw.check(broken, log=said.append) is False
    assert "carries on" in said[0]
    assert gw.check(api([{"id": 1, "status": "queued"}]), log=said.append) is True


@needs_proc
def test_the_watch_stops_the_records_and_gives_way_when_the_trader_waits():
    procs = [_sleeper() for _ in range(3)]
    looks = []
    trader = [{"id": 7, "status": "in_progress"}]
    waiting = {7: [{"status": "queued"}]}

    def get(path):                                   # nothing waits at the first look; the restart at the second
        if path.startswith("actions/workflows"):
            looks.append(path)
        return api(trader if len(looks) >= 2 else [], waiting)(path)

    said = []
    try:
        t0 = time.monotonic()
        rc = gw.watch([p.pid for p in procs], get, every=0.1, grace=10.0, tick=0.02, log=said.append)
        took = time.monotonic() - t0
        assert rc == gw.GIVE_WAY
        assert len(looks) == 2 and took < 5.0
        for p in procs:
            assert p.wait(timeout=5) == -signal.SIGTERM                 # asked to stop, not killed
        assert "Live trading run 7 is waiting" in said[-1]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()


@needs_proc
def test_a_record_that_will_not_stop_is_killed_after_its_grace():
    stubborn = _sleeper(ignore_term=True)
    time.sleep(1.0)                                  # its handler is set before the signal comes
    try:
        rc = gw.watch([stubborn.pid], api([{"id": 1, "status": "queued"}]), every=0.1, grace=0.5, tick=0.02,
                      log=lambda s: None)
        assert rc == gw.GIVE_WAY
        assert stubborn.wait(timeout=5) == -signal.SIGKILL
    finally:
        if stubborn.poll() is None:
            stubborn.kill()


@needs_proc
def test_records_that_end_by_themselves_carry_on_and_errors_never_stop_them():
    procs = [_sleeper(0.3) for _ in range(2)]
    said = []

    def broken(path):
        raise OSError("GitHub unreachable")

    rc = gw.watch([p.pid for p in procs], broken, every=0.05, grace=10.0, tick=0.02, log=said.append)
    assert rc == gw.CARRY_ON
    assert [p.wait(timeout=5) for p in procs] == [0, 0]                 # they ran to their end
    assert len(said) == 1                                                # the failure is told once, not every look


@needs_proc
def test_an_ended_record_not_yet_reaped_counts_as_ended():
    p = _sleeper(0.0)
    deadline = time.monotonic() + 5
    while gw.alive(p.pid) and time.monotonic() < deadline:             # a zombie until its parent reaps it
        time.sleep(0.02)
    assert not gw.alive(p.pid)
    p.wait(timeout=5)
    assert not gw.alive(p.pid)


def _recorder_runs(*runs):
    """GitHub's list of the recorder's dispatched runs: (minutes ago, who) each."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    rows = [{"id": i, "created_at": (now - timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "triggering_actor": {"login": who}} for i, (m, who) in enumerate(runs)]

    def get(path):
        assert path.startswith("actions/workflows/live-record.yml/runs?event=workflow_dispatch"), path
        return {"workflow_runs": rows}
    return get


def test_the_fresh_run_is_queued_on_the_same_branch_to_the_same_time():
    sent = []
    assert gw.requeue("claude/scrape-horse-racing-results-vaHKc", "21:30", get=_recorder_runs(),
                      post=lambda path, body: sent.append((path, body)))
    assert sent == [("actions/workflows/live-record.yml/dispatches",
                     {"ref": "claude/scrape-horse-racing-results-vaHKc", "inputs": {"until": "21:30"}})]


def test_the_recorder_queues_itself_again_at_most_four_times_an_hour():
    """Should GitHub ever hand the runner back to the recorder ahead of the waiting trader, the two do not take turns
    all day."""
    bot = "github-actions[bot]"
    sent, said = [], []
    post = lambda path, body: sent.append(path)               # noqa: E731
    # the owner's own dispatches and the runs of more than an hour ago do not count
    assert gw.requeue("main", "21:30", post=post, log=said.append,
                      get=_recorder_runs((5, bot), (20, bot), (30, bot), (10, "tomwilson41986"), (70, bot), (90, bot)))
    assert len(sent) == 1
    assert not gw.requeue("main", "21:30", post=post, log=said.append,
                          get=_recorder_runs((5, bot), (20, bot), (30, bot), (50, bot)))
    assert len(sent) == 1 and "no more" in said[-1]


def test_the_command_line(monkeypatch):
    calls = []

    def fake_github(method, path, body=None, timeout=20.0):
        calls.append((method, path, body))
        if method == "POST":
            return {}
        if path.startswith("actions/workflows/live-record.yml/runs"):
            return {"workflow_runs": []}
        return api(runs)(path)

    monkeypatch.setattr(gw, "github", fake_github)
    monkeypatch.setenv("GITHUB_REF_NAME", "main-branch")
    runs = [{"id": 3, "status": "queued"}]
    assert gw.main(["--check"]) == gw.GIVE_WAY == 0
    runs = []
    assert gw.main(["--check"]) == gw.CARRY_ON == 3
    calls.clear()
    assert gw.main(["--requeue", "--until", "21:30", "--ref", "main-branch"]) == 0
    assert calls[-1] == ("POST", "actions/workflows/live-record.yml/dispatches",
                         {"ref": "main-branch", "inputs": {"until": "21:30"}})
    with pytest.raises(SystemExit):
        gw.main(["--watch"])                                             # no records to watch


def test_the_script_reads_and_queues_through_githubs_api_with_the_jobs_token():
    """The real calls, end to end, against a stand-in for api.github.com on this machine."""
    import http.server
    import threading

    seen = []

    class GitHub(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, body, code=200):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("Authorization")))
            if self.path.startswith("/repos/o/r/actions/workflows/live-trade.yml/runs"):
                self._reply({"workflow_runs": [{"id": 9, "status": "queued"}]})
            elif self.path == "/repos/o/r/actions/runs/9/jobs":
                self._reply({"jobs": [{"status": "queued"}]})
            elif self.path.startswith("/repos/o/r/actions/workflows/live-record.yml/runs?event=workflow_dispatch"):
                self._reply({"workflow_runs": []})
            else:
                self._reply({"message": "Not Found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            seen.append(("POST", self.path, body))
            self.send_response(204)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), GitHub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.update(GITHUB_API_URL=f"http://127.0.0.1:{server.server_port}", GITHUB_REPOSITORY="o/r",
               GH_TOKEN="test-token", GITHUB_REF_NAME="main", NO_PROXY="*")
    script = str(ROOT / "scripts" / "give_way.py")
    try:
        assert subprocess.run([sys.executable, script, "--check"], env=env, timeout=30).returncode == gw.GIVE_WAY
        assert ("GET", "/repos/o/r/actions/runs/9/jobs", "Bearer test-token") in seen
        assert subprocess.run([sys.executable, script, "--requeue", "--until", "21:30"], env=env,
                              timeout=30).returncode == 0
        assert seen[-1] == ("POST", "/repos/o/r/actions/workflows/live-record.yml/dispatches",
                            {"ref": "main", "inputs": {"until": "21:30"}})
        env["GITHUB_API_URL"] = "http://127.0.0.1:9"                 # GitHub out of reach: the recorder carries on
        assert subprocess.run([sys.executable, script, "--check"], env=env, timeout=60).returncode == gw.CARRY_ON
    finally:
        server.shutdown()

