"""The trader comes first on the UK runner: the market recorder gives way to a Live trading run waiting for it.

One self-hosted runner on the UK server serves the live trader (live-trade.yml) and the market recorder
(live-record.yml). GitHub hands a free runner to the job that has waited longest, so the recorder's 10:10 UTC run,
queued behind the morning's trader, takes the runner the moment the owner cancels the trader to restart it, and
holds it to the last race. On 7 Oct 2026 the trader was cancelled at 11:07 UTC to restart on the new stake; the
recorder took the runner two seconds later, and both restarts (11:06 and 11:54 UTC) sat queued behind it until it
was cancelled at 11:57.

So the recorder steps aside: at its start, and every minute while it records, it asks GitHub whether a Live trading
run has a job waiting for a runner. If one has, it stops its records (the trader's job records the same files, on the
same server, so nothing is lost), ends, and queues a fresh run of itself, which waits behind the trader and does the
evening's part after the last race (the price files, the census, tomorrow's markets).

    python3 scripts/give_way.py --check                      # exit 0 when a Live trading run waits for a runner
    python3 scripts/give_way.py --watch --pids 101,102,103   # watch while those records run: exit 3 when they end
                                                             # by themselves, or stop them and exit 0 when a Live
                                                             # trading run waits
    python3 scripts/give_way.py --requeue --until 21:30      # queue a fresh run of the recorder (live-record.yml)

Nothing here touches the exchange. On GitHub it reads the Live trading runs and dispatches the recorder's own
workflow, with the job's token (GH_TOKEN; the workflow grants actions: write). Standard library only, so it runs
before the job's Python is set up. A read that fails never gives way: the recorder carries on as it would have.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TRADER = "live-trade.yml"
RECORDER = "live-record.yml"
#: a job waiting for a runner, and a run just made with no job yet. A run held by its own concurrency group
#: ("pending": another Live trading run is under way) or awaiting an approval ("waiting") is not waiting for a runner.
JOB_WAITING = {"queued"}
RUN_WAITING = {"queued", "requested"}
#: at most this many fresh recorder runs an hour: should GitHub ever hand the runner back to the recorder ahead of the
#: waiting trader, the two do not take turns all day
MAX_REQUEUES_AN_HOUR = 4
#: exit codes: a Live trading run waits (give way) / it does not, or the records ended by themselves (carry on)
GIVE_WAY, CARRY_ON = 0, 3


def github(method: str, path: str, body: dict | None = None, timeout: float = 20.0) -> dict:
    """One call to the repository's REST API with the job's token."""
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    req = urllib.request.Request(
        f"{base}/repos/{os.environ['GITHUB_REPOSITORY']}/{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def _get(path: str) -> dict:
    return github("GET", path)


def waiting_runs(get=_get) -> list[int]:
    """The Live trading runs, newest first, with a job waiting for a runner, or none given a job yet."""
    out = []
    for run in get(f"actions/workflows/{TRADER}/runs?per_page=10").get("workflow_runs") or []:
        if run.get("status") == "completed":
            continue
        jobs = get(f"actions/runs/{run['id']}/jobs").get("jobs") or []
        if any(j.get("status") in JOB_WAITING for j in jobs) or (not jobs and run.get("status") in RUN_WAITING):
            out.append(int(run["id"]))
    return out


def check(get=_get, log=print) -> bool:
    """Whether a Live trading run waits for a runner; a failed read says no."""
    try:
        waiting = waiting_runs(get)
    except Exception as exc:
        log(f"give way: the Live trading runs could not be read ({exc}); the recorder carries on")
        return False
    if waiting:
        log(f"give way: Live trading run {waiting[0]} is waiting for the runner")
    return bool(waiting)


def alive(pid: int) -> bool:
    """Whether the process still runs. One that has ended but is not yet reaped by its shell counts as ended."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state != "Z"


def stop(pids: list[int], grace: float = 90.0, sleep=time.sleep, clock=time.monotonic) -> None:
    """Ask the records to stop (SIGTERM: each ends cleanly and copies its files to S3); kill any still running after
    ``grace`` seconds."""
    for pid in pids:
        if alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    end = clock() + grace
    while any(alive(p) for p in pids) and clock() < end:
        sleep(0.2)
    for pid in pids:
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def watch(pids: list[int], get=_get, every: float = 60.0, grace: float = 90.0, tick: float = 5.0,
          sleep=time.sleep, clock=time.monotonic, log=print) -> int:
    """Look for a waiting Live trading run now and every ``every`` seconds while any of ``pids`` runs. Returns
    CARRY_ON once they have all ended by themselves, GIVE_WAY once one waits (the records stopped first). A read that
    fails is logged (at most once in ten minutes) and the watch goes on: an error never stops a record."""
    next_look = clock()
    quiet_until = None
    while any(alive(p) for p in pids):
        if clock() >= next_look:
            next_look = clock() + every
            try:
                waiting = waiting_runs(get)
            except Exception as exc:
                waiting = []
                if quiet_until is None or clock() >= quiet_until:
                    log(f"give way: the Live trading runs could not be read ({exc}); the records go on")
                    quiet_until = clock() + 600.0
            if waiting:
                log(f"give way: Live trading run {waiting[0]} is waiting for the runner; the records stop "
                    "(the trader's job records the same files) and this run queues again behind it")
                stop(pids, grace, sleep, clock)
                return GIVE_WAY
        sleep(tick)
    return CARRY_ON


def recent_requeues(get=_get, now: float | None = None) -> int:
    """The fresh recorder runs this script queued (by the job's token: github-actions[bot]) in the last hour."""
    now = time.time() if now is None else now
    runs = get(f"actions/workflows/{RECORDER}/runs?event=workflow_dispatch&per_page=20").get("workflow_runs") or []
    n = 0
    for run in runs:
        who = (run.get("triggering_actor") or run.get("actor") or {}).get("login")
        made = datetime.strptime(run.get("created_at", ""), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        n += who == "github-actions[bot]" and now - made.timestamp() < 3600
    return n


def requeue(ref: str, until: str, post=None, get=_get, log=print) -> bool:
    """Queue a fresh run of the recorder on ``ref``: it waits behind the trader and does the evening's part. Not
    more than MAX_REQUEUES_AN_HOUR in an hour."""
    post = post or (lambda path, body: github("POST", path, body))
    if recent_requeues(get) >= MAX_REQUEUES_AN_HOUR:
        log(f"give way: {MAX_REQUEUES_AN_HOUR} fresh recorder runs queued in the last hour already; no more")
        return False
    post(f"actions/workflows/{RECORDER}/dispatches", {"ref": ref, "inputs": {"until": until}})
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--check", action="store_true", help="exit 0 when a Live trading run waits for a runner")
    what.add_argument("--watch", action="store_true",
                      help="watch while --pids run: stop them and exit 0 when a Live trading run waits, "
                           "exit 3 when they end by themselves")
    what.add_argument("--requeue", action="store_true", help="queue a fresh run of the recorder on --ref")
    ap.add_argument("--pids", default="", help="the records to stop, comma-separated process ids")
    ap.add_argument("--every", type=float, default=60.0, help="seconds between looks at the Live trading runs")
    ap.add_argument("--grace", type=float, default=90.0, help="seconds a record has to stop before it is killed")
    ap.add_argument("--until", default="21:30", help="the fresh run's last UK time to record (--requeue)")
    ap.add_argument("--ref", default=os.environ.get("GITHUB_REF_NAME", ""), help="the branch to queue it on")
    a = ap.parse_args(argv)
    if a.check:
        return GIVE_WAY if check() else CARRY_ON
    if a.watch:
        pids = [int(p) for p in a.pids.split(",") if p.strip()]
        if not pids:
            ap.error("--watch needs --pids")
        return watch(pids, every=a.every, grace=a.grace)
    if not a.ref:
        ap.error("--requeue needs --ref (or GITHUB_REF_NAME)")
    if not requeue(a.ref, a.until):
        return 1
    print(f"give way: a fresh recorder run is queued on {a.ref}, behind the trader")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.exit(main())
