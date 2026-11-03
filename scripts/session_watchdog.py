"""The trading watchdog: restart a Live trading session that dropped, and tell the owner (the owner, 3 Nov 2026).

On 3 Nov 2026 the UK runner received a shutdown signal at 06:09 UTC, before the first bet, and the session died
("failure"); nothing traded until it was restarted by hand at 07:16. The same happened on 23 Oct at 06:53 UTC. The
05:40 UTC morning check could not see it: the session was running then. So this runs all day on GitHub's own runners
(trading-watchdog.yml: every 10 minutes from 05:00 to 09:59 UTC, every 30 minutes to 20:30) and, for today's UK
racing day:

- a trading session queued or running: nothing to do, unless it has waited for a runner for 20 minutes or more
  (the UK runner offline: a restart cannot help, so the owner is told);
- no trading session by 05:45 UTC (the 05:15 UTC schedule missed), or the latest one ended in "failure",
  "timed_out" or "startup_failure": restart it (live-trade.yml, mode trade, today's date), at most 3 times a day,
  never once the S3 kill switch trading/STOP is set, never after 20:30 UK (the last races are near), and tell the
  owner;
- the latest one "cancelled" (the owner stopping it) or "success" (ended by itself): nothing to do.

The owner is told by a GitHub issue, one a day, titled "Live trading: <day>": the first event opens it, each new
event comments on it, so GitHub's notifications (email and the mobile app) carry it. Nothing here touches the
exchange or the trader's code; it reads the Live trading runs and dispatches that workflow with the job's token.

    python3 scripts/session_watchdog.py              # decide, act, report
    python3 scripts/session_watchdog.py --dry-run    # decide and print only

Standard library only. STOP is read by the workflow before this runs (STOP=1 set, 0 not set, unknown when S3 could
not be read: an unknown kill switch never restarts, and the owner is told).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UK = ZoneInfo("Europe/London")
TRADER = "live-trade.yml"
BRANCH = "claude/scrape-horse-racing-results-vaHKc"
BOT = "github-actions[bot]"
#: a run that ended this way dropped; "cancelled" is the owner stopping it, "success" a session that ended itself
DROPPED = {"failure", "timed_out", "startup_failure"}
WAITING = {"queued", "waiting", "requested", "pending"}
#: a trading session is one made between these UK times (the 05:15 UTC schedule, or a restart during the day); the
#: settle runs at 22:25 UTC
TRADE_WINDOW_UK = ("04:30", "21:30")
#: no trading session yet by this UTC time: the schedule missed
MISSING_AFTER_UTC = "05:45"
#: no restart after this UK time
LAST_RESTART_UK = "20:30"
MAX_RESTARTS = 3
QUEUED_TOO_LONG = timedelta(minutes=20)


@dataclass
class Decision:
    action: str                 # "ok", "restart" or "alert"
    reason: str
    key: str = ""               # one alert per key: the issue is commented once for each


def _at(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _hm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def _before(t: datetime, hm: str) -> bool:
    return (t.hour, t.minute) < _hm(hm)


def trading_runs(runs: list[dict], day) -> list[dict]:
    """Today's trading sessions, newest first: the runs of live-trade.yml made in the UK trading window of `day`."""
    out = []
    for r in runs:
        made = _at(r["created_at"]).astimezone(UK)
        if made.date() == day and not _before(made, TRADE_WINDOW_UK[0]) and _before(made, TRADE_WINDOW_UK[1]):
            out.append(r)
    return sorted(out, key=lambda r: r["created_at"], reverse=True)


def restarts_today(sessions: list[dict]) -> int:
    return sum(1 for r in sessions if r.get("event") == "workflow_dispatch"
               and (r.get("triggering_actor") or {}).get("login") == BOT)


def decide(runs: list[dict], now: datetime, stop: str = "0") -> Decision:
    """What to do about today's trading session, from the Live trading runs (newest first or not), at `now` (UTC)."""
    now_uk = now.astimezone(UK)
    day = now_uk.date()
    sessions = trading_runs(runs, day)
    live = [r for r in sessions if r.get("status") != "completed"]
    if live:
        r = live[0]
        waited = now - _at(r["created_at"])
        if r.get("status") in WAITING and waited >= QUEUED_TOO_LONG:
            return Decision("alert", f"the trading session (run {r['id']}) has waited {int(waited.total_seconds() // 60)}"
                            " minutes for the UK runner: the runner looks offline, a restart cannot help",
                            key=f"queued:{r['id']}")
        return Decision("ok", f"the trading session (run {r['id']}) is {r.get('status')}")
    if sessions:
        last = sessions[0]
        if last.get("conclusion") not in DROPPED:
            return Decision("ok", f"the trading session (run {last['id']}) ended: {last.get('conclusion')}")
        why = f"the trading session (run {last['id']}) ended in {last.get('conclusion')} at " \
              f"{_at(last['updated_at']).astimezone(UK):%H:%M} UK"
        key = f"dropped:{last['id']}"
    else:
        if _before(now, MISSING_AFTER_UTC):
            return Decision("ok", "before the morning session is due")
        why, key = "no trading session was started today (the 05:15 UTC schedule did not run)", f"missing:{day}"
    if stop == "1":
        return Decision("alert", why + "; trading/STOP is set, so it is not restarted", key=key)
    if stop != "0":
        return Decision("alert", why + "; the kill switch trading/STOP could not be read, so it is not restarted",
                        key=key)
    if not _before(now_uk, LAST_RESTART_UK):
        return Decision("alert", why + f"; after {LAST_RESTART_UK} UK, so it is not restarted", key=key)
    if restarts_today(sessions) >= MAX_RESTARTS:
        return Decision("alert", why + f"; already restarted {MAX_RESTARTS} times today, so it is left for the owner",
                        key=key)
    return Decision("restart", why, key=key)


# --- GitHub --------------------------------------------------------------------------------------------------------

def github(method: str, path: str, body: dict | None = None, timeout: float = 20.0) -> dict:
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


def restart(day: str) -> None:
    github("POST", f"actions/workflows/{TRADER}/dispatches",
           {"ref": BRANCH, "inputs": {"mode": "trade", "date": day}})


def tell_owner(day: str, d: Decision, restarted: bool, run_url: str) -> str:
    """Open today's issue, or comment on it; once for each alert key. Returns what was done."""
    title = f"Live trading: {day}"
    marker = f"<!-- watchdog:{d.key} -->"
    text = (f"{marker}\n**{'Restarted' if restarted else 'Not restarted'}** at "
            f"{datetime.now(timezone.utc).astimezone(UK):%H:%M} UK: {d.reason}.\n\n"
            + ("A new trading session has been queued (live-trade.yml, mode trade). " if restarted else
               "To start one by hand: GitHub, Actions, \"Live trading (the owner's book)\", Run workflow, branch "
               f"`{BRANCH}`, mode trade, date {day}. ")
            + f"The watchdog's run: {run_url}\n\n---\n_Generated by [Claude Code](https://claude.ai/code)_")
    issues = github("GET", "issues?state=open&per_page=50") or []
    mine = [i for i in issues if i.get("title") == title and not i.get("pull_request")]
    if not mine:
        github("POST", "issues", {"title": title, "body": text})
        return "opened the issue"
    issue = mine[0]
    seen = [issue.get("body") or ""] + [c.get("body") or "" for c in
                                       github("GET", f"issues/{issue['number']}/comments?per_page=100") or []]
    if any(marker in s for s in seen):
        return "already told"
    github("POST", f"issues/{issue['number']}/comments", {"body": text})
    return "commented on the issue"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    runs = github("GET", f"actions/workflows/{TRADER}/runs?per_page=30").get("workflow_runs") or []
    d = decide(runs, now, os.environ.get("STOP", "unknown"))
    day = now.astimezone(UK).date().isoformat()
    print(f"{now:%Y-%m-%d %H:%M} UTC: {d.action}: {d.reason}")
    if a.dry_run or d.action == "ok":
        return 0
    restarted = False
    if d.action == "restart":
        try:
            restart(day)
            restarted = True
            print("restarted: live-trade.yml mode trade", day)
        except Exception as e:                              # the dispatch refused: the owner is told it was not
            d = Decision("alert", f"{d.reason}; the restart was refused ({e})", d.key)
    run_url = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY', '')}"
               f"/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}")
    print(tell_owner(day, d, restarted, run_url))
    return 0 if restarted or d.action == "alert" else 1


if __name__ == "__main__":
    sys.exit(main())
