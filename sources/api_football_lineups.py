"""API-Football line-ups and injuries for the 22 main divisions, with the time each line-up first appeared.

Why (reports/football_model.md s.5): the close beats the open by what arrives between them, mostly team news. To
test whether that is tradeable we need the confirmed elevens and the moment they became public, to set against the
Betfair books the football recorder keeps (betfair_live/<day>/books_football_*.csv). This module only collects; it
places nothing and reads no Betfair data.

Collect (one process for the UK day, football-trade.yml):

- the day's fixtures (``/fixtures?date=<UK day>&timezone=Europe/London``), refreshed hourly, filtered to the 22
  divisions (football.reference COMPETITIONS, their API-Football league ids);
- each match not yet started is polled from ``LEAD_MIN`` minutes before kick-off, in batches of up to 20
  (``/fixtures?ids=a-b-...``, which carries the line-ups), every ``EVERY_MIN`` minutes until both starting elevens
  are out; then once more from ``CONFIRM_MIN`` minutes before kick-off, to catch a late change;
- the day's injuries (``/injuries?date=<UK day>``, every league in one call) at the start and every ``INJURY_EVERY_H``
  hours.

Every reply is kept as it came (bronze), wrapped with the UTC time it was fetched and the call:
``api_football/raw/lineups/<day>/<stamp>_<n>.json.gz``, ``.../raw/day_fixtures/<day>/...``,
``.../raw/injuries/<day>/...``. A poll's fetch time is what dates a line-up: it is the upper bound of when the line-up
was public through this feed, within one polling interval.

Build (tables, rebuilt whole from bronze, safe to rerun):

- ``api_football/lineup_times.parquet``: one row a match: kick-off, the first poll that held both elevens
  (``first_seen_utc``, ``minutes_before_ko``), the poll before it (``last_unseen_utc``: the line-up appeared between
  the two), polls, whether either eleven changed after it first appeared, formations;
- ``api_football/lineups.parquet``: one row a player of the latest line-up a match: side, starter or substitute,
  position, grid;
- ``api_football/injuries.parquet``: one row a player a match in the 22 divisions, with the type and reason and when
  first and last seen.

The key is read from ``API_FOOTBALL_KEY`` and never written anywhere. The collector stops polling when the plan's
remaining daily requests (the reply's header) fall to ``QUOTA_STOP``; the nightly backfill (sources/api_football.py)
leaves its own floor for this.

    python -m sources.api_football_lineups --collect --until 21:30          # the day's loop
    python -m sources.api_football_lineups --ids 1234-5678 --root /tmp/af   # poll given fixtures once (checks)
    python -m sources.api_football_lineups --build
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.common import Store, env, get, gunzip, gz, session  # noqa: E402

log = logging.getLogger(__name__)

API = "https://v3.football.api-sports.io"
UK = ZoneInfo("Europe/London")
PREFIX = "api_football"
LEAD_MIN = 75.0          # start polling a match this long before kick-off (line-ups are due about an hour before)
EVERY_MIN = 2.0          # between polls of a match whose line-up is not yet out
CONFIRM_MIN = 10.0       # the confirming poll, this long before kick-off
FIXTURES_EVERY_MIN = 60.0
INJURY_EVERY_H = 3.0
BATCH = 20               # the most ids /fixtures?ids= takes
QUOTA_STOP = 100
MAX_FAILURES = 60        # failed loop steps (30 s apart) before the day's loop gives up
NOT_STARTED = {"NS", "TBD"}

Fetch = Callable[[str, dict], "tuple[dict | None, int | None]"]


def main_league_ids() -> set[int]:
    """The API-Football league ids of the 22 main divisions (the football paper trial's scope)."""
    from football.reference import COMPETITIONS
    from sources.football import MAIN
    return {c.apif_league_id for c in COMPETITIONS if c.fd_code in MAIN and c.apif_league_id}


def http_fetch(key: str) -> Fetch:
    """GET API-Football with the key in its header -> (body, remaining daily requests)."""
    s = session()
    s.headers.update({"x-apisports-key": key})

    def fetch(path: str, params: dict) -> tuple[dict | None, int | None]:
        r = get(s, f"{API}/{path}", params=params, pause=0.2)
        if r is None:
            return None, None
        left = r.headers.get("x-ratelimit-requests-remaining")
        return r.json(), (int(left) if left and str(left).isdigit() else None)
    return fetch


def _stamp(t: datetime) -> str:
    return t.strftime("%Y%m%dT%H%M%S") + f"{t.microsecond // 1000:03d}Z"


def has_both_elevens(f: dict) -> bool:
    lu = f.get("lineups") or []
    return len(lu) == 2 and all(len(x.get("startXI") or []) == 11 for x in lu)


@dataclass
class Match:
    fixture_id: int
    kickoff: datetime
    status: str
    league_id: int
    seen_at: datetime | None = None
    last_poll: datetime | None = None
    confirmed: bool = False


@dataclass
class Collector:
    """The day's loop, with the clock and the HTTP call injected (the tests' seam)."""

    store: Store
    fetch: Fetch
    day: str
    leagues: set[int]
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    lead_min: float = LEAD_MIN
    every_min: float = EVERY_MIN
    confirm_min: float = CONFIRM_MIN
    quota_stop: int = QUOTA_STOP
    matches: dict[int, Match] = field(default_factory=dict)
    quota_left: int | None = None
    calls: int = 0
    stopped: str | None = None
    rate_limited: int = 0
    _n: int = 0

    def _call(self, kind: str, path: str, params: dict) -> dict | None:
        """One request, its reply kept raw with its fetch time; None (and the loop stopped) on an error reply or
        once the plan's remaining requests reach ``quota_stop``."""
        if self.quota_left is not None and self.quota_left <= self.quota_stop:
            self.stopped = f"quota at {self.quota_left}"
            return None
        t = self.now()
        body, left = self.fetch(path, params)
        self.calls += 1
        if left is not None:
            self.quota_left = left
        if body is None:
            return None
        self._n += 1
        doc = {"fetched_utc": t.isoformat(), "path": path, "params": params, "body": body}
        self.store.put(f"{PREFIX}/raw/{kind}/{self.day}/{_stamp(t)}_{self._n}.json.gz",
                       gz(json.dumps(doc).encode()))
        if body.get("errors"):
            err = str(body["errors"])[:200]
            if "rateLimit" in err:                       # the per-minute limit (another job on the key): skip a beat
                self.rate_limited += 1
                log.warning("api_football_lineups: rate limited (%d); carrying on", self.rate_limited)
                return None
            self.stopped = f"API error: {err}"
            log.warning("api_football_lineups: %s", self.stopped)
            return None
        return body

    def refresh_fixtures(self) -> int:
        body = self._call("day_fixtures", "fixtures", {"date": self.day, "timezone": "Europe/London"})
        if body is None:
            return 0
        n = 0
        for f in body.get("response") or []:
            lg = (f.get("league") or {}).get("id")
            if lg not in self.leagues:
                continue
            fx = f.get("fixture") or {}
            fid, st = fx.get("id"), (fx.get("status") or {}).get("short")
            ko = datetime.fromtimestamp(fx["timestamp"], timezone.utc) if fx.get("timestamp") else None
            if fid is None or ko is None:
                continue
            m = self.matches.get(fid)
            if m is None:
                self.matches[fid] = Match(fid, ko, st, lg)
            else:
                m.kickoff, m.status = ko, st
            n += 1
        return n

    def due(self) -> list[int]:
        """The matches to poll now: not started, inside the lead, and either still without both elevens (and not
        polled for ``every_min``) or with them but not yet confirmed and inside ``confirm_min``."""
        t = self.now()
        out = []
        for m in self.matches.values():
            mins = (m.kickoff - t).total_seconds() / 60.0
            if m.status not in NOT_STARTED or mins < 0 or mins > self.lead_min:
                continue
            since = None if m.last_poll is None else (t - m.last_poll).total_seconds() / 60.0
            if since is not None and since < self.every_min:
                continue
            if m.seen_at is None or (not m.confirmed and mins <= self.confirm_min):
                out.append(m.fixture_id)
        return sorted(out, key=lambda i: self.matches[i].kickoff)

    def poll(self, ids: list[int]) -> int:
        """Poll ``ids`` in batches of ``BATCH``; returns how many replies held both elevens."""
        got = 0
        for i in range(0, len(ids), BATCH):
            chunk = ids[i:i + BATCH]
            t = self.now()
            body = self._call("lineups", "fixtures", {"ids": "-".join(str(x) for x in chunk)})
            if body is None:
                return got
            for f in body.get("response") or []:
                fid = (f.get("fixture") or {}).get("id")
                m = self.matches.get(fid)
                if m is None:
                    continue
                m.last_poll = t
                st = ((f.get("fixture") or {}).get("status") or {}).get("short")
                if st:
                    m.status = st
                if has_both_elevens(f):
                    got += 1
                    if m.seen_at is None:
                        m.seen_at = t
                    elif (m.kickoff - t).total_seconds() / 60.0 <= self.confirm_min:
                        m.confirmed = True
            for fid in chunk:                       # a match the reply left out still counts as polled
                if fid in self.matches and self.matches[fid].last_poll != t:
                    self.matches[fid].last_poll = t
        return got

    def injuries(self) -> int:
        body = self._call("injuries", "injuries", {"date": self.day})
        return 0 if body is None else int(body.get("results") or 0)

    def summary(self) -> dict:
        ms = list(self.matches.values())
        return {"day": self.day, "matches": len(ms), "lineups_seen": sum(m.seen_at is not None for m in ms),
                "confirmed": sum(m.confirmed for m in ms), "calls": self.calls, "quota_left": self.quota_left,
                "stopped": self.stopped, "rate_limited": self.rate_limited}


def collect(store: Store, fetch: Fetch, until: str, leagues: set[int] | None = None, tick_s: float = 30.0,
            now=None, sleep=time.sleep) -> dict:
    """The day's loop to ``until`` (UK clock)."""
    now = now or (lambda: datetime.now(timezone.utc))
    day = now().astimezone(UK).date().isoformat()
    c = Collector(store, fetch, day, leagues or main_league_ids(), now=now)
    end_h, end_m = (int(x) for x in until.split(":"))
    last_fix = last_inj = None
    failures = 0
    while c.stopped is None:
        t = now()
        uk = t.astimezone(UK)
        if (uk.hour, uk.minute) >= (end_h, end_m):
            break
        try:
            if last_fix is None or (t - last_fix) >= timedelta(minutes=FIXTURES_EVERY_MIN):
                n = c.refresh_fixtures()
                last_fix = t
                log.info("api_football_lineups: %d matches today in the 22 divisions", n)
            if last_inj is None or (t - last_inj) >= timedelta(hours=INJURY_EVERY_H):
                c.injuries()
                last_inj = t
            ids = c.due()
            if ids:
                c.poll(ids)
        except Exception as exc:                    # a bad reply must not end the day's record
            failures += 1
            log.warning("api_football_lineups: step failed (%d): %s", failures, str(exc)[:300])
            if failures >= MAX_FAILURES:                 # a fault that repeats all day (a 403, a bad key) is not a blip
                c.stopped = f"{failures} failed steps; last: {str(exc)[:200]}"
        sleep(tick_s)
    out = c.summary()
    store.put(f"{PREFIX}/lineups_runs/{day}/{_stamp(now())}.json", json.dumps(out).encode())
    log.info("api_football_lineups: %s", out)
    return out


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def _docs(store: Store, kind: str):
    for k in sorted(store.listing(f"{PREFIX}/raw/{kind}/")):
        if not k.endswith(".json.gz"):
            continue
        try:
            yield json.loads(gunzip(store.get(k)))
        except ValueError:
            log.warning("api_football_lineups: %s unreadable", k)


def _xi(team: dict) -> tuple[int, ...]:
    return tuple(sorted((p.get("player") or {}).get("id") or -1 for p in team.get("startXI") or []))


def build(store: Store, leagues: set[int] | None = None) -> dict:
    import pandas as pd
    leagues = leagues or main_league_ids()
    polls = []                                      # (fetched, fixture) for every match in every line-up poll
    for doc in _docs(store, "lineups"):
        t = pd.Timestamp(doc["fetched_utc"])
        for f in (doc.get("body") or {}).get("response") or []:
            if (f.get("league") or {}).get("id") in leagues:
                polls.append((t, f))
    report = {"lineup_polls": len(polls)}
    times, players = [], []
    by_fix: dict[int, list] = {}
    for t, f in polls:
        by_fix.setdefault((f.get("fixture") or {}).get("id"), []).append((t, f))
    for fid, seq in by_fix.items():
        seq.sort(key=lambda x: x[0])
        fx, lg, tm = seq[-1][1].get("fixture") or {}, seq[-1][1].get("league") or {}, seq[-1][1].get("teams") or {}
        ko = pd.Timestamp(fx["timestamp"], unit="s", tz="UTC") if fx.get("timestamp") else pd.NaT
        seen = [(t, f) for t, f in seq if has_both_elevens(f)]
        first = seen[0] if seen else None
        unseen_before = [t for t, f in seq if first is not None and t < first[0] and not has_both_elevens(f)]
        changed = False
        if first is not None and len(seen) > 1:
            changed = [_xi(x) for x in first[1]["lineups"]] != [_xi(x) for x in seen[-1][1]["lineups"]]
        lu = (seen[-1][1]["lineups"] if seen else [])
        times.append({"apif_fixture_id": fid, "apif_league_id": lg.get("id"), "apif_season": lg.get("season"),
                      "home": (tm.get("home") or {}).get("name"), "away": (tm.get("away") or {}).get("name"),
                      "apif_home_id": (tm.get("home") or {}).get("id"), "apif_away_id": (tm.get("away") or {}).get("id"),
                      "kickoff_utc": ko, "polls": len(seq), "first_poll_utc": seq[0][0],
                      "first_seen_utc": first[0] if first else pd.NaT,
                      "last_unseen_utc": max(unseen_before) if unseen_before else pd.NaT,
                      "minutes_before_ko": ((ko - first[0]).total_seconds() / 60.0) if first and pd.notna(ko) else None,
                      "xi_changed_after_first": changed,
                      "formation_home": lu[0].get("formation") if lu else None,
                      "formation_away": lu[1].get("formation") if len(lu) > 1 else None})
        home_id, away_id = (tm.get("home") or {}).get("id"), (tm.get("away") or {}).get("id")
        for j, team in enumerate(lu):
            tid = (team.get("team") or {}).get("id")
            # by team id; where it matches neither side, API-Football's order (home first)
            side = "home" if tid == home_id else "away" if tid == away_id else ("home" if j == 0 else "away")
            for role, key in (("starter", "startXI"), ("substitute", "substitutes")):
                for p in team.get(key) or []:
                    pl = p.get("player") or {}
                    players.append({"apif_fixture_id": fid, "side": side, "apif_team_id": (team.get("team") or {}).get("id"),
                                    "team": (team.get("team") or {}).get("name"), "role": role,
                                    "apif_player_id": pl.get("id"), "player": pl.get("name"), "number": pl.get("number"),
                                    "pos": pl.get("pos"), "grid": pl.get("grid"), "formation": team.get("formation"),
                                    "coach": (team.get("coach") or {}).get("name"),
                                    "first_seen_utc": first[0] if first else pd.NaT})
    lt, lp = pd.DataFrame(times), pd.DataFrame(players)
    if len(lt):
        store.put_parquet(f"{PREFIX}/lineup_times.parquet", lt)
    if len(lp):
        store.put_parquet(f"{PREFIX}/lineups.parquet", lp)
    report.update({"matches": int(len(lt)), "with_both_elevens": int(lt["first_seen_utc"].notna().sum()) if len(lt) else 0,
                   "player_rows": int(len(lp)),
                   # API-Football gives some substitutes (mostly youth players) no player id: counted, not duplicates
                   "player_rows_without_id": int(lp["apif_player_id"].isna().sum()) if len(lp) else 0,
                   "duplicate_player_rows": int(lp.dropna(subset=["apif_player_id"]).duplicated(
                       ["apif_fixture_id", "apif_player_id"]).sum()) if len(lp) else 0,
                   "sides_without_eleven_starters": int((lp[lp["role"] == "starter"].groupby(
                       ["apif_fixture_id", "side"]).size() != 11).sum()) if len(lp) else 0,
                   "xi_changed": int(lt["xi_changed_after_first"].sum()) if len(lt) else 0,
                   "median_minutes_before_ko": float(lt["minutes_before_ko"].median()) if len(lt) and
                   lt["minutes_before_ko"].notna().any() else None})
    inj = {}
    rows_in = 0
    for doc in _docs(store, "injuries"):
        t = pd.Timestamp(doc["fetched_utc"])
        for r in (doc.get("body") or {}).get("response") or []:
            rows_in += 1
            if (r.get("league") or {}).get("id") not in leagues:
                continue
            pl, fx = r.get("player") or {}, r.get("fixture") or {}
            k = (fx.get("id"), pl.get("id"))
            rec = inj.get(k)
            if rec is None:
                inj[k] = {"apif_fixture_id": fx.get("id"), "apif_league_id": (r.get("league") or {}).get("id"),
                          "apif_team_id": (r.get("team") or {}).get("id"), "team": (r.get("team") or {}).get("name"),
                          "apif_player_id": pl.get("id"), "player": pl.get("name"), "type": pl.get("type"),
                          "reason": pl.get("reason"), "first_seen_utc": t, "last_seen_utc": t}
            else:
                rec["last_seen_utc"] = max(rec["last_seen_utc"], t)
                rec["type"], rec["reason"] = pl.get("type"), pl.get("reason")
    ij = pd.DataFrame(list(inj.values()))
    if len(ij):
        store.put_parquet(f"{PREFIX}/injuries.parquet", ij)
    report.update({"injury_rows_in": rows_in, "injury_rows_out": int(len(ij)),
                   "injury_null_player_or_fixture": int(ij[["apif_player_id", "apif_fixture_id"]].isna().any(axis=1).sum())
                   if len(ij) else 0})
    log.info("api_football_lineups build: %s", report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--until", default="21:30", help="last UK time to collect, HH:MM")
    ap.add_argument("--ids", default=None, help="poll these fixture ids once (a-b-c), e.g. to check the parse")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--root", default=None, help="a local sources folder in place of S3")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = Store(root=a.root)
    out = {}
    if a.collect or a.ids:
        key = env("API_FOOTBALL_KEY")
        if not key:
            print(json.dumps({"error": "no API_FOOTBALL_KEY; nothing collected"}))
            return 0
        fetch = http_fetch(key)
        if a.ids:
            day = datetime.now(timezone.utc).astimezone(UK).date().isoformat()
            c = Collector(store, fetch, day, main_league_ids())
            ids = [int(x) for x in a.ids.split("-") if x.strip()]
            for i in ids:
                c.matches[i] = Match(i, datetime.max.replace(tzinfo=timezone.utc), "NS", 0)
            got = c.poll(ids)
            out["ids"] = {"polled": len(ids), "with_both_elevens": got, "calls": c.calls, "quota_left": c.quota_left}
        else:
            out["collect"] = collect(store, fetch, a.until)
    if a.build:
        out["build"] = build(store)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
