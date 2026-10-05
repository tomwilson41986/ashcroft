"""Australian racing form and results from Punting Form (api.puntingform.com.au), for the Australian racing model.

A subscription (Starter or above for results; the developer API with ProPunter) gives an API key, kept as the
repository secret ``PUNTINGFORM_API_KEY``. Without one this source does nothing and says so. With one, each day:

- ``/v2/form/meetingslist?meetingDate=`` the day's meetings;
- ``/v2/form/results?meetingId=&raceNumber=0`` each meeting's results, every race;
- ``/v2/form/fields?meetingId=&raceNumber=0`` each meeting's fields as declared (barrier, weight, jockey, trainer,
  rating, the runner's form): what the model would read before the race.

Raw: ``puntingform/raw/<year>/<day>.json.gz`` = {"date", "meetings", "results": {id: ...}, "fields": {id: ...}}, as
served. The table is built once the first files show the subscription's layout.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from sources.common import Budget, Store, env, get, gz, session

log = logging.getLogger(__name__)

API = "https://api.puntingform.com.au/v2/form"
FIRST = date(2018, 1, 1)


def raw_key(d: date) -> str:
    return f"puntingform/raw/{d.year}/{d:%Y-%m-%d}.json.gz"


def _meeting_id(m: dict):
    for k in ("meetingId", "MeetingId", "meetingID", "id"):
        if m.get(k) is not None:
            return m[k]
    return None


def _payload(j):
    """The list Punting Form wraps in {"payLoad": [...]} (or serves bare)."""
    if isinstance(j, dict):
        for k in ("payLoad", "payload", "data", "result"):
            if k in j:
                return j[k]
    return j


def fetch(store: Store, date_from: date = FIRST, date_to: date | None = None, max_minutes: float | None = None,
          refresh_days: int = 3, s=None, key: str | None = None) -> dict:
    key = key or env("PUNTINGFORM_API_KEY")
    if not key:
        log.warning("puntingform: no PUNTINGFORM_API_KEY, nothing fetched (a Punting Form subscription supplies one)")
        return {"fetched": 0, "skipped": "no PUNTINGFORM_API_KEY"}
    s = s or session()
    date_to = date_to or (date.today() - timedelta(days=1))
    have = store.listing("puntingform/raw/")
    budget = Budget(max_minutes)
    out = {"fetched": 0, "meetings": 0, "skipped": 0, "failed": 0, "stopped": False}
    d = date_to
    while d >= date_from:
        if not budget.left():
            out["stopped"] = True
            break
        if raw_key(d) in have and (date_to - d).days >= refresh_days:
            out["skipped"] += 1
            d -= timedelta(days=1)
            continue
        try:
            r = get(s, f"{API}/meetingslist", params={"meetingDate": f"{d:%Y-%m-%d}", "apiKey": key}, pause=0.3)
            meetings = _payload(r.json()) if r is not None else []
            day = {"date": f"{d:%Y-%m-%d}", "meetings": meetings, "results": {}, "fields": {}}
            for m in meetings or []:
                mid = _meeting_id(m)
                if mid is None:
                    continue
                for kind in ("results", "fields"):
                    r = get(s, f"{API}/{kind}", params={"meetingId": mid, "raceNumber": 0, "apiKey": key}, pause=0.3)
                    day[kind][str(mid)] = _payload(r.json()) if r is not None else None
        except Exception as exc:
            msg = str(exc).replace(key, "***")
            log.warning("puntingform %s not fetched (%s)", d, msg)
            out["failed"] += 1
            if out["failed"] >= 5 and out["fetched"] == 0:
                break                                    # a key the API refuses: no point going on
            d -= timedelta(days=1)
            continue
        store.put(raw_key(d), gz(json.dumps(day, separators=(",", ":")).encode()))
        out["fetched"] += 1
        out["meetings"] += len(meetings or [])
        d -= timedelta(days=1)
    return out


def build(store: Store) -> dict:
    n = len(store.listing("puntingform/raw/"))
    return {"raw_days": n, "table": "not built yet: the layout is read from the first files a subscription fetches"}
