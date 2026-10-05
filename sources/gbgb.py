"""UK greyhound results from the Greyhound Board of Great Britain (api.gbgb.org.uk), for the greyhound model.

The public API gives, for a day, its races' winners (``/api/results?date=``, 100 a page, at most 500 a day), and for a
meeting every race in full (``/api/results/meeting/<id>``): trap, finishing position, distance beaten, sectional
time, run time, calculated time, the race comment ("QuickAway,Middle,AlwaysLed"), weight, the starting price with the
market's order, grade and distance, going allowance, the dog's breeding, sex and birth month, trainer and owner. Its
history starts about 2012; the Betfair price files the model is scored on start in 2018, so that is the default.

Raw: one file a day, ``gbgb/raw/<year>/<day>.json.gz`` = {"date", "count", "capped", "meetings": [the meeting
responses as served]}. A day whose list reached 500 is marked ``capped``: some meetings may be missing from it.
Table: ``gbgb/runs_<year>.parquet``, one row a dog a race.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from sources.common import Budget, Store, get, gunzip, gz, session

log = logging.getLogger(__name__)

API = "https://api.gbgb.org.uk/api"
FIRST = date(2018, 1, 1)
LIST_CAP = 500
REFRESH_DAYS = 3                                   # results are corrected for a day or two after the meeting


def raw_key(d: date) -> str:
    return f"gbgb/raw/{d.year}/{d:%Y-%m-%d}.json.gz"


def meeting_ids(s, d: date) -> tuple[list[int], int]:
    ids: list[int] = []
    count, page = 0, 1
    while True:
        r = get(s, f"{API}/results", params={"page": page, "itemsPerPage": 100, "date": f"{d:%Y-%m-%d}"})
        j = r.json() if r is not None else {"items": [], "meta": {}}
        count = int((j.get("meta") or {}).get("count") or 0)
        for it in j.get("items") or []:
            mid = it.get("meetingId")
            if mid is not None and mid not in ids:
                ids.append(int(mid))
        if page >= int((j.get("meta") or {}).get("pageCount") or 0):
            return ids, count
        page += 1


def fetch_day(s, d: date, workers: int = 3) -> dict:
    ids, count = meeting_ids(s, d)

    def one(mid: int):
        r = get(s, f"{API}/results/meeting/{mid}")
        return r.json() if r is not None else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        meetings = [m for m in ex.map(one, ids) if m]
    return {"date": f"{d:%Y-%m-%d}", "count": count, "capped": count >= LIST_CAP, "meetings": meetings}


def fetch(store: Store, date_from: date = FIRST, date_to: date | None = None, max_minutes: float | None = None,
          refresh_days: int = REFRESH_DAYS, workers: int = 3, s=None) -> dict:
    """Every day from ``date_to`` (default yesterday) back to ``date_from`` the store lacks, newest first, and the
    last ``refresh_days`` again."""
    s = s or session()
    date_to = date_to or (date.today() - timedelta(days=1))
    have = store.listing("gbgb/raw/")
    budget = Budget(max_minutes)
    out = {"fetched": 0, "races": 0, "capped": 0, "skipped": 0, "stopped": False}
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
            day = fetch_day(s, d, workers)
        except Exception as exc:
            log.warning("gbgb %s not fetched (%s)", d, exc)
            d -= timedelta(days=1)
            continue
        store.put(raw_key(d), gz(json.dumps(day, separators=(",", ":")).encode()))
        out["fetched"] += 1
        out["races"] += sum(len(m.get("races") or []) for m in _meetings(day))
        out["capped"] += bool(day["capped"])
        if out["fetched"] % 25 == 0:
            log.info("gbgb: %d days fetched, back to %s", out["fetched"], d)
        d -= timedelta(days=1)
    return out


# --------------------------------------------------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------------------------------------------------

def _f(x):
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return v


def _i(x):
    v = _f(x)
    return None if v is None else int(v)


_FRAC = {"": 0.0, "nk": 0.05, "hd": 0.1, "shd": 0.05, "sh": 0.05, "dh": 0.0, "dis": None, "dist": None}


def lengths(text) -> float | None:
    """GBGB's distance beaten ("5 1/2", "3/4", "Hd", "Nk", "Dis") in lengths; None when it is not a distance."""
    t = str(text or "").strip().lower()
    if t in _FRAC:
        return _FRAC[t]
    m = re.fullmatch(r"(\d+)?\s*(?:(\d)/(\d))?", t)
    if not m or not (m.group(1) or m.group(2)):
        return None
    whole = float(m.group(1) or 0)
    part = float(m.group(2)) / float(m.group(3)) if m.group(2) else 0.0
    return whole + part


def sp_decimal(trap: dict) -> float | None:
    """The starting price as a decimal: from the numerator and denominator GBGB gives, else from the text ("11/4F",
    "EvsF")."""
    n, d = _f(trap.get("resultPriceNumerator")), _f(trap.get("resultPriceDenominator"))
    if n is not None and d:
        return round(1.0 + n / d, 4)
    t = str(trap.get("SP") or "").strip().lower().rstrip("fjc").strip()
    if t.startswith("ev"):
        return 2.0
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", t)
    return round(1.0 + float(m.group(1)) / float(m.group(2)), 4) if m and float(m.group(2)) else None


def _day(text) -> str | None:
    try:
        return datetime.strptime(str(text), "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _meetings(day: dict):
    """The day's meetings: each meeting response is a list of one meeting (served that way), or the meeting."""
    for resp in day.get("meetings") or []:
        yield from (resp if isinstance(resp, list) else [resp])


def rows(day: dict) -> list[dict]:
    out = []
    for meet in _meetings(day):
        for race in meet.get("races") or []:
            traps = race.get("traps") or []
            for t in traps:
                out.append({
                    "race_date": _day(race.get("raceDate") or meet.get("meetingDate")),
                    "race_time": race.get("raceTime"),
                    "meeting_id": _i(meet.get("meetingId")),
                    "track": meet.get("trackName"),
                    "race_id": _i(race.get("raceId")),
                    "race_number": _i(race.get("raceNumber")),
                    "race_title": race.get("raceTitle"),
                    "race_type": race.get("raceType"),
                    "race_class": race.get("raceClass"),
                    "handicap": bool(race.get("raceHandicap")),
                    "distance_m": _f(race.get("raceDistance")),
                    "going": _f(race.get("raceGoing")),
                    "prizes": race.get("racePrizes"),
                    "forecast": race.get("raceForecast"),
                    "tricast": race.get("raceTricast"),
                    "runners": len(traps),
                    "trap": _i(t.get("trapNumber")),
                    "trap_handicap": t.get("trapHandicap"),
                    "dog_id": _i(t.get("dogId")),
                    "dog_name": t.get("dogName"),
                    "sire": t.get("dogSire"),
                    "dam": t.get("dogDam"),
                    "origin": t.get("dogOrigin"),
                    "born": t.get("dogBorn"),
                    "colour": t.get("dogColour"),
                    "sex": t.get("dogSex"),
                    "season": t.get("dogSeason"),
                    "trainer": t.get("trainerName"),
                    "owner": t.get("ownerName"),
                    "sp": t.get("SP"),
                    "sp_decimal": sp_decimal(t),
                    "market_pos": _i(t.get("resultMarketPos")),
                    "market_count": _i(t.get("resultMarketCnt")),
                    "position": _i(t.get("resultPosition")),
                    "beaten": t.get("resultBtnDistance"),
                    "beaten_lengths": lengths(t.get("resultBtnDistance")),
                    "sectional": _f(t.get("resultSectionalTime")),
                    "run_time": _f(t.get("resultRunTime")),
                    "adjusted_time": _f(t.get("resultAdjustedTime")),
                    "weight_kg": _f(t.get("resultDogWeight")),
                    "comment": t.get("resultComment"),
                })
    return out


def build(store: Store, years: list[int] | None = None) -> dict:
    """The year tables from the raw days: every year by default, or only ``years``."""
    import pandas as pd
    by_year: dict[int, list[str]] = {}
    for key in store.listing("gbgb/raw/"):
        m = re.search(r"/(\d{4})-\d{2}-\d{2}\.json\.gz$", key)
        if m:
            by_year.setdefault(int(m.group(1)), []).append(key)
    out = {}
    for y in sorted(by_year):
        if years and y not in years:
            continue
        recs = []
        for key in sorted(by_year[y]):
            recs += rows(json.loads(gunzip(store.get(key))))
        df = pd.DataFrame(recs)
        if len(df):
            df = df.drop_duplicates(["race_id", "trap", "dog_id"], keep="last")
        store.put_parquet(f"gbgb/runs_{y}.parquet", df)
        out[y] = {"days": len(by_year[y]), "runs": len(df), "races": int(df["race_id"].nunique()) if len(df) else 0}
        log.info("gbgb %d: %s", y, out[y])
    return out
