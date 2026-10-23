"""Racing Post's odds comparison, the first odds source (the owner, 23 Oct).

The racecards index (`/racecards/<day>`) carries the day's races in its page state (`__NEXT_DATA__`: every meeting
and race with its URL, UK start time, meeting name and country). Each race's `/betting-odds/` page renders the
comparison table once its price feed (Diffusion) has filled it: a row a runner (`Row__<horseId>__Runner`, the name in
`Link__HorseProfile__<horseId>__OddsComparison`), a cell a bookmaker (`Cell__<horseId>_<CODE>__Odds`, fractional
prices), a Best Odds Guaranteed row and an each-way-terms row by the same codes. Ten bookmakers show by default:
William Hill, Ladbrokes, bet365, Coral, Paddy Power, Betfair Sportsbook, Sky Bet, BoyleSports, Betfred and the Tote.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date

from bs4 import BeautifulSoup

from professor.scrapers.base import Browser, Quote, to_decimal, utc_now_iso

log = logging.getLogger(__name__)
NAME = "racingpost"
BASE = "https://www.racingpost.com"
CODES = {
    "WH_OXI": "William Hill", "LADB": "Ladbrokes", "BET365": "bet365", "CORAL": "Coral", "PADDYPOWER": "Paddy Power",
    "BETFAIR": "Betfair Sportsbook", "SKYBET_FLUTTER": "Sky Bet", "BOLEYSPORTS": "BoyleSports", "BETFRED": "Betfred",
    "TOTE": "Tote", "UNIBET": "Unibet", "BETVICTOR": "BetVictor", "888SPORT": "888sport", "BETWAY": "Betway",
}


def _state(html: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("script", id="__NEXT_DATA__")
    if tag is None:
        return None
    try:
        return json.loads(tag.get_text())
    except json.JSONDecodeError:
        return None


def races_from_index(html: str, day: str) -> list[dict]:
    """The day's races from the index page's state: url, race id, venue, UK HH:MM, country, title."""
    st = _state(html)
    if not st:
        return []
    try:
        by_date = st["props"]["pageProps"]["initialState"]["meetings"]["byDate"]
    except KeyError:
        return []
    d = by_date.get(day) or next(iter(by_date.values()), {})
    races = (d.get("races") or {}).get("byRaceId") or {}
    out = []
    for rid, r in races.items():
        if not r.get("raceUrl"):
            continue
        out.append({"race_id": str(rid), "url": BASE + r["raceUrl"].rstrip("/") + "/betting-odds/",
                    "venue": r.get("meetingName") or r.get("venueName") or "", "hhmm": r.get("startTime") or "",
                    "country": r.get("countryCode") or "", "title": r.get("raceTitle") or "",
                    "abandoned": bool(r.get("isAbandoned")), "result": bool(r.get("isResult"))})
    out.sort(key=lambda x: (x["hhmm"], x["venue"]))
    return out


def quotes_from_race(html: str, venue: str, hhmm: str, race_key: str = "") -> list[Quote]:
    """Every bookmaker's price on every runner from a /betting-odds/ page."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find(attrs={"data-testid": "Table__OddsComparison"})
    if table is None:
        return []
    bog = {}
    for e in table.find_all(attrs={"data-testid": re.compile(r"^Cell__[A-Z0-9_]+__BestOddsGuaranteed$")}):
        code = e["data-testid"].split("__")[1]
        txt = e.get_text(" ", strip=True).lower()
        has_mark = bool(e.find("svg") or e.find("img") or e.find(attrs={"data-testid": re.compile("Icon|Tick|Check")}))
        bog[code] = True if ("yes" in txt or "✓" in txt or has_mark) else (False if txt in ("", "no", "-") else None)
    ew = {}
    for e in table.find_all(attrs={"data-testid": re.compile(r"^Cell__[A-Z0-9_]+__EachWayTerms$")}):
        ew[e["data-testid"].split("__")[1]] = e.get_text(" ", strip=True)
    now = utc_now_iso()
    out = []
    for row in table.find_all(attrs={"data-testid": re.compile(r"^Row__\d+__Runner$")}):
        hid = row["data-testid"].split("__")[1]
        link = row.find(attrs={"data-testid": re.compile(r"^Link__HorseProfile__")})
        horse = link.get_text(" ", strip=True) if link else row.get_text(" ", strip=True)[:40]
        for cell in row.find_all(attrs={"data-testid": re.compile(rf"^Cell__{hid}_[A-Z0-9_]+__Odds$")}):
            code = cell["data-testid"].split("__")[1].split("_", 1)[1]
            price = to_decimal(cell.get_text(" ", strip=True))
            if price is None:
                continue
            out.append(Quote(source=NAME, bookmaker=CODES.get(code, code.title()), venue=venue, race_hhmm=hhmm,
                             horse=horse, price=price, bog=bog.get(code), ew_terms=ew.get(code, ""), fetched_utc=now,
                             race_key=race_key))
    return out


class RacingPost:
    name = NAME

    def __init__(self, browser: Browser):
        self.b = browser
        self.raw: dict[str, str] = {}

    def day_races(self, day: date | str) -> list[dict]:
        html = self.b.html(f"{BASE}/racecards/{day}", settle_ms=3000)
        self.raw[f"index_{day}"] = html
        races = races_from_index(html, str(day))
        log.info("racingpost: %d races on %s", len(races), day)
        return races

    def race_quotes(self, race: dict) -> list[Quote]:
        html = self.b.html(race["url"], settle_ms=8000, wait_selector='[data-testid="Table__OddsComparison"]')
        self.raw[f"race_{race['race_id']}"] = html
        q = quotes_from_race(html, race["venue"], race["hhmm"], race.get("race_id", ""))
        log.info("racingpost: %s %s: %d quotes", race["hhmm"], race["venue"], len(q))
        return q
