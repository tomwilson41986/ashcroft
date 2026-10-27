"""Oddschecker, the second source the owner named (23 Oct): a probe until a UK browser is seen to pass its challenge.

From outside the UK the meeting pages answer with a Cloudflare challenge ("Just a moment..."), so no page has been
read to build a parser on. This module fetches a meeting page, keeps the raw HTML for the record, says whether the
challenge stood, and parses what it can by the general shape of an odds-comparison table (a row a runner, a cell a
bookmaker, fractional prices). The first run from the UK runner shows whether a real page comes back; the parser is
then finished against it.
"""
from __future__ import annotations

import logging
import re
from datetime import date

from bs4 import BeautifulSoup

from professor.scrapers.base import Browser, Quote, to_decimal, utc_now_iso

log = logging.getLogger(__name__)
NAME = "oddschecker"
BASE = "https://www.oddschecker.com/horse-racing"
CHALLENGE = re.compile(r"just a moment|cf-chl|challenge-platform|attention required", re.I)


def challenged(html: str) -> bool:
    return bool(CHALLENGE.search(html[:5000] if html else ""))


def quotes_from_race(html: str, venue: str, hhmm: str, race_key: str = "") -> list[Quote]:
    """Best effort on an odds-comparison table: rows with a runner name and fractional prices under bookmaker
    headers (the header read from the table's logos' alt or title text)."""
    if not html or challenged(html):
        return []
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        return []
    heads = []
    for th in table.find_all("th"):
        label = th.get("title") or th.get("data-bk") or (th.find("img") or {}).get("alt") if th.find("img") else None
        heads.append((label or th.get_text(" ", strip=True) or "").strip())
    now, out = utc_now_iso(), []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue
        name = (tr.get("data-bname") or cells[0].get_text(" ", strip=True)).strip()
        if not name:
            continue
        for i, td in enumerate(cells):
            txt = td.get("data-odig") or td.get_text(" ", strip=True)
            price = to_decimal(txt)
            if price is None or i >= len(heads) or not heads[i]:
                continue
            out.append(Quote(source=NAME, bookmaker=heads[i], venue=venue, race_hhmm=hhmm, horse=name, price=price,
                             fetched_utc=now, race_key=race_key))
    return out


class Oddschecker:
    name = NAME

    def __init__(self, browser: Browser):
        self.b = browser
        self.raw: dict[str, str] = {}
        self.blocked = False

    def meeting_quotes(self, day: date | str, venue: str, races: list[str]) -> list[Quote]:
        """One meeting: the race pages /<venue>/<HH:MM>/winner; stops at the first challenge of the run."""
        out = []
        slug = re.sub(r"[^a-z]", "-", venue.lower()).strip("-")
        for hhmm in races:
            if self.blocked:
                break
            url = f"{BASE}/{slug}/{hhmm}/winner"
            try:
                html = self.b.html(url, settle_ms=12000)
            except Exception as e:  # noqa: BLE001
                log.warning("oddschecker %s: %s", url, str(e)[:100])
                continue
            self.raw[f"{slug}_{hhmm.replace(':', '')}"] = html
            if challenged(html):
                self.blocked = True
                log.warning("oddschecker: the challenge stood at %s; no more pages this run", url)
                break
            out.extend(quotes_from_race(html, venue, hhmm, url))
        return out
