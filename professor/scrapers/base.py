"""What every odds source shares: the quote, the polite browser, the price parsing and the name matching.

The sources are web pages built for a browser, so the fetch is Chromium through Playwright (headless, one context a
run, a pause between pages, a retry), never a bare request: Racing Post answers a browser and refuses a plain
request; Oddschecker fronts a Cloudflare challenge that only a UK browser may pass. A source that fails returns no
quotes and a status line; it never stops the run.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

log = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 "
      "Safari/537.36")
PAUSE_S = float(os.getenv("PROFESSOR_SCRAPE_PAUSE", "2.0"))


@dataclass
class Quote:
    source: str
    bookmaker: str          # the canonical name (config major_bookmakers)
    venue: str              # as the source names it
    race_hhmm: str          # UK time, HH:MM
    horse: str              # as the source names it
    price: float            # decimal
    bog: bool | None = None
    ew_terms: str = ""
    fetched_utc: str = ""
    race_key: str = ""      # the source's own race id or url

    def row(self) -> dict:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_decimal(text: str) -> float | None:
    """'7/2' -> 4.5, 'EVS' -> 2.0, '4.50' -> 4.5, 'SP' or '' -> None."""
    t = (text or "").strip().upper().replace(" ", "")
    if not t or t in ("SP", "-", "N/R", "NR", "–"):
        return None
    if t in ("EVS", "EVENS", "EVEN"):
        return 2.0
    m = re.match(r"^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return 1.0 + a / b if b > 0 else None
    m = re.match(r"^(\d+(?:\.\d+)?)$", t)
    if m:
        v = float(m.group(1))
        return v if v > 1 else None
    return None


def norm_name(x: str) -> str:
    """A horse or course name as a key: letters and digits only, lower case, no country suffix."""
    x = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", str(x or ""))
    return re.sub(r"[^a-z0-9]", "", x.lower())


def norm_venue(x: str) -> str:
    x = re.sub(r"\((?:aw|a\.w\.|all weather)\)", "", str(x or ""), flags=re.I)
    return norm_name(x)


class Browser:
    """One headless Chromium for a run; pages fetched with a pause between them, a retry on a failure."""

    def __init__(self, headless: bool = True):
        self._p = None
        self._browser = None
        self._ctx = None
        self.headless = headless
        self.last_fetch = 0.0

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._p = sync_playwright().start()
        exe = os.getenv("PROFESSOR_CHROME")
        kwargs = {"headless": self.headless, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if exe:
            kwargs["executable_path"] = exe
        self._browser = self._p.chromium.launch(**kwargs)
        self._ctx = self._browser.new_context(user_agent=UA, locale="en-GB", timezone_id="Europe/London",
                                              viewport={"width": 1600, "height": 1000})
        return self

    def __exit__(self, *a):
        for closer in (lambda: self._ctx.close(), lambda: self._browser.close(), lambda: self._p.stop()):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass

    def html(self, url: str, settle_ms: int = 6000, wait_selector: str | None = None, tries: int = 2) -> str:
        """The page's HTML once it has settled (an optional selector awaited), with the pause between pages."""
        for attempt in range(tries):
            gap = PAUSE_S - (time.time() - self.last_fetch)
            if gap > 0:
                time.sleep(gap)
            page = self._ctx.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, timeout=settle_ms + 10000)
                    except Exception:  # noqa: BLE001
                        pass
                page.wait_for_timeout(settle_ms)
                html = page.content()
                self.last_fetch = time.time()
                return html
            except Exception as e:  # noqa: BLE001
                log.warning("fetch %s (try %d): %s", url, attempt + 1, str(e)[:120])
                self.last_fetch = time.time()
                if attempt + 1 == tries:
                    raise
            finally:
                try:
                    page.close()
                except Exception:  # noqa: BLE001
                    pass
        return ""
