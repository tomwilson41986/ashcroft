"""Does horseracebase.com serve tomorrow's racecards the evening before? (the owner, 4 Nov 2026: "Run predictions
by night before so we are ready to act then if needed"). The predictions read HRB's onedayracecards.php, which takes
no date in our code: this logs in, reads that page and a few candidate addresses for tomorrow, and prints for each
the races found (time and course), any date written on the page, and the links that name a card or a date. Read-only;
prints no page text beyond those, and masks long numbers (user ids) in the links."""

import re
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import daily_predictions as dp  # noqa: E402

DATE_RX = re.compile(r"(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,?\s+\d{1,2}(st|nd|rd|th)?\s+[A-Z][a-z]+(\s+\d{4})?"
                     r"|\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|"
                     r"October|November|December)\s+\d{4}|20\d\d-\d\d-\d\d")


def races(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        head = [c.get_text(strip=True) for c in rows[0].find_all(["td", "th"])] if rows else []
        if not head or head[0] != "No." or len(head) < 8:
            continue
        prev, label = table.find_previous_sibling(), ""
        for _ in range(5):
            if prev is None or prev.name == "table":
                break
            m = re.match(r"(\d{1,2}\.\d{2})\s+(.+?)(?:\(\d+\s*runners?\))?$", prev.get_text(strip=True)) \
                if prev.name == "span" else None
            if m:
                label = f"{m.group(1)} {m.group(2).strip()}"
                break
            prev = prev.find_previous_sibling()
        out.append(label or "?")
    return out


def links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    seen = []
    for a in soup.find_all(["a", "form", "option"]):
        h = a.get("href") or a.get("action") or a.get("value") or ""
        if re.search(r"card|date|day|tomorrow", h, re.I):
            h = re.sub(r"\d{5,}", "N", urljoin(base, h) if a.name != "option" else h)
            text = re.sub(r"\s+", " ", a.get_text(" ", strip=True))[:40]
            if (h, text) not in seen:
                seen.append((h, text))
    return [f"{h}  [{t}]" for h, t in seen[:60]]


def main() -> int:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    if not dp.login(s):
        print("login failed")
        return 1
    today, tom = date.today(), date.today() + timedelta(days=1)
    t_iso, t_hrb = tom.isoformat(), f"{tom.year}-{tom.month}-{tom.day}"
    base = dp.BASE_URL
    cands = ["onedayracecards.php", f"onedayracecards.php?racedate={t_iso}", f"onedayracecards.php?racedate={t_hrb}",
             f"onedayracecards.php?date={t_iso}", "onedayracecards.php?day=tomorrow", "onedayracecards.php?day=1",
             "onedayracecards.php?tomorrow=1", "tomorrowsracecards.php", f"racecards.php?date={t_iso}",
             "racecards.php"]
    print(f"today {today}, tomorrow {tom}")
    for i, c in enumerate(cands):
        url = f"{base}/{c}"
        try:
            r = s.get(url, timeout=60)
        except requests.RequestException as e:
            print(f"\n== {c}: {type(e).__name__}")
            continue
        rs = races(r.text) if r.ok else []
        dates = sorted(set(m.group(0) for m in DATE_RX.finditer(BeautifulSoup(r.text, "lxml").get_text(" "))))[:8]
        courses = sorted(set(x.split(" ", 1)[1] for x in rs if " " in x))
        print(f"\n== {c}: HTTP {r.status_code}, {len(r.text):,} bytes, {len(rs)} races; courses {courses}; "
              f"dates on the page {dates}")
        if i == 0:
            print("links naming a card, a date or a day:")
            for ln in links(r.text, url):
                print("   ", ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
