"""Does horseracebase.com serve tomorrow's racecards the evening before? (the owner, 4 Nov 2026: "Run predictions
by night before so we are ready to act then if needed"). The predictions read HRB's onedayracecards.php, which takes
no date in our code: this logs in, reads that page and a few candidate addresses for tomorrow, and prints for each
the races found (time and course), any date written on the page, and the links that name a card or a date. Read-only;
prints no page text beyond those, and masks long numbers (user ids) in the links."""

import re
import sys
from datetime import date, datetime, timedelta, timezone
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
    # the 48-hour declarations (the menu's "48hr Decs Cards & Analysis"): which races, and the card's columns
    url = f"{base}/horse-racing-today.php?showearly=1"
    r = s.get(url, timeout=60)
    soup = BeautifulSoup(r.text, "lxml")
    race_links = [(a.get("href"), a.get_text(" ", strip=True)) for a in soup.find_all("a")
                  if "raceid=" in (a.get("href") or "")]
    dates = sorted(set(m.group(0) for m in DATE_RX.finditer(soup.get_text(" "))))[:8]
    courses = sorted(set(t.split(" ", 1)[1] for _, t in race_links if " " in t))
    print(f"\n== horse-racing-today.php?showearly=1: HTTP {r.status_code}, {len(r.text):,} bytes, {len(race_links)} race "
          f"links; courses {courses}; dates on the page {dates}")
    for h, t in race_links[:3]:
        print("   ", re.sub(r"\d{5,}", "N", h), f"[{t}]")
    if race_links:
        r2 = s.get(urljoin(url, race_links[0][0]), timeout=60)
        s2 = BeautifulSoup(r2.text, "lxml")
        dates2 = sorted(set(m.group(0) for m in DATE_RX.finditer(s2.get_text(" "))))[:5]
        print(f"   the first race's page: HTTP {r2.status_code}, {len(r2.text):,} bytes; dates {dates2}")
        # the innermost tables with a runner's row (a row of 8 or more cells, one naming a weight like 9-7 or 11-2)
        inner = [t for t in s2.find_all("table") if not t.find("table")]
        for t in inner:
            rows = t.find_all("tr", recursive=False) or t.find_all("tr")
            cells = [[c.get_text(" ", strip=True)[:22] for c in r.find_all(["td", "th"], recursive=False)] for r in rows]
            if sum(1 for c in cells if len(c) >= 8 and any(re.fullmatch(r"\d{1,2}-\d{1,2}", x) for x in c)) >= 2:
                print(f"   a runner table: {len(rows)} rows; the first 6 rows' cells:")
                for c in cells[:6]:
                    print("     ", c[:32])
                classes = sorted(set(" ".join(td.get("class") or []) for td in t.find_all("td")))[:20]
                print("      cell classes:", classes)
                break
        for sp in s2.find_all(["span", "h1", "h2", "h3", "b"])[:400]:
            tx = sp.get_text(" ", strip=True)
            if re.search(r"Class|Going|runners|\d+m|Win:", tx) and len(tx) < 160:
                print("   race line:", tx)
        n = 0
        for table in s2.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue
            head = [c.get_text(" ", strip=True)[:18] for c in rows[0].find_all(["td", "th"])]
            if len(head) >= 6:
                print(f"   a table of {len(rows)} rows, its columns: {head[:30]}")
                n += 1
            if n >= 4:
                break
    # the formats side by side: the one-day page's (what the morning reads) and the 48-hour race pages'
    one = BeautifulSoup(s.get(f"{base}/onedayracecards.php", timeout=60).text, "lxml")
    shown = 0
    hg = set()
    for table in one.find_all("table"):
        rows = table.find_all("tr")
        head = [c.get_text(strip=True) for c in rows[0].find_all(["td", "th"])] if rows else []
        if not head or head[0] != "No." or len(head) < 8:
            continue
        ix = {h: i for i, h in enumerate(head)}
        for r in rows[1:]:
            cs = r.find_all(["td", "th"])
            if len(cs) > ix.get("Headgear", 99) and cs[ix["Headgear"]].get_text(strip=True):
                hg.add(cs[ix["Headgear"]].get_text(strip=True))
        if shown < 2:
            spans, prev = [], table.find_previous_sibling()
            while prev is not None and prev.name != "table" and len(spans) < 4:
                if prev.name == "span":
                    spans.insert(0, prev.get_text(strip=True))
                prev = prev.find_previous_sibling()
            print(f"\none-day race: spans {spans}\n   columns {head}")
            for r in rows[1:4]:
                cs = r.find_all(["td", "th"])
                print("     ", [c.get_text(strip=True)[:22] for c in cs][:16])
                if len(cs) > ix.get("Horse", 99):
                    print("      horse title:", (cs[ix["Horse"]].get("title") or "")[:120])
            shown += 1
    print("one-day headgear values:", sorted(hg)[:40])
    vals = {"A/S": set(), "W/HG": set(), "D": set(), "Miles": set()}
    lines = []
    for h, _ in race_links[::2][:8]:
        pg = BeautifulSoup(s.get(urljoin(url, h), timeout=60).text, "lxml")
        for sp in pg.find_all("span"):
            tx = sp.get_text(" ", strip=True)
            if re.search(r"Class|runners|Win:", tx) and len(tx) < 200:
                lines.append(tx)
        for t in [t for t in pg.find_all("table") if not t.find("table")]:
            rows = t.find_all("tr")
            cells = [[c for c in r.find_all(["td", "th"], recursive=False)] for r in rows]
            head = [c.get_text(strip=True) for c in cells[0]] if cells else []
            if "No." not in head or "Horse" not in head:
                continue
            ix = {x: i for i, x in enumerate(head)}
            for c in cells[1:]:
                if len(c) < len(head) - 1:
                    continue
                for k in vals:
                    if k in ix and len(c) > ix[k]:
                        vals[k].add(c[ix[k]].get_text(" ", strip=True))
            first = [c for c in cells[1:] if len(c) >= len(head) - 1][:2]
            for c in first:
                print("   early horse title:", (c[ix["Horse"]].get("title") or "")[:120],
                      "| link:", re.sub(r"\d{5,}", "N", (c[ix["Horse"]].find("a") or {}).get("href", "") if c[ix["Horse"]].find("a") else ""))
            break
    for k, v in vals.items():
        print(f"early {k} values ({len(v)}):", sorted(v)[:40])
    print("early race lines:", lines[:10])
    today_page = races(s.get(f"{base}/onedayracecards.php", timeout=60).text)
    print(f"\nonedayracecards.php now ({datetime.now(timezone.utc):%H:%M} UTC): "
          f"{sorted(set(x.split(' ', 1)[1] for x in today_page if ' ' in x))}; the 48-hour page's {courses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
