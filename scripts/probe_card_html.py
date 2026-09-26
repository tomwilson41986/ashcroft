"""What the 06:00 HTML racecard carries beyond what the scraper reads: a one-off probe.

The live card comes from horseracebase's onedayracecards.php whenever the CSV
export is empty, and that path has no pedigree: a debutant's sire, dam and
damsire are missing at 06:00 although training always had them. On the parity
days (28 and 25 March) that one gap moved the 944's price for debutants by a
mean 0.26 in log terms, over half of all the difference between the live path
and training. The scraper skips every row of a race table that does not start
with a runner number ("e.g. stallion notes"); this saves the page and shows
what those rows, and a debutant's horse page, hold, so the gap can be closed at
its source if the card has it.

    python scripts/probe_card_html.py --out out/

Read-only: one login, the day's card page and at most two horse pages.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daily_predictions import BASE_URL, create_session, login  # noqa: E402

PEDIGREE_RX = re.compile(r"\b(sire|dam|by|out of|damsire|pedigree)\b", re.I)


def _cells(row) -> list[str]:
    return [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    ap.add_argument("--horse-pages", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    s = create_session()
    if not login(s):
        print("login failed")
        return 1
    r = s.get(f"{BASE_URL}/onedayracecards.php")
    r.raise_for_status()
    (out / "onedayracecards.html").write_text(r.text, encoding="utf-8")
    print(f"card page: {len(r.text):,} bytes")

    soup = BeautifulSoup(r.text, "lxml")
    tables = [t for t in soup.find_all("table")
              if t.find("tr") and _cells(t.find("tr"))[:1] == ["No."]]
    print(f"race tables: {len(tables)}")
    other_rows, debut_links = [], []
    for ti, t in enumerate(tables):
        rows = t.find_all("tr")
        header = _cells(rows[0])
        if ti < 2:
            print(f"\n--- table {ti}: header {header}")
        for row in rows[1:]:
            cells = _cells(row)
            first = cells[0] if cells else ""
            if ti < 2:
                links = [a.get("href", "") for a in row.find_all("a")]
                print(f"  row: {[c[:40] for c in cells][:14]}  links: {links[:4]}")
            if not first.isdigit():
                other_rows.append(" | ".join(cells)[:300])
                continue
            if "Days" in header and len(cells) > header.index("Days") and not cells[header.index("Days")].strip():
                for a in row.find_all("a"):
                    if "horse" in a.get("href", "").lower():
                        debut_links.append((cells[header.index("Horse")] if "Horse" in header else "", a["href"]))
                        break
    print(f"\nrows not starting with a runner number: {len(other_rows)}")
    for x in other_rows[:12]:
        print("  ", x)
    ped = [x for x in other_rows if PEDIGREE_RX.search(x)]
    print(f"of which mention a pedigree word: {len(ped)}")
    for x in ped[:8]:
        print("  ", x)

    # text beside each runner that the table cells do not show (title attributes, hidden spans)
    titles = [el.get("title") for t in tables[:3] for el in t.find_all(attrs={"title": True})]
    print(f"\ntitle attributes in the first tables: {len(titles)}; e.g. {titles[:6]}")

    print(f"\ndebutants with a horse link: {len(debut_links)}; e.g. {debut_links[:3]}")
    for i, (name, href) in enumerate(debut_links[: args.horse_pages]):
        time.sleep(1.5)
        url = href if href.startswith("http") else f"{BASE_URL}/{href.lstrip('/')}"
        h = s.get(url)
        (out / f"horse_{i}.html").write_text(h.text, encoding="utf-8")
        text = BeautifulSoup(h.text, "lxml").get_text(" ", strip=True)
        hits = [text[max(0, m.start() - 60): m.end() + 80] for m in PEDIGREE_RX.finditer(text)][:8]
        print(f"\nhorse page {i} ({name}, {url}): {len(h.text):,} bytes; pedigree words:")
        for x in hits:
            print("  ", x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
