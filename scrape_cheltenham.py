"""Scrape Cheltenham racecards for today and run predictions."""
import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

BASE_URL = "https://www.horseracebase.com"


def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })

    # Login
    login_url = f"{BASE_URL}/horseracebase_login.php"
    resp = session.get(login_url)
    soup = BeautifulSoup(resp.text, "lxml")
    csrf = (soup.find("input", {"name": "csrf_token"})
            or soup.find("input", {"name": "CSRFtoken"}))
    session.post(
        login_url,
        data={
            "login": os.getenv("HRB_USERNAME"),
            "password": os.getenv("HRB_PASSWORD"),
            csrf.get("name"): csrf.get("value"),
        },
    )

    # Get today page
    resp = session.get(f"{BASE_URL}/horse-racing-today.php")
    soup = BeautifulSoup(resp.text, "lxml")

    # Parse track sections from headers
    track_sections = []
    for h in soup.find_all(["h2", "h3", "h4"]):
        text = h.get_text(strip=True)
        m = re.match(r"^(\w+)\((\d+) races?, (\d+) runners", text)
        if m:
            track_sections.append(
                {"track": m.group(1), "n_races": int(m.group(2))}
            )

    # Collect race links (time-labelled links with raceid)
    race_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        m = re.search(r"raceid=(\d+)", href)
        if m and re.match(r"^\d+\.\d+$", text):
            race_links.append({"race_id": m.group(1), "race_time": text})

    # Assign tracks based on order matching track sections
    idx = 0
    for section in track_sections:
        for i in range(section["n_races"]):
            if idx < len(race_links):
                race_links[idx]["track"] = section["track"]
                idx += 1

    cheltenham_races = [r for r in race_links if r.get("track") == "Cheltenham"]
    print(f"Cheltenham: {len(cheltenham_races)} races")

    all_runners = []
    for race in cheltenham_races:
        url = f"{BASE_URL}/horse-racing-today.php?raceid={race['race_id']}"
        time.sleep(1.5)
        resp = session.get(url)
        race_soup = BeautifulSoup(resp.text, "lxml")

        # Parse race metadata
        page_text = race_soup.get_text()
        going = "Good to Soft"
        distance = ""
        race_class = ""

        going_m = re.search(
            r"(Heavy|Soft|Good to Soft|Good to Firm|Good|Firm|Standard)", page_text
        )
        if going_m:
            going = going_m.group(1)
        dist_m = re.search(r"(\d+m[^,]*?)\s*\((\d+) yards\)", page_text)
        if dist_m:
            distance = dist_m.group(1).strip()
        class_m = re.search(r"(Grade \d|Class \d)", page_text)
        if class_m:
            race_class = class_m.group(1)

        # Find the runner table: header must have No., Horse, Jockey at correct positions
        runner_table = None
        for table in race_soup.find_all("table"):
            first_row = table.find("tr")
            if not first_row:
                continue
            cells = [c.get_text(strip=True) for c in first_row.find_all(["th", "td"])]
            if (
                len(cells) >= 10
                and cells[1] == "No."
                and cells[4] == "Horse"
                and cells[7] == "Jockey"
            ):
                runner_table = table
                break

        if not runner_table:
            print(f"  {race['race_time']}: No runner table found!")
            continue

        rows = runner_table.find_all("tr")
        race_runners = []

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(strip=True) for c in cells]

            if len(cell_texts) < 10:
                continue

            horse_name = cell_texts[4]
            if not horse_name or horse_name == "Horse":
                continue

            # Weight (e.g. "11-7" or "11-7TT")
            weight_str = cell_texts[6]
            weight_match = re.match(r"(\d+)-(\d+)", weight_str)
            pounds = (
                int(weight_match.group(1)) * 14 + int(weight_match.group(2))
                if weight_match
                else None
            )

            # Age from "5yo G"
            age_str = cell_texts[5]
            age_m = re.match(r"(\d+)", age_str)
            horse_age = int(age_m.group(1)) if age_m else None

            # OR
            or_str = cell_texts[10]
            official_rating = int(or_str) if or_str.isdigit() else None

            # Odds (fractional -> decimal)
            odds_str = cell_texts[11]
            odds_m = re.match(r"(\d+)/(\d+)", odds_str)
            odds = (
                int(odds_m.group(1)) / int(odds_m.group(2)) + 1 if odds_m else None
            )

            runner = {
                "horse_name": horse_name,
                "race_date": "2026-03-10",
                "race_time": race["race_time"],
                "track": "Cheltenham",
                "going_description": going,
                "race_class": race_class,
                "race_distance": distance,
                "jockey_name": cell_texts[7],
                "trainer": cell_texts[8],
                "horse_age": horse_age,
                "pounds": pounds,
                "official_rating": official_rating,
                "odds": odds,
                "stall": None,
                "card_no": cell_texts[1],
            }
            race_runners.append(runner)

        for r in race_runners:
            r["number_of_runners"] = len(race_runners)

        all_runners.extend(race_runners)
        print(f"  {race['race_time']}: {len(race_runners)} runners ({going}, {distance})")

    print(f"\nTotal: {len(all_runners)} Cheltenham runners")

    df = pd.DataFrame(all_runners)
    os.makedirs("csv/racecards", exist_ok=True)
    df.to_csv("csv/racecards/racecard_2026-03-10.csv", index=False)

    # Print preview
    for rt in sorted(df["race_time"].unique()):
        rdf = df[df["race_time"] == rt].sort_values("odds", na_position="last")
        print(f"\n{rt} Cheltenham ({len(rdf)} runners)")
        for _, r in rdf.iterrows():
            odds_str = f"{r['odds']:.1f}" if pd.notna(r["odds"]) else "N/A"
            print(
                f"  {r['horse_name']:<30} OR:{r['official_rating'] or '-':>4}  "
                f"Odds:{odds_str:>6}  {r['jockey_name']}"
            )


if __name__ == "__main__":
    main()
