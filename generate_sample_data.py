#!/usr/bin/env python3
"""
Generate Sample Race Data for Model Training.

Creates a realistic SQLite database of horse racing results for testing
the model training pipeline when no live data is available.

The generated data mimics the structure of horseracebase.com results
with realistic distributions for odds, field sizes, and outcomes.
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


TRACKS = [
    "Ascot", "Cheltenham", "Newmarket", "York", "Epsom", "Goodwood",
    "Doncaster", "Sandown", "Haydock", "Kempton", "Lingfield",
    "Wolverhampton", "Newcastle", "Chester", "Aintree", "Newbury",
]

GOING_DESCRIPTIONS = [
    "Good", "Good to Firm", "Good to Soft", "Soft", "Heavy",
    "Firm", "Standard", "Standard to Slow",
]

RACE_TYPES = ["Flat", "Hurdle", "Chase", "NH Flat"]

SURFACE_TYPES = ["Turf", "AW"]

HORSE_NAMES = [
    "Frankel", "Enable", "Stradivarius", "Battaash", "Cityscape",
    "Cracksman", "Crystal Ocean", "Elarqam", "Fairyland", "Ghaiyyath",
    "Hermosa", "Iridessa", "Japanese", "Kameko", "Lah Ti Dar",
    "Magical", "Nashwa", "Onesto", "Pinatubo", "Queen Power",
    "Roaring Lion", "Saxon Warrior", "Taghrooda", "Ulysses", "Veracious",
    "Waldgeist", "Xerxes", "Yeats Legend", "Zarkandar", "Alpine Star",
    "Blue Point", "Calyx", "Dartmouth", "Economics", "Fallen Angel",
    "Goliath", "Harbinger", "Inspiral", "Jubilee Walk", "Kingman",
    "Logician", "Mostahdaf", "Night Of Thunder", "Ouija Board", "Palace Pier",
    "Quest For Rest", "Riverman", "Songline", "Trawlerman", "Uniform",
    "Vauban", "Westover", "Doyen", "Emily Upjohn", "Doyen Star",
    "Rebel Fighter", "Telescope", "Noble Mission", "Highland Reel",
    "Aidan Star", "Baaeed", "Coroebus", "Desert Crown", "Eldar Eldarov",
    "Flightline", "Golden Pal", "Hukum", "Ilaraab", "Just Fine",
    "Kyprios", "Luxembourg", "Mojo Star", "Nashwa Star", "Opera Singer",
    "Passenger", "Queen Anne", "Rosallion", "Siskin", "Too Darn Hot",
    "Ucello", "Voyage Bubble", "William Buick", "Doyen Prince",
    "York Dancer", "Zustar", "Alpinista", "Big Rock", "City Of Troy",
    "Dancing Brave", "Elusive Pimpernel", "Famous Name", "Giant Steps",
    "Harpstar", "Isle Of Jura", "Doyen King", "Karl Burke Star",
    "Liberty Island", "Meditate", "Notable Speech", "Olympic Glory",
    "Pretty Gorgeous", "Doyen Queen", "Running Lion", "Snowfall",
]

JOCKEYS = [
    "R Moore", "W Buick", "T Marquand", "J Doyle", "O Murphy",
    "R Havlin", "D Tudhope", "J Spencer", "A Kirby", "B Curtis",
    "H Bentley", "S De Sousa", "L Dettori", "C Soumillon", "M Barzalona",
    "R Kingscote", "D Egan", "J Watson", "S Levey", "P Dobbs",
]

TRAINERS = [
    "J Gosden", "A O'Brien", "C Appleby", "W Haggas", "R Varian",
    "A Balding", "R Beckett", "Sir M Stoute", "K Ryan", "R Hannon",
    "M Johnston", "J Fanshawe", "S bin Suroor", "H Palmer", "E Walker",
    "T Clover", "C Cox", "D O'Meara", "T Dascombe", "G Scott",
]

STALLIONS = [
    "Frankel", "Dubawi", "Galileo", "Sea The Stars", "Kingman",
    "Dark Angel", "Kodiac", "Lope De Vega", "Night Of Thunder", "Shamardal",
]

SEXES = ["Gelding", "Colt", "Filly", "Mare", "Horse"]
HEADGEAR = ["", "", "", "", "b", "v", "t", "h", "p", "e/s"]


def generate_bfsp(official_rating: float, field_size: int) -> float:
    """Generate realistic BFSP from official rating and field size."""
    base_prob = 0.02 + (official_rating - 40) / 200.0
    base_prob = np.clip(base_prob, 0.02, 0.60)
    noise = np.random.normal(0, 0.05)
    prob = np.clip(base_prob + noise, 0.01, 0.80)
    bfsp = 1.0 / prob
    bfsp = max(1.01, bfsp + np.random.normal(0, bfsp * 0.1))
    return round(bfsp, 2)


def generate_race(
    race_date: datetime,
    race_time: str,
    track: str,
    available_horses: list[dict],
    race_idx: int,
) -> list[dict]:
    """Generate a single race with realistic results."""
    field_size = random.randint(5, 16)
    field_size = min(field_size, len(available_horses))
    runners = random.sample(available_horses, field_size)

    race_class = random.choice([1, 2, 3, 4, 5, 6])
    distance_f = random.choice([5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0])
    going = random.choice(GOING_DESCRIPTIONS)
    race_type = random.choice(RACE_TYPES)
    surface = "AW" if track in ["Kempton", "Lingfield", "Wolverhampton", "Newcastle"] else "Turf"
    prize_money = random.choice([3000, 5000, 7500, 10000, 15000, 25000, 50000, 100000])
    race_name = f"Class {race_class} {race_type}"

    # Assign official ratings
    base_or = 50 + race_class * 15 + random.randint(-10, 10)
    for r in runners:
        r["official_rating"] = max(30, base_or + random.randint(-15, 15))

    # Generate BFSP for each runner
    for r in runners:
        r["bfsp"] = generate_bfsp(r["official_rating"], field_size)

    # Determine finishing order — higher OR + some randomness
    scores = []
    for r in runners:
        ability = r["official_rating"] + np.random.normal(0, 12)
        scores.append(ability)

    order = np.argsort(scores)[::-1]

    median_or = np.median([r["official_rating"] for r in runners])
    max_or = max(r["official_rating"] for r in runners)

    rows = []
    for placing, idx in enumerate(order, 1):
        r = runners[idx]
        jockey = random.choice(JOCKEYS)
        trainer = random.choice(TRAINERS)

        dist_beaten = 0 if placing == 1 else round(random.uniform(0.5, 3.0) * (placing - 1), 2)

        comment_parts = []
        if placing <= 2:
            comment_parts = random.choice([
                ["made all", "ran on well"],
                ["tracked leader", "led close home"],
                ["chased leader", "stayed on"],
                ["prominent", "every chance"],
            ])
        elif placing <= 5:
            comment_parts = random.choice([
                ["held up in midfield", "headway 2f out"],
                ["in touch", "effort 2f out"],
                ["chased leaders", "weakened final furlong"],
            ])
        else:
            comment_parts = random.choice([
                ["towards rear", "never dangerous"],
                ["held up behind", "no impression"],
                ["always rear", "tailed off"],
            ])

        row = {
            "race_date": race_date.strftime("%Y-%m-%d"),
            "race_time": race_time,
            "track": track,
            "race_name": race_name,
            "race_class": race_class,
            "major": 0,
            "race_distance": f"{distance_f}f",
            "dist_furlongs": distance_f,
            "prize_money": prize_money,
            "going_description": going,
            "race_type": race_type,
            "surface_type": surface,
            "number_of_runners": field_size,
            "horse_name": r["name"],
            "stall": random.randint(1, field_size),
            "trainer": trainer,
            "horse_age": random.randint(2, 8),
            "jockey_name": jockey,
            "jockeys_claim": random.choice([0, 0, 0, 0, 3, 5, 7]),
            "pounds": random.randint(112, 140),
            "odds": round(r["bfsp"] - 1, 1),
            "fav": 1 if placing == 1 and r["bfsp"] < 4 else 0,
            "official_rating": r["official_rating"],
            "placing_numerical": placing,
            "distbt": dist_beaten,
            "comptime_numeric": round(60 + distance_f * 12 + random.uniform(-5, 5), 2),
            "total_dst_bt": dist_beaten,
            "bfsp": r["bfsp"],
            "stallion": random.choice(STALLIONS),
            "dam": f"Dam_{random.randint(1, 500)}",
            "dam_stallion": random.choice(STALLIONS),
            "horse_sex": random.choice(SEXES),
            "median_or": median_or,
            "max_or_in_race": max_or,
            "comment": ", ".join(comment_parts),
            "headgear": random.choice(HEADGEAR),
            "days_since_lr": random.randint(7, 120),
            "career_runs": random.randint(0, 50),
            "rail_move": "",
        }
        rows.append(row)

    return rows


def generate_sample_database(db_path: str, n_days: int = 730):
    """Generate a full sample database spanning n_days of racing.

    Args:
        db_path: Path to create the SQLite database.
        n_days: Number of days of racing to generate (default 2 years).
    """
    print(f"Generating {n_days} days of sample racing data...")

    # Create horse pool with persistent identities
    horses = [{"name": name} for name in HORSE_NAMES]

    all_rows = []
    start_date = datetime(2024, 1, 1)

    for day_offset in range(n_days):
        race_date = start_date + timedelta(days=day_offset)

        # Skip some days (no racing)
        if random.random() < 0.15:
            continue

        # 1-3 meetings per day
        n_meetings = random.randint(1, 3)
        tracks_today = random.sample(TRACKS, min(n_meetings, len(TRACKS)))

        for track in tracks_today:
            # 4-8 races per meeting
            n_races = random.randint(4, 8)
            for race_num in range(n_races):
                hour = 13 + race_num
                minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
                race_time = f"{hour:02d}:{minute:02d}"

                rows = generate_race(
                    race_date, race_time, track, horses, race_num
                )
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"Generated {len(df):,} rows across "
          f"{df['race_date'].nunique()} race days")

    # Write to SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS race_results")
    cursor.execute("""
        CREATE TABLE race_results (
            race_date TEXT,
            race_time TEXT,
            track TEXT,
            race_name TEXT,
            race_class INTEGER,
            major INTEGER,
            race_distance TEXT,
            dist_furlongs REAL,
            prize_money REAL,
            going_description TEXT,
            race_type TEXT,
            surface_type TEXT,
            number_of_runners INTEGER,
            horse_name TEXT,
            stall INTEGER,
            trainer TEXT,
            horse_age INTEGER,
            jockey_name TEXT,
            jockeys_claim INTEGER,
            pounds INTEGER,
            odds REAL,
            fav INTEGER,
            official_rating INTEGER,
            placing_numerical INTEGER,
            distbt REAL,
            comptime_numeric REAL,
            total_dst_bt REAL,
            bfsp REAL,
            stallion TEXT,
            dam TEXT,
            dam_stallion TEXT,
            horse_sex TEXT,
            median_or REAL,
            max_or_in_race REAL,
            comment TEXT,
            headgear TEXT,
            days_since_lr INTEGER,
            career_runs INTEGER,
            rail_move TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    df.to_sql("race_results", conn, if_exists="append", index=False)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_race_date ON race_results(race_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_track ON race_results(track)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_horse ON race_results(horse_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trainer ON race_results(trainer)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jockey ON race_results(jockey_name)")

    conn.commit()
    conn.close()

    print(f"Database saved to {db_path}")
    print(f"  Date range: {df['race_date'].min()} to {df['race_date'].max()}")
    print(f"  Unique horses: {df['horse_name'].nunique()}")
    print(f"  Unique tracks: {df['track'].nunique()}")

    return db_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample racing data")
    parser.add_argument("--db", default="horse_racing.db", help="Output database path")
    parser.add_argument("--days", type=int, default=730, help="Days of data to generate")
    args = parser.parse_args()

    generate_sample_database(args.db, args.days)
