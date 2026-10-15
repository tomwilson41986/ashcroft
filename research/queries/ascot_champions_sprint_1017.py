"""The owner's ask (15 Oct): the served model's predicted BFSP for one race, Ascot 2:05 on 17 Oct 2026, the Qatar
Racing British Champions Sprint Stakes (Group 1, 6f, Good), ignoring the market. HRB has no card for 17 Oct yet (its
CSV export returns nothing and its HTML page shows only today), so the card is the Racing Post's declarations of 15 Oct,
written here in HRB's card fields, each horse, trainer and jockey named as the database names them. The model, the
history and the feature build are the 06:00 run's (predict_bfsp_today.py, history from 2018-01-01, data/models).
Read-only: writes nothing to S3."""
import os
import re
import sqlite3
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, ".")
DAY = date(2026, 10, 17)
DB = "horse_racing.db"

# Racing Post, 15 Oct: (horse, age, weight st-lb, OR, draw, jockey, trainer). Samangan has no jockey yet.
CARD = [
    ("Almeraq", 4, "9-4", 123, 8, "Tom Marquand", "William Haggas"),
    ("Binhareer", 4, "9-4", 102, 14, "Cieren Fallon", "William Haggas"),
    ("Great Wish", 5, "9-4", 112, 9, "William Buick", "Bent Olsen"),
    ("James's Delight", 5, "9-4", 106, 16, "Dylan Browne McMonagle", "Clive Cox"),
    ("Jasour", 5, "9-4", 99, 11, "Hector Crouch", "Clive Cox"),
    ("Kind Of Blue", 5, "9-4", 111, 10, "Billy Loughnane", "James Fanshawe"),
    ("Marvelman", 4, "9-4", 116, 13, "Oisin Murphy", "Andrew Balding"),
    ("Mitbaahy", 7, "9-4", 106, 2, "Saffie Osborne", "Charles Hills"),
    ("Paborus", 5, "9-4", 113, 1, "Daniel Tudhope", "Edward Bethell"),
    ("Prince Of India", 4, "9-4", 111, 6, "Pat Dobbs", "Marco Botti"),
    ("Coppull", 3, "9-3", 115, 5, "Rossa Ryan", "Clive Cox"),
    ("Division", 3, "9-3", 117, 12, "James Doyle", "William Haggas"),
    ("Poatan", 3, "9-3", 104, 3, "Rowan Scott", "George Scott"),
    ("Power Blue", 3, "9-3", 110, 4, "David Egan", "Robson De Aguiar"),
    ("Samangan", 3, "9-3", 118, 15, None, "F-H Graffard"),
    ("Soul Love", 3, "9-0", 104, 7, "Pierre-Louis Jamin", "K R Burke"),
]


def norm(s):
    s = re.sub(r"\(.*?\)", "", str(s or "")).lower()
    return re.sub(r"[^a-z]", "", s)


def surname(s):
    return norm(str(s).replace("-", " ").split()[-1]) if s else ""


conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rr = pd.read_sql_query("""SELECT race_date, horse_name, trainer, jockey_name, official_rating, track, race_time
                          FROM race_results WHERE race_date >= '2023-01-01'""", conn)
rr["key"] = rr.horse_name.map(norm)
recent = rr[rr.race_date >= "2025-06-01"]
jockeys = recent.jockey_name.dropna().value_counts()
trainers = recent.trainer.dropna().value_counts()


def match_person(name, counts):
    """The database's spelling of a jockey or trainer: same surname and first initial, the most frequent recently."""
    if not name:
        return None
    sur, ini = surname(name), norm(name)[:1]
    cands = [n for n in counts.index if surname(n) == sur and norm(n)[:1] == ini]
    if not cands:
        cands = [n for n in counts.index if surname(n) == sur]
    return cands[0] if cands else name


rows = []
print("horse | database name | runs since 2023 | last run | last trainer | trainer used | jockey used")
for horse, age, wt, orat, draw, jock, trn in CARD:
    h = rr[rr.key == norm(horse)].sort_values("race_date")
    dbname = h.horse_name.iloc[-1] if len(h) else horse
    last = h.race_date.iloc[-1] if len(h) else None
    last_tr = h.trainer.iloc[-1] if len(h) else None
    tr = last_tr if last_tr and surname(last_tr) == surname(trn) else match_person(trn, trainers)
    jk = match_person(jock, jockeys) if jock else "Not Declared"
    st, lb = (int(x) for x in wt.split("-"))
    days = (DAY - date.fromisoformat(str(last)[:10])).days if last else None
    print(f"{horse} | {dbname} | {len(h)} | {last} | {last_tr} | {tr} | {jk}")
    rows.append({"race_date": DAY.isoformat(), "race_time": "2:05", "track": "Ascot", "going_description": "Good",
                 "race_class": "Class 1", "race_distance": "6f", "prize_money": "283,550",
                 "race_name": "Qatar Racing British Champions Sprint Stakes (Group 1)", "stall": draw,
                 "horse_name": dbname, "horse_age": age, "pounds": st * 14 + lb, "jockey_name": jk, "trainer": tr,
                 "official_rating": orat, "days_since_lr": days, "number_of_runners": len(CARD)})
card = pd.DataFrame(rows)
# the track as the database names Ascot
tracks = rr[rr.track.str.contains("ascot", case=False, na=False)].track.value_counts()
print("Ascot in the database:", tracks.to_dict())
card["track"] = tracks.index[0] if len(tracks) else "Ascot"

import predict_bfsp_today as pbt  # noqa: E402

pbt.fetch_racecard_from_hrb = lambda d: card.copy()
os.environ["HRB_USERNAME"] = os.environ.get("HRB_USERNAME") or "card-from-file"
os.makedirs("out", exist_ok=True)
sys.argv = ["predict_bfsp_today.py", "--date", DAY.isoformat(), "--start-date", "2018-01-01",
            "--model-dir", "data/models", "--output-csv", "out/ascot_1405.csv"]
pbt.main()

p = pd.read_csv("out/ascot_1405.csv")
cols = [c for c in ("horse_name", "stall", "official_rating", "jockey_name", "trainer", "predicted_bfsp",
                    "predicted_win_prob_norm", "model_rank") if c in p.columns]
print("\n== Ascot 2:05, 17 Oct: the model's predicted BFSP (no market input)")
print(p.sort_values("predicted_bfsp")[cols].to_string(index=False))
print("book of the predicted BFSPs (sum of 1/price):", round(float((1 / p.predicted_bfsp).sum()), 3))
