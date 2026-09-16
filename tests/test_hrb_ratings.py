"""Tests for HRB Ratings Machine parsing/matching, features and the rating evaluation."""

import json
import sqlite3

import numpy as np
import pandas as pd

import hrb_ratings as hr
from model.hrb_features import add_hrb_features, load_hrb_frame
from model.rating_eval import evaluate_rating_sets

CSV = "Date,Time,Track,Horse,Rating,Rank\n10/02/2026,2.30,Kempton,Just A Gambler (IRE),84.5,1\n10/02/2026,2.30,Kempton,Tewkesbury,80,2\n10/02/2026,2.30,Kempton,Shafi,71.2,3\n"


def test_parse_csv_standardises_columns():
    df = hr.parse_csv(CSV, 269)
    assert list(df["horse_norm"]) == ["justagambler", "tewkesbury", "shafi"]
    assert df["race_date"].iloc[0] == "2026-02-10" and df["race_time_24"].iloc[0] == "14:30"
    assert df["rating"].iloc[0] == 84.5 and df["set_name"].iloc[0] == "HRB Standard"


# The real Ratings Machine download, verified against horseracebase.com on 2026-09-15.
# Underscored headers, ISO dates, `total` as the rating, and `placing`/`placing_num`
# (the FINISHING position) present -- which must never be read as a rating or a rank.
REAL_CSV = (
    "race_date,time,track,placing,horse_name,last,last2,leveller,speed,jockey,trainer,stallion,today,total,"
    "odds_numerical,racetype,jockey_name,trainer_name,runners,placing_num,HorseAge,Distance,AgeRestrictions,"
    "WinPrizeMoney,UKClass,MajorType,\n"
    '"2026-02-05","12.40.","Doncaster","PU","You Did (IRE)","45.8150","37.8828","3.3903","0.0000","0.0000",'
    '"0.8530","0.0000","3.3000","91.2411","3.50","Handicap Novices Hurdle","Maggs, Charlie","Ellison, B","3",'
    '"0","6","3m.5f ","4yo+","3248","5",,\n'
    '"2026-02-05","12.40.","Doncaster","2nd","Call Your Bluff","62.7714","19.8683","3.9408","0.0000","0.0000",'
    '"1.3739","0.0000","1.2858","89.2402","3.00","Handicap Novices Hurdle","Johnstone-baker, Cameron",'
    '"Lavelle, Miss E C","3","2","6","3m.5f ","4yo+","3248","5",,\n'
    '"2026-02-05","12.40.","Doncaster","1st","Shafi","41.3500","23.5130","4.0894","0.0000","0.0000","0.1515",'
    '"0.0000","0.7000","69.8039","7.00","Handicap Novices Hurdle","Sansom, Daniel","Mullins, J W","3","1","6",'
    '"3m.5f ","4yo+","3248","5",,\n'
)


def test_parse_csv_real_download_layout():
    df = hr.parse_csv(REAL_CSV, 269)
    assert list(df["horse_norm"]) == ["youdid", "callyourbluff", "shafi"]
    assert df["race_date"].iloc[0] == "2026-02-05" and df["race_time_24"].iloc[0] == "12:40"
    # `total` is the rating, not any of the component or context columns
    assert df["rating"].tolist() == [91.2411, 89.2402, 69.8039]
    # rank is derived from the rating within the race, NOT from the finishing position:
    # Shafi won (placing 1st) but is bottom-rated, so it must rank 3.
    assert df["rating_rank"].tolist() == [1.0, 2.0, 3.0]
    # post-race fields stay in extras and are never promoted to rating/rank
    extras = json.loads(df["extras"].iloc[0])
    assert extras["placing"] == "PU" and extras["odds_numerical"] == "3.50"
    assert hr.POST_RACE_COLS <= set(extras)


def test_rate_limit_notice_detected():
    html = "<html><body>Data downloads temporarily unavailable ... Download access will become available again: Tuesday 15th September 2026 at 15:26. The remainder</body></html>"
    assert "15:26" in hr._rate_limit_notice(html)
    assert hr._rate_limit_notice("a,b\n1,2\n") is None


def test_load_match_features(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE race_results (id INTEGER PRIMARY KEY, race_date TEXT, race_time TEXT, track TEXT, horse_name TEXT)")
    conn.executemany("INSERT INTO race_results VALUES (?,?,?,?,?)", [
        (1, "2026-02-10", "2.30", "Kempton", "Just A Gambler (IRE)"), (2, "2026-02-10", "2.30", "Kempton", "Tewkesbury"),
        (3, "2026-02-10", "2.30", "Kempton", "Shafi"), (4, "2026-02-10", "3.00", "Kempton", "Other")])
    conn.commit(); conn.close()
    p = tmp_path / "ratings_269_2026-02-10.csv"; p.write_text(CSV)
    assert hr.load_files([p], str(db))["rows"] == 3
    assert hr.match_to_results(str(db))["matched"] == 3
    rr = pd.read_sql_query("SELECT * FROM race_results", sqlite3.connect(db))
    d, feats = add_hrb_features(rr, frame=load_hrb_frame(str(db)))
    assert "hrb_hrb_standard" in feats and d["hrb_hrb_standard_rank"].iloc[0] == 1
    assert np.isnan(d.loc[d["horse_name"] == "Other", "hrb_hrb_standard"].iloc[0])


def test_evaluate_rating_sets_detects_informative_rating():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(600):
        n = int(rng.integers(6, 12)); strength = rng.normal(0, 1, n)
        p_true = np.exp(strength) / np.exp(strength).sum()
        finish = np.argsort(-(strength + rng.gumbel(0, 1, n)))          # Plackett-Luce order
        pos = np.empty(n, dtype=int); pos[finish] = np.arange(1, n + 1); winner = finish[0]
        for i in range(n):
            rows.append({"race_date": f"2026-01-{1 + r % 28:02d}", "raceid": f"r{r}", "horse_name": f"h{i}",
                         "p_model": p_true[i] * rng.uniform(0.6, 1.4), "p_market": p_true[i] * rng.uniform(0.8, 1.2),
                         "won": float(i == winner), "placing_numerical": int(pos[i]),
                         "good": strength[i] + rng.normal(0, 0.5), "noise": rng.normal(0, 1)})
    d = pd.DataFrame(rows)
    for c in ("p_model", "p_market"):
        d[c] = d[c] / d.groupby("raceid")[c].transform("sum")
    ev = evaluate_rating_sets(d, ["good", "noise"]).set_index("rating")
    assert ev.loc["good", "concordance"] > 0.6 and ev.loc["good", "p_value"] < 1e-6
    assert ev.loc["noise", "p_value"] > 1e-3
    assert ev.loc["good", "stack_gain_logloss"] > ev.loc["noise", "stack_gain_logloss"]
