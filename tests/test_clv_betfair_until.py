"""scripts/clv_betfair.py --until: no price on or after the date is used (the locked holdout)."""
import sqlite3

import pandas as pd

from scripts.clv_betfair import load


def test_until_drops_every_price_on_or_after_the_date(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    rr = pd.DataFrame({"id": [1, 2, 3], "race_date": ["2026-03-30", "2026-03-31", "2026-04-01"],
                       "track": "york", "race_time": "2.30", "horse_name": ["a", "b", "c"]})
    rr.to_sql("race_results", conn, index=False)
    pd.DataFrame({"race_results_id": [1, 2, 3], "market_type": "win", "morningwap": [5.0, 6.0, 7.0],
                  "morning_vol": 500.0, "bsp": [4.0, 5.0, 6.0]}).to_sql("betfair_prices", conn, index=False)
    conn.close()
    preds = rr.drop(columns="id").assign(predicted_bfsp=3.0, won=False)
    path = tmp_path / "p.csv"
    preds.to_csv(path, index=False)
    every = load(str(path), str(db), None)
    capped = load(str(path), str(db), None, until="2026-04-01")
    assert sorted(every["race_date"]) == ["2026-03-30", "2026-03-31", "2026-04-01"]
    assert sorted(capped["race_date"]) == ["2026-03-30", "2026-03-31"]
