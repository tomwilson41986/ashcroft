"""The WIN book against the PLACE book (research/greyhound/cross_market.py) on a synthetic day: the markets paired,
the dislocation's sign, and the walk-forward score finding the move where the PLACE book carries it."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from research.greyhound import cross_market as cm


DEPTH = {c: np.nan for c in ("back2", "back2_size", "back3", "back3_size", "lay2", "lay2_size", "lay3", "lay3_size")}


def synthetic_day(day: date, n_races: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed + day.toordinal())
    books, markets = [], []
    for r in range(n_races):
        off = pd.Timestamp(f"{day} 14:00:00", tz="UTC") + pd.Timedelta(minutes=10 * r)
        wid, pid = f"1.{day.toordinal()}{r:03d}", f"1.{day.toordinal()}{r:03d}9"
        p = rng.dirichlet(np.ones(6) * 3)                       # the true chances
        noise = np.exp(rng.normal(0, 0.25, 6))                   # the WIN mid's error at 3 minutes out
        q_mid = p * noise / (p * noise).sum()
        bsp = 1.0 / p * 1.05                                     # the close knows the truth
        hv = cm.harville_place(p, 2)                             # the PLACE book knows the truth too
        won = rng.choice(6, p=p)
        for mtype, mid in (("WIN", wid), ("PLACE", pid)):
            for i in range(6):
                markets.append({"market_id": mid, "market_type": mtype, "event_id": f"e{r}",
                                "market_start_utc": off.isoformat(), "venue": "Romford", "country": "GB",
                                "selection_id": 100 + i, "runner_name": f"{i + 1}. Dog {i}"})
        for mins in (5.0, 3.0, 1.0):
            t = off - pd.Timedelta(minutes=mins)
            for i in range(6):
                for mtype, mid, price in (("WIN", wid, 1.0 / q_mid[i]), ("PLACE", pid, 1.0 / hv[i])):
                    books.append({"polled_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "source": "recorder",
                                  "market_id": mid, "status": "OPEN", "inplay": 0, "selection_id": 100 + i,
                                  "runner_status": "ACTIVE", "last_traded": round(price, 2), "number_of_active_runners": 6,
                                  "last_match_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  "back1": round(price * 0.98, 2), "back1_size": 50.0, "lay1": round(price * 1.02, 2),
                                  "lay1_size": 50.0, "number_of_winners": 1 if mtype == "WIN" else 2,
                                  "sp_actual": np.nan, "runner_total_matched": 100.0, "market_total_matched": 600.0,
                                  **DEPTH})
        for i in range(6):
            books.append({"polled_utc": (off + pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "source": "final", "market_id": wid, "status": "CLOSED", "inplay": 0,
                          "selection_id": 100 + i, "runner_status": "WINNER" if i == won else "LOSER", "number_of_active_runners": 6,
                          "last_traded": np.nan, "last_match_utc": None, "back1": np.nan, "back1_size": np.nan,
                          "lay1": np.nan, "lay1_size": np.nan, "number_of_winners": 1, "sp_actual": round(bsp[i], 2),
                          "runner_total_matched": 100.0, "market_total_matched": 600.0, **DEPTH})
    return pd.DataFrame(books), pd.DataFrame(markets)


def test_the_markets_pair_and_the_dislocation_reads_the_place_book():
    books, markets = synthetic_day(date(2026, 10, 10))
    pairs = cm.pair_markets(markets)
    assert len(pairs) == 12 and (pairs.place_market_id.str.endswith("9")).all()
    x = cm.features(cm.book_at_marks(books, markets, date(2026, 10, 10)))
    assert set(x["mark"]) >= {"off_5", "off_3", "off_1"}
    assert np.isclose(x.groupby("grp").q_place_mkt.sum().iloc[0], 2.0)              # two places paid
    # the PLACE book knows the truth: a WIN mid too short reads as a negative dislocation and a drift to the close
    assert np.corrcoef(x.disloc, x.y)[0, 1] < -0.5


def test_the_walk_forward_score_finds_the_move():
    frames = []
    for i in range(10):
        d = date(2026, 10, 6 + i)
        books, markets = synthetic_day(d, seed=i)
        frames.append(cm.features(cm.book_at_marks(books, markets, d)))
    panel = pd.concat(frames, ignore_index=True)
    rows = {r["mark"]: r for r in cm.score(panel, n_boot=100)}
    r = rows["off_3"]
    assert r["days"] == 3 and r["rmse_disloc"] < r["rmse_mid"] and r["rmse_both"] < r["rmse_last_trade"]
    assert r["both_vs_last_90%"][1] < 0                                             # helps on every resample
    assert r["result_logloss_with_disloc"] <= r["result_logloss_market"] + 0.01
    assert "| off_3 |" in cm.report_md({"first_day": "a", "last_day": "b", "scores": list(rows.values())})
