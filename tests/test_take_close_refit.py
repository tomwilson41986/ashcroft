"""research/greyhound/take_close_refit.py: the race-traded book (the poll at which every runner has traded, the book
the fixed live trader decides on) and the walk-forward scoring of the current and refitted forecasts. Constructed
books and panels for the unit tests only."""

import numpy as np
import pandas as pd

from research.greyhound import take_close_refit as rf
from tests.test_greyhound_limit_study import synthetic_panel


def test_race_traded_takes_every_runner_at_the_last_first_trade():
    t = pd.Timestamp("2026-11-10 12:00", tz="UTC")
    rows = []
    for k, (b1, b2, lt2) in enumerate([(3.0, 5.0, np.nan), (3.1, 5.5, np.nan), (3.2, 6.0, 6.0), (3.3, 6.5, 6.4)]):
        for sid, back, lt in ((1, b1, 3.0), (2, b2, lt2)):
            rows.append({"market_id": "1.1", "selection_id": sid, "polled": t + pd.Timedelta(seconds=60 * k),
                         "minutes_to_off": 10 - k, "back1": back, "lay1": back * 1.05, "last_traded": lt,
                         **{c: np.nan for c in ("back1_size", "back2", "back2_size", "back3", "back3_size",
                                                "lay1_size", "lay2", "lay2_size", "lay3", "lay3_size",
                                                "runner_total_matched", "market_total_matched")}})
    e = rf.race_traded(pd.DataFrame(rows)).set_index("selection_id")
    # runner 2 first trades at poll 2: both runners are read at poll 2, not runner 1 at its own first trade (poll 0)
    assert e.loc[1, "back1"] == 3.2 and e.loc[2, "back1"] == 6.0
    assert (e.mark == "race_traded").all() and e.loc[1, "back1_prev"] == 3.0     # 90 s or more before: poll 0


def test_score_reads_both_arms_walk_forward():
    p = synthetic_panel(n_days=12, races=20)
    p = p.assign(back2=np.nan, back2_size=np.nan, back3=np.nan, back3_size=np.nan, back1_size=50.0)
    rec = p[p["mark"] == "first_traded"]
    live = p[p["mark"] == p["mark"].iloc[0]].assign(mark="race_traded")
    out = rf.score(rec, live)
    assert len(out["scored_days"]) == 12 - rf.MIN_TRAIN_DAYS
    for arm in ("current", "refit"):
        a = out["arms"][arm]
        assert a["races"] > 0 and "xent" in a and "bias_all_%" in a
    assert set(out["reading"]) == {"clv_interval_above_zero", "taken_unbiased", "net_positive", "dsr_passed",
                                   "min_backs_met", "pass"}
