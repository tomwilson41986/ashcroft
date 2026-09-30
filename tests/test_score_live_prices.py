"""scripts/score_live_prices.py: the owner's staking on live Betfair prices. Mechanics only, on hand-built cards:
names and start times matched to the card, the snapshot formats read, whole races only, the bar, the volume rule,
the stake and the least price that still clears the bar."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import score_live_prices as slp  # noqa: E402

MODEL = slp.load_model(str(ROOT / "data" / "models" / "closing_model.json"))


def test_names_match_without_the_country():
    assert slp.norm_name("Sod Hall Lane (IRE)") == slp.norm_name("Sod Hall Lane") == "sodhalllane"
    assert slp.norm_name("O'Reilly's Star (GB)") == "oreillysstar"


def test_start_times_read_as_the_card_writes_them():
    assert slp.card_time("2026-09-30T12:57:00.000Z") == "1:57"      # summer time: UTC + 1
    assert slp.card_time("2026-01-15T13:30:00Z") == "1:30"          # winter: UTC
    assert slp.card_time("2026-09-30T11:05:00Z") == "12:05"


def test_the_snapshot_drops_removed_runners_and_closed_markets(tmp_path):
    rows = [("2026-09-30T12:57:00Z", "Catterick", "A (IRE)", "ACTIVE", "OPEN"),
            ("2026-09-30T12:57:00Z", "Catterick", "B", "REMOVED", "OPEN"),
            ("2026-09-30T13:13:00Z", "Musselburgh", "C", "ACTIVE", "SUSPENDED")]
    f = tmp_path / "live.csv"
    pd.DataFrame([{"market_start_time": t, "venue": v, "runner_name": n, "runner_status": rs, "market_status": ms,
                   "best_back_price": 5.0, "best_back_size": 20.0, "best_lay_price": 5.2, "best_lay_size": 10.0,
                   "runner_matched": 300.0, "total_matched": 2000.0} for t, v, n, rs, ms in rows]).to_csv(f, index=False)
    live = slp.read_live(str(f))
    assert list(live["horse"]) == ["A (IRE)"]
    assert live.loc[0, "race_time"] == "1:57" and live.loc[0, "runner_matched"] == 300.0


def _card(backs, ours, vols, race_time="1:57", track="Catterick"):
    names = [f"H{i}" for i in range(len(backs))]
    pred = pd.DataFrame({"race_time": race_time, "track": track, "horse_name": names, "predicted_bfsp": ours})
    live = pd.DataFrame({"race_time": race_time, "track": track, "horse": names, "back": backs,
                         "back_avail": 50.0, "lay": np.asarray(backs, float) * 1.04, "lay_avail": 50.0,
                         "runner_matched": vols, "race_matched": np.nan})
    return pred, live


def test_the_bar_the_volume_rule_the_stake_and_the_least_price():
    # H0: the market's 10.0 against our 4.0; H3 the same, but only GBP50 matched
    pred, live = _card([10.0, 3.0, 4.0, 10.0], [4.0, 3.5, 4.5, 4.0], [500.0, 900.0, 800.0, 50.0])
    d = slp.score(pred, live, MODEL, bar=0.03, n_draws=4000, seed=1)
    h0, h3 = d.set_index("horse").loc["H0"], d.set_index("horse").loc["H3"]
    assert h0["ev"] > 0.03 and h0["back_it"]
    assert h3["ev"] > 0.03 and not h3["back_it"]                   # under the GBP100 matched: no bet
    backs = d[d["back_it"]]
    assert np.allclose(backs["stake"], 250.0 / (backs["back"] - 1.0))
    # at the least price the expected CLV is the bar itself
    assert np.allclose((1 + backs["ev"]) * backs["min_back"] / backs["back"] - 1.0, 0.03)
    assert (d.loc[~d["back_it"], "stake"] == 0).all() and d.loc[~d["back_it"], "min_back"].isna().all()


def test_only_whole_races_are_scored():
    pred, live = _card([10.0, 3.0, 4.0], [4.0, 3.5, 4.5], [500.0, 900.0, 800.0])
    pred2, live2 = _card([6.0, 2.5, 5.0], [5.0, 3.0, 6.0], [400.0, 400.0, 400.0], race_time="2:13",
                         track="Musselburgh")
    pred2 = pred2.iloc[:2]                                            # one runner Betfair has open we did not price
    d = slp.score(pd.concat([pred, pred2]), pd.concat([live, live2]), MODEL, n_draws=2000)
    assert set(d["race"]) == {"Catterick 1:57"}


def test_race_money_is_shared_by_price_when_runners_have_none():
    pred, live = _card([2.0, 4.0, 4.0], [2.0, 4.0, 4.0], [np.nan, np.nan, np.nan])
    live["race_matched"] = 1000.0
    d = slp.score(pred, live, MODEL, n_draws=2000).set_index("horse")
    assert d["vol_estimated"].all()
    assert d.loc["H0", "morning_vol"] == pytest.approx(2 * d.loc["H1", "morning_vol"], rel=0.05)
    assert d["morning_vol"].sum() == pytest.approx(1000.0)


def test_a_race_without_matched_money_is_read_by_the_model_fitted_without_volume():
    # the delayed key's feed: no matched money on any runner, and no race total either
    novol = slp.load_model(str(ROOT / "data" / "models" / "closing_model_novol.json"))
    pred, live = _card([10.0, 3.0, 4.0], [4.0, 3.5, 4.5], [np.nan, np.nan, np.nan])
    d = slp.score(pred, live, MODEL, n_draws=4000, novol=novol).set_index("horse")
    assert d["no_volume"].all()
    assert d.loc["H0", "back_it"]                                     # not held to the GBP100 matched
    held = slp.score(pred, live, MODEL, n_draws=4000).set_index("horse")
    assert not held["no_volume"].any() and not held["back_it"].any()  # without it: no matched money, no bet


def test_card_times_order_the_afternoon_after_noon():
    assert sorted(["2:05", "12:30", "1:57", "5:25"], key=slp.card_minutes) == ["12:30", "1:57", "2:05", "5:25"]


def test_the_days_list_marks_what_was_sent_and_keeps_the_days_limit():
    backs = pd.DataFrame({"race": ["A 1:57", "A 1:57", "B 2:13", "C 3:00"], "horse_name": ["x", "y", "z", "w"],
                          "stake": [100.0, 50.0, 200.0, 80.0], "ev": [0.05, 0.10, 0.20, 0.04]})
    seen = pd.DataFrame({"race": ["A 1:57"], "horse_name": ["x"], "stake": [100.0]})
    b = slp.day_list(backs, seen, max_day=400.0).set_index("horse_name")
    assert not b.loc["x", "new"] and b.loc[["y", "z", "w"], "new"].all()
    # GBP300 of room after the GBP100 sent: z (best, 200) then y (50) fit; w (80) would take the day to 430
    assert not b.loc["z", "over_limit"] and not b.loc["y", "over_limit"] and b.loc["w", "over_limit"]


def test_a_bet_is_never_more_than_the_limit():
    pred, live = _card([1.3, 8.0, 12.0], [1.15, 10.0, 15.0], [5000.0, 900.0, 800.0])
    d = slp.score(pred, live, MODEL, n_draws=4000, max_stake=300.0)
    assert (d["stake"] <= 300.0).all()
    short = d.set_index("horse").loc["H0"]
    if short["back_it"]:
        assert short["stake"] == pytest.approx(300.0)           # to win GBP250 at 1.3 would be GBP833
