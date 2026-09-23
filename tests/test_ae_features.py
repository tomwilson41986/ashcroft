"""A/E against the market: lagged by day, shrunk, and blind to today's result."""

import numpy as np
import pandas as pd

from model.ae_features import add_ae_features


def _history():
    rows = []
    # trainer T wins every race at 5.0 (market says 20%) on days 1..3; day 4 is the race being priced
    for day in range(1, 5):
        for i, (trainer, bsp) in enumerate([("T", 5.0), ("U", 2.0), ("V", 4.0), ("W", 10.0)]):
            rows.append({"race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day), "race_time": "2.00",
                         "track": "Ascot", "horse_name": f"h{day}{i}", "trainer": trainer, "jockey_name": f"j{i}",
                         "stallion": f"s{i}", "bfsp": bsp, "placing_numerical": 1 if trainer == "T" else i + 1,
                         "dist_furlongs": 8.0, "stall": i + 1, "number_of_runners": 4})
    return pd.DataFrame(rows)


def test_a_trainer_the_market_underrates_shows_up_on_later_days_only():
    out, names = add_ae_features(_history(), shrink_n=1.0)
    t = out[out["trainer"] == "T"].sort_values("race_date")
    assert np.isnan(t["ae_trainer"].iloc[0])                 # no earlier day
    assert (t["ae_trainer"].iloc[1:] > 0).all()              # beat the price every day since
    u = out[out["trainer"] == "U"].sort_values("race_date")
    assert (u["ae_trainer"].iloc[1:] < 0).all()              # the 2.0 shot that never wins


def test_todays_result_never_enters_todays_features():
    d = _history()
    out1, names = add_ae_features(d)
    d2 = d.copy()
    last = d2["race_date"] == d2["race_date"].max()
    d2.loc[last, "placing_numerical"] = [4, 1, 2, 3]         # a different winner today
    out2, _ = add_ae_features(d2)
    for c in names:
        a = out1.loc[out1["race_date"] == d["race_date"].max(), c].to_numpy()
        b = out2.loc[out2["race_date"] == d["race_date"].max(), c].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), c


def test_shrinkage_pulls_thin_records_toward_zero():
    d = _history()
    loose, _ = add_ae_features(d, shrink_n=0.1)
    tight, _ = add_ae_features(d, shrink_n=100.0)
    t_last = lambda o: o[(o["trainer"] == "T")].sort_values("race_date")["ae_trainer"].iloc[-1]
    assert 0 < t_last(tight) < t_last(loose)
