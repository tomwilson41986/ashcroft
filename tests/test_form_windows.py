"""The form-window block (model/form_windows.py).

Synthetic frames test mechanics only: each window on a hand-built career, each
measure on a hand-built race, and the lag rule. Nothing is trained or evaluated
on them.
"""
import numpy as np
import pandas as pd
import pytest

from model.form_windows import FORM_WINDOW_FEATURES, MEASURES, WINDOWS, add_form_windows, run_measures


def _runs(rows):
    base = dict(race_time="2.30", track="york", number_of_runners=11, dist_furlongs=8.0,
                official_rating=np.nan, total_dst_bt=np.nan, bfsp=np.nan)
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def _career(places):
    days = pd.date_range("2024-01-01", periods=len(places) + 1, freq="14D")
    return _runs([dict(horse_name="x", race_date=d, placing_numerical=p)
                  for d, p in zip(days, list(places) + [np.nan])])


def test_each_window_on_a_hand_built_career():
    # eleven runners, so the normalised position is (11 - place) / 10
    places = [1, 6, 11, 3, 2, 5]
    out, cols = add_form_windows(_career(places))
    assert cols == FORM_WINDOW_FEATURES and set(cols) <= set(out.columns)
    assert len(cols) == len(MEASURES) * len(WINDOWS) + 3
    v = [(11 - p) / 10 for p in places]
    today = out.iloc[-1]
    assert today["fw_nfp_l1"] == pytest.approx(v[-1])
    assert today["fw_nfp_m3"] == pytest.approx(np.mean(v[-3:]))
    assert today["fw_nfp_m5"] == pytest.approx(np.mean(v[-5:]))
    assert today["fw_nfp_w3"] == pytest.approx((3 * v[5] + 2 * v[4] + v[3]) / 6)
    assert today["fw_nfp_w5"] == pytest.approx((5 * v[5] + 4 * v[4] + 3 * v[3] + 2 * v[2] + v[1]) / 15)
    w10 = [10, 9, 8, 7, 6, 5]
    assert today["fw_nfp_w10"] == pytest.approx(np.dot(w10, v[::-1]) / sum(w10))
    assert today["fw_nfp_car"] == pytest.approx(np.mean(v))
    assert today["fw_win_car"] == pytest.approx(1 / 6) and today["fw_win_m5"] == 0
    assert today["fw_plc_m3"] == pytest.approx(2 / 3)
    # the debut has no history; the second run's windows are the debut alone
    first, second = out.iloc[0], out.iloc[1]
    assert first[cols].isna().all()
    assert second["fw_win_l1"] == 1 and second["fw_win_w10"] == 1 and second["fw_win_car"] == 1


def test_a_window_counts_runs_not_known_values():
    # the fifth run is a non-finisher: unknown, but it keeps its place in the window
    places = [1, 6, 11, 3, np.nan, 5]
    out, _ = add_form_windows(_career(places))
    v = [(11 - p) / 10 for p in places]
    today = out.iloc[-1]
    assert today["fw_nfp_m3"] == pytest.approx((v[5] + v[3]) / 2)
    assert today["fw_nfp_w3"] == pytest.approx((3 * v[5] + v[3]) / 4)
    assert today["fw_nfp_car"] == pytest.approx(np.nanmean(v))
    assert np.isnan(out.iloc[5]["fw_nfp_l1"])                   # the run after it: its last is unknown


def test_each_measure_on_a_hand_built_race():
    # one race of four at a mile (2 lbs a length), then each horse runs again
    race = [dict(horse_name="a", placing_numerical=1, total_dst_bt=np.nan, official_rating=80, bfsp=2.0),
            dict(horse_name="b", placing_numerical=2, total_dst_bt="1", official_rating=75, bfsp=4.0),
            dict(horse_name="c", placing_numerical=3, total_dst_bt="3 1/2", official_rating=np.nan, bfsp=5.0),
            dict(horse_name="d", placing_numerical=4, total_dst_bt="20", official_rating=70, bfsp=20.0)]
    rows = [dict(r, race_date="2024-05-01", number_of_runners=4) for r in race]
    rows += [dict(horse_name=h, race_date="2024-05-20", race_time="3.00", number_of_runners=4,
                  placing_numerical=np.nan, official_rating=78) for h in "abcd"]
    out, _ = add_form_windows(_runs(rows))
    last = out[out["race_date"] == "2024-05-20"].set_index("horse_name")

    p = np.array([1 / 2, 1 / 4, 1 / 5, 1 / 20])
    pn = p / p.sum()
    assert last.loc["a", "fw_mkt_l1"] == pytest.approx(np.log(4 * pn[0]))
    assert last.loc["a", "fw_ae_l1"] == pytest.approx(1 - pn[0])
    assert last.loc["d", "fw_ae_l1"] == pytest.approx(-pn[3])
    # the order matched the market's: no residual for anyone
    assert last["fw_nres_l1"].abs().max() == pytest.approx(0.0)

    lbs = np.array([0.0, 2.0, 7.0, 20.0])                  # 20 lengths is 40 lbs, capped at 20
    assert list(last["fw_lbs_l1"]) == pytest.approx(list(lbs))
    assert list(last["fw_lbsc_l1"]) == pytest.approx(list(lbs.mean() - lbs))
    # the winner: its rating plus the winning margin (1 length, 2 lbs); c had no rating
    # and runs off the race's median (75)
    assert last.loc["a", "fw_perf_l1"] == pytest.approx(82.0)
    assert last.loc["b", "fw_perf_l1"] == pytest.approx(73.0)
    assert last.loc["c", "fw_perf_l1"] == pytest.approx(68.0)
    assert last.loc["d", "fw_perf_l1"] == pytest.approx(50.0)
    assert last.loc["a", "fw_perf_l1_vs_or"] == pytest.approx(82.0 - 78.0)
    assert last.loc["a", "fw_wax_l1"] == pytest.approx(0.75)


def test_a_dead_heat_credits_no_margin_and_a_non_finisher_is_unknown():
    rows = [dict(horse_name="a", placing_numerical=1, official_rating=80),
            dict(horse_name="b", placing_numerical=1, official_rating=80),
            dict(horse_name="c", placing_numerical=3, total_dst_bt="2", official_rating=80),
            dict(horse_name="d", placing_numerical=np.nan, official_rating=80, bfsp=9.0)]
    df = _runs([dict(r, race_date="2024-05-01", number_of_runners=4) for r in rows])
    m = run_measures(df)
    assert list(m["perf"][:3]) == pytest.approx([80.0, 80.0, 76.0])
    for k in ("win", "nfp", "lbs", "perf", "ae"):
        assert np.isnan(m[k][3]), k


def _history(days=90, seed=5):
    """A pool of horses running on random days, with results, prices, margins and ratings."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2023-01-01", periods=days, freq="D"):
        for r, track in enumerate(("york", "kempton")):
            horses = rng.choice(50, 8, replace=False)
            place = rng.permutation(8) + 1
            for i, h in enumerate(horses):
                rows.append(dict(race_date=d, race_time=f"{2 + r}.30", track=track, horse_name=f"h{h}",
                                 number_of_runners=8, dist_furlongs=float(5 + 3 * r),
                                 placing_numerical=int(place[i]),
                                 total_dst_bt=np.nan if place[i] == 1 else f"{rng.uniform(0.1, 9):.1f}",
                                 official_rating=float(60 + h), bfsp=float(rng.uniform(2, 30)),
                                 RSR=float(rng.normal())))
    return pd.DataFrame(rows)


KEY = ["race_date", "track", "horse_name"]
RESULT = ["placing_numerical", "total_dst_bt", "bfsp", "RSR"]


@pytest.mark.parametrize("change", ["scramble", "blank"])
def test_a_days_own_results_move_none_of_its_windows(change):
    df = _history()
    day = pd.Timestamp("2023-02-20")
    base, cols = add_form_windows(df.copy())
    alt = df.copy()
    on = alt["race_date"] == day
    rng = np.random.default_rng(2)
    if change == "scramble":
        for _, idx in alt[on].groupby("track").groups.items():
            alt.loc[idx, "placing_numerical"] = rng.permutation(len(idx)) + 1
            alt.loc[idx, ["bfsp", "RSR"]] = alt.loc[idx, ["bfsp", "RSR"]].to_numpy()[rng.permutation(len(idx))]
    else:                                   # the morning card: no result, margin, price or time yet
        alt.loc[on, RESULT] = np.nan
    new, _ = add_form_windows(alt)
    b = base[base.race_date == day].set_index(KEY)[cols].sort_index()
    n = new[new.race_date == day].set_index(KEY)[cols].sort_index()
    pd.testing.assert_frame_equal(b, n, check_exact=True)
    # the control: later days read the day's results, so their windows move
    later = (base.race_date > day) & (base.race_date <= day + pd.Timedelta(days=20))
    bl = base[later].set_index(KEY)[cols].sort_index()
    nl = new[(new.race_date > day) & (new.race_date <= day + pd.Timedelta(days=20))].set_index(KEY)[cols].sort_index()
    moved = [c for c in cols if not np.array_equal(bl[c].to_numpy(float), nl[c].to_numpy(float), equal_nan=True)]
    assert len(moved) > len(cols) // 2, moved


def test_card_rows_get_the_values_the_day_gets_with_results_in():
    df = _history(days=40)
    last = df["race_date"].max()
    full, cols = add_form_windows(df.copy())
    card = df.copy()
    card.loc[card["race_date"] == last, RESULT] = np.nan
    live, _ = add_form_windows(card)
    a = full[full.race_date == last].set_index(KEY)[cols].sort_index()
    b = live[live.race_date == last].set_index(KEY)[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a.notna().mean().min() > 0.5, a.notna().mean().sort_values().head()


def test_two_rows_of_one_horse_on_one_day_see_the_same_history_and_not_each_other():
    df = _runs([dict(horse_name="x", race_date="2024-01-01", placing_numerical=2),
                dict(horse_name="x", race_date="2024-01-08", placing_numerical=1),
                dict(horse_name="x", race_date="2024-01-08", race_time="4.00", placing_numerical=9)])
    out, cols = add_form_windows(df)
    a, b = out.iloc[1], out.iloc[2]
    pd.testing.assert_series_equal(a[cols], b[cols], check_names=False)
    assert a["fw_nfp_l1"] == pytest.approx(0.9)
