"""The freshness block (model/freshness_features.py).

Synthetic frames test mechanics only: each feature on a hand-built career, and the
lag rule. Nothing is trained or evaluated on them.
"""
import numpy as np
import pandas as pd
import pytest

from model.freshness_features import FRESHNESS_FEATURES, add_freshness_features


def _runs(rows):
    base = dict(race_time="2.30", track="york", race_type="Handicap", surface_type="Turf", trainer="a smith",
                placing_numerical=5, days_since_lr=np.nan, career_runs=np.nan)
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def _career():
    # debut 1 Jan; 15 Jan (14 days); won 29 Jan (14); 5 Feb (7, quick after the win);
    # break to 1 Jun (117 days, a leap year); 15 Jun (14); a hurdle on 1 Jul (16)
    return _runs([
        dict(horse_name="x", race_date="2024-01-01", career_runs=0),
        dict(horse_name="x", race_date="2024-01-15"),
        dict(horse_name="x", race_date="2024-01-29", placing_numerical=1),
        dict(horse_name="x", race_date="2024-02-05", placing_numerical=3),
        dict(horse_name="x", race_date="2024-06-01"),
        dict(horse_name="x", race_date="2024-06-15"),
        dict(horse_name="x", race_date="2024-07-01", race_type="Handicap Hurdle"),
    ])


def test_each_feature_on_a_hand_built_career():
    out, cols = add_freshness_features(_career())
    assert cols == FRESHNESS_FEATURES and set(cols) <= set(out.columns)
    o = out.set_index(out["race_date"].dt.strftime("%Y-%m-%d"))

    # runs since a break: the debut starts a sequence, 117 days off starts another
    assert list(out.sort_values("race_date")["fr_runs_since_break"]) == [0, 1, 2, 3, 0, 1, 2]
    assert np.isnan(o.loc["2024-01-01", "fr_break_len"])
    assert o.loc["2024-06-15", "fr_break_len"] == pytest.approx(np.log1p(117))

    # the usual spacing needs two earlier runs; the debut's gap is unknown, so after
    # 15 Jan and 29 Jan it is the mean of log(15) (14 days) weighted by recency
    assert np.isnan(o.loc["2024-01-15", "fr_usual_gap"])
    assert o.loc["2024-01-29", "fr_usual_gap"] == pytest.approx(np.log1p(14))
    assert o.loc["2024-06-01", "fr_gap_vs_usual"] > 1.5          # 117 days against a usual fortnight
    assert o.loc["2024-02-05", "fr_gap_vs_usual"] < 0            # back quicker than usual

    # workload on earlier days: 5 Feb has runs on 15 Jan, 29 Jan in 30 days, and the debut too in 60
    assert o.loc["2024-02-05", "fr_runs_30d"] == 2 and o.loc["2024-02-05", "fr_runs_60d"] == 3
    assert o.loc["2024-06-01", "fr_runs_90d"] == 0

    # since the last win and the last placed run, never counting the day itself
    assert np.isnan(o.loc["2024-01-29", "fr_days_since_win"])    # its own win does not count
    assert o.loc["2024-02-05", "fr_days_since_win"] == 7 and o.loc["2024-02-05", "fr_runs_since_win"] == 1
    assert o.loc["2024-06-01", "fr_days_since_placed"] == 117     # 3rd on 5 Feb
    assert o.loc["2024-02-05", "fr_quick_after_win"] == 1
    assert o.loc["2024-06-01", "fr_quick_after_win"] == 0
    assert np.isnan(o.loc["2024-01-01", "fr_quick_after_win"])

    # since the last run in today's code: the first hurdle has none, the flat runs do
    assert np.isnan(o.loc["2024-07-01", "fr_days_since_code"])
    assert o.loc["2024-06-15", "fr_days_since_code"] == 14


def test_hrbs_gap_is_preferred_to_the_dates_held():
    # the database misses a run: HRB says 20 days, the dates held say 60
    df = _runs([dict(horse_name="x", race_date="2024-01-01", career_runs=0),
                dict(horse_name="x", race_date="2024-03-01", days_since_lr=20, placing_numerical=1),
                dict(horse_name="x", race_date="2024-03-10", days_since_lr=9)])
    out, _ = add_freshness_features(df)
    o = out.set_index(out["race_date"].dt.strftime("%Y-%m-%d"))
    assert o.loc["2024-03-01", "fr_runs_since_break"] == 1         # 20 days is no break
    assert o.loc["2024-03-10", "fr_quick_after_win"] == 1         # the run held 9 days ago was a win


def test_a_history_that_starts_mid_sequence_is_unknown_until_a_break():
    df = _runs([dict(horse_name="x", race_date="2024-01-01", career_runs=12, days_since_lr=21),
                dict(horse_name="x", race_date="2024-01-20", days_since_lr=19),
                dict(horse_name="x", race_date="2024-05-01", days_since_lr=102)])
    out, _ = add_freshness_features(df)
    assert np.isnan(out["fr_runs_since_break"].iloc[:2]).all()
    assert out["fr_runs_since_break"].iloc[2] == 0


def test_the_gap_against_the_field_and_the_yard():
    df = _runs([dict(horse_name="a", race_date="2024-01-01", career_runs=0, trainer="t1"),
                dict(horse_name="b", race_date="2024-01-01", career_runs=0, trainer="t1"),
                dict(horse_name="a", race_date="2024-01-08", trainer="t1"),
                dict(horse_name="b", race_date="2024-03-01", trainer="t1"),
                dict(horse_name="c", race_date="2024-03-01", days_since_lr=10, trainer="t2")])
    out, _ = add_freshness_features(df)
    last = out[out["race_date"] == "2024-03-01"].set_index("horse_name")
    # b (60 days) and c (10) in one race: centred on the race
    assert last.loc["b", "fr_gap_rel_race"] == pytest.approx((np.log1p(60) - np.log1p(10)) / 2)
    # t1's only earlier gap was a's 7 days; b's 60 is well beyond the yard's usual
    assert last.loc["b", "fr_gap_vs_trainer"] == pytest.approx(np.log1p(60) - np.log1p(7))
    assert np.isnan(last.loc["c", "fr_gap_vs_trainer"])            # t2 has no earlier runner


def _history(days=120, seed=3):
    """A pool of horses running on random days, with results."""
    rng = np.random.default_rng(seed)
    rows = []
    last = {}
    for d in pd.date_range("2023-01-01", periods=days, freq="D"):
        for r, track in enumerate(("york", "kempton")):
            horses = rng.choice(60, 8, replace=False)
            place = rng.permutation(8) + 1
            for i, h in enumerate(horses):
                name = f"h{h}"
                rows.append(dict(race_date=d, race_time=f"{2 + r}.30", track=track, horse_name=name,
                                 trainer=f"t{h % 7}", race_type="Handicap Hurdle" if h % 5 == 0 else "Handicap",
                                 surface_type="Turf", placing_numerical=int(place[i]),
                                 days_since_lr=(d - last[name]).days if name in last else np.nan,
                                 career_runs=np.nan if name in last else 0.0, bfsp=float(rng.uniform(2, 30))))
                last[name] = d
    return pd.DataFrame(rows)


KEY = ["race_date", "track", "horse_name"]


@pytest.mark.parametrize("change", ["scramble", "blank"])
def test_a_days_own_results_move_none_of_its_features(change):
    df = _history()
    day = pd.Timestamp("2023-03-15")
    base, cols = add_freshness_features(df.copy())
    alt = df.copy()
    on = alt["race_date"] == day
    rng = np.random.default_rng(1)
    if change == "scramble":
        for _, idx in alt[on].groupby("track").groups.items():
            alt.loc[idx, "placing_numerical"] = rng.permutation(len(idx)) + 1
    else:                                   # the morning card: no result yet
        alt.loc[on, ["placing_numerical", "bfsp"]] = np.nan
    new, _ = add_freshness_features(alt)
    b = base[base.race_date == day].set_index(KEY)[cols].sort_index()
    n = new[new.race_date == day].set_index(KEY)[cols].sort_index()
    pd.testing.assert_frame_equal(b, n, check_exact=True)
    # the control: the next days read today's results, so they can move
    later = (base.race_date > day) & (base.race_date <= day + pd.Timedelta(days=20))
    bl = base[later].set_index(KEY)[cols].sort_index()
    nl = new[(new.race_date > day) & (new.race_date <= day + pd.Timedelta(days=20))].set_index(KEY)[cols].sort_index()
    moved = [c for c in cols if not np.array_equal(bl[c].to_numpy(float), nl[c].to_numpy(float), equal_nan=True)]
    assert moved, "nothing after the day moved: the test cannot see a change"


def test_card_rows_get_the_values_the_day_gets_with_results_in():
    df = _history(days=60)
    last = df["race_date"].max()
    full, cols = add_freshness_features(df.copy())
    card = df.copy()
    card.loc[card["race_date"] == last, ["placing_numerical", "bfsp"]] = np.nan
    live, _ = add_freshness_features(card)
    a = full[full.race_date == last].set_index(KEY)[cols].sort_index()
    b = live[live.race_date == last].set_index(KEY)[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a[["fr_usual_gap", "fr_runs_30d", "fr_gap_rel_race"]].notna().any().all()
