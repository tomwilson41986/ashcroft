"""Guards against features that can see the race they are predicting.

Two leaks lived in the pace and draw blocks and both are easy to reintroduce,
so they get tests rather than comments:

1. `grp.apply(lambda x: x.shift(1).expanding().mean())` lags by ROW. On a key
   every runner in a race shares -- track+distance, going, trainer -- the
   previous row is a rival in the same race, not an earlier race.
2. The EPF pace block parsed the horse's own in-running comment for the race
   being predicted, then aggregated it to race level.
"""

import numpy as np
import pandas as pd
import pytest

from model.custom_metrics import POST_RACE_ONLY, CustomMetricsEngine
from model.draw_metrics import DrawMetricsEngine
from model.lagsafe import race_lagged_expanding_mean
from model.pace_metrics import PaceMetricsEngine


def _card(n_days=6, races_per_day=2, runners=6, seed=0, flip_last_race=False):
    """A small synthetic history with real track/going/draw structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)
        for r in range(races_per_day):
            rid = f"{day.date()}_{13 + r}:00_Ascot"
            winner = int(rng.integers(0, runners))
            for i in range(runners):
                lead = (i + d + r) % 3 == 0
                if flip_last_race and d == n_days - 1 and r == races_per_day - 1:
                    lead = not lead          # only the race being predicted changes
                rows.append({
                    "raceid": rid, "race_date": day, "race_time": f"{13 + r}:00",
                    "track": "Ascot", "horse_name": f"h{i}", "jockey_name": f"j{i}",
                    "trainer": "TR" if i < 2 else f"T{i}",       # TR doubles up in every race
                    "draw": i + 1, "number_of_runners": runners,
                    "dist_furlongs": 6.0, "going_description": "Good",
                    "placing_numerical": 1 if i == winner else int(1 + ((i - winner) % runners)),
                    "comment": "led throughout, ran on well" if lead else "held up in rear, stayed on",
                    "NFP": float(rng.normal(0, 1)),
                    "race_type": "Flat", "bfsp": float(2 + i),
                })
    return pd.DataFrame(rows)


def test_race_lagged_mean_excludes_every_runner_of_its_own_race():
    d = _card(n_days=2, races_per_day=1, runners=3, seed=1)
    d["v"] = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    out = race_lagged_expanding_mean(d, ["track", "dist_furlongs"], "v")
    first, second = d["raceid"].unique()
    assert out[d["raceid"] == first].isna().all()              # nothing earlier exists
    assert np.allclose(out[d["raceid"] == second], 1.0 / 3)    # only race 1 contributes
    assert out[d["raceid"] == second].nunique() == 1           # identical for every runner

    # the idiom this replaces, for contrast: shift(1) steps back one ROW, so the
    # third runner of race 1 already sees the second runner of race 1
    leaky = (d.sort_values(["track", "race_date", "race_time"])
              .groupby("track", group_keys=False)["v"]
              .apply(lambda x: x.shift(1).expanding().mean()))
    assert leaky[d["raceid"] == first].notna().any()


def test_race_lagged_mean_respects_min_races_and_is_index_aligned():
    d = _card(n_days=4, races_per_day=1, runners=4, seed=2)
    d["v"] = np.arange(len(d), dtype=float)
    out = race_lagged_expanding_mean(d, "track", "v", min_races=2)
    assert list(out.index) == list(d.index)
    by_race = out.groupby(d["raceid"]).first()
    assert by_race.isna().sum() == 2 and by_race.notna().sum() == 2


@pytest.mark.parametrize("col", ["td_front_win_share", "td_holdup_win_share", "td_avg_winner_pos",
                                 "track_front_win_share", "td_draw_bias", "td_low_stall_nfp",
                                 "trainer_career_early_pos"])
def test_course_and_trainer_bias_features_do_not_vary_inside_a_race(col):
    card = _card()
    d = DrawMetricsEngine().calculate(PaceMetricsEngine().calculate(card))
    assert col in d.columns
    if col == "trainer_career_early_pos":
        sub = d[d["trainer"] == "TR"]                      # the stable with two runners per race
        spread = sub.groupby("raceid")[col].nunique(dropna=True)
    else:
        spread = d.groupby("raceid")[col].nunique(dropna=True)
    assert spread.max() <= 1, f"{col} varies within a race, which can only come from the race itself"


def test_race_pace_aggregates_ignore_todays_running_comments():
    """Rewriting one race's comments must not change that race's own features.

    Comments from *earlier* races, including earlier races on the same card, are
    fair game -- they are known by the time this race is priced."""
    engine = CustomMetricsEngine()
    base = engine._calc_pace(engine._calc_epf(_card(flip_last_race=False)))
    flip = engine._calc_pace(engine._calc_epf(_card(flip_last_race=True)))
    last = sorted(base["raceid"].unique())[-1]
    key = ["raceid", "horse_name"]
    b = base[base["raceid"] == last].sort_values(key).reset_index(drop=True)
    f = flip[flip["raceid"] == last].sort_values(key).reset_index(drop=True)
    assert len(b) and (b["comment"] != f["comment"]).any()      # the fixture really did change
    for col in ("RPS", "pace_pressure", "prom_runner", "racepacescore", "racepaceindex"):
        assert col in b.columns
        assert np.allclose(b[col].fillna(-1), f[col].fillna(-1)), \
            f"{col} changed when only the predicted race's comments changed"


def test_training_feature_list_contains_no_post_race_column():
    from train_bfsp import ALL_FEATURE_COLS, assert_no_post_race_features
    assert_no_post_race_features(ALL_FEATURE_COLS)
    with pytest.raises(ValueError, match="describe the race being predicted"):
        assert_no_post_race_features(list(ALL_FEATURE_COLS) + ["placing_numerical"])
    assert {"EPF", "placing_numerical", "comment"} <= POST_RACE_ONLY
