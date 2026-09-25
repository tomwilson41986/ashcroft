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


def test_one_beaten_length_table_for_the_whole_repo():
    """Two modules disagreeing on what "nk" means puts two different figures on
    the same race. The spec forbids it; so does this test."""
    import re

    from model.perf_figures import MARGIN_WORDS
    from model.custom_metrics import CustomMetricsEngine

    d = _card(n_days=2, races_per_day=1, runners=4)
    d["total_dst_bt"] = ["0", "nk", "2hd", "shd"] + ["0", "hd", "1nk", "dist"]
    d["placing_numerical"] = [1, 3, 4, 2] + [1, 2, 3, 4]         # the "0" rows won
    out = CustomMetricsEngine()._calc_actual_lengths_beaten(d)   # may reorder rows
    got = dict(zip(out["total_dst_bt"], out["LB"]))
    assert got["nk"] == MARGIN_WORDS["nk"] == 0.3
    assert got["hd"] == MARGIN_WORDS["hd"] == 0.2
    assert got["shd"] == MARGIN_WORDS["shd"] == 0.1
    assert got["2hd"] == 2 + MARGIN_WORDS["hd"]
    assert got["1nk"] == 1 + MARGIN_WORDS["nk"]
    assert got["dist"] == MARGIN_WORDS["dist"] == 30.0


def test_speed_figure_standard_time_uses_only_earlier_races():
    """RSR is a time against a standard. The standard cannot be a median of
    every race in the file, half of which have not been run yet."""
    d = _card(n_days=12, races_per_day=1, runners=4, seed=5)
    d["comptime_numeric"] = 72.0
    late = d["race_date"] == d["race_date"].max()
    d.loc[late, "comptime_numeric"] = 60.0            # a much faster final race

    slow = CustomMetricsEngine()._calc_speed_figures(d.copy())
    d2 = d.copy()
    d2.loc[late, "comptime_numeric"] = 90.0           # ... or a much slower one
    fast = CustomMetricsEngine()._calc_speed_figures(d2)

    key = ["raceid", "horse_name"]
    a = slow[~slow["race_date"].eq(slow["race_date"].max())].sort_values(key)["RSR"].values
    b = fast[~fast["race_date"].eq(fast["race_date"].max())].sort_values(key)["RSR"].values
    assert np.allclose(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999)), \
        "an earlier race's RSR moved when a later race's time changed"


def test_off_times_order_by_the_clock_not_alphabetically():
    """UK cards are written '1.45' for 13:45, so a string sort runs the day
    backwards: '1.45' sorts before '12.40', and the "previous race" is then the
    wrong race."""
    from model.lagsafe import race_minutes

    times = ["11.30", "12.40", "1.45.", "2.20", "3.55", "13:00"]
    mins = race_minutes(pd.Series(times))
    assert list(mins) == [690, 760, 825, 860, 955, 780]
    assert list(mins.iloc[:5]) == sorted(mins.iloc[:5])       # the card runs forwards
    assert sorted(times)[0] == "1.45." and mins.iloc[2] > mins.iloc[0]   # ... but the strings do not


def _same_card_and_next_day():
    d = _card(n_days=1, races_per_day=1, runners=3, seed=9)
    return pd.concat([
        d.assign(raceid="early", race_time="11.30", v=[1.0, 1.0, 1.0]),
        d.assign(raceid="late", race_time="1.45.", v=[0.0, 0.0, 0.0]),
        d.assign(raceid="tomorrow", race_time="2.00", v=[0.5, 0.5, 0.5],
                 race_date=d["race_date"] + pd.Timedelta(days=1)),
    ], ignore_index=True)


def test_nothing_from_the_card_being_priced_enters_a_prior():
    """A 06:00 forecast cannot know the 11.30's result when it prices the 1.45,
    and a live card has no results at all, so priors step back whole days."""
    d = _same_card_and_next_day()
    out = race_lagged_expanding_mean(d, "track", "v")
    assert out[d["raceid"] == "early"].isna().all()           # nothing ran before it
    assert out[d["raceid"] == "late"].isna().all()            # the 11.30 is the same day
    assert np.allclose(out[d["raceid"] == "tomorrow"], 0.5)   # both of yesterday's races


def test_a_live_card_does_not_count_its_own_runners_as_losers():
    """Priced live, today's runners have no placing, so `won` is 0 for all of
    them. Under a race lag the 4.10's trainer strike rate counted his 2.00
    runner as a loser; under the day lag the card never enters its own priors."""
    hist = pd.DataFrame({"raceid": ["h1"], "race_date": pd.to_datetime(["2025-01-01"]),
                         "race_time": ["2.00"], "trainer": ["T"], "won": [1.0]})
    card = pd.DataFrame({"raceid": ["c1", "c2"], "race_date": pd.to_datetime(["2025-01-02"] * 2),
                         "race_time": ["2.00", "4.10"], "trainer": ["T", "T"], "won": [0.0, 0.0]})
    out = race_lagged_expanding_mean(pd.concat([hist, card], ignore_index=True), "trainer", "won")
    assert out.iloc[1:].tolist() == [1.0, 1.0]               # both see yesterday's win and only that


def test_under_the_race_rule_a_later_race_sees_an_earlier_one_by_the_clock(monkeypatch):
    """The old rule, kept for comparison: the clock order has to hold end to
    end, not just in the parser."""
    from model import lagsafe
    monkeypatch.setattr(lagsafe, "LAG_UNIT", "race")
    d = _same_card_and_next_day()
    out = race_lagged_expanding_mean(d, "track", "v")
    assert out[d["raceid"] == "early"].isna().all()
    assert np.allclose(out[d["raceid"] == "late"], 1.0)       # sees only the 11.30


def _result_card(n_days=8, races_per_day=2, runners=8, seed=3, flip_last_race=False):
    """History with real draws, going and classes; optionally the last race's
    finishing order reversed."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)
        for r in range(races_per_day):
            order = list(rng.permutation(runners))
            if flip_last_race and d == n_days - 1 and r == races_per_day - 1:
                order = order[::-1]
            pos = {h: i + 1 for i, h in enumerate(order)}
            for i in range(runners):
                rows.append({
                    "race_date": day, "race_time": f"{1 + r}.30", "track": "Ascot",
                    "raceid": f"{day.date()}_{r}", "horse_name": f"h{i}",
                    "jockey_name": f"j{i}", "trainer": f"T{i % 3}",
                    "stall": i + 1, "draw": i + 1, "number_of_runners": runners,
                    "dist_furlongs": 6.0, "going_description": "Good",
                    "race_type": "Flat", "race_code": "F", "race_class": "4",
                    "placing_numerical": pos[i], "total_dst_bt": "0" if pos[i] == 1 else str(pos[i] - 1),
                    "comment": "led throughout" if i % 3 == 0 else "held up in rear",
                    "comptime_numeric": 72.0, "official_rating": 70 + i,
                    "bfsp": float(2 + i), "won": pos[i] == 1, "horse_age": 4,
                })
    return pd.DataFrame(rows)


def test_no_pace_or_draw_feature_moves_when_the_last_race_is_re_run():
    """The broad guard, over every column the two engines produce.

    The named tests above each cover one known leak. This one does not need to
    know the name: it reverses the finishing order of the final race and asserts
    that nothing the model would see about that race changes. Anything that does
    move is reading the result it is meant to predict."""
    base = DrawMetricsEngine().calculate(PaceMetricsEngine().calculate(_result_card()))
    flip = DrawMetricsEngine().calculate(PaceMetricsEngine().calculate(_result_card(flip_last_race=True)))

    last = sorted(base["raceid"].unique())[-1]
    key = ["raceid", "horse_name"]
    b = base[base["raceid"] == last].sort_values(key).reset_index(drop=True)
    f = flip[flip["raceid"] == last].sort_values(key).reset_index(drop=True)
    assert len(b) and not b["placing_numerical"].equals(f["placing_numerical"]), \
        "the fixture did not actually change the result"

    from model.draw_metrics import ALL_DRAW_FEATURES
    from model.pace_metrics import ALL_PACE_FEATURES

    moved = []
    for col in sorted(set(ALL_PACE_FEATURES) | set(ALL_DRAW_FEATURES)):
        if col not in b.columns or not pd.api.types.is_numeric_dtype(b[col]):
            continue
        if not np.allclose(b[col].fillna(-999.0), f[col].fillna(-999.0), equal_nan=True):
            moved.append(col)
    assert not moved, f"these read the race they are predicting: {moved}"


# ---------------------------------------------------------------------------
# The broad guard: flip the result, and nothing about that race may move.
#
# Every test above names a leak someone already found. This one needs no name.
# It reverses the finishing order of the final race and asserts that not one of
# the features the production model is trained on changes. A feature that moves
# is reading the result it exists to predict.
# ---------------------------------------------------------------------------

def _full_card(n_days=14, races_per_day=2, runners=8, seed=3, flip_last_race=False):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)
        for r in range(races_per_day):
            order = list(rng.permutation(runners))
            if flip_last_race and d == n_days - 1 and r == races_per_day - 1:
                order = order[::-1]
            pos = {h: i + 1 for i, h in enumerate(order)}
            for i in range(runners):
                p = pos[i]
                rows.append({
                    "race_date": day, "race_time": f"{1+r}.30", "track": "Ascot",
                    "raceid": f"{day.date()}_{r}", "horse_name": f"h{i}",
                    "jockey_name": f"j{i}", "trainer": f"T{i%3}",
                    "stall": i+1, "draw": i+1, "number_of_runners": runners,
                    "dist_furlongs": 6.0, "race_distance": "6f", "yards": 1320,
                    "going_description": "Good", "race_type": "Flat", "race_code": "F",
                    "race_class": "4", "race_name": "Handicap", "major": "N",
                    "race_restrictions_age": "3yo+", "prize_money": "5000",
                    "placing_numerical": p, "place": str(p),
                    "total_dst_bt": "0" if p == 1 else str(p-1), "distbt": str(p-1),
                    "comment": "led throughout" if i % 3 == 0 else "held up in rear",
                    "comptime_numeric": 72.0, "comptime": "1:12.00",
                    "official_rating": 70+i, "median_or": 74, "max_or_in_race": 70+runners,
                    "pounds": 126 - i, "jockeys_claim": 0, "card_no": i+1,
                    "odds": float(2+i), "fav": "F" if i == 0 else "",
                    "bfsp": float(2+i), "bfsp_place": float(1.5+i*0.3),
                    "plcs_paid": 3, "bf_plcs_paid": 3,
                    "won": p == 1, "horse_age": 4, "horse_sex": "G",
                    "days_since_lr": 21, "career_runs": 10 + i,
                    "stallion": f"S{i%4}", "dam": f"D{i%6}", "dam_stallion": f"S{(i+1)%4}",
                    "surface_type": "Turf", "horse_prizewin": "1000",
                    "headgear": "", "rail_move": "", "track_direction": "L",
                    "stall_positioning": "Stands", "horse_code": f"hc{i}",
                })
    return pd.DataFrame(rows)



def _deployed_features_for(flip_last_race):
    from model.custom_metrics import CustomMetricsEngine
    import train_bfsp as T

    d = CustomMetricsEngine().calculate_all(_full_card(flip_last_race=flip_last_race))
    return T.build_context_features(d)


def test_no_deployed_feature_reads_the_race_it_is_predicting():
    import train_bfsp as T

    base = _deployed_features_for(False)
    flipped = _deployed_features_for(True)

    last = sorted(base["raceid"].unique())[-1]
    key = ["raceid", "horse_name"]
    b = base[base["raceid"] == last].sort_values(key).reset_index(drop=True)
    f = flipped[flipped["raceid"] == last].sort_values(key).reset_index(drop=True)
    assert len(b) and not b["placing_numerical"].equals(f["placing_numerical"]), \
        "the fixture did not actually change the result"

    checked = [c for c in T.ALL_FEATURE_COLS
               if c in b.columns and pd.api.types.is_numeric_dtype(b[c])]
    assert len(checked) > 300, "the feature list did not build; the fixture is missing columns"
    moved = sorted(c for c in checked
                   if not np.allclose(b[c].fillna(-999.0), f[c].fillna(-999.0), equal_nan=True))
    assert not moved, (
        f"{len(moved)} of {len(checked)} deployed features read the race they predict: {moved}"
    )


# ---------------------------------------------------------------------------
# The same guard over the opt-in blocks.
#
# `ALL_FEATURE_COLS` is what the model trains on today, so the test above
# covers what is deployed and nothing else. The pedigree, connection, primitive
# and performance-figure blocks sit behind flags waiting on the promotion
# protocol, and a leak in one of those is worse than a leak in a deployed
# feature: the protocol would read the leak as the gain that justifies turning
# the block on. They get the same treatment before anyone measures them.
# ---------------------------------------------------------------------------

def _optin_features_for(flip_last_race):
    from model.custom_metrics import CustomMetricsEngine
    from model.primitives import add_run_primitives
    from model.pedigree import add_pedigree_features
    from model.connections import add_connection_features
    from model.perf_figures import add_perf_figure_features, PERF_FIGURE_FEATURES

    d = CustomMetricsEngine().calculate_all(_full_card(flip_last_race=flip_last_race))
    d = add_run_primitives(d)
    d, ped = add_pedigree_features(d)
    d, con = add_connection_features(d)
    d = add_perf_figure_features(d)
    # the engine builds its research blocks by default: shape and draw, form windows, shape form
    from model.bfsp_features import RESEARCH_BLOCKS
    engine = [c for cols in RESEARCH_BLOCKS.values() for c in cols]
    return d, list(ped) + list(con) + list(PERF_FIGURE_FEATURES) + engine


def test_no_optin_feature_reads_the_race_it_is_predicting():
    base, names = _optin_features_for(False)
    flipped, _ = _optin_features_for(True)

    last = sorted(base["raceid"].unique())[-1]
    key = ["raceid", "horse_name"]
    b = base[base["raceid"] == last].sort_values(key).reset_index(drop=True)
    f = flipped[flipped["raceid"] == last].sort_values(key).reset_index(drop=True)

    checked = [c for c in dict.fromkeys(names)
               if c in b.columns and pd.api.types.is_numeric_dtype(b[c])]
    assert len(checked) > 50, f"the opt-in blocks did not build ({len(checked)} columns)"
    moved = sorted(c for c in checked
                   if not np.allclose(b[c].fillna(-999.0), f[c].fillna(-999.0), equal_nan=True))
    assert not moved, (
        f"{len(moved)} of {len(checked)} opt-in features read the race they predict: {moved}"
    )
