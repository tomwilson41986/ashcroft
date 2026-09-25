"""The intent block (model/intent_features.py).

Synthetic frames here test mechanics only: the previous-run lookup, each angle's
definition, the lag rule, and whether a trainer's planted habit of beating the
price with an angle is found. Nothing is trained or evaluated on them.
"""
import numpy as np
import pandas as pd
import pytest

from model.intent_features import ANGLES, INTENT_FEATURES, add_intent_features, previous_run


def _runs(rows):
    base = dict(race_time="2.30", track="york", race_type="Maiden", race_name="", horse_sex="c", headgear="",
                trainer="a smith", jockey_name="j one", jockeys_claim=0, race_class="Class 4", surface_type="Turf",
                career_runs=np.nan, placing_numerical=2, bfsp=5.0)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_previous_run_is_the_last_run_on_an_earlier_day():
    df = _runs([
        dict(horse_name="x", race_date="2024-01-10"),
        dict(horse_name="x", race_date="2024-01-01"),
        dict(horse_name="x", race_date="2024-01-10", race_time="4.00"),     # same day: not "previous"
        dict(horse_name="y", race_date="2024-01-05"),
        dict(horse_name="x", race_date="2024-02-01"),
    ])
    prev, seen = previous_run(df)
    np.testing.assert_array_equal(prev, [1, -1, 1, -1, 2])
    np.testing.assert_array_equal(seen, [1, 0, 1, 0, 3])


def test_each_angle_on_a_hand_built_career():
    df = _runs([
        dict(horse_name="x", race_date="2024-01-01", career_runs=0),
        dict(horse_name="x", race_date="2024-02-01"),
        dict(horse_name="x", race_date="2024-03-01", trainer="b jones", headgear="TT"),
        dict(horse_name="x", race_date="2024-04-01", trainer="b jones", horse_sex="g", race_type="Handicap",
             race_class="Class 5"),
        dict(horse_name="x", race_date="2024-09-01", trainer="b jones", horse_sex="g", race_type="Handicap Hurdle",
             race_class="Class 4", jockeys_claim=7),
    ])
    out, cols = add_intent_features(df.copy())
    assert cols == INTENT_FEATURES and set(cols) <= set(out.columns)
    o = out.set_index("race_date")
    assert o.loc["2024-01-01", "it_debut"] == 1 and np.isnan(o.loc["2024-01-01", "it_new_yard"])
    assert o.loc["2024-02-01", "it_debut"] == 0 and o.loc["2024-02-01", "it_new_yard"] == 0
    assert o.loc["2024-03-01", "it_new_yard"] == 1 and o.loc["2024-03-01", "it_runs_for_yard"] == 0
    assert o.loc["2024-03-01", "it_ft_headgear"] == 1
    assert o.loc["2024-04-01", "it_ft_headgear"] == 0                  # none today
    assert o.loc["2024-04-01", "it_gelded"] == 1 and o.loc["2024-04-01", "it_hcap_debut"] == 1
    assert o.loc["2024-04-01", "it_new_yard"] == 0
    assert o.loc["2024-04-01", "it_class_drop"] == 1 and o.loc["2024-04-01", "it_runs_for_yard"] == 1
    assert o.loc["2024-09-01", "it_gelded"] == 0 and o.loc["2024-09-01", "it_hcap_debut"] == 0
    assert o.loc["2024-09-01", "it_layoff"] == 1 and o.loc["2024-09-01", "it_code_switch"] == 1
    assert o.loc["2024-09-01", "it_class_change"] == -1 and o.loc["2024-09-01", "it_claim_change"] == 7


def test_first_time_headgear_is_any_item_not_worn_before():
    df = _runs([dict(horse_name="x", race_date="2024-01-01", career_runs=0, headgear="TT"),
                dict(horse_name="x", race_date="2024-02-01", career_runs=1, headgear="CkPc TT"),
                dict(horse_name="x", race_date="2024-03-01", career_runs=2, headgear="TT"),
                dict(horse_name="x", race_date="2024-04-01", career_runs=3, headgear="CkPc")])
    out, _ = add_intent_features(df.copy())
    np.testing.assert_array_equal(out.sort_values("race_date")["it_ft_headgear"].to_numpy()[1:], [1, 0, 0])


def test_a_gap_in_the_history_leaves_first_time_angles_unknown():
    # career_runs says two runs came before the last one; the frame holds one
    df = _runs([dict(horse_name="x", race_date="2024-01-01", career_runs=0),
                dict(horse_name="x", race_date="2024-03-01", career_runs=2, race_type="Handicap", headgear="Blnk")])
    out, _ = add_intent_features(df.copy())
    last = out[out.race_date == "2024-03-01"].iloc[0]
    assert np.isnan(last["it_hcap_debut"]) and np.isnan(last["it_ft_headgear"])


def test_first_time_angles_are_unknown_when_the_history_may_be_missing():
    # first row held already has 6 career runs: earlier runs are not in the frame
    df = _runs([dict(horse_name="x", race_date="2024-01-01", career_runs=6),
                dict(horse_name="x", race_date="2024-02-01", race_type="Handicap", headgear="p")])
    out, _ = add_intent_features(df.copy())
    last = out[out.race_date == "2024-02-01"].iloc[0]
    assert np.isnan(last["it_hcap_debut"]) and np.isnan(last["it_ft_headgear"]) and np.isnan(last["it_code_switch"])
    assert last["it_new_yard"] == 0                       # needs only the previous run


def _market(days=300, seed=2, edge=0.0):
    """Six-runner races, each runner priced by a known chance p and winning with it --
    except that trainer 't0''s handicap debutants win with p * (1 + edge)."""
    rng = np.random.default_rng(seed)
    rows, horses = [], {}
    for i, d in enumerate(pd.date_range("2023-01-01", periods=days, freq="D")):
        for r in range(4):
            p = rng.dirichlet(np.ones(6) * 3)
            hcap = r % 2 == 0
            names, boost = [], p.copy()
            for j in range(6):
                t = f"t{rng.integers(0, 6)}"
                h = f"{t}_c{i // 20}_h{rng.integers(0, 8)}"      # a new crop of horses every 20 days
                names.append((h, t))
                if hcap and t == "t0" and not horses.get(h, {}).get("hcap"):
                    boost[j] *= 1 + edge
            w = rng.choice(6, p=boost / boost.sum())
            for j, (h, t) in enumerate(names):
                first = h not in horses
                rows.append(dict(race_date=str(d.date()), race_time=f"{2 + r}.00", track="york", horse_name=h,
                                 trainer=t, race_type="Handicap" if hcap else "Novice", race_name="",
                                 horse_sex="g", headgear="", jockey_name=f"j{j}", jockeys_claim=0,
                                 race_class="Class 4", surface_type="Turf", career_runs=0 if first else np.nan,
                                 placing_numerical=1 if j == w else 2 + j, bfsp=1 / p[j]))
                horses.setdefault(h, {})
                if hcap:
                    horses[h]["hcap"] = True
    return pd.DataFrame(rows).drop_duplicates(["race_date", "race_time", "horse_name"]).reset_index(drop=True)


def test_a_trainers_habit_of_beating_the_price_with_an_angle_is_found():
    out, _ = add_intent_features(_market(edge=1.5))
    late = out[(out.race_date >= "2023-07-01") & (out.it_hcap_debut == 1)]
    t0 = late[late.trainer == "t0"]["it_ae_hcap_debut"].mean()
    rest = late[late.trainer != "t0"]["it_ae_hcap_debut"]
    assert t0 > 0.03
    assert abs(rest.mean()) < 0.015 and t0 > rest.max()


def test_no_habit_no_signal():
    out, _ = add_intent_features(_market(edge=0.0, seed=4))
    late = out[(out.race_date >= "2023-07-01") & (out.it_hcap_debut == 1)]
    assert abs(late["it_ae_hcap_debut"].mean()) < 0.01


@pytest.mark.parametrize("change", ["scramble", "blank"])
def test_a_days_own_results_move_none_of_its_features(change):
    df = _market(days=120, edge=1.0)
    day = "2023-03-15"
    base, cols = add_intent_features(df.copy())
    alt = df.copy()
    on = alt["race_date"] == day
    rng = np.random.default_rng(9)
    if change == "scramble":
        for _, idx in alt[on].groupby("race_time").groups.items():
            alt.loc[idx, "placing_numerical"] = rng.permutation(len(idx)) + 1
            alt.loc[idx, "bfsp"] = rng.uniform(1.5, 30, len(idx))
    else:                                  # the morning card: no result and no price yet
        alt.loc[on, ["placing_numerical", "bfsp"]] = np.nan
    new, _ = add_intent_features(alt)
    k = ["race_date", "race_time", "horse_name"]
    b = base[base.race_date == day].set_index(k)[cols].sort_index()
    n = new[new.race_date == day].set_index(k)[cols].sort_index()
    pd.testing.assert_frame_equal(b, n, check_exact=True)
    # and the next day, which may read today, still gets features
    assert new[new.race_date == "2023-03-16"][cols].notna().any().all()


def test_card_rows_get_features_and_add_nothing():
    df = _market(days=60, edge=1.0)
    card = df[df.race_date == df.race_date.max()].copy()
    hist = df[df.race_date < df.race_date.max()]
    full, cols = add_intent_features(df.copy())
    card[["placing_numerical", "bfsp"]] = np.nan
    live, _ = add_intent_features(pd.concat([hist, card], ignore_index=True))
    k = ["race_date", "race_time", "horse_name"]
    a = full[full.race_date == df.race_date.max()].set_index(k)[cols].sort_index()
    b = live[live.race_date == df.race_date.max()].set_index(k)[cols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_every_angle_has_its_trainer_record_column():
    assert all(f"it_ae_{a}" in INTENT_FEATURES and f"it_{a}" in INTENT_FEATURES for a in ANGLES)
