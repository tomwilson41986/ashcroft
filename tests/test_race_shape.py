"""Run style, race shape, position value and draw curves (model/race_shape.py, model/draw_curve.py).

Synthetic frames here test mechanics only: the parser, the lag rule, the
estimators' ability to recover an effect planted in them. Nothing is trained or
evaluated on them.
"""
import numpy as np
import pandas as pd
import pytest

from model.draw_curve import DRAW_CURVE_FEATURES, DRAW_STYLE_FEATURES, add_draw_curve
from model.race_shape import (
    RACE_SHAPE_FEATURES, add_race_shape_features, asof_decayed_mean, early_style_from_comment,
    horse_decayed_prior, style_class,
)

# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------

PARSER_CASES = [
    ("held up in rear, failed to pick up", 1.0),            # "led to" inside "failed to"
    ("pulled hard, held up in rear, headway 2f out", 1.0),  # "led," inside "pulled,"
    ("travelled well, tracked leaders, led over 1f out, ran on", 4.5),
    ("led, headed over 1f out, weakened", 6.0),
    ("made all, ridden out", 6.0),
    ("dwelt, soon tracking leaders, kept on", 4.5),         # a slow start the comment then places
    ("slowly away, always behind", 1.0),
    ("chased leader, ridden 2f out, one pace", 5.0),        # the leader, not the leaders
    ("chased leaders, no extra final furlong", 4.5),
    ("held up in touch, effort over 2f out", 3.5),          # not "held up", not "in touch"
    ("in touch, ridden and outpaced 3f out", 4.0),
    ("mid-division, kept on final furlong", 3.0),
    ("towards rear, headway over 1f out", 1.5),
    ("prominent, disputed lead 3f out", 4.0),               # the first phrase, not the best
    ("disputed lead until weakened 2f out", 5.5),
    ("jumped left, led to 3rd, weakened 4 out", 6.0),
    ("hit 3rd, behind leaders, weakened", 4.5),
    ("prominent early, led after 1f, took keen hold", 5.8),  # an early lead is a front-runner's race
    ("mid-division, soon led, headed over 2f out", 5.8),
    ("chased leaders, outpaced, left 2nd and hampered last, soon led, driven out", 4.5),  # a late lead is not
    ("slowly into stride, soon improved to chase leaders", 4.5),
    ("waited with in rear, headway 2f out", 1.0),
    ("close-up, pushed along 2f out", 4.0),
    ("mid-divison, ridden and no impression", 3.0),        # the feed's own spelling
    ("raced in 4th, pushed along 3f out", 104.0),          # a place: ORDINAL_BASE + 4
    ("off the pace in 6th, never involved", 106.0),
    ("mistake 4th, soon behind", 1.5),                     # the fourth obstacle, not fourth place
]


@pytest.mark.parametrize("comment,score", PARSER_CASES)
def test_first_positional_phrase(comment, score):
    assert early_style_from_comment(comment) == score


@pytest.mark.parametrize("comment", ["never dangerous", "", None, "ridden 2f out, one pace"])
def test_no_position_is_unknown_not_midfield(comment):
    assert np.isnan(early_style_from_comment(comment))


def test_a_named_place_gives_the_early_position_exactly():
    from model.race_shape import add_run_styles
    df = pd.DataFrame({"comment": ["raced in 4th", "raced in 2nd", "off the pace in 9th", "raced in 4th"],
                       "number_of_runners": [10, 10, 10, 5], "race_date": "2024-01-01", "race_time": "2.00",
                       "track": ["a", "a", "a", "b"], "horse_name": list("wxyz")})
    out = add_run_styles(df)
    np.testing.assert_allclose(out["rs_epf"], [3 / 9, 1 / 9, 8 / 9, 3 / 4])
    np.testing.assert_array_equal(out["rs_class"], [1, 1, 3, 3])     # 4th of ten is prominent, of five is rear


def test_style_classes():
    np.testing.assert_array_equal(style_class([6.0, 5.5, 5.0, 4.0, 3.5, 2.5, 1.5, 1.0]),
                                  [0, 0, 1, 1, 2, 2, 3, 3])
    assert np.isnan(style_class([np.nan])[0])


# ---------------------------------------------------------------------------
# the lookups
# ---------------------------------------------------------------------------

def test_asof_decayed_mean_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 400
    key = rng.integers(0, 5, n)
    day = rng.integers(0, 60, n)
    val = rng.normal(size=n)
    val[rng.random(n) < 0.1] = np.nan
    for hl in (np.inf, 10.0):
        m, ne = asof_decayed_mean(key, day, val, key, day, hl)
        for i in range(0, n, 7):
            sel = (key == key[i]) & (day < day[i]) & np.isfinite(val)
            w = np.exp2(-(day[i] - day[sel]) / hl) if np.isfinite(hl) else np.ones(sel.sum())
            if sel.sum() == 0:
                assert np.isnan(m[i]) and ne[i] == 0
            else:
                assert m[i] == pytest.approx((w * val[sel]).sum() / w.sum())
                assert ne[i] == pytest.approx(w.sum())


def test_horse_prior_is_earlier_days_only_and_decays_by_run():
    horse = np.array(["a"] * 4 + ["b"] * 2)
    day = np.array([1, 5, 9, 12, 5, 9])
    v = np.array([1.0, 0.0, 1.0, 1.0, 1.0, np.nan])
    sums, cnt, runs = horse_decayed_prior(horse, day, {"v": v}, halflife_runs=1.0)
    # a's fourth run sees runs 1-3 at weights 1/8, 1/4, 1/2
    assert sums["v"][3] == pytest.approx(1 / 8 + 0 + 1 / 2)
    assert cnt["v"][3] == pytest.approx(1 / 8 + 1 / 4 + 1 / 2)
    assert sums["v"][0] == 0 and cnt["v"][0] == 0
    assert list(runs) == [0, 1, 2, 3, 0, 1]
    assert cnt["v"][5] == pytest.approx(0.5)       # b's earlier run counts; its own missing value does not


# ---------------------------------------------------------------------------
# lag safety: nothing about a day reaches that day's features
# ---------------------------------------------------------------------------

STYLES = ["made all", "led, headed 1f out", "tracked leaders", "prominent", "mid-division",
          "held up in touch", "held up in rear", "towards rear", "never dangerous"]


def _history(days=150, seed=1, courses=("chester", "kempton", "ascot"), n=9, pace_effect=0.0, draw_effect=0.0,
             forward_draw_effect=0.0):
    """Races on consecutive days with a stable pool of horses, each with a habitual
    style. `pace_effect` makes lone habitual leaders win more; `draw_effect` makes
    low stalls at the first course finish better; `forward_draw_effect` does the
    same for habitual front-runners and prominent racers only."""
    rng = np.random.default_rng(seed)
    pool = {c: [(f"{c}_h{i}", i % len(STYLES[:8])) for i in range(40)] for c in courses}
    rows = []
    for d in pd.date_range("2023-01-01", periods=days, freq="D"):
        for ci, c in enumerate(courses):
            pick = rng.choice(len(pool[c]), n, replace=False)
            horses = [pool[c][k] for k in pick]
            stall = rng.permutation(n) + 1
            leaders = sum(1 for _, s in horses if s <= 1)
            score = rng.normal(size=n)
            for i, (_, s) in enumerate(horses):
                if s <= 1 and leaders == 1:
                    score[i] += pace_effect
                if ci == 0:
                    score[i] -= draw_effect * (stall[i] - 1) / (n - 1)
                    if s <= 3:
                        score[i] -= forward_draw_effect * (stall[i] - 1) / (n - 1)
            place = np.empty(n, int)
            place[np.argsort(-score)] = np.arange(1, n + 1)
            for i, (h, s) in enumerate(horses):
                style = STYLES[s] if rng.random() < 0.8 else STYLES[rng.integers(0, len(STYLES))]
                rows.append(dict(race_date=str(d.date()), race_time=f"{2 + ci}.30", track=c, horse_name=h,
                                 number_of_runners=n, placing_numerical=int(place[i]), stall=int(stall[i]),
                                 total_dst_bt="" if place[i] == 1 else f"{(place[i] - 1) * 0.75:.2f}",
                                 dist_furlongs=6.0, race_type="Handicap", surface_type="Turf", comment=style))
    return pd.DataFrame(rows)


def _features(df):
    a, cols_a = add_race_shape_features(df.copy())
    b, cols_b = add_draw_curve(a)
    return b, cols_a + cols_b


def _key(df):
    return df["race_date"] + "|" + df["track"] + "|" + df["horse_name"]


@pytest.mark.parametrize("change", ["scramble", "blank"])
def test_a_days_own_results_move_none_of_its_features(change):
    df = _history(days=60)
    day = "2023-02-10"
    base, cols = _features(df)
    alt = df.copy()
    on = alt["race_date"] == day
    rng = np.random.default_rng(7)
    if change == "scramble":           # a different result, different margins, different comments
        for (_, _), idx in alt[on].groupby(["race_date", "track"]).groups.items():
            alt.loc[idx, "placing_numerical"] = rng.permutation(len(idx)) + 1
            alt.loc[idx, "comment"] = rng.choice(STYLES, len(idx))
            alt.loc[idx, "total_dst_bt"] = [f"{x:.2f}" for x in rng.uniform(0, 9, len(idx))]
    else:                              # the morning card: no result, no margin, no comment
        alt.loc[on, ["placing_numerical", "total_dst_bt", "comment"]] = None
    new, _ = _features(alt)
    b = base.set_index(_key(base)).loc[_key(base[base.race_date == day])][cols]
    n = new.set_index(_key(new)).loc[b.index][cols]
    diff = [c for c in cols if not np.array_equal(b[c].to_numpy(float), n[c].to_numpy(float), equal_nan=True)]
    assert diff == [], f"features moved by the day's own results: {diff}"


def test_features_on_every_row_and_no_post_race_column():
    df, cols = _features(_history(days=40))
    assert len(cols) == len(set(cols)) == len(RACE_SHAPE_FEATURES) + len(DRAW_CURVE_FEATURES) + len(DRAW_STYLE_FEATURES)
    for c in cols:
        assert c in df.columns
    late = df[df.race_date >= "2023-01-20"]
    assert late[["p_lead", "pred_epf", "shape_bin", "pv_exp_lbs", "dc_edge_lbs"]].notna().all().all()
    probs = late[["p_lead", "p_prom", "p_mid", "p_rear"]].sum(axis=1)
    np.testing.assert_allclose(probs, 1.0)


# ---------------------------------------------------------------------------
# the estimators find what was planted, and nothing where nothing was
# ---------------------------------------------------------------------------

def test_draw_curve_recovers_a_planted_low_draw_bias():
    df, _ = _features(_history(days=200, draw_effect=1.5, seed=3))
    late = df[df.race_date >= "2023-05-01"]
    low = late["dc_draw_pct"] <= 0.2
    high = late["dc_draw_pct"] >= 0.8
    at = late["track"] == "chester"
    assert late.loc[at & low, "dc_edge_lbs"].mean() > 1.0
    assert late.loc[at & high, "dc_edge_lbs"].mean() < -1.0
    assert late.loc[at & low, "dc_edge_nfp"].mean() > late.loc[at & high, "dc_edge_nfp"].mean() + 0.1
    for c in ("kempton", "ascot"):     # no bias planted there
        assert abs(late.loc[(late.track == c) & low, "dc_edge_lbs"].mean()) < 0.6


def test_position_value_recovers_a_lone_leader_edge():
    df, _ = _features(_history(days=200, pace_effect=1.5, seed=5))
    late = df[df.race_date >= "2023-05-01"]
    lone = late["shape_bin"] == 1
    lead = late[["p_lead", "p_prom", "p_mid", "p_rear"]].to_numpy().argmax(axis=1) == 0
    assert late.loc[lone & lead, "pv_exp_lbs"].mean() > 1.0
    assert late.loc[lone & lead, "pv_exp_lbs"].mean() > late.loc[lone & ~lead, "pv_exp_lbs"].mean() + 1.0
    assert late.loc[lone & lead, "pv_act_nfp"].mean() > 0


def test_draw_by_style_finds_a_draw_that_only_helps_forward_runners():
    df, _ = _features(_history(days=250, forward_draw_effect=2.5, seed=11))
    late = df[(df.race_date >= "2023-06-01") & (df.track == "chester")]
    high = late["dc_draw_pct"] >= 0.75
    fwd = (late["p_lead"] + late["p_prom"]) >= 0.5
    # a high draw costs the forward runners far more than the draw-only curve says,
    # and the hold-up horses, which the planted effect spares, nothing
    gap_fwd = (late.loc[high & fwd, "dc_edge_style_lbs"] - late.loc[high & fwd, "dc_edge_lbs"]).mean()
    gap_back = (late.loc[high & ~fwd, "dc_edge_style_lbs"] - late.loc[high & ~fwd, "dc_edge_lbs"]).mean()
    assert gap_fwd < -1.0 and gap_back > 1.0
    assert late.loc[high & fwd, "dc_edge_style_lbs"].mean() < late.loc[high & ~fwd, "dc_edge_style_lbs"].mean() - 2.0
