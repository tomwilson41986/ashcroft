"""Framework §6 — pace, expected position, race shape, draw and bias.

The items under test are the ones the low/high-stall split and the 1–6 style
score cannot express:

* `epf_norm` — early position normalised by field size (§6.1), and the run-style
  profile built on it.
* `predicted_lead_prob` — a softmax over predicted early position (§6.2), so
  "wants to lead" is a probability over the field rather than a count.
* `draw_adj` — the stall after the non-runners are removed (§6.3).
* Draw bias by **expected versus actual** (§6.3). The test that matters here is
  `test_expected_vs_actual_bias_is_zero_when_low_stalls_had_the_better_horses`:
  the framework names estimating bias from raw finishing position by stall band
  as "the standard trap", and that test is the trap, sprung.
* A per-stall surface with hierarchical shrinkage, time decay and a confidence
  flag (§6.3), rather than one number for every stall in the inside half.
* The two §6.4 screens, PCS and split-half, as closed forms.

Every feature above must also be lag-safe: several of these statistics are keyed
on the course, which every runner in a race shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import spatial
from model.draw_metrics import (
    ALL_DRAW_FEATURES,
    DRAW_POST_RACE_ONLY,
    MIN_CONFIDENT_N_EFF,
    DrawMetricsEngine,
    draw_bias_split_half,
    expected_nmfp_from_ratings,
    parse_rail_move,
    parse_track_direction,
    race_lagged_decayed_mean,
)
from model.pace_metrics import (
    ALL_PACE_FEATURES,
    PACE_POST_RACE_ONLY,
    PACE_PRESSURE_THRESHOLD,
    PaceMetricsEngine,
    epf_norm_from_style,
    position_change_statistic,
    run_style_from_epf_norm,
)

LED = "made all, ran on well"
TRACKED = "chased leader, kept on"
MIDFIELD = "held up in mid-division, no extra"
REAR = "held up in rear, stayed on"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _pace_card(n_days: int = 12, races_per_day: int = 2, runners: int = 10,
               seed: int = 0, flip_last: bool = False) -> pd.DataFrame:
    """A history where each horse has a stable running style."""
    rng = np.random.default_rng(seed)
    styles = [LED, TRACKED, MIDFIELD, REAR]
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for r in range(races_per_day):
            rid = f"{day.date()}_{13 + r}:00_Ascot"
            order = rng.permutation(runners)
            last_race = (d == n_days - 1) and (r == races_per_day - 1)
            for i in range(runners):
                style = styles[i % len(styles)]
                if flip_last and last_race:
                    style = styles[(i + 2) % len(styles)]   # only today's comments move
                rows.append({
                    "raceid": rid, "race_date": day, "race_time": f"{13 + r}:00",
                    "track": "Ascot", "dist_furlongs": 6.0, "going_description": "Good",
                    "horse_name": f"h{i}", "jockey_name": f"j{i}", "trainer": f"t{i}",
                    "stall": i + 1, "number_of_runners": runners,
                    "placing_numerical": int(np.argsort(order)[i]) + 1,
                    "comment": style, "race_type": "Flat",
                    "official_rating": 80 - i,
                })
    df = pd.DataFrame(rows)
    df["NFP"] = (df["number_of_runners"] - df["placing_numerical"]) / (df["number_of_runners"] - 1)
    return df


def _draw_card(n_days: int = 60, runners: int = 10, draw_effect: float = 0.0,
               stall1_bonus: float = 0.0, rating_follows_stall: bool = True,
               seed: int = 0, flip_last: bool = False, track_direction: str = "Left",
               rail_move: str = "", going: str = "Good") -> pd.DataFrame:
    """One race a day at one course, with the draw/rating link under control.

    ``rating_follows_stall`` puts the better-rated horse in the lower stall —
    the confound §6.3 calls the standard trap. ``draw_effect`` adds a real
    advantage to low stalls on top of the ratings, and ``stall1_bonus`` gives it
    to stall 1 alone, which only a per-stall estimate can see.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        rid = f"{day.date()}_14:00_Ascot"
        stalls = np.arange(1, runners + 1)
        if rating_follows_stall:
            ratings = 90.0 - 2.0 * (stalls - 1)          # stall 1 holds the best horse
        else:
            ratings = rng.permutation(np.arange(runners, dtype=float)) * 2 + 70
        low = 1 - (stalls - 1) / (runners - 1)
        merit = ratings + draw_effect * low + stall1_bonus * (stalls == 1)
        order = np.argsort(np.argsort(-merit)) + 1       # 1 = best merit finishes first
        if flip_last and d == n_days - 1:
            order = runners + 1 - order                  # only the last race's result moves
        for i, stall in enumerate(stalls):
            rows.append({
                "raceid": rid, "race_date": day, "race_time": "14:00",
                "track": "Ascot", "dist_furlongs": 6.0, "going_description": going,
                "horse_name": f"d{d}_{stall}", "jockey_name": f"j{stall}", "trainer": f"t{stall}",
                "stall": int(stall), "number_of_runners": runners,
                "placing_numerical": int(order[i]), "official_rating": float(ratings[i]),
                "comment": MIDFIELD, "race_type": "Flat",
                "track_direction": track_direction, "rail_move": rail_move,
            })
    df = pd.DataFrame(rows)
    df["NFP"] = (df["number_of_runners"] - df["placing_numerical"]) / (df["number_of_runners"] - 1)
    return df


# ---------------------------------------------------------------------------
# §6.1 epf_norm and the run-style profile
# ---------------------------------------------------------------------------

def test_epf_norm_is_the_field_size_normalised_early_position():
    # a horse that led is position 1 in any field, so epf_norm is 0
    assert epf_norm_from_style([6.0, 6.0], [6, 20]) == pytest.approx([0.0, 0.0])
    # one that chased the leader is position 2: (2 − 1) / (N − 1), field-dependent
    assert epf_norm_from_style([5.0], [6]) == pytest.approx([1 / 5])
    assert epf_norm_from_style([5.0], [21]) == pytest.approx([1 / 20])
    # a horse held up in rear is near the back of the field whatever its size
    assert epf_norm_from_style([1.0], [8])[0] > 0.85
    # undefined without a field size
    assert np.isnan(epf_norm_from_style([6.0], [1])[0])
    assert np.isnan(epf_norm_from_style([np.nan], [10])[0])


@pytest.mark.parametrize("n", [2, 5, 8, 12, 16, 24, 40])
def test_epf_norm_is_monotone_in_style_at_every_field_size(n):
    grid = np.linspace(1.0, 6.0, 401)
    v = epf_norm_from_style(grid, np.full(grid.shape, n))
    assert (np.diff(v) <= 1e-12).all(), "a more forward comment must not score further back"
    assert v.min() >= 0.0 and v.max() <= 1.0


def test_run_style_labels_split_the_field_the_way_6_1_does():
    codes = run_style_from_epf_norm([0.0, 0.03, 0.2, 0.5, 0.8, 0.97, np.nan])
    assert list(codes[:6]) == [4, 4, 3, 2, 1, 0]        # led, led, prominent, mid, held-up, detached
    assert np.isnan(codes[6])


def test_pos_gain_is_ground_made_up():
    d = PaceMetricsEngine().calculate(_pace_card(n_days=4, runners=8))
    won = d["placing_numerical"] == 1
    held_up_winner = won & (d["epf_norm"] > 0.8)
    assert held_up_winner.any()
    assert (d.loc[held_up_winner, "pos_gain"] > 0.5).all()
    # pos_gain is exactly epf_norm minus the normalised finishing position
    expect = d["epf_norm"] - (d["placing_numerical"] - 1) / (d["number_of_runners"] - 1)
    assert np.allclose(d["pos_gain"], expect, equal_nan=True)


def test_run_style_profile_never_contains_todays_run():
    base = PaceMetricsEngine().calculate(_pace_card(flip_last=False))
    flip = PaceMetricsEngine().calculate(_pace_card(flip_last=True))
    last = sorted(base["raceid"].unique())[-1]
    key = ["raceid", "horse_name"]
    b = base[base["raceid"] == last].sort_values(key).reset_index(drop=True)
    f = flip[flip["raceid"] == last].sort_values(key).reset_index(drop=True)
    assert (b["comment"] != f["comment"]).any()                 # the fixture really changed
    for col in ("horse_epf_norm_mean", "horse_epf_norm_sd", "horse_epf_norm_recent",
                "horse_modal_style", "horse_style_mode_share", "pred_epf_norm",
                "predicted_lead_prob", "pred_pace_pressure", "pred_pace_contested"):
        assert np.allclose(b[col].fillna(-9), f[col].fillna(-9)), \
            f"{col} moved when only the predicted race's comments changed"


def test_style_sd_separates_a_one_paced_horse_from_a_versatile_one():
    card = _pace_card(n_days=14, runners=8)
    # h0 always makes the running; h1 alternates between leading and being held up
    alt = np.where(np.arange(len(card)) % 2 == 0, LED, REAR)
    card.loc[card["horse_name"] == "h1", "comment"] = alt[: (card["horse_name"] == "h1").sum()]
    d = PaceMetricsEngine().calculate(card)
    last = d[d["raceid"] == sorted(d["raceid"].unique())[-1]].set_index("horse_name")
    assert last.loc["h0", "horse_epf_norm_sd"] < last.loc["h1", "horse_epf_norm_sd"]
    assert last.loc["h0", "horse_style_inflexible"] > last.loc["h1", "horse_style_inflexible"]


# ---------------------------------------------------------------------------
# §6.2 race shape and lead probability
# ---------------------------------------------------------------------------

def test_predicted_lead_prob_is_a_probability_over_the_field():
    d = PaceMetricsEngine().calculate(_pace_card(n_days=10, runners=10))
    late = d[d["race_date"] > d["race_date"].min()]
    sums = late.groupby("raceid")["predicted_lead_prob"].sum()
    assert np.allclose(sums, 1.0)
    assert (late["predicted_lead_prob"] > 0).all()
    # and it is monotone in the predicted early position: the horse expected to
    # be most forward is the one most likely to lead
    for _, g in late.groupby("raceid"):
        if g["pred_epf_norm"].nunique() > 1:
            assert g.sort_values("pred_epf_norm")["predicted_lead_prob"].is_monotonic_decreasing
    front = late.loc[late["pred_epf_norm"].idxmin()]
    assert front["lead_prob_rank"] == 1


def test_lead_probability_replaces_a_count_of_front_runners():
    """Two confirmed leaders in a race must not both be given the lead."""
    d = PaceMetricsEngine().calculate(_pace_card(n_days=10, runners=8))
    late = d[d["race_date"] == d["race_date"].max()]
    leaders = late[late["pred_epf_norm"] < PACE_PRESSURE_THRESHOLD]
    if len(leaders) >= 2:
        assert leaders["predicted_lead_prob"].max() < 0.95
    # entropy is a race constant and lives in [0, 1]
    ent = late["lead_prob_entropy"]
    assert ent.nunique() == 1 and 0 <= ent.iloc[0] <= 1.0


def test_pace_pressure_counts_runners_predicted_to_go_forward():
    d = PaceMetricsEngine().calculate(_pace_card(n_days=10, runners=12))
    late = d[d["race_date"] == d["race_date"].max()]
    for rid, g in late.groupby("raceid"):
        assert g["pred_pace_pressure"].iloc[0] == (g["pred_epf_norm"] < PACE_PRESSURE_THRESHOLD).sum()
        assert g["pred_pace_contested"].iloc[0] == pytest.approx(g["pred_epf_norm"].std())
        share = g["pred_pace_pressure"].iloc[0] / g["number_of_runners"].iloc[0]
        cls = g["pred_race_pace_class"].iloc[0]
        assert cls == (2.0 if share >= 0.30 else 0.0 if share <= 0.12 else 1.0)


def test_pace_suit_rewards_the_lone_front_runner_and_punishes_the_crowd():
    card = _pace_card(n_days=16, runners=8)
    # every horse but h0 is a hold-up horse: h0 gets an uncontested lead
    card.loc[card["horse_name"] != "h0", "comment"] = REAR
    card.loc[card["horse_name"] == "h0", "comment"] = LED
    lone = PaceMetricsEngine().calculate(card)
    crowd_card = _pace_card(n_days=16, runners=8)
    crowd_card["comment"] = LED                      # everybody wants the front
    crowd = PaceMetricsEngine().calculate(crowd_card)
    a = lone[(lone["race_date"] == lone["race_date"].max()) & (lone["horse_name"] == "h0")]
    b = crowd[(crowd["race_date"] == crowd["race_date"].max()) & (crowd["horse_name"] == "h0")]
    assert a["pace_suit"].iloc[0] > b["pace_suit"].iloc[0]


# ---------------------------------------------------------------------------
# §6.4 the two screens
# ---------------------------------------------------------------------------

def test_position_change_statistic_matches_the_closed_form():
    df = pd.DataFrame({
        "raceid": ["r1"] * 4,
        "number_of_runners": [4] * 4,
        "early_pos_est": [1.0, 2.0, 3.0, 4.0],
        "placing_numerical": [2, 1, 4, 3],
    })
    out = position_change_statistic(df)
    assert out["pcs"].iloc[0] == pytest.approx(np.sqrt(4 * 1.0) / 4)
    # nothing moved at all -> zero churn
    df2 = df.assign(placing_numerical=[1, 2, 3, 4])
    assert position_change_statistic(df2)["pcs"].iloc[0] == 0.0


def test_pcs_baseline_uses_earlier_races_only_and_is_a_race_constant():
    base = PaceMetricsEngine().calculate(_pace_card(n_days=10, runners=8))
    assert base.groupby("raceid")["td_pcs_mean"].nunique(dropna=True).max() <= 1
    first = sorted(base["raceid"].unique())[0]
    assert base.loc[base["raceid"] == first, "td_pcs_mean"].isna().all()

    # rewriting the last race's result cannot move that race's own baseline
    card = _pace_card(n_days=10, runners=8)
    last = sorted(card["raceid"].unique())[-1]
    mask = card["raceid"] == last
    card.loc[mask, "placing_numerical"] = card.loc[mask, "placing_numerical"].values[::-1]
    flip = PaceMetricsEngine().calculate(card)
    a = base[base["raceid"] == last].sort_values("horse_name")["td_pcs_mean"].values
    b = flip[flip["raceid"] == last].sort_values("horse_name")["td_pcs_mean"].values
    assert np.allclose(a, b, equal_nan=True)


def test_split_half_screen_matches_the_closed_form():
    # even field: two halves of two, delta = stall − finishing position
    df = pd.DataFrame({
        "raceid": ["r1"] * 4, "draw_adj": [1.0, 2.0, 3.0, 4.0],
        "placing_numerical": [1, 2, 3, 4], "number_of_runners": [4] * 4,
    })
    out = draw_bias_split_half(df).iloc[0]
    assert out["dsh_front"] == 0.0 and out["dsh_back"] == 0.0 and out["dsh_level"] == 0.0
    # odd field: the median horse is reused in both halves, so each holds N/2 + 1
    df2 = pd.DataFrame({
        "raceid": ["r2"] * 5, "draw_adj": [5.0, 4.0, 3.0, 2.0, 1.0],
        "placing_numerical": [1, 2, 3, 4, 5], "number_of_runners": [5] * 5,
    })
    o2 = draw_bias_split_half(df2).iloc[0]
    assert o2["dsh_front"] == pytest.approx(np.sqrt(16 + 4 + 0) / 3)
    assert o2["dsh_back"] == pytest.approx(np.sqrt(0 + 4 + 16) / 3)
    assert o2["dsh_level"] == pytest.approx(np.sqrt(16 + 4 + 0 + 4 + 16) / 5)


def test_split_half_level_separates_a_draw_decided_race_from_a_random_one():
    ordered = pd.DataFrame({
        "raceid": ["a"] * 8, "draw_adj": np.arange(1.0, 9.0),
        "placing_numerical": np.arange(1, 9), "number_of_runners": [8] * 8,
    })
    reversed_ = ordered.assign(placing_numerical=np.arange(8, 0, -1))
    assert draw_bias_split_half(ordered)["dsh_level"].iloc[0] < \
        draw_bias_split_half(reversed_)["dsh_level"].iloc[0]


def test_split_half_course_baseline_is_lag_safe():
    d = DrawMetricsEngine().calculate(_draw_card(n_days=30))
    assert d.groupby("raceid")["td_draw_screen"].nunique(dropna=True).max() <= 1
    first = sorted(d["raceid"].unique())[0]
    assert d.loc[d["raceid"] == first, "td_draw_screen"].isna().all()


# ---------------------------------------------------------------------------
# §6.3 draw_adj, expected-vs-actual, shrinkage, decay, confidence
# ---------------------------------------------------------------------------

def test_draw_adj_renumbers_the_stalls_after_the_non_runners():
    df = _draw_card(n_days=3, runners=12)
    # three withdrawals inside stall 8, plus one outside it
    df = df[~df["stall"].isin([2, 3, 5, 11])].copy()
    df["number_of_runners"] = 8
    d = DrawMetricsEngine().calculate(df)
    row = d[d["stall"] == 8].iloc[0]
    assert row["draw_adj"] == 5                       # stalls 1,4,6,7 are inside it
    assert row["draw_vacancy_inside"] == 3
    assert row["draw_pct_adj"] == pytest.approx(4 / 7)
    assert row["draw_relative"] > row["draw_pct_adj"]  # the card number overstates how wide it is
    assert d.groupby("raceid")["draw_adj"].min().eq(1).all()


def test_expected_vs_actual_bias_is_zero_when_low_stalls_had_the_better_horses():
    """§6.3's standard trap, sprung.

    Every race here is decided entirely by the ratings, and the best-rated horse
    is always drawn lowest. There is no draw bias whatsoever. A bias estimated
    from raw finishing position by stall band says the inside is worth half a
    field; the expected-versus-actual estimate says, correctly, nothing.
    """
    d = DrawMetricsEngine().calculate(_draw_card(n_days=80, runners=10,
                                                 rating_follows_stall=True, draw_effect=0.0))
    settled = d[d["race_date"] > d["race_date"].min() + pd.Timedelta(days=20)]

    # the naive construction is fooled, and badly
    assert settled["td_draw_bias"].mean() > 0.3
    low = settled[settled["draw_relative"] <= 0.5]
    assert low["draw_bias_alignment"].mean() > 0.3

    # the residual against the rating-implied finish is exactly zero, and so is
    # every cell mean built from it
    assert np.nanmax(np.abs(settled["draw_resid"])) < 1e-9
    assert np.nanmax(np.abs(settled["draw_bias_ev"])) < 1e-9
    assert np.nanmax(np.abs(settled["draw_bias_ev_stall"])) < 1e-9
    assert np.nanmax(np.abs(settled["draw_bias_gradient"].fillna(0))) < 1e-6


def test_expected_vs_actual_bias_still_finds_a_real_draw_advantage():
    """Same machinery, a real effect: ratings say nothing about the stall, and
    low stalls outrun their ratings."""
    d = DrawMetricsEngine().calculate(_draw_card(n_days=200, runners=10, seed=3,
                                                 rating_follows_stall=False, draw_effect=12.0))
    late = d[d["race_date"] > d["race_date"].min() + pd.Timedelta(days=100)]
    by_stall = late.groupby("draw_adj")["draw_bias_ev"].mean()
    assert by_stall.loc[1] > 0.05 and by_stall.loc[10] < -0.05
    assert by_stall.loc[1] > by_stall.loc[5] > by_stall.loc[10]
    # the surface is tilted toward the inside, and the race-level gradient says so
    assert late["draw_bias_gradient"].mean() < 0


def test_the_surface_resolves_single_stalls_not_halves_of_the_field():
    """Stall 1 and stall 3 are both in the inside half; only stall 1 is favoured."""
    d = DrawMetricsEngine().calculate(_draw_card(n_days=250, runners=12, seed=5,
                                                 rating_follows_stall=False,
                                                 stall1_bonus=25.0))
    late = d[d["race_date"] > d["race_date"].min() + pd.Timedelta(days=120)]
    per_stall = late.groupby("draw_adj")["draw_bias_ev"].mean()
    assert per_stall.loc[1] > per_stall.loc[3] + 0.05
    assert abs(per_stall.loc[3] - per_stall.loc[5]) < abs(per_stall.loc[1] - per_stall.loc[3])
    # the old feature cannot tell them apart: it is one number for the whole race
    assert late.groupby("raceid")["td_draw_bias"].nunique(dropna=True).max() <= 1


def test_thin_cells_are_shrunk_hard_and_declared_unconfident():
    d = DrawMetricsEngine().calculate(_draw_card(n_days=14, runners=10, seed=7,
                                                 rating_follows_stall=False, draw_effect=12.0))
    late = d[d["race_date"] == d["race_date"].max()]
    assert (late["draw_bias_confident"] == 0).all()             # nowhere near enough history
    assert late["draw_bias_stall_n_eff"].max() < MIN_CONFIDENT_N_EFF
    # the estimate exists but is pulled far toward zero
    assert late["draw_bias_ev"].abs().max() < 0.25

    plenty = DrawMetricsEngine().calculate(_draw_card(n_days=400, runners=10, seed=7,
                                                      rating_follows_stall=False, draw_effect=12.0))
    tail = plenty[plenty["race_date"] == plenty["race_date"].max()]
    assert (tail["draw_bias_confident"] == 1).all()
    assert tail["draw_bias_ev"].abs().max() > late["draw_bias_ev"].abs().max()


def test_cell_means_are_time_decayed_and_carry_an_effective_sample_size():
    days = [0, 1, 2, 1000]
    df = pd.DataFrame({
        "raceid": [f"r{i}" for i in range(4)],
        "race_date": [pd.Timestamp("2020-01-01") + pd.Timedelta(days=x) for x in days],
        "race_time": ["14:00"] * 4, "track": ["Ascot"] * 4,
        "v": [1.0, 1.0, 1.0, 0.0],
    })
    slow, n_slow = race_lagged_decayed_mean(df, "track", "v", halflife_days=1e9)
    fast, n_fast = race_lagged_decayed_mean(df, "track", "v", halflife_days=30.0)
    # the last row looks back on three 1.0s and no 0.0s either way
    assert slow.iloc[3] == pytest.approx(1.0) and fast.iloc[3] == pytest.approx(1.0)
    # but with a 30-day half-life they are ancient history: n_eff collapses
    assert n_slow.iloc[3] == pytest.approx(3.0)
    assert n_fast.iloc[3] < 0.01
    # and nothing ever sees its own race
    assert np.isnan(slow.iloc[0]) and n_slow.iloc[0] == 0.0


def test_the_decayed_mean_matches_a_brute_force_weighted_mean():
    """The fast path is flat numpy; this is the definition, written out slowly."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(50):
        day = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
        key = rng.choice(["a", "b"])
        for _ in range(int(rng.integers(2, 5))):
            rows.append({"raceid": f"r{i}", "race_date": day, "race_time": "14:00",
                         "k": key, "v": float(rng.normal())})
    d = pd.DataFrame(rows).sort_values(["race_date", "raceid"]).reset_index(drop=True)
    halflife = 200.0
    mean, n_eff = race_lagged_decayed_mean(d, "k", "v", halflife_days=halflife)

    per_race = d.groupby(["k", "raceid"]).agg(s=("v", "sum"), c=("v", "size"),
                                              day=("race_date", "min")).reset_index()
    for i in range(len(d)):
        row = d.iloc[i]
        me = per_race[(per_race["k"] == row["k"]) & (per_race["raceid"] == row["raceid"])].iloc[0]
        prior = per_race[(per_race["k"] == row["k"]) & (per_race["day"] < me["day"])]
        if prior.empty:
            assert np.isnan(mean.iloc[i]) and n_eff.iloc[i] == 0
            continue
        w = 2.0 ** (-(me["day"] - prior["day"]).dt.days / halflife)
        assert mean.iloc[i] == pytest.approx((w * prior["s"]).sum() / (w * prior["c"]).sum())
        assert n_eff.iloc[i] == pytest.approx((w * prior["c"]).sum())


def test_the_decayed_mean_is_the_canonical_lag_safe_mean_when_nothing_decays():
    """The fast path must not quietly disagree with model/lagsafe.py."""
    from model.lagsafe import race_lagged_expanding_mean

    d = _pace_card(n_days=9, races_per_day=2, runners=6)
    d["v"] = np.arange(len(d), dtype=float)
    for key, min_races in (("track", 0), ("track", 3), ("jockey_name", 0)):
        fast, _ = race_lagged_decayed_mean(d, key, "v", halflife_days=np.inf, min_races=min_races)
        canon = race_lagged_expanding_mean(d, key, "v", min_races=max(min_races, 1))
        if min_races == 0:                       # the canonical helper has no zero setting
            canon = race_lagged_expanding_mean(d, key, "v")
        assert np.allclose(fast.fillna(-9), canon.fillna(-9)), key


def test_a_reversed_draw_bias_is_followed_rather_than_averaged_away():
    old = _draw_card(n_days=200, runners=10, seed=11, rating_follows_stall=False, draw_effect=14.0)
    new = _draw_card(n_days=200, runners=10, seed=12, rating_follows_stall=False, draw_effect=-14.0)
    new["race_date"] = new["race_date"] + pd.Timedelta(days=400)
    new["raceid"] = new["race_date"].astype(str) + "_14:00_Ascot"
    new["horse_name"] = new["horse_name"] + "_b"
    d = DrawMetricsEngine().calculate(pd.concat([old, new], ignore_index=True))
    tail = d[d["race_date"] == d["race_date"].max()]
    inside = tail[tail["draw_adj"] == 1]["draw_bias_ev"].iloc[0]
    assert inside < 0, "the two-year half-life must let the new regime overturn the old one"


def test_rail_movement_and_handedness_are_read_and_gate_confidence():
    handed = parse_track_direction(["Left", "right-handed", "Straight", ""])
    assert list(handed[:3]) == [1.0, -1.0, 0.0] and np.isnan(handed.iloc[3])
    yards, moved = parse_rail_move(["", "rail moved out 4 yards", "rail in 6yds", "no change"])
    assert list(yards) == [0.0, 4.0, -6.0, 0.0]
    assert list(moved) == [0.0, 1.0, 1.0, 0.0]

    left = DrawMetricsEngine().calculate(_draw_card(n_days=400, runners=10, seed=9,
                                                    rating_follows_stall=False, draw_effect=12.0))
    moved_rail = DrawMetricsEngine().calculate(
        _draw_card(n_days=400, runners=10, seed=9, rating_follows_stall=False,
                   draw_effect=12.0, rail_move="rail moved out 12 yards",
                   track_direction="Right"))
    a = left[left["race_date"] == left["race_date"].max()]
    b = moved_rail[moved_rail["race_date"] == moved_rail["race_date"].max()]
    assert (a["draw_bias_confident"] == 1).all()
    assert (b["draw_bias_confident"] == 0).all(), "a moved rail cannot leave the bias confident"
    assert b["draw_bias_ev_adj"].abs().max() < a["draw_bias_ev_adj"].abs().max()
    # handedness is signed, so the interaction can carry opposite signs per course
    assert np.allclose(a["draw_x_handed"], a["draw_pct_adj"])
    assert np.allclose(b["draw_x_handed"], -b["draw_pct_adj"])


def test_expected_nmfp_needs_a_rating_and_says_so():
    df = _draw_card(n_days=4, runners=10)
    exp, src = expected_nmfp_from_ratings(df)
    assert exp.notna().all() and (src == "official_rating").all()
    blank = df.assign(official_rating=np.nan)
    exp2, src2 = expected_nmfp_from_ratings(blank)
    assert exp2.isna().all(), "no rating means no expectation, not a guessed one"


def test_draw_interactions_are_the_products_6_5_asks_for():
    card = _pace_card(n_days=12, runners=10)
    card["track_direction"] = "Left"
    card["rail_move"] = ""
    d = DrawMetricsEngine().calculate(PaceMetricsEngine().calculate(card))
    late = d[d["race_date"] == d["race_date"].max()]
    assert np.allclose(late["draw_x_style"], late["draw_pct_adj"] * late["horse_epf_norm_mean"])
    assert np.allclose(late["draw_x_pace"], late["draw_pct_adj"] * late["pred_pace_pressure"])
    assert np.allclose(late["draw_x_fieldsize"], late["draw_pct_adj"] * late["number_of_runners"])
    assert np.allclose(late["draw_pos_fit"],
                       late["draw_bias_ev_adj"] * (1 - 2 * late["pred_epf_norm"]))


# ---------------------------------------------------------------------------
# lag safety of everything new
# ---------------------------------------------------------------------------

def test_the_bias_surface_cannot_see_the_race_it_describes():
    base = DrawMetricsEngine().calculate(_draw_card(n_days=120, seed=2, draw_effect=8.0,
                                                    rating_follows_stall=False))
    flip = DrawMetricsEngine().calculate(_draw_card(n_days=120, seed=2, draw_effect=8.0,
                                                    rating_follows_stall=False, flip_last=True))
    last = base["race_date"].max()
    b = base[base["race_date"] == last].sort_values("stall")
    f = flip[flip["race_date"] == last].sort_values("stall")
    assert (b["placing_numerical"].values != f["placing_numerical"].values).any()
    for col in ("draw_bias_ev", "draw_bias_ev_stall", "draw_bias_ev_course", "draw_bias_ev_adj",
                "draw_bias_ev_n_eff", "draw_bias_confident", "draw_bias_gradient",
                "td_draw_screen", "td_dsh_level", "draw_pos_fit"):
        assert np.allclose(b[col].fillna(-9).values, f[col].fillna(-9).values), \
            f"{col} changed when only the predicted race's result changed"


@pytest.mark.parametrize("col", ["td_pcs_mean", "track_pcs_mean", "pred_pace_pressure",
                                 "pred_pace_contested", "lead_prob_entropy"])
def test_new_race_level_pace_columns_do_not_vary_inside_a_race(col):
    d = PaceMetricsEngine().calculate(_pace_card(n_days=10, runners=8))
    assert d.groupby("raceid")[col].nunique(dropna=True).max() <= 1


@pytest.mark.parametrize("col", ["td_draw_screen", "track_draw_screen", "td_dsh_level",
                                 "draw_bias_gradient", "draw_bias_spread", "rating_slope"])
def test_new_race_level_draw_columns_do_not_vary_inside_a_race(col):
    d = DrawMetricsEngine().calculate(_draw_card(n_days=40))
    assert d.groupby("raceid")[col].nunique(dropna=True).max() <= 1


def test_no_post_race_column_is_exported_as_a_feature():
    from model.custom_metrics import POST_RACE_ONLY

    assert not (set(ALL_PACE_FEATURES) & PACE_POST_RACE_ONLY)
    assert not (set(ALL_DRAW_FEATURES) & DRAW_POST_RACE_ONLY)
    assert not (set(ALL_PACE_FEATURES) & POST_RACE_ONLY)
    assert not (set(ALL_DRAW_FEATURES) & POST_RACE_ONLY)
    assert {"epf_norm", "pos_gain", "run_style"} <= PACE_POST_RACE_ONLY
    assert {"draw_resid", "nmfp_actual"} <= DRAW_POST_RACE_ONLY


def test_every_exported_feature_is_actually_produced():
    card = _pace_card(n_days=8, runners=10)
    card["track_direction"] = "Left"
    card["rail_move"] = ""
    d = DrawMetricsEngine().calculate(PaceMetricsEngine().calculate(card))
    missing = [c for c in list(ALL_PACE_FEATURES) + list(ALL_DRAW_FEATURES) if c not in d.columns]
    assert missing == []


# ---------------------------------------------------------------------------
# the GP per-stall surface in model/spatial.py
# ---------------------------------------------------------------------------

def test_smooth_draw_bias_honours_an_as_of_cut():
    rng = np.random.default_rng(1)
    n = 4000
    d = pd.DataFrame({
        "track": ["Chester"] * n, "dist_furlongs": 5.0, "number_of_runners": 12,
        "stall": rng.integers(1, 13, n),
        "race_date": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1400, n), "D"),
    })
    early = d["race_date"] < pd.Timestamp("2022-01-01")
    # the bias reverses in 2022: a surface cut at 2022 must not know that
    p_low = np.where(early, 0.20 - 0.012 * d["stall"], 0.02 + 0.012 * d["stall"])
    d["won"] = (rng.random(n) < p_low).astype(int)
    cut = spatial.smooth_draw_bias(d, "chester", 5, as_of="2022-01-01")
    assert cut["smooth_mean"].iloc[0] > cut["smooth_mean"].iloc[-1]
    assert cut["n"].sum() < len(d)
    whole = spatial.smooth_draw_bias(d, "chester", 5)
    assert whole["n"].sum() == len(d)


def test_walk_forward_stall_surface_is_lag_safe_and_shrinks_thin_cells():
    base = _draw_card(n_days=900, runners=10, seed=4, rating_follows_stall=False, draw_effect=10.0)
    base["draw_adj"] = base["stall"]
    base["draw_resid"] = (base["number_of_runners"] - base["placing_numerical"]) / \
        (base["number_of_runners"] - 1) - 0.5
    out = spatial.walk_forward_stall_surface(base, keys=["track"], min_history_rows=200)
    scored = out["gp_draw_bias"].notna()
    assert scored.any()
    # the first year has no history to fit on
    first_year = base["race_date"] < pd.Timestamp("2025-01-01")
    assert not out.loc[first_year & (base["race_date"] < pd.Timestamp("2024-06-01")),
                       "gp_draw_bias"].notna().any()
    # inside beats outside once there is history
    tail = base.assign(v=out["gp_draw_bias"])[scored]
    assert tail.groupby("draw_adj")["v"].mean().loc[1] > tail.groupby("draw_adj")["v"].mean().loc[10]

    # changing only the final year cannot move an earlier year's surface
    changed = base.copy()
    late = changed["race_date"] >= pd.Timestamp("2026-01-01")
    changed.loc[late, "draw_resid"] = -changed.loc[late, "draw_resid"]
    out2 = spatial.walk_forward_stall_surface(changed, keys=["track"], min_history_rows=200)
    early = base["race_date"] < pd.Timestamp("2026-01-01")
    assert np.allclose(out.loc[early, "gp_draw_bias"].fillna(-9),
                       out2.loc[early, "gp_draw_bias"].fillna(-9))
