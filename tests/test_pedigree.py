"""
Tests for the Part 5 connection and pedigree suites (model/pedigree.py and the
rebuilt model/connections.py).

The two things worth testing here are the two things that are easy to get
wrong:

1. **Lag safety.** A sire, a dam, a trainer and a jockey can all have several
   runners in the same race and several more on the same card. Every test that
   matters below changes a race's *result* and asserts that the features for
   that race do not move.
2. **Shrinkage and residuals.** §5.3 says the alpha is in the aptitude
   *residual*, not the conditional level. `test_aptitude_is_a_residual_not_a_level`
   is the test that a levels implementation fails.
"""

import numpy as np
import pandas as pd
import pytest

from model.connections import (aptitude_residual, add_connection_features, is_black_type, is_handicap,
                               going_group, shrink)
from model.custom_metrics import POST_RACE_ONLY
from model.pedigree import add_pedigree_features, class_rating

SIRE_COLS = ["sire_nmfp_shrunk", "sire_sr_shrunk", "sire_going_apt", "sire_dist_apt", "sire_surface_apt",
             "sire_2yo_apt", "sire_first_time_out_apt", "sire_improvement", "sire_elite_share",
             "sire_class_rating", "sire_track_strength", "nick_apt", "sib_mean_nmfp",
             "dam_best_progeny_rating", "family_bt_density"]


def _race(day, time, runners, track="Ascot", going="Good", dist=6.0, surface="Turf", race_name="Handicap"):
    """One race: `runners` is a list of (horse, sire, dam, damsire, position) tuples."""
    n = len(runners)
    return [dict(race_date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=day), race_time=time,
                 track=track, raceid=f"{day}_{time}_{track}", horse_name=h, stallion=s, dam=dm,
                 dam_stallion=ds, placing_numerical=float(p), number_of_runners=float(n),
                 going_description=going, dist_furlongs=dist, surface_type=surface,
                 official_rating=80.0, horse_age=3.0, prize_money=10000.0, race_class="Class 4",
                 race_name=race_name, race_type="Flat", trainer=f"TR{i % 3}", jockey_name=f"JK{i % 3}")
            for i, (h, s, dm, ds, p) in enumerate(runners)]


def _fill(prefix, positions, dam="filler_dam"):
    return [(f"{prefix}_{p}", f"S_{prefix}_{p}", f"{dam}_{p}", "DS_filler", p) for p in positions]


# ---------------------------------------------------------------------------
# The credibility formulas, by hand
# ---------------------------------------------------------------------------

def test_shrink_is_the_credibility_mean():
    # 6 runs totalling 3.0 (mean 0.5), prior 0.1, k = 4:  (3 + 4·0.1) / (6 + 4)
    assert shrink(3.0, 6.0, 0.1, 4.0) == pytest.approx(0.34)
    assert shrink(0.0, 0.0, 0.1, 4.0) == pytest.approx(0.1)        # no evidence -> the prior
    assert shrink(300.0, 600.0, 0.1, 4.0) == pytest.approx(0.5, abs=3e-3)   # lots -> the sample


def test_aptitude_residual_removes_the_entity_level():
    # cell of 6 runs at mean 0.5 against an overall of 0.1, k = 4:
    #   (3 + 0.4)/10 − 0.1 = 0.24 = 6/10 · (0.5 − 0.1)
    assert aptitude_residual(3.0, 6.0, 0.1, 4.0) == pytest.approx(0.24)
    assert aptitude_residual(3.0, 6.0, 0.1, 4.0) == pytest.approx(6 / 10 * (0.5 - 0.1))
    assert aptitude_residual(0.0, 0.0, 0.9, 4.0) == 0.0            # empty cell -> no aptitude
    # a cell that matches the entity's own level carries no information
    assert aptitude_residual(0.5 * 20, 20.0, 0.5, 4.0) == pytest.approx(0.0)


def test_class_rating_matches_the_published_formula():
    # LN(geomean WPMCV) · LN(geomean WPMCV · %RB²/100), %RB² passed as a fraction
    got = class_rating([10000.0], [0.36])[0]
    assert got == pytest.approx(np.log(10000.0) * np.log(3600.0))
    assert np.isnan(class_rating([0.0], [0.36])[0])                # log of nothing
    assert np.isnan(class_rating([10000.0], [np.nan])[0])


def test_race_classification_helpers():
    d = pd.DataFrame({"race_name": ["Group 2 Sprint Stakes", "Novice Handicap", "Listed Fillies Stakes", "Maiden"],
                      "race_class": ["Class 1", "Class 4", "Class 1", "Class 5"],
                      "major": ["", "", "", ""]})
    assert list(is_black_type(d)) == [True, False, True, False]
    assert list(is_handicap(d)) == [False, True, False, False]
    assert list(going_group(pd.Series(["Good To Firm", "Heavy", "Good", "Standard"]))) == ["firm", "soft", "good", "good"]


# ---------------------------------------------------------------------------
# Lag safety: the same race, and the same card
# ---------------------------------------------------------------------------

def _two_runners_per_sire_card(flip_last=False):
    """Sire "BIG" has two runners in every race; the last race can be re-run."""
    rows = []
    for day in range(1, 9):
        for t, track in (("13:00", "Ascot"), ("15:00", "Ascot")):
            last = (day == 8) and (t == "15:00")
            a, b = (2, 1) if (flip_last and last) else (1, 2)
            runners = [(f"big_a{day}{t}", "BIG", f"dam_a{day}", "DSX", a),
                       (f"big_b{day}{t}", "BIG", f"dam_b{day}", "DSX", b)]
            runners += _fill(f"f{day}{t}", [3, 4, 5, 6])
            rows += _race(day, t, runners, track=track)
    return pd.DataFrame(rows)


def test_a_sire_with_two_runners_in_one_race_does_not_leak():
    base, _ = add_pedigree_features(_two_runners_per_sire_card(False))
    flip, _ = add_pedigree_features(_two_runners_per_sire_card(True))
    last = base["raceid"].iloc[-1]
    b = base[(base["raceid"] == last) & (base["stallion"] == "BIG")].sort_values("horse_name")
    f = flip[(flip["raceid"] == last) & (flip["stallion"] == "BIG")].sort_values("horse_name")
    assert len(b) == 2 and len(f) == 2
    # the fixture really did change the race
    assert not np.array_equal(b["placing_numerical"].values, f["placing_numerical"].values)
    for col in SIRE_COLS:
        assert col in b.columns
        # both of the sire's runners see exactly the same figure ...
        assert b[col].nunique(dropna=False) == 1, f"{col} varies between two runners of one sire in one race"
        # ... and it is the same figure whichever of them won
        assert np.allclose(b[col].fillna(-999).values, f[col].fillna(-999).values), \
            f"{col} moved when only the predicted race's result changed"


def test_a_sire_does_not_see_the_rest_of_todays_card():
    """A dam's or sire's other progeny running earlier on the same card must not
    contribute to a horse's figure for this race."""
    base = _two_runners_per_sire_card(False)
    flip = base.copy()
    first_of_last_day = (flip["race_date"] == flip["race_date"].max()) & (flip["race_time"] == "13:00")
    # reverse the earlier race on the final card
    flip.loc[first_of_last_day, "placing_numerical"] = (7.0 - flip.loc[first_of_last_day, "placing_numerical"])
    out_b, _ = add_pedigree_features(base)
    out_f, _ = add_pedigree_features(flip)
    late = (out_b["race_date"] == out_b["race_date"].max()) & (out_b["race_time"] == "15:00")
    for col in SIRE_COLS:
        assert np.allclose(out_b.loc[late, col].fillna(-999).values, out_f.loc[late, col].fillna(-999).values), \
            f"{col} in the 15:00 saw the 13:00 on the same card"


def test_a_trainer_with_two_runners_in_one_race_does_not_leak():
    def card(flip=False):
        rows = []
        for day in range(1, 10):
            runners = [(f"t_a{day}", "S1", f"d_a{day}", "DS1", 2 if flip and day == 9 else 1),
                       (f"t_b{day}", "S2", f"d_b{day}", "DS2", 1 if flip and day == 9 else 2)]
            runners += _fill(f"f{day}", [3, 4, 5, 6])
            r = _race(day, "13:00", runners)
            for row in r[:2]:
                row["trainer"] = "TR_double"      # one yard, two runners, every race
                row["jockey_name"] = "JK_double"
            rows += r
        return pd.DataFrame(rows)

    base, feats = add_connection_features(card(False), nmfp_col="nmfp_missing")
    flip, _ = add_connection_features(card(True), nmfp_col="nmfp_missing")
    last = base["raceid"].iloc[-1]
    b = base[(base["raceid"] == last) & (base["trainer"] == "TR_double")].sort_values("horse_name")
    f = flip[(flip["raceid"] == last) & (flip["trainer"] == "TR_double")].sort_values("horse_name")
    assert len(b) == 2
    for col in ("trainer_runs", "trainer_sr_shrunk", "trainer_wins", "trainer_place_rate_shrunk",
                "trainer_prb2_fsa_shrunk", "tj_sr_shrunk", "jockey_sr_shrunk", "trainer_lr_place_rating"):
        assert b[col].nunique(dropna=False) == 1, f"{col} varies between two runners of one yard in one race"
        assert np.allclose(b[col].fillna(-999).values, f[col].fillna(-999).values), \
            f"{col} moved when only the predicted race's result changed"


def test_no_feature_moves_when_the_last_race_is_re_run():
    """The broad version: scramble the final race's finishing order and nothing
    either module produces for that race may change."""
    base = _two_runners_per_sire_card(False)
    scram = base.copy()
    last = scram["raceid"] == scram["raceid"].iloc[-1]
    scram.loc[last, "placing_numerical"] = scram.loc[last, "placing_numerical"].values[::-1]

    pb, pf = add_pedigree_features(base)
    ps, _ = add_pedigree_features(scram)
    cb, cf = add_connection_features(base, nmfp_col="none")
    cs, _ = add_connection_features(scram, nmfp_col="none")
    for out_a, out_b, feats in ((pb, ps, pf), (cb, cs, cf)):
        m = out_a["raceid"] == out_a["raceid"].iloc[-1]
        for col in feats:
            a = pd.to_numeric(out_a.loc[m, col], errors="coerce").fillna(-999).values
            z = pd.to_numeric(out_b.loc[m, col], errors="coerce").fillna(-999).values
            assert np.allclose(a, z), f"{col} depends on the result of the race it is predicting"


def test_no_post_race_column_is_shipped_as_a_feature():
    _, pf = add_pedigree_features(_two_runners_per_sire_card())
    _, cf = add_connection_features(_two_runners_per_sire_card(), nmfp_col="none")
    assert not (set(pf) | set(cf)) & set(POST_RACE_ONLY)


# ---------------------------------------------------------------------------
# §5.3: the aptitude residual is the point, not the conditional level
# ---------------------------------------------------------------------------

def _going_aptitude_card(days=40):
    """`UNIFORM` wins on every going. `SPECIALIST` wins on soft and is beaten
    out of sight on good. Their *levels* say UNIFORM is the better sire on soft;
    their *residuals* say only SPECIALIST has a soft aptitude."""
    rows = []
    for day in range(1, days + 1):
        soft = day % 2 == 0
        going = "Soft" if soft else "Good"
        runners = [(f"u{day}", "UNIFORM", f"du{day}", "DSU", 1),
                   (f"s{day}", "SPECIALIST", f"ds{day}", "DSS", 2 if soft else 6)]
        runners += _fill(f"f{day}", [3, 4, 5, 6] if soft else [2, 3, 4, 5])
        rows += _race(day, "13:00", runners, going=going)
    return pd.DataFrame(rows)


def test_aptitude_is_a_residual_not_a_level():
    out, _ = add_pedigree_features(_going_aptitude_card())
    soft_last = out[(out["going_description"] == "Soft")].iloc[-8:]
    u = soft_last[soft_last["stallion"] == "UNIFORM"].iloc[0]
    s = soft_last[soft_last["stallion"] == "SPECIALIST"].iloc[0]

    # the level: UNIFORM is simply the better sire, on soft and overall
    assert u["sire_nmfp_shrunk"] > s["sire_nmfp_shrunk"]
    u_soft_level = u["sire_going_apt"] + u["sire_nmfp_shrunk"]
    s_soft_level = s["sire_going_apt"] + s["sire_nmfp_shrunk"]
    assert u_soft_level > s_soft_level          # a *levels* feature ranks UNIFORM first

    # the residual: UNIFORM has no going preference, SPECIALIST has a large one.
    # UNIFORM's soft mean *is* its overall mean, so the residual is exactly zero.
    assert u["sire_going_apt"] == pytest.approx(0.0, abs=1e-9)
    assert s["sire_going_apt"] > 0.15
    assert s["sire_going_apt"] > u["sire_going_apt"]     # the residual reverses the ranking


def test_aptitude_cell_with_no_history_is_zero_not_a_guess():
    d = _going_aptitude_card(days=6)
    out, _ = add_pedigree_features(d)
    first = out.sort_values(["race_date", "race_time"]).iloc[0]
    assert first["sire_going_apt"] == pytest.approx(0.0)
    assert first["sire_prog_runs"] == 0.0


def test_elite_share_counts_runners_that_beat_78pc_of_the_field():
    rows = []
    for day in range(1, 21):
        runners = [(f"e{day}", "ELITE", f"de{day}", "DS1", 1),       # %RB = 1.0  -> elite
                   (f"m{day}", "MODEST", f"dm{day}", "DS1", 4)]      # %RB = 0.4  -> not
        runners += _fill(f"f{day}", [2, 3, 5, 6])
        rows += _race(day, "13:00", runners)
    out, _ = add_pedigree_features(pd.DataFrame(rows), k_sr=5.0)
    last = out.iloc[-6:]
    assert last[last["stallion"] == "ELITE"]["sire_elite_share"].iloc[0] > 0.7
    assert last[last["stallion"] == "MODEST"]["sire_elite_share"].iloc[0] < 0.25


# ---------------------------------------------------------------------------
# §5.4: siblings, and the horse is not its own sibling
# ---------------------------------------------------------------------------

def _sibling_card():
    """Dam D has two progeny: `good` always wins, `bad` always finishes last.
    They never run in the same race."""
    rows = []
    for day in range(1, 12):
        if day % 2:
            runners = [("good", "S1", "D", "DS1", 1)] + _fill(f"f{day}", [2, 3, 4, 5, 6])
        else:
            runners = [("bad", "S1", "D", "DS1", 6)] + _fill(f"f{day}", [1, 2, 3, 4, 5])
        rows += _race(day, "13:00", runners)
    d = pd.DataFrame(rows)
    d.loc[d["horse_name"] == "good", "official_rating"] = 110.0
    d.loc[d["horse_name"] == "bad", "official_rating"] = 85.0
    return d


def test_sibling_features_exclude_the_horse_itself():
    out, _ = add_pedigree_features(_sibling_card(), k_dam=10.0)
    last_good = out[out["horse_name"] == "good"].sort_values("race_date").iloc[-1]
    # `bad` has run 5 times before that day, always last of 6:
    #   nmfp = (6 + 1 − 12) / sqrt(3·35) = −0.487950
    bad_nmfp = (6 + 1 - 2 * 6) / np.sqrt(3 * (36 - 1))
    assert last_good["sib_count"] == 1.0
    assert last_good["dam_runners"] == 5.0
    assert last_good["sib_mean_nmfp"] == pytest.approx((5 * bad_nmfp + 10 * 0.0) / (5 + 10), abs=1e-6)
    assert last_good["sib_mean_nmfp"] < 0        # the horse's own winning form is not in there


def test_dam_best_progeny_rating_is_the_best_sibling_not_the_horse():
    out, _ = add_pedigree_features(_sibling_card())
    good = out[out["horse_name"] == "good"].sort_values("race_date").iloc[-1]
    bad = out[out["horse_name"] == "bad"].sort_values("race_date").iloc[-1]
    assert good["dam_best_progeny_rating"] == 85.0     # its sibling's mark, not its own 110
    assert bad["dam_best_progeny_rating"] == 110.0


def test_dam_own_racing_record_joins_the_mare_by_name():
    rows = []
    for day in range(1, 7):        # the mare races ...
        rows += _race(day, "13:00", [("Mare", "S9", "GrannyDam", "DS9", 1)] + _fill(f"m{day}", [2, 3, 4, 5, 6]))
    for day in range(40, 43):      # ... and later her foal runs
        rows += _race(day, "13:00", [(f"foal{day}", "S1", "Mare", "DS1", 3)] + _fill(f"p{day}", [1, 2, 4, 5, 6]))
    out, _ = add_pedigree_features(pd.DataFrame(rows))
    foal = out[out["horse_name"].str.startswith("foal")].iloc[-1]
    assert foal["dam_own_runs"] == 6.0
    assert foal["dam_own_nmfp"] > 0.4          # she won all six
    assert foal["dam_own_best_or"] == 80.0


# ---------------------------------------------------------------------------
# §5.4: the nick gets the hardest shrinkage in the suite
# ---------------------------------------------------------------------------

def test_nick_is_shrunk_hardest_and_toward_the_sires_own_mean():
    rows = []
    for day in range(1, 31):
        # the cross S_HOT x DS_HOT wins; the same sire's other runners do not
        runners = [(f"cross{day}", "S_HOT", f"dc{day}", "DS_HOT", 1),
                   (f"other{day}", "S_HOT", f"do{day}", "DS_COLD", 5)]
        runners += _fill(f"f{day}", [2, 3, 4, 6])
        rows += _race(day, "13:00", runners)
    out, feats = add_pedigree_features(pd.DataFrame(rows), k_nick=60.0)
    row = out[out["dam_stallion"] == "DS_HOT"].sort_values("race_date").iloc[-1]
    n = row["nick_runs"]
    win_nmfp = (6 + 1 - 2) / np.sqrt(3 * 35)          # the cross always wins
    fifth_nmfp = (6 + 1 - 10) / np.sqrt(3 * 35)       # the sire's other runners are 5th
    sire_raw = (win_nmfp + fifth_nmfp) / 2            # so the sire's own mean is the midpoint
    expected = n / (n + 60.0) * (win_nmfp - sire_raw)
    assert n == 29.0
    assert row["nick_apt"] == pytest.approx(expected, abs=1e-6)
    # hard shrinkage: 29 runs against k = 60 keeps under a third of the raw gap
    assert 0 < row["nick_apt"] < 0.35 * (win_nmfp - sire_raw)


def test_median_handicap_debut_or_needs_enough_debutants():
    rows = []
    for i, mark in enumerate((60.0, 70.0, 80.0, 90.0)):        # January: four handicap debuts
        r = _race(1 + i, "13:00", [(f"hd{i}", "S_HD", f"dh{i}", "DS1", 1)] + _fill(f"f{i}", [2, 3, 4, 5, 6]),
                  race_name="Novice Handicap")
        r[0]["official_rating"] = mark
        rows += r
    rows += _race(70, "13:00", [("later", "S_HD", "dlater", "DS1", 3)] + _fill("g", [1, 2, 4, 5, 6]),
                  race_name="Maiden Stakes")
    d = pd.DataFrame(rows)
    loose, _ = add_pedigree_features(d, min_hcap_debuts=3)
    strict, _ = add_pedigree_features(d, min_hcap_debuts=30)
    row = loose[loose["horse_name"] == "later"].iloc[0]
    assert row["sire_median_hcap_debut_or"] == pytest.approx(75.0)
    assert np.isnan(strict[strict["horse_name"] == "later"]["sire_median_hcap_debut_or"].iloc[0])


# ---------------------------------------------------------------------------
# §5.1 - 5.2: trainer and jockey
# ---------------------------------------------------------------------------

def test_ae_ratio_and_roi_are_the_stage_c_formulas():
    rows = []
    for day, (pos, price) in enumerate(((1, 5.0), (3, 3.0), (2, 4.0)), start=1):
        r = _race(day, "13:00", [(f"h{day}", "S1", f"d{day}", "DS1", pos)]
                  + _fill(f"f{day}", [p for p in (1, 2, 3, 4) if p != pos]))
        for row in r:
            row["trainer"] = "T" if row["horse_name"].startswith("h") else row["horse_name"]
            row["bfsp"] = price if row["horse_name"].startswith("h") else 9.0
        rows += r
    out, feats = add_connection_features(pd.DataFrame(rows), price_col="bfsp",
                                         context_splits=False, strength=False, seasons=False)
    t = out[(out["trainer"] == "T") & (out["race_date"] == out["race_date"].max())].iloc[0]
    # two priced prior runs: a 5.0 winner and a 3.0 loser
    expected_wins, expected_exp = 1.0, 1 / 5 + 1 / 3
    assert t["trainer_ae_ratio"] == pytest.approx((expected_wins + 20.0) / (expected_exp + 20.0))
    assert t["trainer_roi_bsp"] == pytest.approx(((5.0 - 1.0) - 1.0) / (2.0 + 50.0))
    assert {"trainer_ae_ratio", "trainer_roi_bsp"} <= set(feats)
    # and they stay out of the default (Stage F) feature set
    _, plain = add_connection_features(pd.DataFrame(rows), context_splits=False, strength=False, seasons=False)
    assert not [c for c in plain if c.endswith(("_ae_ratio", "_roi_bsp", "_roi_sp"))]


def test_strike_rate_is_scored_against_the_schedule_entered():
    """§5.1: "a trainer with a 22% strike rate in Class 6 sellers is not
    stronger than one with 12% in Class 2 handicaps"."""
    rows = []
    for day in range(1, 51):
        small = day % 2 == 0
        n = 6 if small else 16
        who = "T_SMALL" if small else "T_BIG"
        wins = (day // 2) % 5 == 0                     # both yards win exactly 1 in 5
        pos = 1 if wins else n
        others = [p for p in range(1, n + 1) if p != pos]
        r = _race(day, "13:00", [(f"h{day}", "S1", f"d{day}", "DS1", pos)] + _fill(f"f{day}", others))
        for i, row in enumerate(r):
            row["trainer"] = who if i == 0 else f"F{day}_{i}"
        rows += r
    out, _ = add_connection_features(pd.DataFrame(rows), context_splits=False, seasons=False)
    small = out[out["trainer"] == "T_SMALL"].sort_values("race_date").iloc[-1]
    big = out[out["trainer"] == "T_BIG"].sort_values("race_date").iloc[-1]
    assert small["trainer_sr_shrunk"] == pytest.approx(big["trainer_sr_shrunk"], abs=0.02)   # same raw rate
    assert small["trainer_sr_expected"] > big["trainer_sr_expected"]   # small fields are easier
    assert big["trainer_sr_resid"] > small["trainer_sr_resid"]         # so the big-field yard is better
    assert np.isfinite(big["trainer_strength"])


def test_context_splits_are_residuals_against_the_yards_own_level():
    rows = []
    for day in range(1, 31):
        ascot = day % 2 == 0
        pos = 1 if ascot else 6
        others = [p for p in range(1, 7) if p != pos]
        r = _race(day, "13:00", [(f"h{day}", "S1", f"d{day}", "DS1", pos)] + _fill(f"f{day}", others),
                  track="Ascot" if ascot else "York")
        for i, row in enumerate(r):
            row["trainer"] = "TC" if i == 0 else f"F{day}_{i}"
        rows += r
    out, feats = add_connection_features(pd.DataFrame(rows), strength=False, seasons=False)
    tc = out[out["trainer"] == "TC"].sort_values("race_date")
    ascot = tc[tc["track"] == "Ascot"].iloc[-1]
    york = tc[tc["track"] == "York"].iloc[-1]
    assert ascot["trainer_course_apt"] > 0.1 and york["trainer_course_apt"] < -0.1
    assert ascot["trainer_course_n"] == 14.0
    assert ascot["trainer_course_sr_shrunk"] > york["trainer_course_sr_shrunk"]
    assert {"trainer_course_apt", "trainer_course_sr_shrunk", "trainer_going_apt",
            "trainer_dist_apt", "trainer_class_apt", "trainer_fieldsize_apt"} <= set(feats)


def test_jockey_rides_today_at_meeting_counts_declarations_not_results():
    rows = _race(1, "13:00", [(f"a{i}", "S1", f"d{i}", "DS1", i + 1) for i in range(6)])
    rows += _race(1, "15:00", [(f"b{i}", "S1", f"e{i}", "DS1", i + 1) for i in range(6)])
    d = pd.DataFrame(rows)
    d["jockey_name"] = "JK"                                   # one jockey, both races
    d.loc[d["race_time"] == "15:00", "track"] = "Ascot"
    out, feats = add_connection_features(d, context_splits=False, strength=False, seasons=False)
    assert "jockey_rides_today_at_meeting" in feats
    assert set(out["jockey_rides_today_at_meeting"]) == {12.0}


def test_form_vs_career_uses_the_short_window_and_the_index_survives():
    d = _two_runners_per_sire_card()
    d.index = pd.RangeIndex(100, 100 + len(d))                # a non-default index
    out, feats = add_connection_features(d, context_splits=False, strength=False)
    assert list(out.index) == list(d.index)
    assert list(out["horse_name"]) == list(d["horse_name"])   # row order preserved
    assert {"trainer_form_14d", "trainer_form_30d", "trainer_form_90d",
            "trainer_runner_count_14d", "trainer_runs_season", "trainer_sr_last_season_shrunk"} <= set(feats)
    expected = out["trainer_form_14d"] - out["trainer_nmfp_shrunk"]
    assert np.allclose(out["trainer_form_vs_career"], expected)
