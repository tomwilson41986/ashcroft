"""The audit of 11 Nov 2026's foundation faults, each held by a test: what data.clean keeps and how it reads a class,
a distance and an allowance; the engine's beaten distances, its meeting variant, its pace map; the card's meetings;
random draws that are a race's own. Frames here are built for the mechanics - not data a model is trained or scored
on."""

import numpy as np
import pandas as pd
import pytest

from greyhound import card as gcard
from greyhound.data import clean, grade_scale, handicap_start
from greyhound.metrics import GreyhoundMetricsEngine


def _race(race_id, day, time, cls="A5", track="Romford", meeting=1, n=6, **over):
    rows = []
    for i in range(n):
        rows.append({"race_date": day, "race_time": time, "meeting_id": meeting, "track": track, "race_id": race_id,
                     "race_number": 1, "race_type": "Flat", "race_class": cls, "handicap": cls == "HP", "distance_m": 480.0,
                     "going": 0.0, "trap": i + 1, "trap_handicap": None, "dog_id": race_id * 10 + i, "dog_name": f"D{race_id}{i}",
                     "sire": "S", "trainer": "T", "born": "Jan-2024", "sex": "d", "sp_decimal": 6.0, "position": float(i + 1),
                     "beaten": "" if i == 0 else "1", "beaten_lengths": 0.0 if i == 0 else 1.0, "sectional": 4.6,
                     "run_time": 29.0 + 0.08 * i, "adjusted_time": 29.0 + 0.08 * i, "weight_kg": 32.0, "comment": "Mid"})
    df = pd.DataFrame(rows)
    for k, v in over.items():
        df[k] = v
    return df


def test_a_void_race_is_no_run_and_a_non_finisher_lost():
    void = _race(1, "2026-01-01", "18:00:00", position=0.0, comment="(NoRace)")
    run = _race(2, "2026-01-01", "18:20:00")
    run.loc[5, ["position", "beaten", "run_time", "adjusted_time"]] = [0.0, "", np.nan, np.nan]      # started, no result
    out = clean(pd.concat([void, run], ignore_index=True))
    assert set(out.race_id) == {2} and len(out) == 6                       # the void race is gone, the starter is kept
    dnf = out[out.dnf].iloc[0]
    assert dnf.won == 0 and dnf.placed2 == 0 and np.isnan(dnf.position) and out.field.eq(6).all()
    assert out.position.min() == 1 and out.placed2.sum() == 2


def test_cleaning_twice_changes_nothing():
    """The live card is priced on clean(history + card), the history already clean: the second pass must keep every
    row and value of the first (a non-finisher's row and flag, a faulted allowance, the distances)."""
    void = _race(1, "2026-01-01", "18:00:00", position=0.0, comment="(NoRace)")
    run = _race(2, "2026-01-01", "18:20:00", going=20.0)                               # allowance never applied
    run.loc[5, ["position", "beaten", "run_time", "adjusted_time"]] = [0.0, "", np.nan, np.nan]
    dis = _race(3, "2026-01-02", "18:00:00", cls="HP")
    dis["beaten"] = ["", "SH", "Nk", "Dis", "2", ""]
    dis["trap_handicap"] = ["Scr", "R2", "R4", "R6", "R8", "R10"]
    once = clean(pd.concat([void, run, dis], ignore_index=True))
    twice = clean(once)
    pd.testing.assert_frame_equal(once, twice)
    assert once.dnf.sum() == 1 and once.allowance_fault.sum() == 5 + 5 and len(once) == 12
    # and with a card on top: the history rows are the same rows
    card = once[once.race_id == 3].assign(race_id=-5, race_date="2026-01-03", is_card=True, position=np.nan, beaten=np.nan,
                                           beaten_lengths=np.nan, run_time=np.nan, adjusted_time=np.nan, meeting_id=-1)
    both = clean(pd.concat([once, card[[c for c in card.columns if c not in ("won", "placed2", "t")]]], ignore_index=True))
    hist = both[~both.is_card].reset_index(drop=True)
    pd.testing.assert_frame_equal(hist[once.columns], once, check_dtype=False)
    assert both[both.is_card].beaten_lengths.isna().all() and both[both.is_card].handicap.all()


def test_an_intertrack_race_is_a_race_and_a_trial_is_not():
    it = _race(1, "2026-01-01", "18:00:00", cls="IT ")
    out = clean(it)
    assert not out.is_trial.any() and (out.race_class == "IT").all() and out.grade.isna().all()
    assert clean(_race(2, "2026-01-01", "18:00:00", cls="T3")).is_trial.all()


def test_the_class_ladders_and_the_handicap():
    assert grade_scale("OR2") == grade_scale("A4") == grade_scale("IV") == "A" and grade_scale("D2") == "D"
    assert grade_scale("HP") is None and grade_scale("IT") is None
    assert handicap_start("R7") == 7 and handicap_start("Scr") == 0 and np.isnan(handicap_start(None))
    hp = _race(1, "2026-01-01", "18:00:00", cls="HP")
    hp["trap_handicap"] = ["Scr", "R2", "R4", "R6", "R8", "R10"]
    hp["adjusted_time"] = hp.run_time + 0.08 * np.array([0, 2, 4, 6, 8, 10])          # GBGB's own sum
    out = clean(hp)
    assert out.handicap.all() and list(out.hcp_start) == [0, 2, 4, 6, 8, 10] and not out.allowance_fault.any()
    card_like = hp.drop(columns=["handicap", "trap_handicap"])                         # a card has only the class
    assert clean(card_like.assign(adjusted_time=np.nan)).handicap.all()
    hurdles = clean(_race(2, "2026-01-01", "18:00:00", cls="OR", race_type="Hurdles"))
    assert hurdles.hurdles.all() and not out.hurdles.any()


def test_an_allowance_that_does_not_add_up_is_no_time():
    ok = _race(1, "2026-01-01", "18:00:00", going=20.0)
    ok["adjusted_time"] = ok.run_time + 0.20
    unapplied = _race(2, "2026-01-01", "18:20:00", going=20.0)                         # the allowance never applied
    typo = _race(3, "2026-01-01", "18:40:00", going=2020.0)
    typo["adjusted_time"] = typo.run_time + 20.20
    out = clean(pd.concat([ok, unapplied, typo], ignore_index=True))
    assert out[out.race_id == 1].adjusted_time.notna().all() and not out[out.race_id == 1].allowance_fault.any()
    assert out[out.race_id == 2].adjusted_time.isna().all() and out[out.race_id == 3].adjusted_time.isna().all()


def test_a_missing_distance_is_unknown_not_nothing():
    r = _race(1, "2026-01-01", "18:00:00")
    r["beaten"] = ["", "SH", "Nk", "Dis", "2", "DNF"]
    r.loc[3, "run_time"] = 29.0 + 0.08 * 30                    # the dog beaten a distance: 30 lengths back by the clock
    r.loc[4, "run_time"] = 29.0 + 0.08 * 32
    r.loc[5, "run_time"] = np.nan
    out = clean(r)
    assert list(out.beaten_lengths[:3]) == [0.0, 0.125, 0.375] and out.beaten_lengths[3:].isna().tolist() == [True, False, True]
    eng = GreyhoundMetricsEngine(blocks=())
    lbw = eng._per_run(out).lbw.round(3).tolist()
    # the first three by the distances; from the "Dis" on by the clock, capped; no clock, unknown
    assert lbw[:3] == [0.0, 0.125, 0.5] and lbw[3] == lbw[4] == 15.0 and np.isnan(lbw[5])
    r2 = r.copy()
    r2.loc[3, "run_time"], r2.loc[4, "run_time"] = 29.0 + 0.08 * 6, 29.0 + 0.08 * 8      # a "Dis" the clock puts 6 back
    lbw2 = eng._per_run(clean(r2)).lbw.round(3).tolist()
    assert lbw2[3] == pytest.approx(6.0) and lbw2[4] == pytest.approx(8.0)
    blank = _race(2, "2026-01-02", "18:00:00")
    blank["beaten"], blank["beaten_lengths"] = "", 0.0          # no text on a beaten dog: no distance, not a dead heat
    assert clean(blank).beaten_lengths.isna().sum() == 5


def test_the_meeting_variant_is_not_the_class_of_the_card():
    """Two tracks' worth of history where A1 winners run 0.40 s faster than A8 winners and every night is the same
    going: a night of A1 races and a night of A8 races must both read as no variant."""
    rng = np.random.default_rng(0)
    frames, rid = [], 0
    for d in range(60):
        day = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        classes = ["A1"] * 6 if d % 2 == 0 else ["A8"] * 6     # a top-grade night, then a low-grade night
        for k, cls in enumerate(classes):
            rid += 1
            r = _race(rid, day, f"{18 + k // 3}:{20 * (k % 3):02d}:00", cls=cls, meeting=d)
            base = 28.8 if cls == "A1" else 29.2
            r["run_time"] = r["adjusted_time"] = base + 0.08 * np.arange(6) + rng.normal(0, 0.02, 6)
            frames.append(r)
    out = GreyhoundMetricsEngine(blocks=())._per_run(clean(pd.concat(frames, ignore_index=True)))
    late = out[out.race_date >= "2026-02-10"].drop_duplicates("meeting_id")
    top, low = late[late.race_class == "A1"].variant, late[late.race_class == "A8"].variant
    assert abs(top.mean()) < 0.03 and abs(low.mean()) < 0.03, (top.mean(), low.mean())   # was about -0.2 and +0.2
    # and the figure still tells the classes apart: the A1 winner is the faster dog
    w = out[(out.won == 1) & (out.race_date >= "2026-02-10")]
    assert w[w.race_class == "A1"].gsr.mean() - w[w.race_class == "A8"].gsr.mean() > 3.0


def test_the_pace_map_knows_an_empty_box_and_an_unknown_dog():
    eng = GreyhoundMetricsEngine(blocks=())
    df = pd.DataFrame({"race_id": [1] * 5, "trap": [1, 2, 4, 5, 6], "esr_mean3": [1.0, 3.0, 2.0, np.nan, 0.5],
                       "path_mean6": [1.0, 2.0, 2.0, 2.0, 3.0], "grade_family": ["A"] * 5, "grade_par": 0.0,
                       "gsr_max3": 0.0, "gsr_trouble_adj_max3": 0.0})
    for c in ("gsr_mean3", "gsr_max6", "gsr_ewm", "esr_max3", "fsr_mean3", "position_mean3", "won_mean6", "trainer_win",
              "brk_mean6", "led_mean6"):
        df[c] = 0.0
    out = eng._race_relative(df, [])
    # trap 2's neighbour outside is the empty box 3 (room), trap 4's inside too; trap 5 has no early speed behind it
    sq = out["squeeze"]
    assert sq.tolist()[:3] == [1.0, 0.0, 0.0] and np.isnan(sq[3]) and sq[4] == 0.0
    assert np.isnan(out.early_rank[3]) and out.early_rank[1] == 1.0
    assert out.faster_inside.tolist()[:3] == [0.0, 0.0, 1.0] and out.faster_inside[4] == 1.0
    # a railer in trap 1 and a wide runner in trap 6 fit their boxes
    assert out.path_vs_trap[0] == 0.0 and out.path_vs_trap[4] == 0.0 and "is_handicap" in out and "hurdle" not in out


def test_a_race_s_draws_are_its_own():
    """The simulated chances of one race are the same whatever other races share the frame."""
    from greyhound.ability import p_best
    from greyhound.blocks import sim
    m = np.array([1.0, 0.5, 0.0, -0.5, 2.0, 1.0, 0.0])
    sd = np.ones(7)
    alone = p_best(m[:4], sd[:4], np.array([9, 9, 9, 9]))
    among = p_best(m, sd, np.array([9, 9, 9, 9, 3, 3, 3]))
    assert np.array_equal(alone, among[:4])
    one = pd.DataFrame({"race_id": [9] * 4, "trap": [1, 2, 3, 4], "esr_mean3": m[:4], "esr_max3": m[:4],
                        "fsr_mean3": m[:4], "gsr_mean6": m[:4], "gsr_std6": 1.0})
    other = one.assign(race_id=3, esr_mean3=[0.0, 2.0, 1.0, 1.0])
    a = one.copy()
    sim.build(a)
    b = pd.concat([other, one], ignore_index=True)
    sim.build(b)
    assert np.array_equal(a.sim_win_p.to_numpy(), b[b.race_id == 9].sim_win_p.to_numpy())


def test_the_card_numbers_a_second_meeting_from_one():
    rows = []
    for k, (hh, mm) in enumerate([(11, 0), (11, 18), (11, 36), (18, 30), (18, 48)]):
        for trap in range(1, 7):
            rows.append({"market_id": f"1.{k}", "selection_id": 100 * k + trap, "market_type": "WIN", "country": "GB",
                         "market_start_utc": f"2026-01-10T{hh:02d}:{mm:02d}:00Z", "venue": "Romford",
                         "market_name": "A5 400m", "runner_name": f"{trap}. Dog {k}{trap}"})
    hist = pd.DataFrame({"dog_id": [1], "dog_name": ["x"], "race_date": ["2025-01-01"], "race_time": ["10:00"], "sire": ["s"],
                         "dam": ["d"], "trainer": ["t"], "born": ["Jan-2024"], "sex": ["d"], "season": [None]})
    c = gcard.cards(pd.DataFrame(rows), hist).drop_duplicates("market_id").sort_values("race_time")
    assert c.race_number.tolist() == [1, 2, 3, 1, 2] and c.meeting_id.nunique() == 2
    # and the history is numbered the same way: by the off within the meeting, whatever GBGB's own number
    h = clean(pd.concat([_race(1, "2026-01-01", "18:00:00", race_number=7), _race(2, "2026-01-01", "18:20:00", race_number=9)]))
    assert h.drop_duplicates("race_id").race_number.tolist() == [1, 2]


def test_one_list_of_card_unsafe_columns():
    from greyhound import fastloop as fl, live, stack_model as sm
    assert all(live.is_card_unsafe(f) for f in ("lf_wt_change", "hb_days_since_season", "ti_new_trainer_ae", "weight_change"))
    assert not live.is_card_unsafe("gsr_mean3") and sm.DROP is live.CARD_UNSAFE_PREFIXES
    assert live.card_safe(["gsr_mean3", "lf_wt_vs_mean", "ti_new_trainer"]) == ["gsr_mean3"]
    meta = {"features": {"base": ["gsr_mean3", "weight_change"], "intent": ["ti_new_trainer", "ti_class_up"]},
            "card_unsafe": sorted(live.CARD_UNSAFE), "card_unsafe_prefixes": list(live.CARD_UNSAFE_PREFIXES)}
    assert fl.feature_set(meta, "card_safe") == ["gsr_mean3", "ti_class_up"]
    assert fl.feature_set(meta, "all") == ["gsr_mean3", "weight_change", "ti_new_trainer", "ti_class_up"]
    assert "model/lagsafe.py" in fl.CODE_FILES and {"won", "bsp", "position", "lbw"} <= set(fl.LABELS)


def test_race_rel2_reads_every_input_in_a_fixed_order():
    from greyhound.blocks import race_rel2
    cols = [f"gp_{m}_{w}" for m in ("nfpz", "gsr", "ep", "lbsc", "ae", "mkt", "bsp_mkt", "rs", "shape", "wax")
            for w in ("w3", "w5", "w10", "car")]
    a = pd.DataFrame({c: [0.0, 1.0] for c in ["lb_kr", "lb_elo"] + cols})
    b = a[a.columns[::-1]]
    assert race_rel2.columns(a) == race_rel2.columns(b) and len(race_rel2.columns(a)) == 42


def test_one_name_a_track():
    from greyhound.data import canonical_track
    from greyhound.parity import attach_bsp
    old = _race(1, "2024-12-01", "18:00:00", track="Pelaw Grange")
    new = _race(2, "2024-12-20", "18:00:00", track="Star Pelaw ")
    runs = clean(pd.concat([old, new], ignore_index=True))
    assert set(runs.track) == {"Star Pelaw"}
    assert canonical_track(pd.Series(["The Valley", "Romford", None])).tolist()[:2] == ["Valley", "Romford"]
    # Betfair's files under the old name still join
    prices = pd.DataFrame({"market": "win", "country": "GB", "race_date": "2024-12-20", "race_time": "18:00",
                           "track": "Pelaw Grange", "trap": [1, 2, 3, 4, 5, 6], "bsp": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]})
    with_bsp = attach_bsp(runs, prices)
    assert with_bsp[with_bsp.race_id == 2].bsp.notna().all() and with_bsp[with_bsp.race_id == 1].bsp.isna().all()


def test_the_last_handicap_start_is_read_in_a_handicap_only():
    """A dog's start at its last handicap, known before the race, as today's (which no card gives)."""
    frames = []
    for k, (day, cls) in enumerate([("2026-01-01", "HP"), ("2026-01-05", "A5"), ("2026-01-09", "HP"), ("2026-01-13", "A5")]):
        r = _race(k + 1, day, "18:00:00", cls=cls)
        r["dog_id"] = np.arange(6) + 100                       # the same six dogs each time
        r["trap_handicap"] = ["Scr", "R2", "R4", "R6", "R8", "R10"] if cls == "HP" else None
        if cls == "HP":
            r["adjusted_time"] = r.run_time + 0.08 * np.array([0, 2, 4, 6, 8, 10])
        frames.append(r)
    eng = GreyhoundMetricsEngine(blocks=())
    out = eng.calculate_all(clean(pd.concat(frames, ignore_index=True)))
    by = lambda rid: out[out.race_id == rid].sort_values("trap")
    assert by(1).hcp_last_start.isna().all()                   # the first handicap: no start behind any dog
    assert by(3).hcp_last_start.tolist() == [0, 2, 4, 6, 8, 10]            # the second: the first one's, past the A5 between
    assert by(3).hcp_last_vs_field.round(2).tolist() == [-6.0, -3.6, -1.2, 1.2, 3.6, 6.0]
    assert by(2).hcp_last_start.isna().all() and by(4).hcp_last_vs_field.isna().all()     # not read outside a handicap
    assert {"hcp_last_start", "hcp_last_vs_field", "is_handicap"} <= set(eng.features) and "_hcp_last" not in eng.features
    # and the start its trap usually carries at the track, from the earlier handicaps there (the trap is the order)
    assert by(1).hcp_trap_start.isna().all() and by(3).hcp_trap_start.tolist() == [0, 2, 4, 6, 8, 10]
    assert by(3).hcp_trap_vs_field.tolist() == [-5, -3, -1, 1, 3, 5] and by(4).hcp_trap_start.isna().all()
    assert "_hcp_trap" not in eng.features and "hcp_trap_start" in eng.features
