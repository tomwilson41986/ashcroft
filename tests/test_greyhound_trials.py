"""greyhound/pit/trials.py: the trials data.clean drops, read on their own channel. The trial rows are selected from
the raw frame; a debutant gets its trial figure and a level on the races' scale; the counts and the recency are
right; the day's own results move nothing; a card debutant is matched to its trial rows by name."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from greyhound import card as gcard
from greyhound.data import clean
from greyhound.pit import core, trials
from tests.test_greyhound_pit import cardify_last_day, perturb_last_day

L = core.LENGTH_S


def raw(days=60, seed=2):
    """One track, 500 m. Dogs 1-12 race weekly from day 0 (winning time 29.50 on average). Dog 50 trials on days 30
    and 37 (29.30 and 29.20: 2.5 and 3.75 lengths inside the standard) and has its first race on day 44. Dog 3 is
    off from day 20 and trials back on day 40. Trials are served with no off time, as GBGB serves them."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 3000

    def run(day, rt, cls, dog, trap, t, pos, wt=30.0, comment="QAw,Rls"):
        return {"race_date": day, "race_time": rt, "meeting_id": 1, "track": "Hove", "race_id": rid, "race_class": cls,
                "distance_m": 500.0, "trap": trap, "dog_id": dog, "dog_name": f"Dog {dog}", "sire": "S", "dam": "D",
                "born": "Jan-2023", "sex": "d", "trainer": "T", "sp_decimal": 6.0 if rt else np.nan, "position": pos,
                "beaten_lengths": 0.5, "sectional": 4.5, "run_time": t, "adjusted_time": t, "weight_kg": wt,
                "comment": comment}
    for d in range(0, days, 7):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        field = [x for x in range(1, 13) if not (x == 3 and d >= 20)]
        if d >= 44:
            field = field[:11] + [50] if len(field) >= 12 else field + [50]
        for r in range(0, 12, 6):
            rid += 1
            f = field[r:r + 6]
            t = np.sort(29.5 + np.abs(rng.normal(0, 0.08, len(f))))
            t[0] = 29.5
            for k, dog in enumerate(f):
                rows.append(run(day, f"{18 + r // 6}:00", "A3", dog, k + 1, float(t[k]), k + 1))
    for d, dog, t, wt in ((30, 50, 29.30, 31.0), (37, 50, 29.20, 31.5), (40, 3, 29.60, 30.4)):
        rid += 1
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        rows.append(run(day, "", "T1", dog, 1, t, 1, wt, "Mid"))
    rid += 1
    rows.append(run("2025-01-20", "", "T2", 60, 1, np.nan, 0))            # a no-trial: no clock, position 0
    return pd.DataFrame(rows)


@pytest.fixture()
def built():
    r = raw()
    tr = trials.select(r)
    runs = clean(r)
    tab, notes = trials.build(runs, core.per_run_figures(runs), {"trial_runs": tr})
    return r, runs, runs.merge(tab, on=["race_id", "dog_id"]), notes, tr


def test_clean_drops_the_trials_and_select_keeps_the_timed_ones(built):
    r, runs, _, notes, tr = built
    assert not trials.is_trial_class(runs.race_class).any()                # what the audit found
    assert len(tr) == 3 and set(tr.dog_id) == {50, 3} and notes["trial_rows"] == 3


def test_every_column_is_registered(built):
    _, runs, m, _, _ = built
    assert {c for c in m.columns if c.startswith(trials.PREFIX)} == set(trials.COLUMNS) and len(m) == len(runs)


def test_a_debutant_reads_its_trials(built):
    _, _, m, notes, _ = built
    first = m[m.dog_id == 50].sort_values("race_date").iloc[0]
    assert first.race_date == "2025-02-19" and first.tl_n_all == 2 and first.tl_n28 == 2 and first.tl_days == 12
    assert first.tl_last_was_trial == 1 and first.tl_races_since == 0 and first.tl_out_days == 12
    assert np.isclose(first.tl_fig_last, (29.5 - 29.2) / L, atol=0.01) and np.isclose(first.tl_fig_best3, 3.75, atol=0.01)
    assert np.isclose(first.tl_fig_mean3, (2.5 + 3.75) / 2, atol=0.01) and first.tl_solo_last == 1
    assert first.tl_same_track == 1 and first.tl_same_trip == 1 and np.isclose(first.tl_fig_here, 3.75, atol=0.01)
    assert np.isclose(first.tl_wt_last, 31.5) and np.isnan(first.tl_wt_vs_race) and first.tl_path_last == 2
    # a level on the races' scale from the trials alone, and well clear of a field that runs about a length off it
    assert 2.5 < first.tl_lvl < 3.75 and np.isnan(first.tl_lvl_gain) and first.tl_lvl_vs_field > 2
    assert notes["first_races_with_a_trial"] > 0


def test_a_dog_back_from_a_break_and_a_dog_that_never_trialled(built):
    _, _, m, _, _ = built
    three = m[(m.dog_id == 3)].sort_values("race_date")
    assert (three[three.race_date < "2025-02-10"].tl_n_all == 0).all()
    one = m[m.dog_id == 1]
    assert (one.tl_n_all == 0).all() and one.tl_days.isna().all() and (one.tl_last_was_trial == 0).all()
    assert one.tl_lvl.notna().mean() > 0.7 and one.tl_lvl_gain.isna().all()


@pytest.mark.parametrize("change", [perturb_last_day, cardify_last_day])
def test_the_days_own_results_move_nothing(change):
    r = raw()
    tr = trials.select(r)
    runs = clean(r)
    keys = runs.loc[runs.race_date == runs.race_date.max(), ["race_id", "dog_id"]]
    a = trials.build(runs, core.per_run_figures(runs), {"trial_runs": tr})[0]
    r2 = change(runs)
    b = trials.build(r2, core.per_run_figures(r2), {"trial_runs": tr})[0]
    cols = list(trials.COLUMNS)
    x = keys.merge(a, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    z = keys.merge(b, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])[cols].to_numpy(float)
    assert np.allclose(x, z, atol=1e-4, equal_nan=True)


def test_a_trial_on_the_day_or_after_is_not_read():
    r = raw()
    late = r.assign(race_date=r.race_date.where(~((r.dog_id == 50) & (r.race_class == "T1")), "2025-02-19"))
    runs = clean(late)
    tab = trials.build(runs, core.per_run_figures(runs), {"trial_runs": trials.select(late)})[0]
    m = runs.merge(tab, on=["race_id", "dog_id"])
    first = m[m.dog_id == 50].sort_values("race_date").iloc[0]
    assert first.tl_n_all == 0 and np.isnan(first.tl_fig_last)             # trialled the same day: not known before


def test_a_card_debutant_is_matched_to_its_trial_rows():
    r = raw()
    hist, tr = clean(r[r.race_date < "2025-02-19"]), trials.select(r)
    markets = pd.DataFrame({"market_id": ["1.5"] * 2, "selection_id": [901, 902], "market_type": "WIN", "country": "GB",
                            "market_start_utc": "2025-02-19T19:00:00Z", "venue": "Hove", "market_name": "A3 500m",
                            "runner_name": ["1. Dog 50", "2. Dog 999"], "trainer": [None, "X"]})
    plain = gcard.cards(markets, hist).set_index("dog_name")
    with_tr = gcard.cards(markets, hist, trials=tr).set_index("dog_name")
    assert plain.loc["Dog 50", "dog_id"] == -901 and pd.isna(plain.loc["Dog 50", "sex"])
    assert with_tr.loc["Dog 50", "dog_id"] == 50 and with_tr.loc["Dog 50", "sex"] == "d" and with_tr.loc["Dog 50", "born"] == "Jan-2023"
    assert with_tr.loc["Dog 999", "dog_id"] == -902
