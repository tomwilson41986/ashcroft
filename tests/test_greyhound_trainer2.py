"""greyhound/pit/trainers.py and trainer2.py (the feature review of 17 Nov 2026, B3, B4, D1, D5): the trainer at the
time from GreyhoundAPI's per-run trainer, GBGB read fresh and the card; the trainer2 table's planted effects, its
point-in-time rule and its card row. Constructed frames only (unit tests; nothing is evaluated on them)."""
import numpy as np
import pandas as pd

from greyhound.data import clean
from greyhound.pit import core, trainer2, trainers

from tests.test_greyhound_pit import cardify_last_day, perturb_last_day, synth


def api_from(runs, trainer=None, days=None):
    """GreyhoundAPI GB rows for ``runs`` (the slot, the name, the trainer), optionally only on ``days``."""
    r = runs if days is None else runs[runs.race_date.isin(days)]
    return pd.DataFrame({"race_id": r.race_id.to_numpy() + 10 ** 6, "dog_id": r.dog_id.to_numpy() + 5000,
                         "dog_name": r.dog_name.str.upper().to_numpy(), "track": r.track.to_numpy(),
                         "start_local": (r.race_date + "T" + r.race_time.astype(str).str[:5] + ":00").to_numpy(),
                         "trap": r.trap.to_numpy(), "trainer_id": 1,
                         "trainer": (r.trainer if trainer is None else trainer.loc[r.index]).to_numpy()})


def mover(days=40, move_day=20, seed=1):
    """Dog 7 trains with 'Trainer 2' until ``move_day`` and with 'Trainer 3' after; GBGB serves 'Trainer 3' (the
    current trainer) on every run, the API the trainer of the day."""
    runs = synth(days=days, seed=seed)
    d = sorted(runs.race_date.unique())
    true = runs.trainer.copy()
    dog = runs.dog_id == 7
    true[dog] = np.where(runs.race_date[dog] < d[move_day], "Trainer 2", "Trainer 3")
    runs.loc[dog, "trainer"] = "Trainer 3"
    return runs, true, d


def test_the_api_trainer_is_the_trainer_at_the_time_and_gbgb_stale_is_not():
    runs, true, d = mover()
    api_names = true.str.upper().str.replace("TRAINER ", "TRAINER  ", regex=False)      # the API's own spelling
    ctx = {"fetch_lag": pd.Series(30, index=d), "api_gb_trainer": api_from(runs, api_names)}
    tt, notes = trainers.at_the_time(runs, ctx)
    assert (tt.trainer == true).all() and (tt.source == trainers.SRC["api"]).all()
    assert notes["link"]["runs_linked"] == len(runs) and notes["names"]["names_by_name"] == 5
    # no API and a stale fetch: unknown, not GBGB's current trainer
    tt2, _ = trainers.at_the_time(runs, {"fetch_lag": pd.Series(30, index=d)})
    assert tt2.trainer.isna().all()


def test_names_map_by_cooccurrence_and_initials():
    gb = pd.Series(["John Smith"] * 12 + ["Mary Jones"] * 3 + ["A Brown"] * 2)
    api = pd.Series(["Yard Twelve"] * 12 + ["M Jones"] * 3 + ["Zed Unknown"] * 2)
    m, notes = trainers.map_names(api, gb)
    assert list(m[:12].unique()) == ["John Smith"]                  # 12 shared runs: the co-occurrence
    assert list(m[12:15].unique()) == ["Mary Jones"]                # 3 runs, too few: first initial and surname
    assert list(m[15:].unique()) == ["?Zed Unknown"]                # unmapped: a key of its own
    assert notes["names_by_cooccur"] == 1 and notes["names_by_initial"] == 1 and notes["names_by_unmapped"] == 1


def test_the_move_is_seen_on_the_day_it_happens():
    runs, true, d = mover()
    ctx = {"fetch_lag": pd.Series(30, index=d), "api_gb_trainer": api_from(runs, true)}
    tab, _ = trainer2.build(runs, None, ctx)
    t = runs[["race_date", "t"]].join(tab)[runs.dog_id == 7].sort_values("t")
    after = t[t.race_date >= d[20]]
    move = after.race_date == after.race_date.iloc[0]               # every run of the move's day (a dog may run twice)
    assert (after.ta_new_trainer[move] == 1).all() and (after.ta_new_trainer[~move] == 0).all()
    assert (after.ta_runs_since_move[move] == 0).all() and after.ta_runs_since_move.is_monotonic_increasing
    assert after.ta_first_for_yard.iloc[0] == 1 and np.isfinite(after.ta_first_for_yard_ae_log.iloc[0])
    before = t[t.race_date < d[20]]
    assert (before.ta_new_trainer.dropna() == 0).all()


def planted(days=80, seed=3, boost=2.5):
    """A card where 'Trainer 1's runners win ``boost`` times what the SP says (the SP knows only the dog's ability)."""
    rng = np.random.default_rng(seed)
    rows, rid = [], 1
    for dd in range(days):
        day = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=dd)).strftime("%Y-%m-%d")
        for r in range(10):
            rid += 1
            field = rng.choice(np.arange(1, 121), 6, replace=False)
            q = np.full(6, 1 / 6)
            sp = 1 / q * 1.15
            tr = field % 6
            w = q * np.where(tr == 1, boost, 1.0)
            win = rng.choice(6, p=w / w.sum())
            pos = np.empty(6, int)
            pos[win] = 1
            pos[np.delete(np.arange(6), win)] = rng.permutation(np.arange(2, 7))
            for k, dog in enumerate(field):
                rows.append({"race_date": day, "race_time": f"{18 + r // 6}:{(r % 6) * 9:02d}", "meeting_id": dd,
                             "track": "Romford", "race_id": rid, "race_class": "A3", "distance_m": 400.0,
                             "trap": k + 1, "dog_id": int(dog), "dog_name": f"Dog {dog}",
                             "trainer": f"Trainer {tr[k]}", "sp_decimal": float(sp[k]), "position": int(pos[k]),
                             "beaten_lengths": 1.0, "sectional": 3.5, "run_time": 24.0, "adjusted_time": 24.0,
                             "weight_kg": 30.0, "comment": "Mid"})
    return clean(pd.DataFrame(rows))


def test_a_planted_yard_reads_above_the_market():
    runs = planted()
    ctx = {"fetch_lag": pd.Series(0, index=sorted(runs.race_date.unique()))}
    tab, _ = trainer2.build(runs, None, ctx)
    late = runs.race_date >= sorted(runs.race_date.unique())[60]
    by = tab[late.to_numpy()].groupby(runs.trainer[late].to_numpy()).ta_ae_log.mean()
    assert by["Trainer 1"] > 0.35 and by["Trainer 1"] > by.drop("Trainer 1").max() + 0.4
    assert tab.ta_ae_sd[late.to_numpy()].max() < tab.ta_ae_sd[~late.to_numpy()].max()     # evidence narrows it


def test_decayed_prior_counts_earlier_days_only_and_halves():
    df = pd.DataFrame({"race_date": ["2025-01-01", "2025-01-01", "2025-01-11", "2025-01-11"],
                       "g": ["a", "a", "a", "b"], "x": [1.0, 1.0, 5.0, 7.0]})
    out = trainer2.decayed_prior(df, ["g"], ["x"], half_life=10.0)
    assert list(out.x) == [0.0, 0.0, 1.0, 0.0]                      # 2 ten days back, halved; same day not counted


def test_no_row_reads_its_own_days_results():
    runs = synth()
    d = sorted(runs.race_date.unique())
    ctx = {"fetch_lag": pd.Series(0, index=d), "api_gb_trainer": api_from(runs)}
    a, _ = trainer2.build(runs, None, ctx)
    b, _ = trainer2.build(perturb_last_day(runs), None, ctx)
    last = (runs.race_date == d[-1]).to_numpy()
    pd.testing.assert_frame_equal(a[last].reset_index(drop=True), b[last].reset_index(drop=True))


def test_the_card_row_gets_the_features_its_run_had():
    """The card has no API row and no fetch for today (B3): its trainer is the card's, and its row equals the
    nightly build's once the day is in."""
    runs = synth()
    d = sorted(runs.race_date.unique())
    full_ctx = {"fetch_lag": pd.Series(30, index=d), "api_gb_trainer": api_from(runs)}
    card_ctx = {"fetch_lag": pd.Series(30, index=d[:-1]), "api_gb_trainer": api_from(runs, days=d[:-1])}
    full, _ = trainer2.build(runs, None, full_ctx)
    card_runs = cardify_last_day(runs)
    card, notes = trainer2.build(card_runs, None, card_ctx)
    keys = runs.loc[runs.race_date == d[-1], ["race_id", "dog_id"]]
    x = keys.merge(full, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])
    y = keys.merge(card, on=["race_id", "dog_id"]).sort_values(["race_id", "dog_id"])
    cols = list(trainer2.COLUMNS)
    assert len(y) == len(keys) and y.ta_known.eq(1).all()
    assert np.allclose(x[cols].to_numpy(float), y[cols].to_numpy(float), equal_nan=True, atol=1e-8)
    assert notes["by_source"]["card"] == len(keys)


def test_betfairs_card_trainer_shows_a_move_on_the_day():
    """D5: the catalogue's trainer on the card row (``bf_trainer``) wins over GBGB's last served one."""
    runs = synth()
    d = sorted(runs.race_date.unique())
    card_runs = cardify_last_day(runs)
    last = card_runs.race_date == d[-1]
    card_runs["bf_trainer"] = np.where(last, card_runs.trainer.str.upper(), None)
    dog = card_runs.loc[last, "dog_id"].iloc[0]
    other = "TRAINER 4" if card_runs.loc[last & (card_runs.dog_id == dog), "trainer"].iloc[0] != "Trainer 4" \
        else "TRAINER 0"
    card_runs.loc[last & (card_runs.dog_id == dog), "bf_trainer"] = other
    tt, notes = trainers.at_the_time(card_runs, {"fetch_lag": pd.Series(0, index=d[:-1])})
    assert tt.trainer[last & (card_runs.dog_id == dog)].iloc[0] == other.title()
    tab, _ = trainer2.build(card_runs, None, {"fetch_lag": pd.Series(0, index=d[:-1])})
    assert tab.ta_new_trainer[(last & (card_runs.dog_id == dog)).to_numpy()].iloc[0] == 1


def test_carried_reads_earlier_days_only():
    runs = synth(days=3)
    tr = pd.Series(np.where(runs.race_date == runs.race_date.max(), "New", "Old"), index=runs.index)
    c = trainers.carried(runs, tr)
    assert (c[runs.race_date == runs.race_date.max()].dropna() == "Old").all()
    assert c[runs.race_date == runs.race_date.min()].isna().all()
    assert core.FRESH_DAYS == 4


def _store(tmp_path, historic: bool):
    """A local sources store: GBGB serves each dog's current trainer; the API the trainer of the day (``historic``)
    or the current one. 12 of 80 dogs move yard on day 20."""
    from sources import gbgb_dogs
    from sources.common import Store
    runs = synth(days=40, seed=5)
    d = sorted(runs.race_date.unique())
    movers = runs.dog_id.isin(range(1, 13))
    true = runs.trainer.where(~movers | (runs.race_date >= d[20]), "Old Yard " + (runs.dog_id % 3).astype(str))
    store = Store(root=tmp_path)
    raw = runs.drop(columns=[c for c in runs.columns if c not in ("race_date", "race_time", "track", "trap", "dog_id",
                                                                  "dog_name", "trainer")])
    store.put_parquet("gbgb/runs_2025.parquet", raw)
    store.put_parquet("greyhoundapi/runs_GB_2025.parquet", api_from(runs, true if historic else runs.trainer))
    from datetime import date
    for day, tr in ((date(2025, 1, 10), "old"), (date(2025, 2, 5), "new")):
        lines = [{"fetched_utc": f"{day}T08:00:00.000000Z", "dog_id": int(k), "status": 200,
                  "body": {"trainerName": (f"Old Yard {k % 3}" if tr == "old" else f"Trainer {k % 5}")}}
                 for k in range(1, 13)]
        gbgb_dogs.write_lines(store, day, lines)
    return store


def test_the_check_calls_a_historic_api_historic_and_a_current_one_current(tmp_path, monkeypatch):
    from research.greyhound import trainer_check as tc
    monkeypatch.setattr(tc, "MIN_MOVERS", 10)
    res = tc.run(_store(tmp_path / "h", historic=True), first_year=2025)
    assert res["verdict"] == "historic", res["why"]
    assert res["api_movers"]["dogs_api_trainer_changes"] == 12
    assert res["api_movers"]["earlier_spells_agree_current"] == 0.0
    assert res["snapshot_movers"]["api_old_share_before"] == 1.0
    assert res["snapshot_movers"]["api_new_share_after"] == 1.0
    cur = tc.run(_store(tmp_path / "c", historic=False), first_year=2025)
    assert cur["verdict"] == "current" and cur["agreement"]["all_days_agree"] == 1.0
    empty = tc.run(__import__("sources.common", fromlist=["Store"]).Store(root=tmp_path / "e"), first_year=2025)
    assert empty["verdict"] == "no data found"
