"""The career-before block (model/blocks/career_before.py) against the history it stands in for.

The block's promise is that the table of runs before the cut-off, added to the runs
since, gives what the engine's own windows would give on a history that had never been
cut. So the test cuts a synthetic history at 2021-01-01, builds the table from the runs
before (the block's pre_window_table, as the research query does on race_results), runs
the block on the whole frame, and compares it with form_windows on the uncut history.
Synthetic frames test mechanics only; nothing is trained or evaluated on them.
"""
import numpy as np
import pandas as pd
import pytest

from model.blocks import career_before as cb
from model.form_windows import add_form_windows

COURSES = [("York", "Turf", "Handicap", 8.0), ("Kempton", "Standard", "Handicap", 7.0),
           ("Kelso", "Turf", "Handicap Hurdle", 16.0)]
RESULTS = ["placing_numerical", "bfsp"]


def _name(h: int) -> str:
    return "Null" if h == 7 else f"Horse {h}" + (" (IRE)" if h % 4 == 0 else "")


def _history(start="2020-09-01", n_days=150, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    born = {h: 2013 + h % 6 for h in range(48)}
    rows = []
    for d in range(n_days):
        day = pd.Timestamp(start) + pd.Timedelta(days=d)
        free = list(range(48))                                   # one run a day per horse
        for t, (track, surface, rtype, dist) in enumerate(COURSES):
            if rng.random() < 0.4:
                continue
            n = int(rng.integers(5, 9))
            horses = rng.choice(free, n, replace=False)
            free = [h for h in free if h not in set(horses)]
            finishers = n - int(rng.random() < 0.3)
            for rank, h in enumerate(horses):
                rows.append({
                    "race_date": day, "race_time": f"{1 + t}:30", "track": track, "horse_name": _name(int(h)),
                    "horse_age": day.year - born[int(h)], "number_of_runners": n,
                    "placing_numerical": rank + 1 if rank < finishers else np.nan,
                    "bfsp": float(rng.uniform(1.5, 40)),
                    "official_rating": 0 if h % 5 == 0 else int(50 + h + rng.integers(-3, 4)),
                    "race_type": rtype, "surface_type": surface, "dist_furlongs": dist,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def hist():
    h = _history()
    cb.use_table(cb.pre_window_table(h))
    yield h
    cb.use_table(None)


def _after(df):
    return (pd.to_datetime(df["race_date"]) >= pd.Timestamp(cb.CUTOFF)).to_numpy()


def test_the_table_and_the_runs_since_give_the_uncut_historys_windows(hist):
    out = cb.build(hist.copy())
    whole, _ = add_form_windows(hist.copy())
    cut, _ = add_form_windows(hist[_after(hist)].copy())
    after = _after(hist)
    for m in cb.MEASURES:
        np.testing.assert_allclose(out.loc[after, f"cb_car_{m}"].to_numpy(float),
                                   whole.loc[after, f"fw_{m}_car"].to_numpy(float), rtol=1e-9, atol=1e-12,
                                   err_msg=m)
    for m in cb.STITCHED:
        np.testing.assert_allclose(out.loc[after, f"cb_{m}_m{cb.LAST}"].to_numpy(float),
                                   whole.loc[after, f"fw_{m}_m{cb.LAST}"].to_numpy(float), rtol=1e-9, atol=1e-12,
                                   err_msg=m)
    # and the cut history alone would not: the table is what closes the gap
    differs = ~np.isclose(cut["fw_nfp_car"].to_numpy(float), whole.loc[after, "fw_nfp_car"].to_numpy(float),
                          equal_nan=True)
    assert differs.sum() > 50


def test_peaks_share_and_career_length_against_a_direct_count(hist):
    out = cb.build(hist.copy())
    d = hist.assign(date=pd.to_datetime(hist["race_date"]), key=cb.horse_key(hist["horse_name"]))
    jumps = d["race_type"].str.contains("Hurdle").to_numpy()
    orr = d["official_rating"].where(d["official_rating"] > 0).to_numpy(float)
    after = _after(hist)
    checked = 0
    for i in np.flatnonzero(after)[::7]:
        earlier = (d["key"] == d.at[i, "key"]).to_numpy() & (d["date"] < d.at[i, "date"]).to_numpy()
        flat = orr[earlier & ~jumps]
        jmp = orr[earlier & jumps]
        flat_peak = np.nanmax(flat) if np.isfinite(flat).any() else np.nan
        jumps_peak = np.nanmax(jmp) if np.isfinite(jmp).any() else np.nan
        np.testing.assert_equal(out.at[i, "cb_flat_or_peak"], flat_peak)
        np.testing.assert_equal(out.at[i, "cb_or_peak"], jumps_peak if jumps[i] else flat_peak)
        before_cut = earlier & ~after
        runs = earlier.sum()
        assert out.at[i, "cb_pw_runs"] == before_cut.sum()
        np.testing.assert_equal(out.at[i, "cb_pw_share"], before_cut.sum() / runs if runs else np.nan)
        np.testing.assert_allclose(out.at[i, "cb_jumps_share"], (earlier & jumps).sum() / runs if runs else np.nan)
        first = d.loc[earlier, "date"].min()
        years = (d.at[i, "date"] - first).days / 365.25 if runs else np.nan
        np.testing.assert_allclose(out.at[i, "cb_years_since_first"], years, rtol=1e-12)
        checked += int(before_cut.any())
    assert checked > 20


def test_rows_before_the_cutoff_get_nothing_and_an_earlier_start_changes_nothing(hist):
    out = cb.build(hist.copy())
    after = _after(hist)
    assert out.loc[~after, cb.FEATURES].isna().all().all()
    since = cb.build(hist[after].copy())
    pd.testing.assert_frame_equal(out.loc[after, cb.FEATURES], since[cb.FEATURES], check_exact=True)


def test_a_namesake_by_age_reads_as_a_horse_with_no_earlier_runs():
    t = pd.DataFrame({"horse_key": ["old timer", "old timer 2"], "pw_runs": [30, 30], "pw_jumps_runs": [0, 0],
                      **{f"pw_{m}_{s}": [3.0, 3.0] for m in cb.MEASURES for s in ("sum", "n")},
                      **{f"pw_{m}_l{j}": [0.5, 0.5] for m in cb.STITCHED for j in range(1, cb.LAST + 1)},
                      "pw_or_max_flat": [90.0, 90.0], "pw_or_max_jumps": [np.nan, np.nan],
                      "pw_first_run": ["2012-05-01", "2018-05-01"], "pw_last_run": ["2015-09-01", "2020-09-01"]})
    card = pd.DataFrame({"race_date": pd.Timestamp("2022-03-01"), "race_time": "2:00", "track": "York",
                         "horse_name": ["Old Timer", "Old Timer 2", "Somebody"], "horse_age": [3, 5, 5],
                         "number_of_runners": 3, "placing_numerical": np.nan, "bfsp": np.nan,
                         "official_rating": [70, 80, 60], "race_type": "Handicap", "surface_type": "Turf"})
    cb.use_table(t)
    try:
        out = cb.build(card)
    finally:
        cb.use_table(None)
    # born 2019 cannot have run in 2012: a namesake; born 2017, a record from 2018 may be its own
    # (a year's grace on the two-year-old's first season)
    assert out["cb_pw_runs"].tolist() == [0.0, 30.0, 0.0]
    assert np.isnan(out.at[0, "cb_flat_or_peak"]) and out.at[1, "cb_flat_or_peak"] == 90.0
    assert out.at[1, "cb_or_vs_peak"] == -10.0
    assert out.at[1, "cb_car_nfp"] == 1.0 and np.isnan(out.at[0, "cb_car_nfp"])


def test_a_days_own_results_move_none_of_its_features_after_the_cutoff(hist):
    base = cb.build(hist.copy())
    days = pd.to_datetime(hist["race_date"])
    flip = days[_after(hist)].drop_duplicates().sort_values().iloc[10]
    blank = hist.copy()
    blank.loc[days == flip, RESULTS] = np.nan
    out = cb.build(blank)
    on = (days == flip).to_numpy()
    pd.testing.assert_frame_equal(base.loc[on, cb.FEATURES], out.loc[on, cb.FEATURES], check_exact=True)
    later = (days > flip).to_numpy()
    assert not base.loc[later, "cb_car_nfp"].equals(out.loc[later, "cb_car_nfp"])


def test_the_table_reads_back_from_its_file_as_built(hist, tmp_path, monkeypatch):
    built = cb.build(hist.copy())
    t = cb.pre_window_table(hist)
    assert "null" in set(t["horse_key"])                      # a horse called Null is a name, not a gap
    path = tmp_path / "careers.csv.gz"
    t.to_csv(path, index=False, float_format="%.9g")
    monkeypatch.setattr(cb, "TABLE", path)
    cb.use_table(None)
    read = cb.build(hist.copy())
    for c in cb.FEATURES:
        np.testing.assert_allclose(read[c].to_numpy(float), built[c].to_numpy(float), rtol=1e-7, atol=1e-8,
                                   err_msg=c)                  # the file keeps nine significant figures
