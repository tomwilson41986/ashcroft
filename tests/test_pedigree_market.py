"""The pedigree market block (model/blocks/pedigree_market.py) on hand-built histories.

Synthetic frames test mechanics only; nothing is trained or evaluated on them.
"""
import numpy as np
import pandas as pd

from model.blocks import pedigree_market as pdm
from model.form_windows import run_measures


def _race(day, time, runners):
    """runners: (horse, sire, dam, career_runs, bsp)."""
    return [{"race_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=day), "race_time": time, "track": "York",
             "horse_name": h, "stallion": s, "dam_stallion": "DS", "dam": d, "career_runs": cr,
             "number_of_runners": len(runners), "placing_numerical": i + 1, "bfsp": b}
            for i, (h, s, d, cr, b) in enumerate(runners)]


def _history():
    rows = []
    rows += _race(0, "1:00", [("A", "S1", "D1", 0, 2.0), ("X", "S2", "D9", 0, 5.0), ("Y", "S2", "D8", 3, 9.0)])
    rows += _race(5, "1:00", [("A", "S1", "D1", 1, 3.0), ("Z", "S3", "D7", 0, 4.0), ("Y", "S2", "D8", 4, 6.0)])
    rows += _race(9, "2:00", [("B", "S1", "D1", 0, 3.5), ("W", "S2", "D6", 0, 3.5), ("Y", "S2", "D8", 5, 8.0)])
    rows += _race(14, "2:00", [("B", "S1", "D1", 1, 2.5), ("A", "S1", "D1", 2, 4.0), ("Q", "S3", "D5", 7, 7.0)])
    return pd.DataFrame(rows)


def test_the_dams_other_foals_leave_the_horses_own_runs_out():
    h = _history()
    out = pdm.build(h.copy())
    m = run_measures(h)["mkt"]
    day = (h["race_date"] - h["race_date"].min()).dt.days.to_numpy()
    # B on day 14: the dam D1's other foal A ran on days 0 and 5 (and runs again on day 14, not before);
    # B's own day-9 run is left out
    i = int(np.flatnonzero((h["horse_name"] == "B").to_numpy() & (day == 14))[0])
    a_runs = np.flatnonzero((h["horse_name"] == "A").to_numpy() & (day < 14))
    w = np.exp2(-(14 - day[a_runs]) / pdm.HALFLIFE)
    total, count = float((w * m[a_runs]).sum()), float(w.sum())
    assert abs(out.at[i, "pdm_dam_n"] - count) < 1e-9
    prior = out.at[i, "pdm_sire_mkt"]
    want = (total + pdm.K_DAM * prior) / (count + pdm.K_DAM)
    assert abs(out.at[i, "pdm_dam_mkt"] - want) < 1e-9
    # a first foal (no other foal has run) reads the sire's stock
    j = int(np.flatnonzero((h["horse_name"] == "X").to_numpy())[0])
    assert out.at[j, "pdm_dam_n"] == 0 and np.isclose(out.at[j, "pdm_dam_mkt"], out.at[j, "pdm_sire_mkt"],
                                                       equal_nan=True)


def test_a_sires_debutants_are_read_on_earlier_days_only_and_by_stage():
    h = _history()
    out = pdm.build(h.copy())
    day = (h["race_date"] - h["race_date"].min()).dt.days.to_numpy()
    # W (S2) debuts on day 9: S2's debutant X ran on day 0; the stage reading is the sire's debut level
    i = int(np.flatnonzero((h["horse_name"] == "W").to_numpy())[0])
    assert out.at[i, "pdm_sire_debut_n"] > 0 and out.at[i, "pdm_stage_mkt"] == out.at[i, "pdm_sire_debut_mkt"]
    # on day 0 nothing earlier exists: every sire level is the population prior of nothing (0) or missing
    first = day == 0
    assert (out.loc[first, "pdm_sire_debut_n"].fillna(0) == 0).all()
    # an exposed horse (Q, seven runs) has no stage reading
    q = int(np.flatnonzero((h["horse_name"] == "Q").to_numpy())[0])
    assert np.isnan(out.at[q, "pdm_stage_mkt"])
    # a runner with no sire recorded reads nothing from the sire
    h2 = h.copy()
    h2.loc[h2["horse_name"] == "W", "stallion"] = ""
    assert np.isnan(pdm.build(h2).at[i, "pdm_sire_debut_mkt"])
