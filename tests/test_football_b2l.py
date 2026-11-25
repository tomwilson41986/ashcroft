"""The back-to-lay model frozen for the trader: the kept ridge matches the research one, a save and load predicts the
same, the cells."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from football import b2l
from research.open_close import football_bf_close_model as bf


def frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    d.loc[rng.random(n) < 0.1, "b"] = np.nan
    d["y_logit"] = 0.3 * d["a"] - 0.1 * d["b"].fillna(0) + rng.normal(scale=0.2, size=n)
    d["role"] = np.array(["home", "draw", "away"])[np.arange(n) % 3]
    d["usable"] = True
    d["kickoff"] = pd.Timestamp("2026-10-01", tz="UTC")
    return d


def test_the_kept_ridge_predicts_as_the_research_ridge_does():
    d = frame()
    tr, te = d.iloc[:2000], d.iloc[2000:]
    y = tr["y_logit"].to_numpy(float)
    r = b2l.ridge_fit(tr[["a", "b"]].to_numpy(float), y, bf.RIDGE_ALPHA, bf.CLIP)
    ours = b2l.ridge_predict(r, te[["a", "b"]].to_numpy(float))
    theirs = bf.ridge_fit_predict(tr, te, ["a", "b"], y)
    assert ours == pytest.approx(theirs)


def test_a_saved_and_loaded_model_forecasts_the_same():
    d = frame()
    model = b2l.fit(d, ["a", "b"])
    again = json.loads(json.dumps(model))
    p1, p2 = b2l.predict(model, d.head(30)), b2l.predict(again, d.head(30))
    assert np.allclose(p1, p2) and np.isfinite(p1).all()
    assert np.corrcoef(p1, d.head(30)["y_logit"])[0, 1] > 0.5


def test_a_forecast_takes_the_strictest_cell_it_clears():
    assert b2l.cell_for(360, 0.06) == "360 over 0.05"
    assert b2l.cell_for(360, 0.04) == "360 over 0.03"
    assert b2l.cell_for(15, 0.09) == "15 over 0.08" and b2l.cell_for(15, 0.02) is None
    assert b2l.cell_for(60, 0.5) is None


def test_the_live_feature_list_is_the_re_scored_models():
    f = b2l.features()
    assert f[:5] == ["x_cp", "cp_book_sum", "cp_min", "ko_hour", "ko_dow"] and "idl_own_dp" in f
    assert not any("sup_gap" in c or c.startswith("rate_") or "n_ticks" in c for c in f) and len(f) == len(set(f))


def test_live_memory_reads_only_matches_a_week_old_and_shrinks_as_the_research_memory_does():
    ko = pd.to_datetime(["2026-10-01", "2026-10-10", "2026-10-20"], utc=True)
    d = pd.DataFrame({"kickoff": ko, "role": "home", "home_id": 7, "away_id": [8, 9, 10], "country_code": "GB",
                      "cp_min": 360, "usable": True, "y_logit": [0.1, 0.2, 0.9]})
    hist = b2l.memory_history(d)
    now = int((pd.Timestamp("2026-10-22", tz="UTC") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1))
    got = b2l.memory_features(hist, 7, 11, "GB", "home", 360, now)
    # the 20 Oct match is not a week old on 22 Oct: the mean of 0.1 and 0.2 over (2 + 5)
    assert got["lv_mm_own"] == pytest.approx(0.3 / 7) and got["lv_mm_own_n"] == 2
    assert got["lv_mm_opp_n"] == 0 and got["lv_mm_country_n"] == 2
