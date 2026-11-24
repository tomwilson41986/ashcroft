"""predict_bfsp_today's serving checks (audit MS-3, MS-4, BF-3; 24 Nov 2026): toy frames and a toy booster only.

They say what on the card the model will read wrongly, in the log, a JSON sidecar and the job summary; they never
change a value or a price."""
import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

import predict_bfsp_today as P


def _toy_booster(tmp_path, xz=False):
    """A tiny booster on two features: `w` never missing in training, `m` with NaNs (a missing direction)."""
    rng = np.random.default_rng(0)
    n = 400
    w = rng.uniform(105, 170, n)
    m = rng.uniform(0, 10, n)
    m[::5] = np.nan
    y = (w - 140) / 10 + np.nan_to_num(m, nan=5.0) / 5 + rng.normal(0, 0.1, n)
    b = lgb.train({"objective": "regression", "verbose": -1, "num_leaves": 4, "min_data_in_leaf": 5},
                  lgb.Dataset(pd.DataFrame({"w": w, "m": m}), y), num_boost_round=20)
    path = tmp_path / "toy.lgb"
    b.save_model(str(path))
    if xz:
        import lzma
        with lzma.open(str(path) + ".xz", "wt") as f:
            f.write(path.read_text())
        return P._read_booster(str(path) + ".xz")
    return P._read_booster(str(path))


@pytest.mark.parametrize("xz", [False, True])
def test_the_booster_profile_names_features_without_a_missing_direction(tmp_path, xz):
    b = _toy_booster(tmp_path, xz)
    prof = b.serving_profile
    assert prof["never_missing"] == ["w"]
    lo, hi = prof["ranges"]["w"]
    assert 105 <= lo < 106 and 169 < hi <= 170
    assert P.model_profile(b) == prof


def test_the_checks_count_what_the_model_reads_wrongly_and_change_nothing(tmp_path):
    b = _toy_booster(tmp_path)
    card = pd.DataFrame({"w": [120.0, np.nan, 200.0], "m": [1.0, np.nan, 2.0], "e": [np.nan] * 3,
                         "track": ["Ascot", "Newtrack", None], "track_cat": [0, -1, -1]})
    before = card.copy()
    fresh = {"gap_days": 8, "last_result_day": "2026-11-16", "runners_in_gap": 1, "runners": 3}
    out = P.serving_checks(card, ["w", "m", "e", "track_cat", "gone"], b, missing=["gone"], freshness=fresh)
    pd.testing.assert_frame_equal(card, before)                      # nothing in the frame is changed
    assert out["missing_features"] == ["gone"] and out["all_nan_features"] == ["e"]
    assert out["unseen_categories"] == {"track_cat": {"Newtrack": 1, "<missing>": 1}}
    assert out["nan_in_never_missing"] == {"w": 1}                   # m holds a missing direction: not counted
    assert out["out_of_range"] == {"w": 1}
    flags = " | ".join(out["flags"])
    for word in ("not built", "NaN for every runner", "outside the vocabulary", "read as 0.0", "14 and 30-day"):
        assert word in flags
    assert "out of range" not in flags                               # counted, never flagged


def test_a_clean_card_flags_nothing(tmp_path):
    b = _toy_booster(tmp_path)
    card = pd.DataFrame({"w": [120.0, 130.0], "m": [np.nan, 2.0]})
    out = P.serving_checks(card, ["w", "m"], b, missing=[], freshness={"gap_days": 1})
    assert out["flags"] == [] and out["nan_in_never_missing"] == {} and out["profile_read"]


def test_the_checks_are_logged_written_and_summarised(tmp_path, caplog):
    checks = {"flags": ["1 served features not built (priced as missing): gone"], "missing_features": ["gone"]}
    side, summ = tmp_path / "d" / "checks.json", tmp_path / "summary.md"
    with caplog.at_level(logging.INFO, logger=P.log.name):
        P.report_serving_checks(checks, "2026-11-24", str(side), str(summ))
    doc = json.loads(side.read_text())
    assert doc["target_date"] == "2026-11-24" and doc["missing_features"] == ["gone"]
    assert "not built" in summ.read_text()
    line = next(r.message for r in caplog.records if r.message.startswith("SERVING CHECKS "))
    assert json.loads(line[len("SERVING CHECKS "):])["flags"] == checks["flags"]
    assert any(r.levelno == logging.ERROR and "not built" in r.message for r in caplog.records)
    P.report_serving_checks(checks, "2026-11-24", str(tmp_path / "x" / "\0bad"))   # never raises


def test_the_gap_message_names_the_short_calendar_windows(caplog):
    hist = pd.Series(pd.to_datetime(["2026-11-16"]))
    with caplog.at_level(logging.INFO, logger=P.log.name):
        P.results_freshness(hist, pd.Timestamp("2026-11-24").date())
    assert any("14 and 30-day window" in r.message for r in caplog.records if r.levelno == logging.WARNING)
