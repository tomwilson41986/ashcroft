"""The stacked move rule's trial code: the signal's sign, the frozen models' round trip, the hash check (synthetic rows)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from research.open_close import football_stack_trial as st
from research.open_close.football_exchange import read_prereg
from tests.test_football_lab import matrix  # noqa: F401  (the lab's synthetic matrix fixture)


def test_a_forecast_rise_in_chance_is_a_back_and_a_fall_a_lay():
    d = pd.DataFrame({"match_id": ["m1", "m2", "m3"], "fd_code": ["E0", "D1", "E1"],
                      "match_date": ["2026-11-07", "2026-11-07", "2026-11-04"],
                      "bfe_open_h": [2.0, 2.0, 2.0], "bfe_open_d": [3.6, 3.6, 3.6], "bfe_open_a": [4.2, 4.2, 4.2]})
    pred = pd.DataFrame({"match_id": ["m1", "m2", "m3"], "chance_h": [0.10, -0.10, 0.01],
                         "chance_a": [-0.05, 0.03, 0.0]})
    s = st.signals(d, pred, 0.04).set_index(["event_id", "selection_id"])
    assert s.loc[("m1", "H"), "side"] == 1          # chance up 0.10: price shortens, back early, lay at the close
    assert s.loc[("m1", "A"), "side"] == -1         # chance down 0.05: lay early
    assert s.loc[("m2", "H"), "side"] == -1
    assert s.loc[("m2", "A"), "side"] == 0 and s.loc[("m3", "H"), "side"] == 0   # inside the threshold
    assert s.loc[("m1", "H"), "tier"] == "top five" and s.loc[("m3", "H"), "tier"] == "rest"
    assert s.loc[("m3", "H"), "collection"] == "2026-11-03"   # a Wednesday match: Tuesday's collection
    assert s.loc[("m1", "H"), "collection"] == "2026-11-06"   # a Saturday match: Friday's


def test_freeze_then_predict_round_trips_and_the_hashes_hold(matrix, tmp_path):  # noqa: F811
    cache = tmp_path / "cache"
    cache.mkdir()
    matrix.to_parquet(cache / "matrix_t.parquet", index=False)
    (cache / "LATEST").write_text("matrix_t.parquet")
    out = tmp_path / "models"
    info = st.freeze(str(cache), str(out))
    assert info["rows"] > 0 and info["ridge_features"] <= info["features"]
    sums = dict(reversed(line.split("  ")) for line in (out / "SHA256SUMS").read_text().splitlines())
    assert set(sums) == {"spec.json", "gbm_h.txt", "gbm_a.txt"}
    st.verify_models(str(out), sums)
    d, _ = st._matrix(str(cache))
    p1, p2 = st.predict(str(out), d), st.predict(str(out), d)
    assert np.isfinite(p1[["chance_h", "chance_a"]].to_numpy()).all()
    pd.testing.assert_frame_equal(p1, p2)
    spec = json.loads((out / "spec.json").read_text())
    assert spec["arm"] == st.ARM and spec["train"] == [st.TRAIN_FROM, st.TRAIN_TO]
    (out / "gbm_h.txt").write_text((out / "gbm_h.txt").read_text() + "\n")
    with pytest.raises(SystemExit, match="pre-registered"):
        st.verify_models(str(out), sums)


def test_the_preregistration_names_the_committed_models():
    rule, _ = read_prereg(st.PREREG)
    assert rule["arm"] == st.ARM and rule["models_dir"] == "models/football/stack_v1"
    st.verify_models(rule["models_dir"], rule["model_sha256"])
    assert rule["threshold"] in st.THRESHOLDS and len(rule["leagues_fd"]) == 22
    sweep = json.load(open("reports/football_runs/stack_threshold_sweep_2024.json"))
    assert sweep["chosen"] == rule["threshold"] and sweep["arm"] == rule["arm"]
    assert {k for k in rule["gate"] if k in st.GATE} == set(st.GATE)
    assert all(rule["gate"][k] == v for k, v in st.GATE.items())
    for name, h in rule["model_sha256"].items():
        assert hashlib.sha256(open(f"{rule['models_dir']}/{name}", "rb").read()).hexdigest() == h
