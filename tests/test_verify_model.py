"""The pre-publish gate (scripts/verify_model.py): a sound artefact passes, a broken one fails.

A tiny model trained on a synthetic matrix, served next to a copy of itself.
Mechanics only; nothing is trained for use.
"""
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import train_bfsp
from model import feature_cache
from model.bfsp_model import DEFAULT_PARAMS, TrainConfig
from tests.test_fast_train import SMALL, _matrix

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def setup(tmp_path):
    db = tmp_path / "h.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE race_results (race_date TEXT)")
        c.execute("INSERT INTO race_results VALUES ('2024-01-01')")
    df = _matrix()
    feature_cache.save(df, str(tmp_path / "cache"), feature_cache.cache_key(str(db), "2021-01-01"))
    new = tmp_path / "new"
    train_bfsp.BFSPTrainer(cfg=TrainConfig(fixed_rounds=15, params=SMALL), prob_model=False).train(
        df, output_dir=str(new), prepared=True)
    served = tmp_path / "served"
    shutil.copytree(new, served)
    return tmp_path, db, new, served


def _verify(tmp_path, db, model_dir, served):
    out = tmp_path / "verify.md"
    r = subprocess.run([sys.executable, "scripts/verify_model.py", "--model-dir", str(model_dir),
                        "--served-dir", str(served), "--feature-cache", str(tmp_path / "cache"),
                        "--db", str(db), "--start-date", "2021-01-01", "--days", "30", "--out", str(out)],
                       cwd=REPO, capture_output=True, text=True)
    return r.returncode, out.read_text() if out.exists() else r.stdout + r.stderr


def test_a_sound_artefact_passes(setup):
    tmp_path, db, new, served = setup
    code, report = _verify(tmp_path, db, new, served)
    assert code == 0, report
    assert "PASS" in report and "correlation of log prices 1.0000" in report


def test_a_post_race_input_fails(setup):
    tmp_path, db, new, served = setup
    meta_path = new / "bfsp_model_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["feature_cols"] = meta["feature_cols"][:-1] + ["rs_class"]        # a run's own position
    meta_path.write_text(json.dumps(meta))
    code, report = _verify(tmp_path, db, new, served)
    assert code == 1 and "FAIL" in report and "not servable" in report


def test_a_drop_in_feature_fails_until_it_is_promoted(setup):
    tmp_path, db, new, served = setup
    meta_path = new / "bfsp_model_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["feature_cols"] = meta["feature_cols"][:-1] + ["fv_nfp_e3"]
    meta_path.write_text(json.dumps(meta))
    code, report = _verify(tmp_path, db, new, served)
    assert code == 1 and "drop-in block form_variants" in report
