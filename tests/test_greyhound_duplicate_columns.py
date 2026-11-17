"""research/greyhound/duplicate_columns.py: on a constructed matrix it finds an exact copy (lr_ae_last = gp_ae_l1,
float32 against float64), a near copy (r above 0.999), a copy on a column with few rows present and a negated
copy; it leaves alone a correlated pair under the threshold, keys and labels; it suggests dropping the table's column
against the engine's; and it flags a suggested name that is the prefix of another column. Constructed data only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.greyhound import duplicate_columns as dc


def frame(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    sparse = np.where(rng.random(n) < 0.05, rng.normal(size=n), np.nan)
    gp = np.where(rng.random(n) < 0.1, np.nan, x)
    return pd.DataFrame({
        "race_id": np.arange(n) // 6, "dog_id": np.arange(n),
        "gp_ae_l1": gp, "lr_ae_last": gp.astype(np.float32),                          # exact, float32
        "hb_speed": x + rng.normal(0, 0.01, n), "zz_speed_copy": x,                 # near (r ~ 0.99995)
        "kf_level": -x * 2.0 + 1.0,                                                   # an exact negated rescale
        "fg_sparse": sparse, "op_sparse_copy": sparse * 3.0 + 0.5,                    # few rows: still found
        "ab_related": x + rng.normal(0, 0.2, n),                                      # r ~ 0.98: left alone
        "won": (x > 1).astype(float), "won_copy_label": (x > 1).astype(float),
        "lr_ae": rng.normal(size=n), "lr_ae_x": rng.normal(size=n),
        "const": np.ones(n),
    })


def pairs(res, kind):
    return {tuple(sorted((p["a"], p["b"]))): p for p in res[kind]}


def test_finds_exact_and_near_duplicates():
    df = frame()
    res = dc.find_duplicates(df, exclude=("won", "won_copy_label"))
    ex, near = pairs(res, "exact"), pairs(res, "near")
    assert ("gp_ae_l1", "lr_ae_last") in ex and ex[("gp_ae_l1", "lr_ae_last")]["drop"] == "lr_ae_last"
    allp = {**ex, **near}
    for p in [("hb_speed", "zz_speed_copy"), ("kf_level", "zz_speed_copy"), ("fg_sparse", "op_sparse_copy")]:
        assert p in allp, p
    assert near[("kf_level", "zz_speed_copy")]["corr"] < -0.999
    assert not any("ab_related" in p for p in allp)
    assert not any("won" in c for p in allp for c in p) and not any("race_id" in c for p in allp for c in p)
    assert "op_sparse_copy" in res["drop"] and "lr_ae_last" in res["drop"]
    assert "const" in res["constant_in_sample"]
    json.dumps(res)


def test_flags_a_dropped_name_that_is_a_prefix():
    df = frame()
    df["gp_ae_mean"] = df.lr_ae.to_numpy()                             # the table's lr_ae copies an engine column
    res = dc.find_duplicates(df, exclude=("won", "won_copy_label"))
    assert "lr_ae" in res["drop"] and res["prefix_clash"]["lr_ae"] == ["lr_ae_last", "lr_ae_x"]


def test_the_cli_reads_a_matrix_and_tables(tmp_path):
    df = frame()
    eng = df.drop(columns=["lr_ae_last"])
    eng.to_parquet(tmp_path / "features.parquet", index=False)
    df[["race_id", "dog_id", "lr_ae_last"]].sample(frac=1.0, random_state=1).to_parquet(tmp_path / "lr.parquet")
    res = dc.main(["--matrix", str(tmp_path / "features.parquet"), "--tables", str(tmp_path / "lr.parquet"),
                   "--out", str(tmp_path / "d.json")])
    assert ("gp_ae_l1", "lr_ae_last") in pairs(res, "exact")
    assert not any("won" in c for p in res["exact"] + res["near"] for c in (p["a"], p["b"]))     # labels left out
    assert json.loads((tmp_path / "d.json").read_text())["drop"] == res["drop"]
