"""model/blocks/connection_relative.py by hand: connection windows read against today's field."""
import numpy as np
import pandas as pd

from model import blocks
from model.blocks import connection_relative as cr
from tests.test_connection_windows_block import _race


def test_each_runners_connection_form_against_the_field():
    rows = (_race(0, [("a1", "T1", "J1"), ("b1", "T2", "J2"), ("c1", "T3", "J3")])
            + _race(1, [("a2", "T1", "J1"), ("b2", "T2", "J2"), ("c2", "T3", "J3")])
            + _race(2, [("a3", "T1", "J1"), ("b3", "T2", "J2"), ("c3", "T3", "J3")], result=False))
    out, cols = blocks.attach(pd.DataFrame(rows), ["connection_relative"])
    assert cols == cr.FEATURES
    today = out[out["race_date"] == pd.Timestamp("2025-06-03")].set_index("horse_name")
    v = today["cw_tr_nfp_l20"]                          # T1 won twice (1), T2 second twice (0.5), T3 last (0)
    assert np.allclose(v[["a3", "b3", "c3"]], [1.0, 0.5, 0.0])
    z = (v - v.mean()) / v.std()
    assert np.allclose(today["cwr_tr_nfp_l20_z"], z[today.index])
    assert np.allclose(today["cwr_tr_nfp_l20_gap"], v - v.max())
    assert today.loc["a3", "cwr_tr_nfp_l20_gap"] == 0


def test_a_field_with_no_readings_gets_none_and_a_level_field_gets_zero():
    rows = _race(0, [("a1", "T1", "J1"), ("b1", "T2", "J2")], result=False)
    out, _ = blocks.attach(pd.DataFrame(rows), ["connection_relative"])
    assert out["cwr_tr_nfp_l20_z"].isna().all()         # no earlier runners: no last-20 reading
    assert (out["cwr_tr_nfp_car_z"] == 0).all()          # the career NFP is the prior for both: level
