"""The owner's NFP (z-score over the field, divided by 3) by hand (made-up runs; nothing scored)."""

import numpy as np
import pandas as pd
import pytest

from model.blocks import nfp_z


def test_the_formula_by_hand():
    n = np.array([2, 5, 20, 10, 10, 1])
    f = np.array([1, 5, 1, 4, np.nan, 1])
    z = nfp_z.nfp_z(n, f)
    assert z[0] == pytest.approx(1 / 3)                                   # a two-runner win
    assert z[1] == pytest.approx(-np.sqrt(4 / 18))                        # last of five
    assert z[2] == pytest.approx(np.sqrt(19 / 63))                        # a twenty-runner win
    nfp = (10 - 4) / 9
    assert z[3] == pytest.approx((2 * nfp - 1) * np.sqrt(9 / 33))
    assert np.isnan(z[4]) and np.isnan(z[5])                              # no finish; a field of one
    for size in (3, 8, 16):
        zz = nfp_z.nfp_z(np.full(size, size), np.arange(1, size + 1))
        assert zz.mean() == pytest.approx(0.0, abs=1e-12) and zz.std() == pytest.approx(1 / 3)


def test_a_horse_by_hand():
    days = pd.to_datetime(["2026-01-01", "2026-01-08", "2026-01-15", "2026-01-22"])
    rows = []
    for k, (d, place) in enumerate(zip(days, [1, 3, 2, 4])):
        for j in range(5):                                                # five-runner races; ours is runner 0
            rows.append(dict(raceid=f"r{k}", race_date=d, race_time="2.00", track="York",
                             horse_name="Ours" if j == 0 else f"Other{k}{j}", number_of_runners=5,
                             placing_numerical=place if j == 0 else [p for p in range(1, 6) if p != place][j - 1]))
    df = pd.DataFrame(rows)
    out = nfp_z.build(df)
    ours = out[out.horse_name.eq("Ours")].reset_index(drop=True)
    v = nfp_z.nfp_z(np.full(4, 5), np.array([1, 3, 2, 4]))
    assert np.isnan(ours.loc[0, "nz_l1"]) and np.isnan(ours.loc[0, "nz_car"])   # no history before the first run
    assert ours.loc[1, "nz_l1"] == pytest.approx(v[0])
    assert ours.loc[3, "nz_l1"] == pytest.approx(v[2])
    assert ours.loc[3, "nz_car"] == pytest.approx(v[:3].mean())
    assert ours.loc[3, "nz_m3"] == pytest.approx(v[:3].mean())
    assert ours.loc[3, "nz_w5"] == pytest.approx((5 * v[2] + 4 * v[1] + 3 * v[0]) / 12)
