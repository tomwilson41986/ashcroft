"""The two screens of the Blandford repositories' data: the sale-price reader and the
point-in-time lag of the sectionals. Fabricated rows exercise the code paths only."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from external_sales_screen import parse_sale  # noqa: E402
from external_sectionals_screen import lag_features  # noqa: E402


@pytest.mark.parametrize("comment, price, kind", [
    ("€42,000Y: second foal", 42000 * 0.85, "yearling"),
    ("180,000Y, €400,000 2-y-o: fourth foal", 400000 * 0.85, "2yo"),     # the last sale before the colon
    ("11,000F: fourth foal", 11000 * 1.05, "foal"),                       # bare prices are guineas
    ("£52,000Y: first foal", 52000.0, "yearling"),
    ("35,000 gns Y: dam unraced", 35000 * 1.05, "yearling"),
    ("fourth foal: half-sister to 3 winners, 50,000Y", None, None),       # a price after the colon is not the sale
    (None, None, None),
])
def test_parse_sale(comment, price, kind):
    p, k = parse_sale(comment)
    assert k == kind
    if price is None:
        assert p is None
    else:
        assert p == pytest.approx(price)


def _history(rows):
    cols = ["fsp", "fsp_rel", "fsp_vs_pos", "early_rel", "late_rel", "s0p", "secPct", "timePct", "sl", "slPct",
            "cad", "cadPct", "pos_pct", "r_tf", "r_tfig", "r_rpr"]
    h = pd.DataFrame([{"horse_norm": hn, "date": pd.Timestamp(d), **{c: v for c in cols}} for hn, d, v in rows])
    return h.sort_values(["horse_norm", "date"]).reset_index(drop=True)


def test_lag_reads_only_earlier_days():
    h = _history([("a", "2024-07-01", 1.0), ("a", "2024-07-10", 5.0), ("b", "2024-07-10", 9.0)])
    sample = pd.DataFrame({"row": [0, 1, 2, 3], "horse_norm": ["a", "a", "b", "c"],
                           "date": pd.to_datetime(["2024-07-10", "2024-07-11", "2024-07-10", "2024-07-11"])})
    m = lag_features(sample, h)
    # On 10 Jul horse a sees only its 1 Jul run: the run that day is the race itself
    assert m.loc[0, "fsp_rel_last"] == 1.0 and m.loc[0, "sec_n"] == 1 and m.loc[0, "sec_days"] == 9
    # The next day it sees both
    assert m.loc[1, "fsp_rel_last"] == 5.0 and m.loc[1, "sec_n"] == 2 and m.loc[1, "fsp_rel_mean3"] == 3.0
    # A run on the race day is never read, and a horse with no runs has none
    assert np.isnan(m.loc[2, "fsp_rel_last"]) and np.isnan(m.loc[3, "fsp_rel_last"])
