"""The sales block by hand: which sales the table keeps, how a runner finds its horse, and the
field-relative readings. Fabricated rows exercise the code paths only; nothing is scored on them."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from betfair_prices import normalise_horse  # noqa: E402
from build_timeform_sales import sales_in  # noqa: E402
from model.blocks import sales  # noqa: E402


@pytest.mark.parametrize("name", ["1. Casts Tasha (IRE)", "ADMIRAL HALSEY (IRE)", "Raging Bull", "Mrs O'Hara (FR)", ""])
def test_the_block_names_horses_as_betfair_prices_does(name):
    assert sales.norm_name(name) == normalise_horse(name)


def test_only_sales_before_racing_are_kept():
    # Flat: foal, yearling, breeze-up kept, the last of them is the last sale; a three-year-old sale is not
    s = sales_in("€42,000Y, 400,000 2-y-o, €50,000 3-y-o: fourth foal", jumps=False)
    assert s == {"yearling": pytest.approx(42000 * 0.85), "breeze": pytest.approx(400000 * 1.05), "_last": "breeze"}
    # Jumps: three- and four-year-old stores kept, a five-year-old sale is not
    assert sales_in("£20,000 4-y-o, £60,000 5-y-o: dam winner", jumps=True) == {"store": 20000.0, "_last": "store"}
    assert sales_in("€8,500 3-y-o: fifth foal", jumps=True)["store"] == pytest.approx(8500 * 0.85)
    assert sales_in("€8,500 3-y-o: fifth foal", jumps=False) == {}
    # Only the opening clause is read
    assert sales_in("half-sister to 3 winners: sold 50,000Y at the sales", jumps=False) == {}


@pytest.fixture
def table(tmp_path, monkeypatch):
    t = pd.DataFrame([
        # name, foaled, foal, yearling, breeze, store, last, last kind, comment
        ("alpha", 2022, np.nan, 100000, np.nan, np.nan, 100000, 1, 1),
        ("beta", 2022, np.nan, 10000, 50000, np.nan, 50000, 2, 1),
        ("gamma", 2022, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1),     # known, never sold
        ("delta", 2021, 3000, np.nan, np.nan, np.nan, 3000, 0, 1),             # found by name, a year out
        ("echo", 2015, np.nan, 9000, np.nan, np.nan, 9000, 1, 1),              # two echoes: never by name alone
        ("echo", 2019, np.nan, 7000, np.nan, np.nan, 7000, 1, 1),
    ], columns=["name_norm", "foal_year", "foal_gbp", "yearling_gbp", "breeze_gbp", "store_gbp", "last_gbp",
                "last_kind", "has_comment"])
    path = tmp_path / "sales.csv"
    t.to_csv(path, index=False)
    monkeypatch.setattr(sales, "TABLE_PATH", str(path))
    return t


def test_a_race_by_hand(table):
    race = pd.DataFrame({"raceid": "r1", "race_date": pd.Timestamp("2025-08-01"), "race_time": "2.00",
                         "track": "York", "horse_name": ["Alpha (IRE)", "Beta", "Gamma", "Delta", "Echo", "Zulu"],
                         "horse_age": [3, 3, 3, 3, 3, 3]})
    out = sales.build(race).set_index("horse_name")
    np.testing.assert_array_equal(out["sl_known"].to_numpy(), [1, 1, 1, 1, np.nan, np.nan])
    np.testing.assert_array_equal(out["sl_sold"].to_numpy(), [1, 1, 0, 1, np.nan, np.nan])
    assert out.loc["Alpha (IRE)", "sl_ln_price"] == pytest.approx(np.log(100000))
    assert out.loc["Beta", "sl_ln_breeze"] == pytest.approx(np.log(50000))
    assert out.loc["Beta", "sl_ln_yearling"] == pytest.approx(np.log(10000))
    assert out.loc["Beta", "sl_kind"] == 2 and out.loc["Delta", "sl_kind"] == 0
    # Delta's age puts it in 2022's crop; the one Delta in the file is 2021's: found, a year out
    assert out.loc["Delta", "sl_ln_price"] == pytest.approx(np.log(3000))
    # Three priced runners: Alpha dearest, Delta cheapest
    assert out.loc["Alpha (IRE)", "sl_price_rank"] == pytest.approx(1 / 3)
    assert out.loc["Delta", "sl_price_rank"] == pytest.approx(1.0)
    ln = np.log([100000, 50000, 3000])
    assert out.loc["Beta", "sl_price_z"] == pytest.approx((ln[1] - ln.mean()) / ln.std(ddof=1))
    assert np.isnan(out.loc["Gamma", "sl_price_z"])
    # Four known, three of them sold
    assert out["sl_race_sold_share"].iloc[0] == pytest.approx(0.75)


def test_nothing_is_read_on_or_before_the_snapshot(table):
    """The file is a snapshot whose sale clauses depend on careers run after a race before it."""
    race = pd.DataFrame({"raceid": ["a", "a", "b", "b"], "race_time": "2.00", "track": "York",
                         "race_date": pd.to_datetime(["2025-07-03", "2025-07-03", "2025-07-04", "2025-07-04"]),
                         "horse_name": ["Alpha", "Beta", "Alpha", "Beta"], "horse_age": 3})
    out = sales.build(race)
    assert out.loc[:1, sales.FEATURES].isna().all().all()
    assert out.loc[2, "sl_known"] == 1 and out.loc[2, "sl_ln_price"] == pytest.approx(np.log(100000))
