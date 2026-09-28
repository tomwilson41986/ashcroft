"""The market-history and seasonal blocks on small hand-built histories.

Synthetic frames test mechanics only; nothing is trained or evaluated on them.
tests/test_feature_blocks.py runs both through the generic lag, card, order and READS tests (the
seasonal block at a short reach); these check the values, and the seasonal block's real reach.
"""
import numpy as np
import pandas as pd
import pytest

from model.blocks import market_history, seasonal


def _run(date, horse, *, track="York", time="2.00", trainer="T1", jockey="J1", pos=2, runners=10, bsp=5.0,
         sp=None, place=None, places=3):
    row = {"race_date": pd.Timestamp(date), "race_time": time, "track": track, "horse_name": horse,
           "trainer": trainer, "jockey_name": jockey, "placing_numerical": pos, "number_of_runners": runners,
           "bfsp": bsp, "odds": (bsp - 1.0) if sp is None else sp, "race_code": "Flat",
           "bfsp_place": (1.0 + (bsp - 1.0) / 4.0) if place is None else place, "bf_plcs_paid": places,
           "plcs_paid": places}
    row["raceid"] = f"{row['race_date']:%Y-%m-%d}|{track}|{time}"
    return row


def _field(date, time, horses, **kw):
    return [_run(date, h, time=time, pos=i + 1, **kw) for i, h in enumerate(horses)]


# --------------------------------------------------------------------------- market history

def _market_history(x_sp=None, x_place=None):
    """Forty days of ten-runner races at one price: every runner's SP and place price at the usual
    spread except horse X's, which runs every fifth day."""
    rows = []
    for d in range(40):
        date = pd.Timestamp("2025-03-01") + pd.Timedelta(days=d)
        horses = [f"h{d}_{i}" for i in range(10)]
        if d % 5 == 0:
            horses[0] = "X"
        for i, h in enumerate(horses):
            kw = {}
            if h == "X":
                kw = {"sp": x_sp, "place": x_place}
            rows.append(_run(date, h, time="2.00", pos=i + 1, trainer="TX" if h == "X" else f"T{i}", **kw))
    return pd.DataFrame(rows)


def test_a_horse_the_bookmakers_backed_reads_a_positive_spread_and_the_rest_nothing():
    df = _market_history(x_sp=2.0)                       # X's SP 2/1 when its BSP (5.0) says 4/1
    out = market_history.build(df)
    x = out[df["horse_name"] == "X"]
    last = x.iloc[-1]
    assert last["mh_spb_h"] > 0.5                        # log(4/2) = 0.69 less a cell mean near zero
    assert last["mh_spb_h_n"] > 3
    assert last["mh_spb_tr"] > 0                         # its yard's (only X) the same way, shrunk
    assert abs(last["mh_spb_tr"]) < abs(last["mh_spb_h"])
    others = out[(df["horse_name"] != "X") & (df["race_date"] > "2025-03-20")]
    assert np.nanmax(np.abs(others["mh_spb_jk"].to_numpy(float))) < 0.1   # one rider: the whole field, near 0


def test_a_horse_the_place_market_liked_reads_a_positive_ratio():
    df = _market_history(x_place=1.5)                    # place 1/2 when the win price is 4/1: short
    out = market_history.build(df)
    last = out[df["horse_name"] == "X"].iloc[-1]
    assert last["mh_wp_h"] > 0.5                         # log(4 / 0.5) - log(4 / 1) = 0.69, less the cell's
    assert np.isnan(out.iloc[0]["mh_wp_h"])              # nothing before the first day


def test_the_cell_baseline_needs_earlier_days():
    df = _market_history(x_sp=2.0)
    out = market_history.build(df)
    first_x = out[df["horse_name"] == "X"].iloc[0]
    assert np.isnan(first_x["mh_spb_h"]) and first_x["mh_spb_h_n"] == 0.0


# --------------------------------------------------------------------------- seasonal

def _season_history():
    """A horse that wins in May and runs down the field in the autumn, over two years, among
    fixed-size fields of others."""
    rows = []
    for date, pos in (("2024-05-10", 1), ("2024-05-30", 1), ("2024-10-05", 9), ("2024-11-02", 10),
                      ("2025-05-15", 3), ("2025-06-20", 2), ("2025-10-10", 8)):
        rows += _field(date, "2.00", [f"o{date}_{i}" for i in range(9)] + ["S"], trainer="TS")
        # put S at its position: swap placings so S finishes `pos`
        block = rows[-10:]
        s = next(r for r in block if r["horse_name"] == "S")
        other = next(r for r in block if r["placing_numerical"] == pos)
        s["placing_numerical"], other["placing_numerical"] = pos, s["placing_numerical"]
    return pd.DataFrame(rows)


def test_the_horses_form_at_this_time_of_year_in_earlier_years():
    df = _season_history()
    out = seasonal.build(df)
    s = out[df["horse_name"] == "S"].set_index(df.loc[df["horse_name"] == "S", "race_date"])
    may25 = s.loc[pd.Timestamp("2025-05-15")]
    assert may25["sn_n"] == 2 and may25["sn_nfp"] == pytest.approx(1.0)       # the two May 2024 wins
    all_before = np.mean([1.0, 1.0, 1 / 9, 0.0])                              # its four 2024 runs
    assert may25["sn_nfp_vs_all"] == pytest.approx(1.0 - all_before)
    jun25 = s.loc[pd.Timestamp("2025-06-20")]
    assert jun25["sn_n"] == 2                            # May 2024 is inside June's window; May 2025 is too recent
    oct25 = s.loc[pd.Timestamp("2025-10-10")]
    assert oct25["sn_n"] == 2 and oct25["sn_nfp"] == pytest.approx((1 / 9 + 0.0) / 2)
    assert np.isnan(s.loc[pd.Timestamp("2024-05-30")]["sn_nfp"])              # 20 days is not an earlier year


def test_a_run_less_than_the_reach_before_does_not_count():
    rows = _field("2025-05-01", "2.00", ["S"] + [f"a{i}" for i in range(9)])   # S wins on 1 May
    rows += _field("2025-06-15", "2.00", [f"b{i}" for i in range(9)] + ["S"])
    rows += _field("2026-05-20", "2.00", [f"c{i}" for i in range(9)] + ["S"])
    df = pd.DataFrame(rows)
    out = seasonal.build(df)
    s = out[df["horse_name"] == "S"]
    assert s["sn_n"].tolist() == [0.0, 0.0, 2.0]        # 45 days is inside the reach; a year is not
    assert s.iloc[2]["sn_nfp"] == pytest.approx((1.0 + 0.0) / 2)


def test_the_yards_winners_against_the_price_in_these_months_shrunk():
    rows = []
    for y in (2023, 2024):
        rows += _field(f"{y}-04-10", "2.00", ["w1"] + [f"x{y}_{i}" for i in range(9)], trainer="TY", bsp=5.0)
    rows += _field("2025-04-12", "2.00", ["q"] + [f"z{i}" for i in range(9)], trainer="TY", bsp=5.0)
    df = pd.DataFrame(rows)
    out = seasonal.build(df)
    q = out[df["horse_name"] == "q"].iloc[0]
    # 20 runners in April of earlier years, 2 winners, each priced at 1/5: A-E per runner 0.1 - 0.2
    ae = (2 * (1 - 0.2) + 18 * (0 - 0.2)) / 20
    assert q["sn_tr_ae"] == pytest.approx(ae * 20 / (20 + seasonal.K_TRAINER))
