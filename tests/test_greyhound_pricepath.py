"""greyhound/pit/pricepath.py (review D3, D4): planted steamers, attention and place ratios are recovered from past
runs only; a race's own prices and result never move its row; a price column not held stays NaN with "no data
found". Constructed frames for the unit tests only."""
import numpy as np
import pandas as pd

from greyhound.pit import core, pricepath
from tests.test_greyhound_pit import synth

STEAMER, LOUD, LOOSE = 3, 5, 7


def prices_for(runs, seed=1):
    """GB win and place price rows for every race run: the STEAMER's BSP 30% under its morning WAP (others level),
    LOUD gets 6x the matched volume, LOOSE's place BSP twice as long (less 1) as the others'."""
    rng = np.random.default_rng(seed)
    r = runs[~runs.is_trial.fillna(False)].copy()
    bsp = r.sp_decimal.to_numpy(float) * np.exp(rng.normal(0, 0.05, len(r)))
    morning = bsp * np.exp(rng.normal(0, 0.05, len(r)))
    morning = np.where(r.dog_id == STEAMER, bsp / 0.7, morning)
    pp = bsp * np.exp(rng.normal(0, 0.03, len(r)))
    vol = 100.0 / bsp * np.where(r.dog_id == LOUD, 6.0, 1.0) * np.exp(rng.normal(0, 0.1, len(r)))
    place = 1.0 + (bsp - 1.0) / 3.0 * np.where(r.dog_id == LOOSE, 2.0, 1.0) * np.exp(rng.normal(0, 0.03, len(r)))
    base = {"race_date": r.race_date.to_numpy(), "race_time": r.race_time.to_numpy(), "track": r.track.to_numpy(),
            "trap": r.trap.to_numpy(), "country": "GB"}
    win = pd.DataFrame({**base, "market": "win", "bsp": bsp, "morningwap": morning, "ppwap": pp, "pp_vol": vol})
    plc = pd.DataFrame({**base, "market": "place", "bsp": place})
    return win, plc


def build(runs, win, plc):
    return pricepath.build(runs, None, {"prices": win, "place_prices": plc})


def late_rows(runs, tab, dog):
    return tab[(runs.dog_id == dog).to_numpy() & (runs.race_date > "2025-01-20").to_numpy()]


def test_planted_effects_are_recovered(monkeypatch):
    monkeypatch.setattr(pricepath, "MIN_EXPECT", 10)
    runs = synth()
    win, plc = prices_for(runs)
    tab, notes = build(runs, win, plc)
    assert len(tab) == len(runs) and (tab.index == runs.index).all()
    others = tab[~runs.dog_id.isin([STEAMER, LOUD, LOOSE]).to_numpy() & (runs.race_date > "2025-01-20").to_numpy()]
    st = late_rows(runs, tab, STEAMER)
    assert st.px_steam_ewm.mean() < -0.25 and abs(others.px_steam_ewm.mean()) < 0.05
    assert late_rows(runs, tab, LOUD).px_attn_ewm.mean() > others.px_attn_ewm.mean() + 1.0
    assert late_rows(runs, tab, LOUD).px_attn_xs_ewm.mean() > 0.9 and abs(others.px_attn_xs_ewm.mean()) < 0.3
    assert late_rows(runs, tab, LOOSE).px_plr_ewm.mean() > 0.5 and abs(others.px_plr_ewm.mean()) < 0.1
    assert notes["place_bsp"] > 0 and notes["win_morningwap"] > 0
    first = runs.groupby("dog_id").race_date.transform("min") == runs.race_date
    assert tab.px_steam_ewm[first.to_numpy()].isna().all() and (tab.px_n[first.to_numpy()] == 0).all()


def test_a_races_own_prices_and_result_never_move_its_row():
    runs = synth()
    win, plc = prices_for(runs)
    last = runs.race_date.max()
    a, _ = build(runs, win, plc)
    w2, p2 = win.copy(), plc.copy()
    on = w2.race_date == last
    w2.loc[on, ["bsp", "morningwap", "ppwap", "pp_vol"]] = [50.0, 2.0, 3.0, 9999.0]
    p2.loc[p2.race_date == last, "bsp"] = 40.0
    r2 = runs.copy()
    r2.loc[r2.race_date == last, "position"] = 7 - r2.loc[r2.race_date == last, "position"]
    b, _ = build(r2, w2, p2)
    pd.testing.assert_frame_equal(a, b)
    # the card: the last day's races with no result and no prices give the rows the historic build gave
    card = runs.copy()
    lastm = (card.race_date == last).to_numpy()
    card.loc[lastm, "is_card"] = True
    card.loc[lastm, ["position", "won", "sp_decimal"]] = np.nan
    c, _ = build(card, win[win.race_date != last], plc[plc.race_date != last])
    pd.testing.assert_frame_equal(a[lastm], c[lastm])


def test_a_column_not_held_is_nan_with_no_data_found():
    runs = synth(days=15)
    win, _ = prices_for(runs)
    tab, notes = build(runs, win.drop(columns=["morningwap"]), None)
    assert tab.px_steam_ewm.isna().all() and tab.px_plr_ewm.isna().all()
    assert notes["win_morningwap"] == "no data found" and notes["place_bsp"] == "no data found"
    assert tab.px_late_ewm.notna().any()
    tab, notes = pricepath.build(runs, None, {})
    assert notes["win_bsp"] == "no data found" and tab.drop(columns=["race_id", "dog_id", "px_n", "px_plr_n"]).isna().all().all()


def test_history_reads_earlier_days_only():
    df = pd.DataFrame({"dog_id": [1, 1, 1, 1], "race_date": ["2025-01-01", "2025-01-02", "2025-01-02", "2025-01-05"],
                       "v": [1.0, 3.0, 5.0, np.nan]})
    h = pricepath.history(df, ["v"], hl=1.0)
    assert np.isnan(h.v_last[0]) and h.v_n[0] == 0
    assert h.v_last[1] == 1.0 and h.v_last[2] == 1.0                     # never the same day
    assert h.v_last[3] == 4.0 and h.v_n[3] == 2                           # the day's mean
    # half-life 1 run: weights 1/2 and 1 over (1, 4)
    assert np.isclose(h.v_ewm[3], (0.5 * 1 + 1 * 4) / 1.5)


def test_prefix():
    from greyhound.pit import racectx
    assert pricepath.PREFIX == "px_" and racectx.PREFIX == "rx_"
    assert all(c.startswith("px_") for c in pricepath.COLUMNS)
