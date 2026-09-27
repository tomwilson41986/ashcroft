"""model/blocks/future_form.py by hand: how the rivals from a horse's recent races ran next, back and forward.

Synthetic frames test mechanics only; nothing is trained or evaluated on them.
"""
import numpy as np
import pandas as pd
import pytest

from model.blocks import form_lines, form_uplift, future_form as ff


def _race(day, order, dnf=(), bsp=None, rid=None, rating=None, beaten=None):
    """One race: `order` finishes in that order, `dnf` did not finish; BSPs equal unless given."""
    d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=day)
    horses = list(order) + list(dnf)
    bsp = bsp or {h: float(len(horses)) for h in horses}
    rows = []
    for i, h in enumerate(horses):
        fin = i < len(order)
        rows.append(dict(raceid=rid or f"r{day}", race_date=d, race_time="2.30", track="York", horse_name=h,
                         number_of_runners=len(horses), placing_numerical=float(i + 1) if fin else np.nan,
                         bfsp=bsp[h], official_rating=(rating or {}).get(h, np.nan),
                         LB=(0.0 if i == 0 else float((beaten or {}).get(h, i))) if fin else np.nan,
                         dist_furlongs=8.0))
    return rows


def _card(day, horses, rid=None):
    rows = _race(day, horses, rid=rid)
    for r in rows:
        r.update(placing_numerical=np.nan, bfsp=np.nan, LB=np.nan)
    return rows


def _at(out, day, horse):
    return out.set_index(["race_date", "horse_name"]).loc[(pd.Timestamp("2025-06-01") + pd.Timedelta(days=day), horse)]


def _rivals_frame():
    """A (day 0) with B, C, D; B then wins its next five races, C runs twice unplaced, D never again."""
    rows = _race(0, ["A", "B", "C", "D"])
    for d in range(1, 6):
        rows += _race(d, ["B", f"x{d}"] if d > 2 else ["B", "C", f"x{d}"])
    return rows + _card(6, ["A", "Z"])


def test_each_forward_window_counts_that_many_of_each_rivals_next_runs_before_today():
    out = ff.build(pd.DataFrame(_rivals_frame()))
    a = _at(out, 6, "A")
    # B's next runs: days 1-5, all won; C's: days 1-2, second both times; D: none
    assert a["ff_l1_n1_n"] == 2 and a["ff_l1_n2_n"] == 4 and a["ff_l1_n3_n"] == 5 and a["ff_l1_n5_n"] == 7
    wins = {1: 1, 2: 2, 3: 3, 5: 5}
    for f, n in ((1, 2), (2, 4), (3, 5), (5, 7)):
        assert np.isclose(a[f"ff_l1_n{f}_wr"], (wins[f] + ff.K * ff.P_WIN) / (n + ff.K))
    z = _at(out, 6, "Z")
    assert all(np.isnan(z[c]) for c in ff.FEATURES)              # a debutant has no race to read
    assert _at(out, 0, "A")[ff.FEATURES].isna().all()           # nor has anyone on their first day


def test_the_day_itself_never_counts():
    rows = _race(0, ["A", "B"]) + _race(1, ["A", "B"])          # B wins again on the day A runs next
    out = ff.build(pd.DataFrame(rows))
    a = _at(out, 1, "A")
    assert a["ff_l1_n5_n"] == 0 and np.isclose(a["ff_l1_n5_wr"], ff.P_WIN)


def test_the_windows_back_pool_the_last_races_and_leave_out_the_horses_own_runs():
    rows = (_race(0, ["A", "C"]) + _race(1, ["A", "D"]) + _race(2, ["A", "E"]) + _race(3, ["A", "F"])
            + _race(4, ["A", "G"]) + _race(5, ["C", "D", "E", "F", "G"]) + _card(6, ["A", "Q"]))
    out = ff.build(pd.DataFrame(rows))
    a = _at(out, 6, "A")
    # every rival ran once more (day 5, one race); A's own runs since each race never count
    assert a["ff_l1_n1_n"] == 1 and a["ff_l3_n1_n"] == 3 and a["ff_l5_n1_n"] == 5
    assert a["ff_l5_n5_n"] == 5                                 # A's own later runs are left out at every depth
    # C, first on day 5, came from A's oldest race: only the five-race window sees its win
    assert np.isclose(a["ff_l5_n1_wr"], (1 + ff.K * ff.P_WIN) / (5 + ff.K))
    assert np.isclose(a["ff_l3_n1_wr"], (0 + ff.K * ff.P_WIN) / (3 + ff.K))


def test_the_cells_form_lines_and_form_uplift_already_read_agree_with_them():
    rng = np.random.default_rng(7)
    rows, horses = [], [f"h{i}" for i in range(30)]
    for d in range(40):
        field = list(rng.choice(horses, size=6, replace=False))
        rating = {h: float(rng.integers(50, 90)) for h in field}
        beaten = {h: float(rng.uniform(0.5, 6)) * (i + 1) / 2 for i, h in enumerate(field)}
        bsp = {h: float(rng.uniform(2, 30)) for h in field}
        rows += _race(d, field[:5], dnf=field[5:], bsp=bsp, rating=rating, beaten=beaten)
    df = pd.DataFrame(rows)
    got = ff.build(df)
    fl = form_lines.build(df)
    fu = form_uplift.build(df)
    for w in ("l1", "l3"):
        np.testing.assert_allclose(got[f"ff_{w}_n3_n"], fl[f"fl_{w}_n"], equal_nan=True)
        np.testing.assert_allclose(got[f"ff_{w}_n3_wr"], fl[f"fl_{w}_wr"], equal_nan=True)
        np.testing.assert_allclose(got[f"ff_{w}_n3_ae"], fl[f"fl_{w}_ae"], equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(got[f"ff_{w}_n3_perf"], fu[f"fu_{w}_perf"], equal_nan=True, atol=1e-9)


def test_finishing_positions_and_figures_count_only_runs_that_have_one():
    rows = _race(0, ["A", "B", "C"]) + _race(1, ["B", "x"], dnf=["C"]) + _card(2, ["A", "Q"])
    out = ff.build(pd.DataFrame(rows))
    a = _at(out, 2, "A")
    assert a["ff_l1_n1_n"] == 2                                 # B won, C did not finish: two runs
    assert np.isclose(a["ff_l1_n1_wr"], (1 + ff.K * ff.P_WIN) / (2 + ff.K))
    # one finishing position known (B's win, NFP 1): (1 + 5 x 0.5) / (1 + 5)
    assert np.isclose(a["ff_l1_n1_nfp"], (1 + ff.K * ff.NFP0) / (1 + ff.K))


def test_the_revised_figure_is_the_horses_own_plus_how_its_race_has_worked_out():
    rating = {"A": 70.0, "B": 60.0}
    rows = (_race(0, ["B", "A"], rating=rating, beaten={"A": 2.0})         # A beaten 2 lengths by B
            + _race(1, ["B", "y"], rating={"B": 60.0, "y": 60.0})          # B wins again
            + _card(2, ["A", "Q"]))
    out = ff.build(pd.DataFrame(rows))
    a = _at(out, 2, "A")
    assert np.isfinite(a["ff_l1_n1_perf"]) and np.isfinite(a["ff_l1_n1_adj"])
    from model.form_windows import run_measures
    m = run_measures(pd.DataFrame(rows))
    a_perf_day0 = m["perf"][1]
    assert np.isclose(a["ff_l1_n1_adj"], a_perf_day0 + a["ff_l1_n1_perf"])


@pytest.mark.parametrize("shuffle_seed", [1, 2])
def test_row_order_does_not_move_a_value(shuffle_seed):
    df = pd.DataFrame(_rivals_frame())
    a = ff.build(df)
    b = ff.build(df.sample(frac=1.0, random_state=shuffle_seed)).loc[a.index]
    pd.testing.assert_frame_equal(a[ff.FEATURES], b[ff.FEATURES])
