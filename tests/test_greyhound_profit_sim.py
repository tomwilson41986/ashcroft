"""The profit simulator (research/greyhound/profit_sim.py) on a small constructed day: the green-up price down the
book, the P&L of a matched lay (greened, closed at the SP, held), a lay matched past its queue and greened, the live
ledger replayed beside the simulation, and the commission on each race's net win."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.greyhound import profit_sim as ps
from sources.common import Store

OFF = pd.Timestamp("2026-11-03 15:00:00", tz="UTC")
MID = "1.250"


def at(s):
    return (OFF + pd.Timedelta(seconds=s)).isoformat()


def book(s, sid, back, lay, status="OPEN", source="live", runner_status="ACTIVE", sp=np.nan, inplay=False):
    r = {"polled_utc": at(s), "source": source, "market_id": MID, "status": status, "inplay": inplay,
         "selection_id": sid, "runner_status": runner_status, "last_traded": back[0][0] if back else np.nan,
         "runner_total_matched": 100.0, "market_total_matched": 600.0, "sp_actual": sp}
    for i in range(3):
        r[f"back{i + 1}"], r[f"back{i + 1}_size"] = back[i] if i < len(back) else (np.nan, np.nan)
        r[f"lay{i + 1}"], r[f"lay{i + 1}_size"] = lay[i] if i < len(lay) else (np.nan, np.nan)
    return r


def day():
    rows = [book(-120, 7, [(4.0, 30.0), (3.9, 50.0)], [(4.2, 20.0)]),          # we lay at 3.9: GBP50 ahead
            book(-30, 7, [(4.6, 40.0), (4.5, 40.0), (4.4, 100.0)], [(4.8, 20.0)]),
            book(-6, 7, [(5.0, 2.0), (4.9, 3.0), (4.8, 100.0)], [(5.1, 20.0)]),  # the green-up book: drifted
            book(-2, 7, [(5.5, 50.0)], [(5.6, 20.0)], inplay=True),               # in play: never read
            book(300, 7, [], [], status="CLOSED", source="final", runner_status="LOSER", sp=5.2),
            book(-120, 8, [(2.0, 50.0)], [(2.02, 50.0)]),
            book(300, 8, [], [], status="CLOSED", source="final", runner_status="WINNER", sp=1.9)]
    markets = pd.DataFrame([{"market_id": MID, "market_start_utc": OFF.isoformat()}])
    ladder = pd.DataFrame([{"polled_utc": at(s), "market_id": MID, "selection_id": 7, "price": 3.9, "traded": v}
                           for s, v in [(-200, 100.0), (-100, 130.0), (-60, 160.0)]])
    return ps.prepare(pd.DataFrame(rows), markets, ladder)


def test_the_green_price_goes_down_the_book_by_value():
    b = {"back": [(5.0, 2.0), (4.9, 3.0), (4.8, 100.0)]}
    # GBP39 of value: GBP10 at 5.0, GBP14.70 at 4.9, GBP14.30 at 4.8
    stake = 10 / 5.0 + 14.7 / 4.9 + 14.3 / 4.8
    assert np.isclose(ps.green_price(b, 39.0), 39.0 / stake)
    assert ps.green_price({"back": [(5.0, 2.0)]}, 20.0) == 5.0                 # beyond the depth: the last price
    assert ps.green_price({"back": []}, 10.0) is None


def test_a_matched_lay_greened_closed_at_the_sp_or_held():
    pnl, how = ps.lay_pnl(10.0, 3.9, 5.0, 5.2, 1)
    assert how == "green" and np.isclose(pnl, 10 * (1 - 3.9 / 5.0))              # the same whatever wins
    assert ps.lay_pnl(10.0, 3.9, None, 5.2, 1) == (10 * (5.2 - 3.9), "sp_close")
    assert ps.lay_pnl(10.0, 3.9, None, 5.2, 0) == (0.0, "sp_close")
    assert ps.lay_pnl(10.0, 3.9, None, None, 1) == (-10 * 2.9, "held")


def test_a_lay_matched_past_its_queue_and_greened():
    d = day()
    assert d["pre"].polled.max() < OFF                                          # nothing in play or after the off
    assert d["result"].set_index("selection_id").loc[8, "won"] == 1
    s = ps.simulate_lay(d, MID, 7, OFF - pd.Timedelta(seconds=110), 3.9, 10.0)
    # GBP50 queued ahead at 3.9; GBP60 traded there after we joined (100 -> 160): GBP10 past the queue, ours in full
    assert s["queue_ahead"] == 50.0 and s["matched"] == 10.0 and s["how"] == "green"
    s = ps.simulate_lay(d, MID, 7, OFF - pd.Timedelta(seconds=70), 3.9, 10.0)
    assert s["matched"] == 0.0 and s["pnl"] == 0.0                              # joined later: GBP30 traded since
    s = ps.simulate_lay(d, MID, 7, OFF - pd.Timedelta(seconds=110), 4.0, 10.0)
    assert s["queue_ahead"] == 30.0 and s["matched"] == 10.0                   # money traded beyond it, at 3.9


def test_a_trade_beyond_the_lay_fills_it_and_it_is_greened():
    d = day()
    d["ladder"] = pd.concat([d["ladder"], pd.DataFrame([{"polled_utc": at(s), "market_id": MID, "selection_id": 7,
                                                         "price": 3.8, "traded": v} for s, v in [(-200, 5.0),
                                                                                                 (-50, 15.0)]])])
    d["ladder"]["selection_id"] = d["ladder"].selection_id.astype(float)
    s = ps.simulate_lay(d, MID, 7, OFF - pd.Timedelta(seconds=110), 3.9, 10.0)
    assert s["matched"] == 10.0 and s["how"] == "green"
    B = ps.green_price(ps._book_at(d["pre"], MID, 7, OFF - pd.Timedelta(seconds=5)), 39.0)
    assert np.isclose(s["pnl"], round(10 * (1 - 3.9 / B), 2)) and s["pnl"] > 0


def test_the_live_ledger_replayed_beside_the_simulation(tmp_path, monkeypatch):
    d = day()
    d["ladder"] = pd.concat([d["ladder"], pd.DataFrame([{"polled_utc": at(-50), "market_id": MID, "selection_id": 7.0,
                                                         "price": 3.8, "traded": 15.0},
                                                        {"polled_utc": at(-200), "market_id": MID,
                                                         "selection_id": 7.0, "price": 3.8, "traded": 5.0}])])
    store = Store(root=tmp_path)
    led = pd.DataFrame([
        {"time_utc": at(-110), "action": "lay", "market_id": MID, "selection_id": 7, "track": "Romford",
         "race_time": "15:00", "price": 3.9, "size": 10.0},
        {"time_utc": at(-5), "action": "green", "market_id": MID, "selection_id": 7, "lay_matched": 10.0,
         "lay_price": 3.9, "locked_pnl": 2.0},
        {"time_utc": at(-110), "action": "lay", "market_id": MID, "selection_id": 8, "track": "Romford",
         "race_time": "15:00", "price": 1.95, "size": 10.0}])
    store.put("greyhound/early_lay/live/2026-11-03/ledger.csv", led.to_csv(index=False).encode())
    from datetime import date
    rp = ps.replay(store, date(2026, 11, 3), d).set_index("selection_id")
    assert rp.loc[7, "real_matched"] == 10.0 and rp.loc[7, "real_pnl"] == 2.0 and rp.loc[7, "matched"] == 10.0
    assert rp.loc[8, "real_matched"] == 0.0 and rp.loc[8, "real_pnl"] == 0.0     # never matched live
    assert rp.loc[8, "matched"] == 0.0                                          # nor in the simulation


def test_commission_on_each_races_net_win():
    rows = pd.DataFrame({"market_id": ["a", "a", "b"], "pnl": [5.0, -1.0, -3.0]})
    n = ps.race_net(rows).set_index("market_id").net
    assert np.isclose(n["a"], 4.0 * 0.98) and n["b"] == -3.0


def test_the_report_reads_without_live_orders():
    md = ps.report_md({"day": "2026-11-03", "strategies": {"live_rule": {"lays": 0}}})
    assert "Calibration" in md and "| live_rule | 0 |" in md


def test_the_early_lay_rule_runs_on_the_first_traded_books(tmp_path, monkeypatch):
    from datetime import date
    from greyhound import early_lay as el
    d = day()
    d["ladder"] = pd.concat([d["ladder"], pd.DataFrame([{"polled_utc": at(-50), "market_id": MID, "selection_id": 7.0,
                                                         "price": 3.8, "traded": 15.0}])])
    store = Store(root=tmp_path)
    preds = pd.DataFrame({"market_id": [MID, MID], "selection_id": [7, 8], "p_model": [0.2, 0.8]})
    store.put("greyhound/live/predictions/2026-11-03.csv", preds.to_csv(index=False).encode())
    seen = {}

    def decide(rows, fit, cfg):
        seen["rows"] = rows
        return rows.assign(lay=rows.selection_id == 7, lay_price=3.9, f_bsp=5.0, mid=4.1)
    monkeypatch.setattr(el, "decide", decide)
    out = ps.early_lay(store, date(2026, 11, 3), d, dict(el.CONFIG), fit={})
    assert set(seen["rows"].selection_id) == {7, 8}                             # decided once the field had traded
    assert len(out) == 1 and out.selection_id.iloc[0] == 7 and out.matched.iloc[0] > 0
    s = ps.summarise(out)
    assert s["lays"] == 1 and s["matched_lays"] == 1 and s["races"] == 1


def test_a_two_sided_position_greened_is_the_same_whatever_wins():
    fills = [("lay", 3.9, 10.0), ("back", 5.5, 4.0)]
    win, lose = ps.outcomes_of(fills)
    assert np.isclose(win, -10 * 2.9 + 4 * 4.5) and np.isclose(lose, 10 - 4)
    b = {"back": [(5.0, 100.0)], "lay": [(5.1, 100.0)]}
    pnl, how = ps.hedge(fills, b, None, 0)
    G = 5.0 if win < lose else 5.1
    h = (win - lose) / G
    assert how == "green" and np.isclose(pnl, lose + h) and np.isclose(pnl, win - h * (G - 1))
    # a lone lay greened this way is the familiar S(1 - m/B)
    assert np.isclose(ps.hedge([("lay", 3.9, 10.0)], b, None, 1)[0], 10 * (1 - 3.9 / 5.0))
    # a lone back that drifted is closed by a lay, at a loss
    assert np.isclose(ps.hedge([("back", 3.9, 10.0)], b, None, 1)[0], 10 * (3.9 / 5.1 - 1))
    assert ps.hedge([("lay", 3.9, 10.0)], None, 5.2, 1) == (10 * (1 - 3.9 / 5.2), "sp_close")


def test_both_sides_quoted_requoted_and_greened(monkeypatch):
    d = day()
    d["ladder"] = pd.concat([d["ladder"], pd.DataFrame([{"polled_utc": at(-50), "market_id": MID, "selection_id": 7.0,
                                                         "price": 3.8, "traded": 15.0}])])
    calls = []

    def fair(pre_m, t, preds_m, fit, cfg):
        calls.append(t)
        return pd.DataFrame({"selection_id": [7, 8], "f_bsp": [4.9, 2.0], "mid": [4.1, 2.01]})
    monkeypatch.setattr(ps, "fair_values", fair)
    from greyhound import early_lay as el
    preds = pd.DataFrame({"market_id": [MID, MID], "selection_id": [7, 8], "p_model": [0.2, 0.8]})
    rows = ps.quote_race(d, MID, preds, {}, dict(el.CONFIG), {"sides": ("back", "lay"), "margin": 0.2,
                                                              "max_loss": 60.0})
    assert max(calls) <= OFF - pd.Timedelta(seconds=10)                         # no new quotes in the last 10 s
    r = pd.DataFrame(rows).set_index("selection_id")
    # dog 7: the lay at 3.9 (4.9 x 0.8) fills (money traded at 3.8); the back at 5.9 (4.9 x 1.2) never does
    assert r.loc[7, "lay_matched"] > 0 and r.loc[7, "back_matched"] == 0 and r.loc[7, "how"] == "green"
    assert 8 not in r.index                                                     # quoted, never matched


def test_the_quote_widens_with_the_forecasts_error():
    assert ps.sigma_at(60) == 0.146 and ps.sigma_at(30) == 0.146 and ps.sigma_at(900) == 0.356
    assert ps.sigma_at(120) == (0.146 + 0.244) / 2
    lay_far, back_far = ps.quote_prices(4.0, {"k": 1.0}, 600)
    lay_near, back_near = ps.quote_prices(4.0, {"k": 1.0}, 60)
    assert lay_far < lay_near < 4.0 < back_near < back_far
    assert np.isclose(np.log(back_near / 4.0), 0.146) and np.isclose(np.log(4.0 / lay_near), 0.146)
    assert ps.quote_prices(4.0, {"margin": 0.2}, 60) == (4.0 * 0.8, 4.0 * 1.2)
