"""research/greyhound/sp_offset_forward.py: the frozen residual served before the off, the first serve kept, settled
at the SP in wholly priced races only, read alone and with the study's days."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.greyhound import sp_offset as so
from research.greyhound import sp_offset_forward as fw
from tests.test_greyhound_sp_offset import SMALL, _frame


def test_the_freeze_fits_once_on_every_day_and_its_cuts_order_the_rules():
    d = _frame(0.5, races=3000)
    orig, so.PARAMS = so.PARAMS, SMALL
    try:
        m, meta = fw.freeze_fit(d, ["x", "noise"], seeds=(1, 2, 3))
    finally:
        so.PARAMS = orig
    assert meta["fit_to"] == d.day.max() and meta["fit_from"] == d.day.min()
    assert len(m) == 3 == len(meta["rounds"]) and min(meta["rounds"]) > 0 and meta["seeds"] == [1, 2, 3]
    assert meta["early_stop_from"] > "2024-06"                 # the last tenth of the days
    c = meta["cuts"]
    assert c["back_top2"] > c["back_top5"] > c["back_top10"] > 0 > c["lay_bottom10"] > c["lay_bottom5"] > c["lay_bottom2"]
    # the residual follows the feature the BSP misses, on rows scored as a card would be
    f = fw.score(m, d[["x", "noise"]].to_numpy(np.float32))
    assert np.corrcoef(f, d.x)[0, 1] > 0.9
    assert fw.SEEDS == tuple(range(1, 21)) and fw.DROP_FEATURES == ("gp_bsp_ae_car", "gp_bsp_mkt_car")


def test_the_signal_is_centred_in_its_race_and_a_short_field_is_left_out():
    card = pd.DataFrame({"race_id": ["a"] * 4 + ["b"] * 3, "trap": [1, 2, 3, 4, 1, 2, 3]})
    s = fw.signal(card, np.array([0.3, 0.1, -0.1, 0.1, 5.0, 0.0, 0.0]))
    assert set(s.race_id) == {"a"} and abs(s.s.sum()) < 1e-12 and np.isclose(s.s.iloc[0], 0.2)


def _served(race, off, served, s=(0.2, 0.0, -0.1, -0.1)):
    return pd.DataFrame({"race_id": race, "trap": [1, 2, 3, 4], "off_utc": off, "served_utc": served, "s": list(s)})


def test_the_first_serve_wins_and_a_race_too_near_its_off_is_not_served():
    now = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
    first = fw.first_serve_wins(None, pd.concat([_served("r1", "2026-11-21T12:30:00Z", "12:00"),
                                                 _served("r2", "2026-11-21T12:05:00Z", "12:00")]), now)
    assert set(first.race_id) == {"r1"}                         # r2 is off in five minutes
    later = datetime(2026, 11, 21, 12, 10, tzinfo=timezone.utc)
    again = pd.concat([_served("r1", "2026-11-21T12:30:00Z", "12:10", s=(9, 9, 9, 9)),
                       _served("r3", "2026-11-21T18:00:00Z", "12:10")])
    out = fw.first_serve_wins(first, again, later)
    assert set(out.race_id) == {"r1", "r3"}
    assert out[out.race_id == "r1"].s.tolist() == [0.2, 0.0, -0.1, -0.1] and set(out[out.race_id == "r1"].served_utc) == {"12:00"}


CUTS = {"back_top10": 0.15, "back_top5": 0.5, "back_top2": 0.9, "lay_bottom10": -0.15, "lay_bottom5": -0.5,
        "lay_bottom2": -0.9}


def test_a_day_settles_at_the_sp_in_wholly_priced_races_only():
    rows = []
    for race, ss in (("m1", (0.2, 0.0, 0.0, -0.2)), ("m2", (0.3, -0.1, -0.1, -0.1))):
        for i, s in enumerate(ss):
            rows.append({"day": "2026-11-21", "race_id": race, "market_id": race, "selection_id": i + 1, "s": s})
    served = pd.DataFrame(rows)
    close = pd.DataFrame({"market_id": ["m1"] * 4 + ["m2"] * 3, "selection_id": [1, 2, 3, 4, 1, 2, 3],
                          "bsp": [3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 4.0], "won": [1, 0, 0, 0, 1, 0, 0]})
    per_race, counts = fw.settle_day(served, close, CUTS, 0.02)
    assert counts == {"races_served": 2, "races_wholly_priced": 1, "races_left_out": 1}     # m2: a runner with no SP
    r = per_race.set_index("rule")
    assert set(r.index) == {"back_top10", "lay_bottom10"} and set(per_race.race_id) == {"m1"}
    assert np.isclose(r.loc["back_top10", "net"], 2.0 * 0.98)            # a back of 1 at 3.0 that won, 2% off
    assert np.isclose(r.loc["lay_bottom10", "net"], 0.2 * 0.98)          # a lay risking 1 at 6.0 that lost: stake 0.2


def test_the_read_needs_every_forward_day_a_forward_gain_and_the_pooled_bar():
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-01", periods=600).strftime("%Y-%m-%d")
    dev = pd.concat([pd.DataFrame({"rule": n, "day": days, "risk": 50.0,
                                   "net": 50.0 * (0.02 + 0.1 * rng.standard_normal(600))}) for n, _, _ in so.STAGE2])
    fdays = pd.date_range("2026-11-21", periods=fw.READ_DAYS).strftime("%Y-%m-%d")
    good = pd.DataFrame({"rule": fw.PRIMARY, "day": fdays, "race_id": "r", "risk": 50.0,
                         "net": 50.0 * (0.05 + 0.1 * rng.standard_normal(fw.READ_DAYS))})
    bad = good.assign(rule="back_top10", net=-abs(good.net))
    res = fw.read(pd.concat([good, bad]), dev, 1)
    by = {r["rule"]: r for r in res["rules"]}
    assert by[fw.PRIMARY]["pass"] and by[fw.PRIMARY]["pooled"]["days"] == 600 + fw.READ_DAYS
    assert res["gate"]["state"] == "PASS" and not by["back_top10"]["pass"]      # its forward days lost
    assert by["back_top2"]["forward"] == {"bets": 0, "note": "no data found"} and not by["back_top2"]["pass"]
    hard = fw.read(pd.concat([good, bad]), dev, 10 ** 9)                          # the pooled bar bites
    assert hard["passed"] == [] and hard["gate"]["state"] == "FAIL" and "| lay_bottom10 |" in fw.markdown(res)
    # before the last day nothing passes, however good; a forward interval wholly below zero at the look ends it
    early = fw.read(good[good.day <= fdays[59]], dev, 1)
    assert early["passed"] == [] and early["gate"]["state"] == f"running: 60 of {fw.READ_DAYS} days"
    lost = fw.read(good[good.day <= fdays[49]].assign(net=lambda f: -abs(f.net)), dev, 1)
    assert lost["gate"]["state"].startswith("FAIL (futility")


def test_the_studys_days_are_the_stage_two_bets_summed():
    r = pd.DataFrame({"day": ["d1"] * 4 + ["d2"] * 4, "race_id": ["a"] * 4 + ["b"] * 4, "won": [1, 0, 0, 0, 0, 0, 0, 1],
                      "bsp": [3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0], "s": [0.2, 0.0, 0.0, -0.2, 0.2, 0.0, 0.0, -0.2]})
    folds = [{"_runners": r, "cuts": CUTS, "test": ["d1", "d2"]}]
    daily = so.stage2_daily(folds, 0.02)
    st = so.stage2(folds, 0.02, 1)
    for rule in st["rules"]:
        got = daily[daily.rule == rule["rule"]]
        assert np.isclose(got.net.sum(), rule["pooled"].get("net", 0.0), atol=0.006) and got.risk.sum() == rule["pooled"]["bets"]
    assert set(daily.rule) == {"back_top10", "lay_bottom10"}


def test_the_forward_workflow_holds_no_betfair_key_and_sends_no_order():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    wf = (root / ".github" / "workflows" / "greyhound-sp-offset.yml").read_text()
    src = (root / "research" / "greyhound" / "sp_offset_forward.py").read_text()
    assert "BETFAIR" not in wf and "--live" not in wf and "--trade" not in wf
    for word in ("place_orders", "placeOrders", "betfair_client", "BetfairClient"):
        assert word not in src, word
    assert fw.FIRST_DAY == "2026-11-21"


def test_the_frozen_fits_come_back_from_the_store_and_the_seed_halves_days_are_averaged(tmp_path):
    import json

    from sources.common import Store
    d = _frame(0.5, races=1500)
    orig, so.PARAMS = so.PARAMS, SMALL
    try:
        models, meta = fw.freeze_fit(d, ["x", "noise"], seeds=(1, 2))
    finally:
        so.PARAMS = orig
    store = Store(root=tmp_path)
    assert fw.load_frozen(store) == (None, None)
    store.put(fw.MODEL_KEY, json.dumps([m.model_to_string(num_iteration=m.best_iteration) for m in models]).encode())
    store.put(fw.META_KEY, json.dumps(meta).encode())
    back, m2 = fw.load_frozen(store)
    x = d[["x", "noise"]].to_numpy(np.float32)[:500]
    want = np.mean([m.predict(x, num_iteration=m.best_iteration, raw_score=True) for m in models], axis=0)
    assert len(back) == 2 and m2["cuts"] == meta["cuts"] and np.allclose(fw.score(back, x), want)
    a = pd.DataFrame({"rule": ["r", "r"], "day": ["d1", "d2"], "net": [2.0, -1.0], "risk": [10.0, 10.0]})
    b = pd.DataFrame({"rule": ["r"], "day": ["d1"], "net": [4.0], "risk": [20.0]})
    got = fw.mean_daily([a, b]).set_index("day")
    assert got.loc["d1", "net"] == 3.0 and got.loc["d1", "risk"] == 15.0 and got.loc["d2", "net"] == -0.5
