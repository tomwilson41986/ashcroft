"""research/overfit.py and the gates that read it (the owner, 5 Nov 2026: "wire in"): the deflated Sharpe that a
promotion gate needs, the trial ledger that keeps the count honest, and the horse strategy gates that now say
"not deflated" where the best of many rules clears every other bar by luck."""
import json

import numpy as np

from research import overfit as ov
from research.strat import strat_lib as L


def test_a_real_edge_passes_and_the_same_sharpe_fails_once_enough_rules_were_tried():
    rng = np.random.default_rng(0)
    r = rng.normal(0.15, 1.0, 600)                   # a Sharpe of about 0.13 a period over 600 periods
    one = ov.dsr_check(r, n_trials=1)
    many = ov.dsr_check(r, n_trials=500, trial_sharpes=rng.normal(0, 0.05, 500))
    assert one["passed"] and one["dsr"] >= 0.95
    assert not many["passed"] and many["dsr"] < one["dsr"] and "below" in many["reason"]


def test_too_few_periods_or_no_spread_never_passes():
    assert not ov.dsr_check([0.1] * 10, 1)["passed"]
    assert not ov.dsr_check([0.05] * 100, 1)["passed"]            # no spread: no Sharpe, no pass
    assert not ov.dsr_check([], 1)["passed"]


def test_the_ledger_counts_each_rule_once_and_outlives_the_run(tmp_path):
    p = tmp_path / "trials.json"
    led = ov.TrialLedger(p, programme="test")
    rng = np.random.default_rng(1)
    for _ in range(3):                                         # the same two rules scored three nights running
        led.record("a", returns=rng.normal(0, 1, 50))
        led.record("b", returns=rng.normal(0, 1, 50))
    led.save()
    again = ov.TrialLedger(p)
    assert again.n_trials == 2 and json.loads(p.read_text())["n_trials"] == 2
    again.record("c", sharpe=0.2)
    assert again.n_trials == 3 and np.isfinite(again.sharpes()).sum() == 3


def test_deflate_table_uses_the_larger_of_the_table_and_the_ledger(tmp_path):
    rng = np.random.default_rng(2)
    series = {f"r{i}": rng.normal(0, 1, 100) for i in range(5)}
    assert {d["n_trials"] for d in ov.deflate_table(series).values()} == {5}
    led = ov.TrialLedger(tmp_path / "l.json")
    for i in range(20):
        led.record(f"old{i}", sharpe=0.0)
    assert {d["n_trials"] for d in ov.deflate_table(series, ledger=led).values()} == {25}
    assert (tmp_path / "l.json").exists()


def _row(name, t_ret, **kw):
    base = {"rule": name, "n": 3000, "A/E": 1.1, "p": 1e-4, "A/E_pl": 1.05, "p_pl": 1e-3, "t_n": 800, "t_WAX": 5.0,
            "t_p": 0.01, "t_A/E_pl": 1.02, "t_roi_both%": 2.0, "t_years_both>1": "4/5", "n_am": 0,
            "t_years_am_both>1": "0/0", "_t_ret": t_ret, "_t_ret_am": np.array([])}
    base.update(kw)
    return base


def test_the_strategy_gate_says_not_deflated_where_the_best_of_many_clears_the_old_bar():
    rng = np.random.default_rng(3)
    strong = rng.normal(0.25, 1.0, 800)                          # clears the deflated bar on its own
    alone = L.rules_table([_row("strong", strong)], ledger=None)
    assert alone.iloc[0]["gate"] == "PASS" and alone.iloc[0]["t_dsr"] >= 0.95
    weak = rng.normal(0.05, 1.0, 800)                            # every other bar met, but the best of 200 by luck
    rows = [_row("weak", weak)] + [_row(f"noise{i}", rng.normal(0, 1, 800), t_WAX=-1.0) for i in range(199)]
    t = L.rules_table(rows, ledger=None)
    w = t[t["rule"] == "weak"].iloc[0]
    assert w["gate"] == "not deflated" and w["t_dsr_trials"] == 200
    assert not any(c.startswith("_") for c in t.columns)
