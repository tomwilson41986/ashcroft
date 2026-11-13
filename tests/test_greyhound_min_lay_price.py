"""The minimum lay price test (research/greyhound/min_lay_price.py): planted lays, never the real data."""

import numpy as np
import pandas as pd

from research.greyhound import min_lay_price as mlp


def _lays(short_edge: float, long_edge: float, n_days=30, per_day=40, seed=0) -> tuple[pd.DataFrame, list[str]]:
    """Matched lays greened up: under 5 the price moves by ``short_edge`` on average, at 5 and over by ``long_edge``."""
    rng = np.random.default_rng(seed)
    days = list(pd.date_range("2026-10-10", periods=n_days).strftime("%Y-%m-%d"))
    rows = []
    for d in days:
        for i in range(per_day):
            price = float(rng.choice([2.5, 3.5, 4.5, 6.0, 9.0, 14.0]))
            move = rng.normal(short_edge if price < 5 else long_edge, 0.08)
            stake = 100.0 / (price - 1.0)
            out = price * (1.0 + move)
            rows.append({"race_date": d, "market_id": f"{d}-{i}", "selection_id": i, "price": price, "stake": stake,
                         "risk": stake * (price - 1.0), "pnl": stake * (1.0 - price / out), "held": 0.0,
                         "clv": 1.0 - price / out, "clv_value": 0.0, "filled": True})
    return pd.DataFrame(rows), days


def test_lays_that_lose_under_the_cut_pass_the_reading():
    o, days = _lays(short_edge=-0.03, long_edge=0.04)
    r = mlp.split(o, days, 5.0, n_boot=500)
    assert r["bands"]["under"]["roi%"] < 0 < r["bands"]["over"]["roi%"]
    assert r["under_net_a_day"]["interval_90"][1] < 0 and r["over_less_all_roi"]["diff"] > 0
    assert mlp.verdict(r) == "PASS"
    assert r["bands"]["all"]["trades"] == r["bands"]["under"]["trades"] + r["bands"]["over"]["trades"] == len(o)


def test_no_difference_by_price_fails_the_reading():
    o, days = _lays(short_edge=0.02, long_edge=0.02, seed=1)
    r = mlp.split(o, days, 5.0, n_boot=500)
    assert mlp.verdict(r) == "FAIL"


def test_a_day_without_a_lay_under_the_cut_counts_as_a_day():
    o, days = _lays(short_edge=-0.03, long_edge=0.04, n_days=10, seed=2)
    o = o[~((o.race_date == days[0]) & (o.price < 5))]
    r = mlp.split(o, days, 5.0, n_boot=200)
    assert r["days"] == 10 and r["under_net_a_day"]["days_down"] + r["under_net_a_day"]["days_up"] <= 9


def test_the_markdown_names_the_verdict():
    o, days = _lays(short_edge=-0.03, long_edge=0.04, n_days=8, seed=3)
    s = mlp.split(o, days, 5.0, n_boot=200)
    r = {"first_scored_day": days[0], "last_scored_day": days[-1], "scored_days": 8, "source": "retimed",
         "orders": len(o), "matched": len(o), "primary": s, "verdict": mlp.verdict(s), "sensitivity": {"3.0": s}}
    md = mlp.markdown(r)
    assert "**PASS**" in md and "| under |" in md
