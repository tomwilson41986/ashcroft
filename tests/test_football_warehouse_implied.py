"""gold/implied: the goal expectancies are read off the prices exactly, whatever the low-score correction; the
handicap's fair chance matches a brute-force score matrix on every kind of line."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import poisson

from football.ratings import markets, score_matrix
from football.warehouse import gold_implied as gi


def test_total_supremacy_and_rho_are_recovered_from_a_dixon_coles_market():
    rng = np.random.default_rng(0)
    lh, la, rho = rng.uniform(0.4, 3.2, 60), rng.uniform(0.3, 2.6, 60), rng.uniform(-0.18, 0.05, 60)
    m = [markets(score_matrix(a, b, r, n=25)) for a, b, r in zip(lh, la, rho)]
    ph, pd_, pa_, po = (np.array([x[k] for x in m]) for k in ("p_h", "p_d", "p_a", "p_o25"))
    total = gi.total_from_over(po)
    sup = gi.supremacy_from_1x2(ph, pa_, total)
    assert np.abs(total - (lh + la)).max() < 1e-9 and np.abs(sup - (lh - la)).max() < 1e-9
    d0 = gi.draw_independent(total, sup)
    rho_hat = (d0 - pd_) / (2 * 0.5 * (total + sup) * 0.5 * (total - sup) * np.exp(-total))
    assert np.abs(rho_hat - rho).max() < 1e-9


def _brute(a: float, b: float, line: float) -> float:
    k = np.arange(0, 40)
    pm = np.outer(poisson.pmf(k, a), poisson.pmf(k, b))
    d = k[:, None] - k[None, :]

    def wl(x):
        return pm[d + x > 1e-9].sum(), pm[d + x < -1e-9].sum()
    if abs((abs(line) * 4) % 2 - 1) < 1e-9:                    # a quarter line: half a stake on each neighbour
        (w1, x1), (w2, x2) = wl(line - 0.25), wl(line + 0.25)
        w, x = w1 + w2, x1 + x2
    else:
        w, x = wl(line)
    return w / (w + x)


@pytest.mark.parametrize("line", [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_handicap_chance_matches_a_score_matrix_on_half_whole_and_quarter_lines(line):
    for a, b in ((1.6, 1.1), (2.4, 0.7), (0.9, 1.5)):
        fair = _brute(a, b, line)
        got = float(gi._cover(np.array([a + b]), np.array([a - b]), np.array([line]))[0])
        assert abs(got - fair) < 1e-12
        sup = float(gi.supremacy_from_handicap(np.array([line]), np.array([fair]), np.array([a + b]))[0])
        assert abs(sup - (a - b)) < 1e-9


def test_out_of_range_prices_give_blanks_not_numbers():
    assert np.isnan(gi.total_from_over(np.array([np.nan, 0.0, 1.0]))).all()
    assert np.isnan(gi.supremacy_from_1x2(np.array([0.5]), np.array([0.2]), np.array([np.nan]))).all()


def _checkpoint(rows) -> pa.Table:
    cols = ["market_id", "event_id", "market_type", "role", "cp_min", "line", "cp_time", "p_norm", "book_complete",
            "book_sum", "book_max_age_s"]
    data = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    data["cp_time"] = pa.array(data["cp_time"], pa.int64()).cast(pa.timestamp("ms", tz="UTC"))
    return pa.table(data)


def test_the_table_joins_the_three_markets_and_picks_the_handicap_line_nearest_evens():
    m = markets(score_matrix(1.7, 1.0, -0.1, n=25))
    t = 1_769_716_800_000
    rows = [("mo", 1, "MATCH_ODDS", "home", 0, 0.0, t, m["p_h"], True, 1.01, 30.0),
            ("mo", 1, "MATCH_ODDS", "draw", 0, 0.0, t, m["p_d"], True, 1.01, 30.0),
            ("mo", 1, "MATCH_ODDS", "away", 0, 0.0, t, m["p_a"], True, 1.01, 30.0),
            ("ou", 1, "OVER_UNDER_25", "over", 0, 0.0, t, m["p_o25"], True, 1.0, 60.0),
            ("ou", 1, "OVER_UNDER_25", "under", 0, 0.0, t, 1 - m["p_o25"], True, 1.0, 60.0),
            ("ah", 1, "ASIAN_HANDICAP", "home", 0, -0.5, t, _brute(1.7, 1.0, -0.5), True, 1.0, 10.0),
            ("ah", 1, "ASIAN_HANDICAP", "home", 0, -1.5, t, _brute(1.7, 1.0, -1.5), True, 1.0, 10.0),
            ("ah", 1, "ASIAN_HANDICAP", "home", 0, 0.5, t, 0.9, False, 0.9, 10.0),     # one side only: not a book
            ("mo", 1, "MATCH_ODDS", "home", 60, 0.0, t - 3_600_000, 0.6, False, 0.6, 5.0)]
    event = pa.table({"event_id": [1], "mo_market_id": ["mo"], "ou_market_id": ["ou"], "ah_market_id": ["ah"]})
    out = gi.build({"gold/checkpoint": _checkpoint(rows), "gold/event": event})["gold/implied"].to_pandas()
    assert out["cp_min"].tolist() == [0]                       # the incomplete T-60 book gives no row
    r = out.iloc[0]
    assert abs(r["lam_home"] - 1.7) < 1e-9 and abs(r["lam_away"] - 1.0) < 1e-9 and abs(r["rho"] + 0.1) < 1e-9
    assert r["ah_line"] == -0.5 and r["ah_lines"] == 2        # -0.5 is nearer evens than -1.5
    # the handicap priced off the same rates (independent Poisson) agrees with the 1X2: no gap
    assert abs(r["sup_ah"] - 0.7) < 1e-9 and abs(r["sup_gap"]) < 1e-9
    assert (r["mo_age_s"], r["ou_age_s"], r["ah_age_s"]) == (30.0, 60.0, 10.0)
