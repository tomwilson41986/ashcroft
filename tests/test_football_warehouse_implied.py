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


def _brute(a: float, b: float, line: float, rho: float = 0.0) -> float:
    """The home side's fair chance on a handicap line, summed over a score matrix (Dixon-Coles with ``rho``)."""
    k = np.arange(0, 40)
    pm = np.outer(poisson.pmf(k, a), poisson.pmf(k, b))
    pm[0, 0] *= 1 - a * b * rho
    pm[0, 1] *= 1 + a * rho
    pm[1, 0] *= 1 + b * rho
    pm[1, 1] *= 1 - rho
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
        for rho in (0.0, -0.0857, -0.2, 0.06):                 # independent Poisson, the market's usual, extremes
            fair = _brute(a, b, line, rho)
            got = float(gi._cover(np.array([a + b]), np.array([a - b]), np.array([line]), rho)[0])
            assert abs(got - fair) < 1e-12
            sup = float(gi.supremacy_from_handicap(np.array([line]), np.array([fair]), np.array([a + b]), rho)[0])
            assert abs(sup - (a - b)) < 1e-9


def test_the_brute_force_matrix_is_the_repositorys_dixon_coles():
    m = markets(score_matrix(1.6, 1.1, -0.1, n=40))
    assert abs(_brute(1.6, 1.1, -0.5, -0.1) - m["p_h"]) < 1e-9      # home -0.5 wins when the home side does


def test_reading_a_dixon_coles_handicap_without_its_rho_misses_by_the_lines_sign():
    # what the module did before: a market priced by one model with the usual rho, read as independent Poisson
    a, b, rho = 1.5, 1.1, -0.0857
    gap = {line: float(gi.supremacy_from_handicap(np.array([line]), np.array([_brute(a, b, line, rho)]),
                                                  np.array([a + b]), 0.0)[0]) - (a - b) for line in (-0.5, 0.5)}
    assert gap[-0.5] < -0.03 and gap[0.5] > 0.03


def test_a_rho_no_dixon_coles_market_allows_is_not_valid():
    lh, la = np.array([1.5, 1.5, 1.5, 0.5, 2.0]), np.array([1.0, 1.0, 1.0, 0.5, 2.0])
    rho = np.array([-0.1, -0.7, 0.7, 1.5, 0.2])
    # -0.7 < -1/1.5; 0.7 > 1/(1.5 x 1.0); 1.5 > 1; 0.2 <= 1/(2 x 2)
    assert gi.rho_valid(lh, la, rho).tolist() == [True, False, False, False, True]
    assert not gi.rho_valid(np.array([np.nan]), np.array([1.0]), np.array([0.0]))[0]


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
            ("ah", 1, "ASIAN_HANDICAP", "home", 0, -0.5, t, _brute(1.7, 1.0, -0.5, -0.1), True, 1.0, 10.0),
            ("ah", 1, "ASIAN_HANDICAP", "home", 0, -1.5, t, _brute(1.7, 1.0, -1.5, -0.1), True, 1.0, 10.0),
            ("ah", 1, "ASIAN_HANDICAP", "home", 0, 0.5, t, 0.9, False, 0.9, 10.0),     # one side only: not a book
            ("mo", 1, "MATCH_ODDS", "home", 60, 0.0, t - 3_600_000, 0.6, False, 0.6, 5.0)]
    event = pa.table({"event_id": [1], "mo_market_id": ["mo"], "ou_market_id": ["ou"], "ah_market_id": ["ah"]})
    out = gi.build({"gold/checkpoint": _checkpoint(rows), "gold/event": event})["gold/implied"].to_pandas()
    assert out["cp_min"].tolist() == [0]                       # the incomplete T-60 book gives no row
    r = out.iloc[0]
    assert abs(r["lam_home"] - 1.7) < 1e-9 and abs(r["lam_away"] - 1.0) < 1e-9 and abs(r["rho"] + 0.1) < 1e-9
    assert r["ah_line"] == -0.5 and r["ah_lines"] == 2        # -0.5 is nearer evens than -1.5
    # the handicap priced by the same model as the 1X2 (the same rates, the same rho) agrees with it: no gap
    assert abs(r["sup_ah"] - 0.7) < 1e-9 and abs(r["sup_gap"]) < 1e-9
    assert (r["mo_age_s"], r["ou_age_s"], r["ah_age_s"]) == (30.0, 60.0, 10.0)


def _one_event(mo, ah_rows, t=1_769_716_800_000) -> dict:
    rows = [("mo", 1, "MATCH_ODDS", "home", 0, 0.0, t, mo["p_h"], True, 1.0, 30.0),
            ("mo", 1, "MATCH_ODDS", "draw", 0, 0.0, t, mo["p_d"], True, 1.0, 30.0),
            ("mo", 1, "MATCH_ODDS", "away", 0, 0.0, t, mo["p_a"], True, 1.0, 30.0),
            ("ou", 1, "OVER_UNDER_25", "over", 0, 0.0, t, mo["p_o25"], True, 1.0, 60.0)]
    rows += [("ah", 1, "ASIAN_HANDICAP", "home", 0, line, t, p, True, 1.0, age) for line, p, age in ah_rows]
    event = pa.table({"event_id": [1], "mo_market_id": ["mo"], "ou_market_id": ["ou"], "ah_market_id": ["ah"]})
    return gi.build({"gold/checkpoint": _checkpoint(rows), "gold/event": event})["gold/implied"].to_pandas().iloc[0]


def test_the_main_handicap_line_is_a_freshly_traded_one():
    m = markets(score_matrix(1.7, 1.0, -0.1, n=25))
    near, far = _brute(1.7, 1.0, -0.75, -0.1), _brute(1.7, 1.0, -1.5, -0.1)     # -0.75 is the nearer to evens
    assert abs(near - 0.5) < abs(far - 0.5)
    both_fresh = _one_event(m, [(-0.75, near, 100.0), (-1.5, far, 100.0)])
    assert both_fresh["ah_line"] == -0.75
    # ... but it last traded eight hours ago and -1.5 a minute ago: the market is at -1.5
    stale = _one_event(m, [(-0.75, 0.30, 28_000.0), (-1.5, far, 99.0)])
    assert stale["ah_line"] == -1.5 and stale["ah_age_s"] == 99.0 and abs(stale["sup_gap"]) < 1e-9
    # no line is fresh: the nearest evens of what there is
    old = _one_event(m, [(-0.75, near, 5_000.0), (-1.5, far, 4_000.0)])
    assert old["ah_line"] == -0.75 and old["ah_lines"] == 2


def test_books_that_are_not_one_markets_prices_give_no_rho():
    # a 1X2 with a draw far too short for its goals (an in-running draw price beside pre-off sides)
    m = dict(markets(score_matrix(1.7, 1.0, -0.1, n=25)))
    m["p_d"], m["p_h"], m["p_a"] = 0.80, 0.12, 0.08
    r = _one_event(m, [(-0.5, 0.55, 30.0)])
    assert np.isnan(r["rho"]) and np.isfinite(r["total"]) and np.isfinite(r["supremacy"])
    assert np.isfinite(r["sup_ah"])                            # read with the usual rho, for what it is worth
