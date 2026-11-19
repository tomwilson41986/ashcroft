"""research/harness.py: the Phase 1 scoring contracts, on constructed frames (code tests, not evaluations)."""
import numpy as np
import pandas as pd

from research import harness as h


def _frame(n_races=300, field=8, seed=1, model_noise=0.3):
    """Races whose winners are drawn from known chances; the BSP is the chances with a 5% book, the model the BSP
    with noise. Built to test the code paths, never to evaluate anything."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        day = pd.Timestamp("2025-01-01") + pd.Timedelta(days=r // 6)
        p = rng.dirichlet(np.ones(field) * 2)
        win = rng.choice(field, p=p)
        for i in range(field):
            bsp = 1.0 / (p[i] * 1.05)
            rows.append({"race_date": day.date().isoformat(), "race_time": f"{12 + r % 6}:00", "track": "T",
                         "horse_name": f"h{r}_{i}", "bfsp": bsp,
                         "predicted_bfsp": bsp * np.exp(rng.normal(0, model_noise)), "won": i == win,
                         "fold_idx": r * 4 // n_races, "number_of_runners": field,
                         "race_code": "Hurdle" if r % 2 else "Flat", "surface_type": "Turf",
                         "outcome": rng.normal(p[i] * 10, 1.0)})
    return pd.DataFrame(rows)


def test_normalised_probabilities_sum_to_one_in_every_race():
    df = _frame(50)
    race = h.race_ids(df)
    p = h.normalise(df["predicted_bfsp"].to_numpy(), race)
    assert h.check_race_sums(p, race) == 0
    bad = p.copy()
    bad[0] += 0.01
    assert h.check_race_sums(bad, race) == 1


def test_day_bootstrap_is_the_mean_with_an_interval_around_it_and_repeatable():
    day = np.repeat(np.arange(40), 5)
    x = np.random.default_rng(3).normal(2.0, 1.0, size=len(day))
    est, lo, hi = h.day_bootstrap(day, x)
    assert abs(est - x.mean()) < 1e-12 and lo < est < hi
    assert h.day_bootstrap(day, x) == (est, lo, hi)


def test_bfsp_error_is_zero_for_a_perfect_forecast_and_segments_cover_every_runner():
    df = _frame(60)
    df["predicted_bfsp"] = df["bfsp"]
    t = h.bfsp_error(df)
    assert t.loc[t.segment == "all", "error"].item() == 0.0
    for seg in ("code", "band", "field", "quarter"):
        assert t.loc[t.segment == seg, "runners"].sum() == len(df)


def test_win_scores_prefer_the_true_chances_to_a_noisy_copy():
    s = h.win_scores(_frame(400, model_noise=0.6))
    assert s["logloss_model_minus_market"][0] > 0          # the noisy model loses to the market built on the truth
    assert 0 <= s["ece"] < 0.1


def test_benter_gives_the_market_the_weight_when_the_model_is_only_noise():
    b = h.benter(_frame(600, model_noise=0.8, seed=4))
    assert b["races"] > 0
    for w in b["weights"]:
        assert w["beta_market"] > 0.5 and abs(w["beta_model"]) < 0.3


def test_clogit_recovers_a_known_weight():
    rng = np.random.default_rng(7)
    races, field, beta = 3000, 6, 1.5
    x = rng.normal(size=(races, field))
    p = np.exp(beta * x) / np.exp(beta * x).sum(axis=1, keepdims=True)
    win = np.array([rng.choice(field, p=pi) for pi in p])
    y = np.zeros((races, field), dtype=int)
    y[np.arange(races), win] = 1
    b = h.clogit_fit(x.reshape(-1, 1), np.repeat(np.arange(races), field), y.reshape(-1))
    assert abs(b[0] - beta) < 0.1


def test_order_agreement_reads_a_perfect_order_as_one():
    df = _frame(80)
    df["outcome"] = -np.log(df["predicted_bfsp"])           # the outcome follows the model's order exactly
    r = h.order_agreement(df, "outcome")
    assert r["races"] == 80 and abs(r["model"][0] - 1.0) < 1e-12
    assert r["model_minus_market"][0] > 0
