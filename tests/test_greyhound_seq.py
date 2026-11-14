"""greyhound/pit/seq.py: the sequence table. One late day's results shuffled moves no row of that day or before it
(and does move later ones); two builds are identical; a card row gets the features its run had; the rows from the
first scored month are filled and the earlier ones are not; the columns are the named forecasts; nothing reads the
trainer. Planted data only (the pit suite's synthetic card), a small network and few epochs."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from greyhound import pit  # noqa: E402
from greyhound.data import clean  # noqa: E402
from greyhound.pit import seq  # noqa: E402
from tests.test_greyhound_pit import cardify_last_day, synth  # noqa: E402

KEY = ["race_id", "dog_id"]
COLS = list(seq.COLUMNS)
SHUFFLED = "2025-04-05"                       # a day inside the last scored month, before the last day


@pytest.fixture(autouse=True)
def small(monkeypatch):
    monkeypatch.setattr(seq, "START", "2025-03-01")
    monkeypatch.setattr(seq, "FIRST_MONTHS", 1)
    monkeypatch.setattr(seq, "MIN_FIT", 50)
    monkeypatch.setattr(seq, "HIDDEN", 8)
    monkeypatch.setattr(seq, "EMB", 4)
    monkeypatch.setattr(seq, "EPOCHS0", 2)
    monkeypatch.setattr(seq, "EPOCHS", 1)
    monkeypatch.setattr(seq, "BATCH", 64)


@pytest.fixture(scope="module")
def runs():
    return synth(days=100, seed=11)           # 1 Jan to 10 Apr 2025: fits for March and April


def shuffle_day(runs: pd.DataFrame, day: str) -> pd.DataFrame:
    """The results of ``day`` shuffled across its rows (positions, times, sectionals, SPs, comments, margins)."""
    r = runs.copy()
    ix = np.flatnonzero(r.race_date.to_numpy() == day)
    pm = np.random.default_rng(7).permutation(ix)
    for c in ("position", "beaten_lengths", "sectional", "run_time", "adjusted_time", "sp_decimal", "market_pos",
              "comment"):
        r.iloc[ix, r.columns.get_loc(c)] = r[c].to_numpy()[pm]
    r["won"] = (r.position == 1).astype(float)
    r["sp_p"] = 1.0 / r.sp_decimal
    r["sp_p_norm"] = r.sp_p / r.groupby("race_id").sp_p.transform("sum")
    return r


def by_key(tab, runs, mask=None):
    d = runs[KEY + ["race_date"]].merge(tab, on=KEY)
    if mask is not None:
        d = d[mask(d)]
    return d.sort_values(KEY).reset_index(drop=True)


def test_a_days_results_reach_no_row_of_that_day_or_before(runs):
    a, _ = seq.build(runs)
    b, _ = seq.build(shuffle_day(runs, SHUFFLED))
    early = lambda d: d.race_date <= SHUFFLED                                  # noqa: E731
    x, y = by_key(a, runs, early), by_key(b, runs, early)
    assert x.se_lshare.notna().sum() > 1000
    pd.testing.assert_frame_equal(x, y, check_exact=True)
    later = lambda d: d.race_date > SHUFFLED                                   # noqa: E731
    xl, yl = by_key(a, runs, later)[COLS].to_numpy(), by_key(b, runs, later)[COLS].to_numpy()
    assert not np.array_equal(xl, yl, equal_nan=True)                         # the shuffle bites where it may


def test_two_builds_are_identical(runs):
    a, na = seq.build(runs)
    b, nb = seq.build(runs)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert [f["seed0"].get("val_ce") for f in na["folds"].values()] == \
        [f["seed0"].get("val_ce") for f in nb["folds"].values()]


def test_a_card_gets_the_features_its_run_had(runs):
    last = runs.race_date.max()
    full, _ = seq.build(runs)
    card, _ = seq.build(cardify_last_day(runs))
    on = lambda d: d.race_date == last                                         # noqa: E731
    x, y = by_key(full, runs, on), by_key(card, runs, on)
    assert len(x) == (runs.race_date == last).sum() and x.se_lshare.notna().all()
    pd.testing.assert_frame_equal(x, y, check_exact=True)


def test_the_rows_from_the_start_are_filled_and_the_columns_named(runs):
    tab, notes = seq.build(runs)
    assert list(tab.columns) == KEY + COLS
    assert len(tab) == len(runs) and (tab[KEY].to_numpy() == runs[KEY].to_numpy()).all()   # one row a run, in order
    d = runs[["race_date", "is_trial"]].join(tab)
    late = d[d.race_date >= "2025-03-01"]
    assert late[COLS].drop(columns="se_vs_last").notna().all().all()
    assert late.se_vs_last.notna().mean() > 0.9
    assert d[d.race_date < "2025-03-01"].se_lshare.isna().all()
    assert (tab.se_nhist.between(0, seq.L)).all()
    assert not any(c.startswith("se_pc") or c == "se_top_gap" for c in tab.columns)
    assert set(notes["folds"]) == {"2025-03", "2025-04"}
    p = np.exp(late.se_lshare).groupby(runs.race_id[late.index]).sum()
    assert np.allclose(p, 1.0, atol=1e-4)                                     # a softmax over the field


def test_the_forecast_follows_the_market():
    """On the planted card (each dog a fixed ability the SP reflects) the forecast SP share ranks a race's field like
    the SP did."""
    runs = synth(days=100, seed=12)
    tab, _ = seq.build(runs)
    d = runs[KEY + ["race_date", "sp_p_norm"]].merge(tab, on=KEY)
    d = d[d.race_date >= "2025-04-01"]
    rho = d.groupby("race_id").apply(lambda g: g.se_lshare.corr(np.log(g.sp_p_norm), method="spearman"))
    assert rho.mean() > 0.3


def test_no_trainer_kennel_or_owner_and_the_table_is_registered():
    src = inspect.getsource(seq.build) + inspect.getsource(seq._token_facts)
    assert not any(w in src for w in ("trainer", "owner", "kennel"))
    assert "seq" in pit.TABLES and "seq" in pit.ON_DEMAND
    from greyhound import stack_model as sm
    assert "seq" not in sm.PIT_TABLES
    assert {r["column"] for r in pit.registry() if r["table"] == "seq"} == set(COLS)


def test_a_cleaned_frame_twice_gives_the_same_table(runs):
    a, _ = seq.build(runs)
    b, _ = seq.build(clean(runs))
    pd.testing.assert_frame_equal(a, b, check_exact=True)
