"""The fold filter is plumbing, and plumbing is where the confident wrong
numbers come from.

Every defect this review turned up was of one shape: a mechanism that looked
like it was working and silently was not -- `weight=` discarded by a custom
objective, `--decay-rate` inert whenever that objective was on,
`folds_early_stopped` counting a stop at 2989 of 3000. A fold filter that
applied to one run and not the other, or that selected a fold meaning one
period in one run and another period in the other, would fail exactly that way:
a well-formed report, correctly formatted, about the wrong rows.

So these tests are about the filter's mechanics -- which rows it selects, and
what it refuses. They say nothing about whether any variant beats any other;
that comes from the real prediction CSVs, because a synthetic frame can be made
to say whatever its author expects.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "compare_oos_runs.py"


def _module():
    """`scripts/` is not a package, so load the file directly."""
    spec = importlib.util.spec_from_file_location("compare_oos_runs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cor = _module()


def _run(path, n_races=30, runners=6, folds=3, shift=1.0, first_date="2025-01-01"):
    """A prediction CSV with `folds` chronological folds of equal length.

    `shift` scales the predicted price away from the settled one, so two runs
    written with different shifts differ by a known amount.
    """
    per_fold = n_races // folds
    rows = []
    for r in range(n_races):
        fold = min(r // per_fold, folds - 1)
        date = pd.Timestamp(first_date) + pd.Timedelta(days=r)
        for h in range(runners):
            rows.append({
                "race_date": date.strftime("%Y-%m-%d"),
                "race_time": "14:30",
                "track": f"T{r % 3}",
                "horse_name": f"h{r}_{h}",
                "bfsp": 2.0 + h,
                "predicted_bfsp": 1.0 + (1.0 + h) * shift,
                "won": 1 if h == 0 else 0,
                "fold_idx": fold,
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,want", [
    ("0-4", {0, 1, 2, 3, 4}),
    ("5-10", {5, 6, 7, 8, 9, 10}),
    ("3", {3}),
    ("0,2,5-7", {0, 2, 5, 6, 7}),
    (" 1 , 2 ", {1, 2}),
    ("2-2", {2}),
])
def test_fold_spec_reads_ranges_singletons_and_lists(spec, want):
    assert cor.parse_folds(spec) == want


def test_no_spec_means_every_fold():
    assert cor.parse_folds(None) is None


@pytest.mark.parametrize("spec", ["", "first-half", "1-", "1,,x", "0-four"])
def test_an_unreadable_spec_is_refused_rather_than_guessed_at(spec):
    with pytest.raises(SystemExit):
        cor.parse_folds(spec)


# ---------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------

def test_the_filter_selects_exactly_the_folds_named(tmp_path):
    p = _run(tmp_path / "a.csv", folds=3)
    d = cor.load(str(p), {1})
    assert set(d["fold_idx"]) == {1}
    assert len(d) == len(cor.load(str(p))) // 3


def test_the_same_filter_selects_the_same_folds_in_both_runs(tmp_path):
    """The whole point: one spec, two files, the same rows out of each."""
    a = _run(tmp_path / "a.csv", shift=1.0)
    b = _run(tmp_path / "b.csv", shift=1.2)
    da, db = cor.load(str(a), {0, 1}), cor.load(str(b), {0, 1})
    assert set(da["fold_idx"]) == set(db["fold_idx"]) == {0, 1}
    assert list(da["horse_name"]) == list(db["horse_name"])


def test_naming_every_fold_reproduces_the_unfiltered_frame(tmp_path):
    p = _run(tmp_path / "a.csv", folds=3)
    pd.testing.assert_frame_equal(cor.load(str(p)), cor.load(str(p), {0, 1, 2}))


def test_a_selection_that_matches_nothing_is_refused(tmp_path):
    p = _run(tmp_path / "a.csv", folds=3)
    with pytest.raises(SystemExit, match="no usable rows"):
        cor.load(str(p), {9})


def test_selecting_folds_from_a_csv_that_has_none_is_refused(tmp_path):
    p = _run(tmp_path / "a.csv")
    d = pd.read_csv(p).drop(columns=["fold_idx"])
    d.to_csv(p, index=False)
    with pytest.raises(SystemExit, match="no fold_idx column"):
        cor.load(str(p), {0})
    cor.load(str(p))          # ...but the whole-sample comparison still works


# ---------------------------------------------------------------------------
# fold k must be the same period in both runs
# ---------------------------------------------------------------------------

def test_matching_geometry_passes(tmp_path):
    a = _run(tmp_path / "a.csv", shift=1.0)
    b = _run(tmp_path / "b.csv", shift=1.2)
    cor.assert_comparable_folds(cor.load(str(a)), cor.load(str(b)))


def test_folds_covering_different_dates_are_refused(tmp_path):
    """An 11-fold run against a 22-fold one reads fold 3 as two periods."""
    a = _run(tmp_path / "a.csv", first_date="2025-01-01")
    b = _run(tmp_path / "b.csv", first_date="2025-06-01")
    with pytest.raises(SystemExit, match="different dates"):
        cor.assert_comparable_folds(cor.load(str(a)), cor.load(str(b)))


def test_the_refusal_names_the_fold_it_disagreed_on(tmp_path):
    a = _run(tmp_path / "a.csv", n_races=30, folds=3)
    d = pd.read_csv(a)
    b = tmp_path / "b.csv"
    # fold 2 alone slides a year forward; folds 0 and 1 still line up.
    late = d["fold_idx"] == 2
    d.loc[late, "race_date"] = (pd.to_datetime(d.loc[late, "race_date"])
                                + pd.DateOffset(years=1)).dt.strftime("%Y-%m-%d")
    d.to_csv(b, index=False)
    with pytest.raises(SystemExit, match="fold 2"):
        cor.assert_comparable_folds(cor.load(str(a)), cor.load(str(b)))


def test_geometry_is_only_checked_over_the_selected_folds(tmp_path):
    """Fold 2 disagreeing is no reason to refuse a comparison of folds 0 and 1."""
    a = _run(tmp_path / "a.csv", n_races=30, folds=3)
    d = pd.read_csv(a)
    b = tmp_path / "b.csv"
    late = d["fold_idx"] == 2
    d.loc[late, "race_date"] = (pd.to_datetime(d.loc[late, "race_date"])
                                + pd.DateOffset(years=1)).dt.strftime("%Y-%m-%d")
    d.to_csv(b, index=False)
    cor.assert_comparable_folds(cor.load(str(a), {0, 1}), cor.load(str(b), {0, 1}))


# ---------------------------------------------------------------------------
# end to end: the report has to say which rows it is about
# ---------------------------------------------------------------------------

def _report(tmp_path, *args):
    a = _run(tmp_path / "a.csv", shift=1.00)
    b = _run(tmp_path / "b.csv", shift=1.02)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--base", str(a), "--variant", str(b),
         "--n-boot", "50", *args],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_a_half_sample_report_says_so_in_its_header(tmp_path):
    text = _report(tmp_path, "--folds", "0-1")
    assert "**Folds 0-1.**" in text
    # 20 of 30 races, and the dates stop before the third fold begins
    assert "over 20 races" in text
    assert "2025-01-01 to 2025-01-20" in text


def test_a_whole_sample_report_still_names_its_span(tmp_path):
    text = _report(tmp_path)
    assert "Folds" not in text.split("## Mean")[0]
    assert "over 30 races" in text
    assert "2025-01-01 to 2025-01-30" in text


@pytest.mark.parametrize("brier_ci, conc_ci, want", [
    ((-0.0005, +0.0010), (-0.0010, +0.0020), True),     # both unresolved: the price decides
    ((+0.0005, +0.0020), (+0.0001, +0.0030), True),     # both resolved better
    ((-0.0041, -0.0009), (-0.0038, +0.0003), False),    # Brier skill resolved worse (iteration 62's fl_fu)
    ((-0.0010, +0.0010), (-0.0040, -0.0001), False),    # concordance resolved worse
])
def test_the_guards_refuse_a_resolved_loss_against_the_market(brier_ci, conc_ci, want):
    primary, brier_ok, conc_ok, replaces = cor.decide(-0.0005, brier_ci, conc_ci)
    assert primary and replaces is want
    assert brier_ok is (brier_ci[1] >= 0) and conc_ok is (conc_ci[1] >= 0)


def test_no_guard_passes_a_price_forecast_that_is_not_better():
    assert cor.decide(+0.0001, (0.001, 0.002), (0.001, 0.002))[3] is False
