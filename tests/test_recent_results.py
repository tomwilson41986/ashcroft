"""The recent-results report is plumbing between two stores that disagree on types.

The morning files and the database both write a race time as "2.30". Read as a
number it becomes 2.3, which joins to nothing, and a report over zero rows
still prints a table -- just an empty one. That failure mode is why these tests
exist: they check the join finds the rows, that a non-runner and a day without
results fall out rather than corrupting the counts, and that the money is
computed the way the rest of the repo computes it.

Synthetic frames exercise the mechanics only. What the model actually did on
real days comes from the workflow, run against the real files.
"""

import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "recent_results.py"


def _module():
    spec = importlib.util.spec_from_file_location("recent_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr = _module()


def _fixture(tmp_path: Path, with_morning: bool = True):
    """Two days of two six-runner races, a scratched runner, and a day not yet run.

    Runner 0 is the shortest price in every race and finishes second; runner 1
    wins at 4.5-ish. The model's price is the true price with a known tilt, so
    the ranks and returns below can be checked by hand.
    """
    live = tmp_path / "live"
    live.mkdir()
    res = []
    for day in ("2026-09-18", "2026-09-19"):
        rows = []
        for trk, tm in (("Ascot", "2.30"), ("York", "3.05")):
            for h in range(6):
                price = 2.0 + 2.5 * h
                row = dict(date=day, venue=trk, race_time=tm, runner_name=f"{trk} {day} {h}",
                           predicted_bfsp=price, predicted_win_prob=1 / price)
                if with_morning:
                    row["bf_best_back"] = price * 1.1          # always 10% long of BSP
                rows.append(row)
                res.append(dict(race_date=day, track=trk, race_time=tm,
                                horse_name=f"{trk} {day} {h}", bfsp=price,
                                placing_numerical=1 if h == 1 else (2 if h == 0 else h + 1)))
            rows.append(dict(date=day, venue=trk, race_time=tm, runner_name=f"Scratched {trk}",
                             predicted_bfsp=40.0, predicted_win_prob=0.025))
        pd.DataFrame(rows).to_csv(live / f"{day}.csv", index=False)
    pd.DataFrame([dict(date="2026-09-20", venue="Ascot", race_time="1.10", runner_name="Tomorrow",
                       predicted_bfsp=3.0, predicted_win_prob=0.33)]).to_csv(
        live / "2026-09-20.csv", index=False)
    db = tmp_path / "db.sqlite"
    con = sqlite3.connect(db)
    pd.DataFrame(res).to_sql("race_results", con, index=False)
    con.close()
    return live, db


def _joined(tmp_path, **kw):
    from research_lab import _live_clv_frame
    live, db = _fixture(tmp_path, **kw)
    return rr.score(_live_clv_frame(str(live), str(db)))


def test_a_race_time_written_as_2_30_still_joins(tmp_path):
    """The whole point of reading the morning files as text."""
    d = _joined(tmp_path)
    assert len(d) == 24                          # 2 days x 2 races x 6 runners
    assert set(d["race_time"]) == {"2.30", "3.05"}


def test_a_scratched_runner_and_an_unrun_day_fall_out(tmp_path):
    d = _joined(tmp_path)
    assert not d["horse_name"].str.startswith("Scratched").any()
    assert "2026-09-20" not in set(d["race_date"])


def test_ranks_are_whole_numbers_within_each_race(tmp_path):
    d = _joined(tmp_path)
    assert d["model_rank"].dtype.kind == "i"
    for _, g in d.groupby("raceid"):
        assert sorted(g["model_rank"]) == list(range(1, len(g) + 1))


def test_returns_are_net_of_commission_at_bsp(tmp_path):
    d = _joined(tmp_path)
    winner = d[d["won"]].iloc[0]
    assert winner["ret"] == pytest.approx((winner["bfsp"] - 1.0) * 0.95)
    assert (d.loc[~d["won"], "ret"] == -1.0).all()


def test_the_all_row_is_the_sum_of_the_days(tmp_path):
    s = rr.day_summary(_joined(tmp_path))
    days, total = s[s["date"] != "all"], s[s["date"] == "all"].iloc[0]
    assert total["races"] == days["races"].sum() == 4
    assert total["runners"] == days["runners"].sum() == 24
    # The shortest price in every race finished second, so top pick and
    # favourite are the same horse and both lose all four.
    assert total["top_pick"] == "0/4" and total["top_pick_pnl"] == -4.0
    assert total["favourite"] == total["top_pick"]


def test_the_morning_comparison_scores_both_on_the_same_rows(tmp_path):
    fm = rr.forecast_vs_morning(_joined(tmp_path))
    assert fm["runners_with_morning_price"] == 24 and fm["coverage_%"] == 100.0
    # The model's price equals BSP exactly; the morning price is 10% long of it.
    assert fm["model_median_abs_log_err"] == 0.0
    assert fm["morning_median_abs_log_err"] == pytest.approx(round(np.log(1.1), 3))
    assert fm["top_pick_median_morning/bsp"] == pytest.approx(1.1)
    assert fm["top_pick_morning_beat_bsp_%"] == 100.0


def test_no_morning_prices_is_said_rather_than_scored(tmp_path):
    assert rr.forecast_vs_morning(_joined(tmp_path, with_morning=False)) is None


def test_the_report_runs_end_to_end_and_names_what_it_matched(tmp_path, capsys):
    live, db = _fixture(tmp_path)
    out = tmp_path / "out"
    assert rr.main(["--live-dir", str(live), "--db", str(db), "--out", str(out)]) == 0
    text = (out / "summary.md").read_text()
    assert "2026-09-18: 14 predicted, 12 settled (86%)" in text
    assert "2026-09-20: 1 predicted, 0 settled" in text
    # A day that joined nothing is diagnosed, not excused as "no racing".
    assert "## Days that joined nothing" in text
    assert "database rows dated 2026-09-20: 0" in text
    assert (out / "top_picks.csv").exists() and (out / "runners.csv").exists()
    assert len(pd.read_csv(out / "top_picks.csv")) == 4


def test_nothing_settled_is_a_failure_not_an_empty_table(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    pd.DataFrame([dict(date="2026-09-20", venue="Ascot", race_time="1.10", runner_name="X",
                       predicted_bfsp=3.0, predicted_win_prob=0.33)]).to_csv(live / "2026-09-20.csv",
                                                                              index=False)
    db = tmp_path / "db.sqlite"
    con = sqlite3.connect(db)
    pd.DataFrame([dict(race_date="2026-09-18", track="York", race_time="2.30",
                       horse_name="Y", bfsp=3.0, placing_numerical=1)]).to_sql("race_results", con, index=False)
    con.close()
    assert rr.main(["--live-dir", str(live), "--db", str(db), "--out", str(tmp_path / "o")]) == 1


def test_the_diagnosis_tells_no_results_from_keys_that_disagree(tmp_path):
    """The two ways a day joins nothing need different fixes, so say which.

    Here the database has the day's results, but under a different track name
    than the morning file used: the rows exist and the keys still miss.
    """
    live = tmp_path / "live"
    live.mkdir()
    pd.DataFrame([dict(date="2026-09-19", venue="Newmarket (July)", race_time="2.30",
                       runner_name="Some Horse", predicted_bfsp=3.0,
                       predicted_win_prob=0.33)]).to_csv(live / "2026-09-19.csv", index=False)
    db = tmp_path / "db.sqlite"
    con = sqlite3.connect(db)
    pd.DataFrame([dict(race_date="2026-09-19", track="Kempton", race_time="2.30",
                       horse_name="Some Horse", bfsp=3.0, placing_numerical=1)]).to_sql(
        "race_results", con, index=False)
    con.close()

    lines = rr.unmatched_diagnosis(str(live), str(db), "2026-09-19")
    text = "\n".join(lines)
    assert "database rows dated 2026-09-19: 1" in text      # results are there...
    assert "morning keys" in text and "database keys" in text  # ...so show both sides
    assert "Kempton" in text and "Newmarket" in text

    # And a day the database has never heard of says exactly that.
    assert "database rows dated 2026-09-25: 0" in "\n".join(
        rr.unmatched_diagnosis(str(live), str(db), "2026-09-25"))
