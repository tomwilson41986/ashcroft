"""gold/inplay_minute and gold/inplay: the match as the market saw it, minute by minute (a hand-written match through
the real parser for the grid, the as-of rule and the quality rule; a plain reference for the summary's arithmetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from football.warehouse import bronze, gold, gold_inplay, silver
from football.warehouse.schema import TS, conform
from football.warehouse.store_io import read_part
from sources.common import Store
from tests.test_football_warehouse import KO_MS, MO_RUNNERS, OU_RUNNERS, _bundle, _defn, _file, _msg

M = 60_000
START = KO_MS + 90_000                    # the market goes in play 90 seconds late
HOME, AWAY, DRAW, UNDER, OVER = 71085, 31317, 58805, 47972, 47973
MO, OU, EVENT = "1.252787713", "1.252787714", 35155405


def match_odds_file(in_play=True) -> bytes:
    settled = [dict(r, status=s) for r, s in zip(MO_RUNNERS, ("LOSER", "LOSER", "WINNER"))]       # a draw
    rc = lambda **p: [{"id": {"h": HOME, "d": DRAW, "a": AWAY}[k], "ltp": v} for k, v in p.items()]   # noqa: E731
    msgs = [_msg(KO_MS - 40 * 60 * M, MO, defn=_defn()),
            _msg(KO_MS - M, MO, rc=rc(h=2.0, d=4.0, a=4.0))]                   # the close: a book of exactly one
    if in_play:
        msgs += [_msg(START, MO, defn=_defn(in_play=True, betDelay=5)),
                 _msg(START + 2 * M, MO, rc=rc(h=1.9)),                         # on minute 2 exactly: it counts
                 _msg(START + 3 * M + 1, MO, rc=rc(h=1.8)),                     # a millisecond after minute 3: not yet
                 _msg(START + 9 * M + 30_000, MO, rc=rc(h=1.25, d=8.0, a=20.0)),    # the home side scores
                 _msg(START + 29 * M + 30_000, MO, rc=rc(h=1.2, d=15.0, a=100.0)),
                 _msg(START + 39 * M + 30_000, MO, rc=rc(h=1.15)),              # only the favourite trades
                 _msg(START + 49 * M + 30_000, MO, rc=rc(h=3.0)),               # an equaliser: one leg re-prices ...
                 _msg(START + 50 * M + 30_000, MO, rc=rc(d=2.0, a=8.0))]        # ... the others a minute later
    msgs.append(_msg(START + 55 * M, MO, defn=_defn("CLOSED", in_play, runners=settled, numberOfActiveRunners=0)))
    return _file(msgs)


def over_under_file() -> bytes:
    kw = {"market_type": "OVER_UNDER_25", "name": "Over/Under 2.5 Goals", "runners": OU_RUNNERS}
    return _file([_msg(KO_MS - 40 * 60 * M, OU, _defn(**kw)),
                  _msg(KO_MS - M, OU, rc=[{"id": UNDER, "ltp": 2.0}, {"id": OVER, "ltp": 2.0}]),
                  _msg(START, OU, _defn(in_play=True, **kw)),
                  _msg(START + 9 * M + 30_000, OU, rc=[{"id": UNDER, "ltp": 2.5}, {"id": OVER, "ltp": 1.6}]),
                  _msg(START + 20 * M, OU, _defn("CLOSED", True, **kw))])          # closed early: the third goal


def goals(rows) -> pa.Table:
    """A hand-written gold/goal: (goal_no, ms, home_score, away_score)."""
    return pa.table({"event_id": pa.array([EVENT] * len(rows), pa.int64()),
                     "goal_no": pa.array([r[0] for r in rows], pa.int16()),
                     "pt": pa.array([r[1] for r in rows], pa.int64()).cast(TS),
                     "home_score": pa.array([r[2] for r in rows], pa.int16()),
                     "away_score": pa.array([r[3] for r in rows], pa.int16())})


def timeline(rows) -> pa.Table:
    """A hand-written gold/timeline: (usable, consistent, ht_start_min, ht_end_min[, reconciled, n_unplaced])."""
    rows = [tuple(r) + (False, 0)[len(r) - 4:] for r in rows]
    return pa.table({"event_id": pa.array([EVENT] * len(rows), pa.int64()),
                     "usable": pa.array([r[0] for r in rows], pa.bool_()),
                     "consistent": pa.array([r[1] for r in rows], pa.bool_()),
                     "reconciled": pa.array([r[4] for r in rows], pa.bool_()),
                     "n_unplaced": pa.array([r[5] for r in rows], pa.int16()),
                     "ht_start_min": pa.array([r[2] for r in rows], pa.float64()),
                     "ht_end_min": pa.array([r[3] for r in rows], pa.float64())})


GOALS = [(1, START + 9 * M + 10_000, 1, 0), (2, START + 49 * M, 1, 1)]
TIMELINE = [(True, True, 24.0, 34.0)]


def inputs_of(tmp_path, name="base", mo=None, goal_rows=GOALS, timeline_rows=TIMELINE) -> dict:
    """The module's inputs for a one-match warehouse: silver and gold/event through the real parser and build, the
    inferred goals and timeline written by hand."""
    root = tmp_path / name
    root.mkdir()
    store = Store(root=root / "wh")
    bronze.ingest_bundle(store, _bundle(root, {f"2026/Jan/29/{EVENT}/{MO}.bz2": mo or match_odds_file(),
                                               f"2026/Jan/29/{EVENT}/{OU}.bz2": over_under_file()}), "b")
    silver.build(store, workers=1)
    assert gold.build(store, only=["gold_event"])["failed"] == []
    inp = {t: read_part(store, t, 2026, 1, gold_inplay.COLUMNS.get(t))
           for t in gold_inplay.INPUTS if t not in ("gold/goal", "gold/timeline")}
    inp["gold/goal"], inp["gold/timeline"] = goals(goal_rows), timeline(timeline_rows)
    return inp


@pytest.fixture()
def built(tmp_path):
    out = gold_inplay.build(inputs_of(tmp_path))
    return {k: conform(spec, out[spec.name]).to_pandas() for k, spec in (("minute", gold_inplay.MINUTE),
                                                                         ("match", gold_inplay.MATCH))}


def test_one_row_a_minute_from_the_kick_off_to_the_close(built):
    m = built["minute"]
    assert m["minute"].tolist() == list(range(56)) and (m["event_id"] == EVENT).all()        # closed at 55:00
    ms = (m["pt"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    assert (ms == START + m["minute"].astype("int64") * M).all()
    assert not m.duplicated(["event_id", "minute"]).any()


def test_a_minute_reads_the_last_price_at_or_before_it_the_close_until_the_first_in_play_trade(built):
    m = built["minute"].set_index("minute")
    assert m.loc[0, ["p_home", "p_draw", "p_away"]].tolist() == [0.5, 0.25, 0.25]            # the pre-off close
    assert m.loc[0, "mo_max_age_s"] == 150.0 and m.loc[0, "mo_legs"] == 3
    book2 = 1 / 1.9 + 0.5                                           # the 1.9 was published on minute 2 exactly
    assert m.loc[2, "p_home"] == pytest.approx((1 / 1.9) / book2) and m.loc[2, "mo_book_sum"] == pytest.approx(book2)
    assert m.loc[3, "p_home"] == m.loc[2, "p_home"]                 # 1.8 came a millisecond after minute 3
    assert m.loc[4, "p_home"] == pytest.approx((1 / 1.8) / (1 / 1.8 + 0.5))
    assert m.loc[10, ["p_home", "p_draw", "p_away"]].tolist() == pytest.approx([0.8 / 0.975, 0.125 / 0.975,
                                                                                 0.05 / 0.975])


def test_a_reading_is_ok_when_the_book_is_whole_and_stale_legs_carry_little_of_it(built):
    m = built["minute"].set_index("minute")
    ok = set(m.index[m["ok"]])
    # 0-2: the close, under five minutes old; 10-14 and 30-34: all three legs traded together; 40-44: only the
    # favourite trades and the stale legs carry 8 per cent; 51-54: after the equaliser has reached every leg
    assert ok == set(range(0, 3)) | set(range(10, 15)) | set(range(30, 35)) | set(range(40, 45)) | set(range(51, 55))
    assert m.loc[3, "mo_stale_mass"] == pytest.approx(0.5 / (1 / 1.9 + 0.5))       # draw and away five minutes old
    assert m.loc[40, "mo_stale_mass"] == pytest.approx((1 / 15 + 0.01) / (1 / 1.15 + 1 / 15 + 0.01))
    assert m.loc[40, "mo_max_age_s"] == 630.0 and bool(m.loc[40, "ok"])            # stale, small, harmless
    # minute 50: the equaliser has reached one leg only; last traded prices of two scores do not make a book
    assert m.loc[50, "mo_book_sum"] == pytest.approx(1 / 3 + 1 / 15 + 0.01) and not m.loc[50, "ok"]
    assert m.loc[51, ["p_home", "p_draw", "p_away"]].tolist() == pytest.approx(
        [x / (1 / 3 + 0.5 + 0.125) for x in (1 / 3, 0.5, 0.125)])


def test_the_over_chance_stands_while_its_market_is_open_and_fresh(built):
    m = built["minute"].set_index("minute")
    assert m.loc[0, "p_over"] == 0.5 and m.loc[10, "p_over"] == pytest.approx(0.625 / 1.025)
    assert m.loc[14, "p_over"] == pytest.approx(0.625 / 1.025) and np.isnan(m.loc[15, "p_over"])   # five minutes old
    assert m.loc[19, "ou_open"] and not m.loc[20, "ou_open"] and np.isnan(m.loc[30, "p_over"])     # closed at 20:00


def test_the_phase_and_the_football_minute_follow_the_matchs_half_time(tmp_path, built):
    m = built["minute"].set_index("minute")
    assert m.loc[23, "phase"] == "first_half" and m.loc[24, "phase"] == "half_time"
    assert m.loc[33, "phase"] == "half_time" and m.loc[34, "phase"] == "second_half"
    assert (m.loc[10, "match_min"], m.loc[28, "match_min"], m.loc[34, "match_min"], m.loc[40, "match_min"]) == \
        (10.0, 45.0, 45.0, 51.0)
    # no timeline for the match: the 2026 medians stand in, the score is blank and nothing claims to be checked
    bare = gold_inplay.build(inputs_of(tmp_path, "bare", goal_rows=[], timeline_rows=[]))["gold/inplay_minute"]
    b = bare.to_pandas().set_index("minute")
    assert b.loc[47, "phase"] == "first_half" and b.loc[48, "phase"] == "half_time" and b.loc[47, "match_min"] == 45.0
    assert b["home_score"].isna().all() and not b["score_ok"].any()


def test_the_score_is_the_inferred_one_as_of_the_minute(tmp_path, built):
    m = built["minute"].set_index("minute")
    assert (m.loc[9, "home_score"], m.loc[10, "home_score"], m.loc[10, "away_score"]) == (0, 1, 0)   # scored at 9:10
    assert (m.loc[48, "away_score"], m.loc[49, "away_score"]) == (0, 1)        # a goal on the minute counts for it
    assert m["score_ok"].all()
    unusable = gold_inplay.build(inputs_of(tmp_path, "u", timeline_rows=[(False, True, 24.0, 34.0)]))
    u = unusable["gold/inplay_minute"].to_pandas()
    assert u["home_score"].isna().all() and u["away_score"].isna().all() and not u["score_ok"].any()
    # a score put right by the settlement counts; one with a goal that has no minute keeps its score, not the flag
    fixed = gold_inplay.build(inputs_of(tmp_path, "r", timeline_rows=[(True, False, 24.0, 34.0, True, 0)]))
    assert fixed["gold/inplay_minute"].to_pandas()["score_ok"].all()
    loose = gold_inplay.build(inputs_of(tmp_path, "n", timeline_rows=[(True, True, 24.0, 34.0, False, 1)]))
    n = loose["gold/inplay_minute"].to_pandas()
    assert not n["score_ok"].any() and n["home_score"].notna().all()
    wrong = gold_inplay.build(inputs_of(tmp_path, "w", timeline_rows=[(True, False, 24.0, 34.0, False, 0)]))
    assert not wrong["gold/inplay_minute"].to_pandas()["score_ok"].any()


def reference(minute: pd.DataFrame, close, result: str) -> dict:
    """The summary, slowly: one pass over a match's minutes."""
    held, last, hold = [], None, list(close)
    P = minute[["p_home", "p_draw", "p_away"]].to_numpy(float)
    travel, changes, seen, low, decided = 0.0, 0, [], 1.0, None
    r = "HDA".index(result)
    for i, ok in enumerate(minute["ok"].to_numpy()):
        if ok:
            if last is not None:
                travel += 0.5 * abs(P[i] - P[last]).sum()
                changes += int(P[i].argmax() != P[last].argmax())
            last, hold = i, list(P[i])
            seen.append(P[i])
            low = min(low, P[i][r])
            if P[i][r] < gold_inplay.DECIDED:
                decided = None
            elif decided is None:
                decided = int(minute["minute"].iloc[i])
        held.append(hold)
    avg = np.mean(held, axis=0)
    seen = np.array(seen)
    return {"avg": avg, "xh": 3 * avg[0] + avg[1], "xa": 3 * avg[2] + avg[1], "travel": travel, "changes": changes,
            "max_h": seen[:, 0].max(), "min_h": seen[:, 0].min(), "max_a": seen[:, 2].max(), "min_a": seen[:, 2].min(),
            "low": low, "decided": decided, "n_ok": int(minute["ok"].sum())}


def test_the_summary_counts_every_minute_and_reads_the_path(tmp_path, monkeypatch):
    monkeypatch.setattr(gold_inplay, "MIN_OK", 10)                   # the hand-written match has 22 ok minutes
    out = gold_inplay.build(inputs_of(tmp_path))
    m, s = out["gold/inplay_minute"].to_pandas(), out["gold/inplay"].to_pandas().iloc[0]
    want = reference(m, (0.5, 0.25, 0.25), "D")
    assert (s["minutes"], s["n_ok"], bool(s["filled"])) == (56, 22, True) and want["n_ok"] == 22
    assert [s["avg_p_home"], s["avg_p_draw"], s["avg_p_away"]] == pytest.approx(list(want["avg"]))
    assert s["avg_p_home"] + s["avg_p_draw"] + s["avg_p_away"] == pytest.approx(1.0)
    assert (s["xpts_pre_home"], s["xpts_pre_away"]) == (1.75, 1.0)   # 3 x 0.5 + 0.25; 3 x 0.25 + 0.25
    assert s["xpts_inplay_home"] == pytest.approx(want["xh"]) and s["xpts_inplay_away"] == pytest.approx(want["xa"])
    assert s["perf_home"] == pytest.approx(want["xh"] - 1.75) and s["perf_home"] > 0 > s["perf_away"]
    # half time ran from minute 24 to 33: its middle is 28 or 29, and the nearest ok reading within three is 30
    assert [s["ht_p_home"], s["ht_p_draw"], s["ht_p_away"]] == pytest.approx(
        m.loc[m["minute"] == 30, ["p_home", "p_draw", "p_away"]].iloc[0].tolist())
    assert (s["max_p_home"], s["min_p_home"]) == pytest.approx((want["max_h"], want["min_h"]))
    assert (s["max_p_away"], s["min_p_away"]) == pytest.approx((want["max_a"], want["min_a"]))
    assert s["travel"] == pytest.approx(want["travel"]) and s["fav_changes"] == want["changes"] == 1
    # the draw was 5 per cent at its lowest and never reached 90 per cent: no minute decided it
    assert s["result_min_p"] == pytest.approx(want["low"]) and np.isnan(s["decided_min"]) and want["decided"] is None


def test_decided_is_the_first_minute_from_which_the_result_stayed_likely():
    n = 40
    p_home = np.r_[np.full(10, 0.5), np.full(5, 0.95), np.full(5, 0.6), np.full(20, 0.97)]      # dips at 15-19
    minute = pd.DataFrame({"event_id": 1, "minute": np.arange(n), "phase": "first_half", "ok": True,
                           "p_home": p_home, "p_draw": (1 - p_home) * 0.6, "p_away": (1 - p_home) * 0.4})
    ev = pd.DataFrame({"event_id": [1], "result": ["H"], "p_home": [0.5], "p_draw": [0.3], "p_away": [0.2]})
    s = gold_inplay.summary_frame(minute, ev).iloc[0]
    assert s["decided_min"] == 20 and s["result_min_p"] == 0.5 and s["fav_changes"] == 0
    assert s["travel"] == pytest.approx(0.45 + 0.35 + 0.37)
    # no result (a void match): nothing to decide, the rest stands
    ev["result"] = None
    v = gold_inplay.summary_frame(minute, ev).iloc[0]
    assert np.isnan(v["decided_min"]) and np.isnan(v["result_min_p"])
    assert v["avg_p_home"] == pytest.approx(p_home.mean())


def test_a_thin_match_keeps_its_minutes_and_leaves_the_summary_blank(built):
    s = built["match"].iloc[0]                                       # 22 ok minutes: under the thirty needed
    assert not s["filled"] and s["n_ok"] == 22 and s["minutes"] == 56
    assert np.isnan(s["avg_p_home"]) and np.isnan(s["perf_home"]) and np.isnan(s["travel"]) and s["fav_changes"] == 0
    assert s["xpts_pre_home"] == 1.75                                # the pre-match expectation needs no path


def test_a_match_that_never_went_in_play_has_no_rows_and_the_build_is_pure(tmp_path):
    never = gold_inplay.build(inputs_of(tmp_path, "never", mo=match_odds_file(in_play=False)))
    assert never["gold/inplay_minute"].num_rows == 0 and never["gold/inplay"].num_rows == 0
    assert never["gold/inplay_minute"].schema.names == [c.name for c in gold_inplay.MINUTE.columns]
    inp = inputs_of(tmp_path, "pure")
    a, b = gold_inplay.build(inp), gold_inplay.build({k: v.slice(0) for k, v in inp.items()})
    for name in ("gold/inplay_minute", "gold/inplay"):
        assert a[name].equals(b[name])
        conform(dict((s.name, s) for s in gold_inplay.SPECS)[name], a[name])       # keys non-null, types as declared
