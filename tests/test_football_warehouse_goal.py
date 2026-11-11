"""gold/goal and gold/timeline: goals read off in-play prices. Each test scripts one match: its goals go into a
small Poisson pricer that writes the match's price messages minute by minute (match odds, over/under 2.5 and the
Asian handicap lines, a message only when a price changes, as Betfair's files do), and the real build reads them."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import poisson

from football.warehouse import gold_goal as gg
from football.warehouse.schema import conform

KO = 1_769_716_800_000                       # 2026-01-29 20:00 UTC
MIN = 60_000
HT = (47, 62)                                # wall-clock minutes of the break
PLAY = 98                                    # minutes of play
SPAN = PLAY + HT[1] - HT[0]                  # wall-clock minutes in play
LINES = [x + 0.5 for x in range(-6, 6)]      # the handicap's half lines, -5.5 to 5.5
HOME, AWAY, DRAW, UNDER, OVER = 1, 2, 3, 11, 12
TS = pa.timestamp("ms", tz="UTC")


def played(wall: float) -> float:
    return wall if wall <= HT[0] else HT[0] if wall <= HT[1] else HT[0] + wall - HT[1]


def chances(left: float, home: int, away: int, lh: float, la: float) -> dict:
    """Fair chances with ``left`` of the match to play at home-away: (selection, handicap) -> chance."""
    k = np.arange(0, 14)
    pm = np.outer(poisson.pmf(k, lh * left), poisson.pmf(k, la * left))
    diff, total = k[:, None] - k[None, :] + home - away, k[:, None] + k[None, :] + home + away
    out = {(HOME, 0.0): pm[diff > 0].sum(), (DRAW, 0.0): pm[diff == 0].sum(), (AWAY, 0.0): pm[diff < 0].sum()}
    if home + away < 3:
        out[(UNDER, 0.0)] = pm[total <= 2].sum()
        out[(OVER, 0.0)] = 1.0 - out[(UNDER, 0.0)]
    for line in LINES:
        out[(HOME, line)] = pm[diff + line > 0].sum()
        out[(AWAY, -line)] = 1.0 - out[(HOME, line)]
    return out


def match(event_id: int = 1, goals=(), lh: float = 1.5, la: float = 1.1, shocks=(), spells=(), fake=(),
          last_price: int = SPAN - 2, first_price: int = 1, skip=(), settle=None) -> dict:
    """One match's rows for every input table. ``goals``: (wall minute, +1 home / -1 away); ``shocks``: (wall
    minute, factor on the home rate, factor on the away rate): a red card; ``fake``: (from, to, side): the prices
    of minutes from..to-1 show a goal that is then taken back; ``spells``: (wall minute, seconds) suspensions of
    the match odds and over/under; ``skip``: wall minutes with no message; ``settle``: the final (home, away) the
    markets settle on, where that is not the scripted goals' (a goal after the last price)."""
    mo, ou, ah = f"mo{event_id}", f"ou{event_id}", f"ah{event_id}"
    market_of = {HOME: mo, AWAY: mo, DRAW: mo, UNDER: ou, OVER: ou}
    ticks, last = [], {}
    third = None
    for w in range(first_price, last_price + 1):
        h = sum(1 for m, s in goals if m <= w and s > 0) + sum(1 for a, b, s in fake if a <= w < b and s > 0)
        a = sum(1 for m, s in goals if m <= w and s < 0) + sum(1 for a_, b, s in fake if a_ <= w < b and s < 0)
        if third is None and h + a >= 3:
            third = w
        fh = float(np.prod([f for m, f, _ in shocks if m <= w] or [1.0]))
        fa = float(np.prod([f for m, _, f in shocks if m <= w] or [1.0]))
        left = max(1.0 - played(w) / PLAY, 0.01)
        in_break = HT[0] < w < HT[1]
        if w in skip:
            continue
        for (sel, line), p in chances(left, h, a, lh * fh, la * fa).items():
            price = float(np.clip(round(1.0 / max(p, 1e-9), 2), 1.01, 1000.0))
            if in_break and sel == DRAW and line == 0.0 and w % 3 == 0:
                price = round(price + 0.01 * (1 if w % 2 else -1), 2)          # a flicker: the break is not silent
            market = ah if line != 0.0 else market_of[sel]
            key = (market, sel, line)
            if last.get(key) != price:
                last[key] = price
                ticks.append((market, sel, line, KO + w * MIN + {mo: 0, ou: 2000, ah: 4000}[market], w, price, True))
    fh_, fa_ = settle or (sum(1 for _, s in goals if s > 0), sum(1 for _, s in goals if s < 0))
    ft = KO + SPAN * MIN
    result = "H" if fh_ > fa_ else "A" if fa_ > fh_ else "D"
    ou_closed = KO + third * MIN + 60_000 if third is not None else ft
    event = {"event_id": event_id, "mo_market_id": mo, "ou_market_id": ou, "ah_market_id": ah, "went_in_play": True,
             "kickoff_actual": KO, "ft_pt": ft, "result": result, "over25": int(fh_ + fa_ >= 3), "void": False}
    close = chances(1.0, 0, 0, lh, la)
    event.update(p_home=close[(HOME, 0.0)], p_draw=close[(DRAW, 0.0)], p_away=close[(AWAY, 0.0)])
    runners = [(mo, "MATCH_ODDS", HOME, 0.0, "home", int(result == "H")),
               (mo, "MATCH_ODDS", AWAY, 0.0, "away", int(result == "A")),
               (mo, "MATCH_ODDS", DRAW, 0.0, "draw", int(result == "D")),
               (ou, "OVER_UNDER_25", UNDER, 0.0, "under", int(fh_ + fa_ < 3)),
               (ou, "OVER_UNDER_25", OVER, 0.0, "over", int(fh_ + fa_ >= 3))]
    for line in LINES:
        runners.append((ah, "ASIAN_HANDICAP", HOME, line, "home", int(fh_ - fa_ + line > 0)))
        runners.append((ah, "ASIAN_HANDICAP", AWAY, -line, "away", int(fh_ - fa_ + line < 0)))
    stops = [(m, KO + w * MIN - 20_000, KO + w * MIN - 20_000 + 1000 * secs, True, "SUSPENDED")
             for w, secs in spells for m in (mo, ou)]
    stops += [(mo, ft - 2 * MIN, ft, True, "SUSPENDED"), (ah, ft - 2 * MIN, ft, True, "SUSPENDED"),
              (ou, (KO + third * MIN - 20_000) if third is not None else ft - 2 * MIN, ou_closed, True, "SUSPENDED")]
    return {"event": [event], "market": [(mo, ft), (ou, ou_closed), (ah, ft)], "runner": runners, "spell": stops,
            "tick": ticks}


def inputs(*matches) -> dict:
    def col(key, i):
        return [r[i] for m in matches for r in m[key]]

    events = [m["event"][0] for m in matches]
    ev = {k: [e[k] for e in events] for k in gg.COLUMNS["gold/event"]} if events else {}
    stamp = (lambda v: pa.array(v, pa.int64()).cast(TS))
    event = pa.table({**ev, "kickoff_actual": stamp(ev["kickoff_actual"]), "ft_pt": stamp(ev["ft_pt"])}) if events \
        else pa.table({k: pa.array([], t) for k, t in (
            ("event_id", pa.int64()), ("mo_market_id", pa.string()), ("ou_market_id", pa.string()),
            ("ah_market_id", pa.string()), ("went_in_play", pa.bool_()), ("kickoff_actual", TS), ("ft_pt", TS),
            ("result", pa.string()), ("over25", pa.int8()), ("void", pa.bool_()), ("p_home", pa.float64()),
            ("p_draw", pa.float64()), ("p_away", pa.float64()))})
    return {
        "gold/event": event,
        "silver/market": pa.table({"market_id": pa.array(col("market", 0), pa.string()),
                                   "closed_pt": stamp(col("market", 1))}),
        "silver/runner": pa.table({"market_id": pa.array(col("runner", 0), pa.string()),
                                   "market_type": pa.array(col("runner", 1), pa.string()),
                                   "selection_id": pa.array(col("runner", 2), pa.int64()),
                                   "handicap": pa.array(col("runner", 3), pa.float64()),
                                   "role": pa.array(col("runner", 4), pa.string()),
                                   "won": pa.array(col("runner", 5), pa.int8())}),
        "silver/spell": pa.table({"market_id": pa.array(col("spell", 0), pa.string()),
                                  "start_pt": stamp(col("spell", 1)), "end_pt": stamp(col("spell", 2)),
                                  "in_play": pa.array(col("spell", 3), pa.bool_()),
                                  "status": pa.array(col("spell", 4), pa.string())}),
        "silver/tick": pa.table({"market_id": pa.array(col("tick", 0), pa.string()),
                                 "selection_id": pa.array(col("tick", 1), pa.int64()),
                                 "handicap": pa.array(col("tick", 2), pa.float64()),
                                 "pt": stamp(col("tick", 3)), "seq": pa.array(col("tick", 4), pa.int32()),
                                 "ltp": pa.array(col("tick", 5), pa.float64()),
                                 "in_play": pa.array(col("tick", 6), pa.bool_())}),
    }


def run(*matches):
    """(goals, timeline) as pandas frames, through the real build and the tables' contracts."""
    out = gg.build(inputs(*matches))
    return (conform(gg.GOAL, out["gold/goal"]).to_pandas(), conform(gg.TIMELINE, out["gold/timeline"]).to_pandas())


def wall(g) -> list:
    return [round(x) for x in g["elapsed_min"]]


def test_one_goal_is_seen_by_all_three_markets_and_timed_to_the_minute():
    g, t = run(match(goals=[(30, +1)]))
    assert len(g) == 1 and len(t) == 1
    r = g.iloc[0]
    assert (r["goal_no"], r["side"], r["evidence"], r["home_score"], r["away_score"]) == (1, "home", "ou+ah", 1, 0)
    assert r["pt"].value // 10**6 == KO + 30 * MIN and r["pt_before"].value // 10**6 == KO + 29 * MIN
    assert r["half"] == 1 and r["match_min"] == 30.0 and r["timed"] and not r["by_settlement"]
    assert r["mo_jump"] > 1.0 and r["ou_jump"] > 0.7 and 0.6 < r["ah_jump"] < 1.4
    assert r["home_after"] < r["home_before"] and r["under_after"] > r["under_before"]
    assert r["confidence"] == gg.CONFIDENCE["ou+ah"][0]
    m = t.iloc[0]
    assert (m["n_goals"], m["home_goals"], m["away_goals"], m["n_side_unknown"]) == (1, 1, 0, 0)
    assert m["usable"] and m["consistent"] and not m["reconciled"] and m["n_edits"] == 0
    assert m["agrees_result"] and m["agrees_over25"] and m["agrees_margin"] and m["final_margin"] == 1
    assert m["reliability"] == gg.RELIABILITY["agreed"][0]                  # agreed, under three goals


def test_a_goal_with_no_suspension_is_a_jump_between_two_messages_and_one_with_a_suspension_starts_there():
    g, t = run(match(1, goals=[(30, -1)]), match(2, goals=[(30, -1)], spells=[(30, 45)]))
    plain, marked = g[g["event_id"] == 1].iloc[0], g[g["event_id"] == 2].iloc[0]
    assert not plain["suspended"] and plain["side"] == "away" and plain["mo_jump"] < -1.0
    assert marked["suspended"] and marked["pt"].value // 10**6 == KO + 30 * MIN - 20_000     # the spell's start
    assert t.set_index("event_id").loc[2, ["n_stoppages", "n_goal_stoppages", "n_other_stoppages"]].tolist() \
        == [1, 1, 0]
    assert t.set_index("event_id").loc[1, "n_stoppages"] == 0


def test_two_goals_two_minutes_apart_stay_two():
    g, t = run(match(goals=[(20, +1), (22, -1)]))
    assert wall(g) == [20, 22] and g["side"].tolist() == ["home", "away"]
    assert g[["home_score", "away_score"]].to_numpy().tolist() == [[1, 0], [1, 1]]
    assert t.iloc[0]["consistent"]


def test_the_third_goal_closes_the_over_under_market_and_is_read_from_that():
    g, t = run(match(goals=[(15, +1), (40, +1), (70, -1)]))
    assert wall(g) == [15, 40, 70] and g["evidence"].tolist()[:2] == ["ou+ah", "ou+ah"]
    third = g.iloc[2]
    assert third["evidence"] == "third" and third["side"] == "away" and np.isnan(third["ou_jump"])
    assert (third["home_score"], third["away_score"]) == (2, 1) and third["half"] == 2
    m = t.iloc[0]
    assert abs(m["ou_closed_min"] - 70) < 1.0 and m["agrees_over25"] and m["consistent"]
    assert m["reliability"] == gg.RELIABILITY["agreed"][1]                  # agreed, three goals


def test_a_red_card_moves_match_odds_and_not_under_so_it_is_not_a_goal():
    g, t = run(match(goals=[(70, +1)], shocks=[(30, 0.6, 1.5)], spells=[(30, 60)]))
    assert wall(g) == [70]                                   # the card at 30 moved the home price out by a third
    m = t.iloc[0]
    assert (m["n_goals"], m["n_stoppages"], m["n_goal_stoppages"], m["n_other_stoppages"]) == (1, 1, 0, 1)
    assert m["consistent"] and not m["reconciled"]


def test_a_goal_taken_back_is_undone_when_under_comes_back_in():
    g, t = run(match(goals=[(70, -1)], fake=[(30, 33, +1)]))
    assert wall(g) == [70] and g["side"].tolist() == ["away"]
    assert t.iloc[0]["n_disallowed"] == 1 and t.iloc[0]["consistent"]


def test_a_goal_at_three_nil_is_seen_only_by_the_handicap_margin():
    g, t = run(match(goals=[(10, +1), (25, +1), (40, +1), (70, +1)], lh=1.8, la=0.8))
    assert wall(g)[:3] == [10, 25, 40] and len(g) == 4
    fourth = g.iloc[3]
    assert fourth["evidence"] == "ah" and fourth["side"] == "home" and abs(fourth["elapsed_min"] - 70) <= 2.0
    assert np.isnan(fourth["ou_jump"]) and 0.6 < fourth["ah_jump"] < 1.4
    assert (fourth["home_score"], fourth["away_score"]) == (4, 0)
    m = t.iloc[0]
    assert m["final_margin"] == 4 and m["agrees_margin"] and m["consistent"]
    # match odds are pinned at 3-0 and the over/under is closed: no price row for over 12 minutes, so not usable
    assert m["max_gap_min"] > gg.USABLE_GAP_MIN and not m["usable"] and np.isnan(m["reliability"])


def test_a_goal_after_the_over_under_closed_is_read_from_match_odds_and_the_handicap():
    g, t = run(match(goals=[(10, +1), (25, -1), (40, +1), (70, -1)]))
    assert wall(g) == [10, 25, 40, 70] and g["evidence"].tolist() == ["ou+ah", "ou+ah", "third", "ah+mo"]
    fourth = g.iloc[3]
    assert fourth["side"] == "away" and np.isnan(fourth["ou_jump"]) and fourth["mo_jump"] < -1.0
    assert (fourth["home_score"], fourth["away_score"]) == (2, 2)
    m = t.iloc[0]
    assert m["usable"] and m["consistent"] and m["final_margin"] == 0
    assert m["reliability"] == gg.RELIABILITY["agreed"][2]                  # agreed, four goals


def test_half_time_is_found_and_minutes_are_football_minutes():
    g, t = run(match(goals=[(30, +1), (46, -1), (80, +1)]))
    m = t.iloc[0]
    assert m["ht_found"] and abs(m["ht_end_min"] - HT[1]) < 1.5 and abs(m["ht_start_min"] - HT[0]) < 2.5
    assert abs(m["inplay_min"] - SPAN) < 1e-9
    assert g["half"].tolist() == [1, 1, 2]
    assert g["match_min"].tolist()[:2] == [30.0, 45.0]                       # first-half stoppage time is 45
    assert abs(g["match_min"].iloc[2] - (45 + 80 - HT[1])) < 1.5
    assert (m["first_goal_min"], m["last_goal_min"]) == (30.0, g["match_min"].iloc[2])


def test_the_restart_is_read_from_the_minutes_in_play_where_no_quiet_stretch_is_found():
    found = np.array([True, True, False, False])
    span = np.array([113.0, np.nan, 113.0, np.nan])
    _, start, end = gg._restart(found, np.full(4, 40.0), np.full(4, 63.0), span)
    a, b, c = gg.RESTART_FOUND
    assert end[0] == pytest.approx(a + b * 63.0 + c * 113.0) and end[1] == gg.RESTART_END[0] + gg.RESTART_END[1] * 63.0
    assert end[2] == gg.RESTART_SPAN[0] + gg.RESTART_SPAN[1] * 113.0 and end[3] == gg.HT_END_MIN
    assert start[0] == end[0] - gg.HT_BREAK_MIN[2] and start[2] == end[2] - gg.HT_BREAK_MIN[1]


def test_scores_after_each_goal_and_the_minutes_leading_level_and_trailing():
    g, t = run(match(goals=[(20, +1), (40, -1), (75, -1)]))
    assert g[["home_score", "away_score"]].to_numpy().tolist() == [[1, 0], [1, 1], [1, 2]]
    assert g["elapsed_min"].tolist() == pytest.approx([20.0, 40.0, 75.0 - 1 / 3])     # the third: its suspension
    m = t.iloc[0]
    at_third = m["ht_start_min"] + g["elapsed_min"].iloc[2] - m["ht_end_min"]        # minutes of play by then
    whole = m["ht_start_min"] + SPAN - m["ht_end_min"]
    assert m["min_home_leading"] == pytest.approx(20.0)                              # from 20 to 40
    assert m["min_level"] == pytest.approx(20.0 + at_third - 40.0)                   # to 20, and 40 to the third
    assert m["min_home_trailing"] == pytest.approx(whole - at_third)
    assert abs(whole - PLAY) < 3.0 and abs(at_third - (HT[0] + 75 - HT[1])) < 3.0


def test_the_usable_rule():
    whole = match(1, goals=[(30, +1)])
    gap = match(2, goals=[(30, +1)], skip=range(70, 84))                      # 14 minutes with no message
    late = match(3, goals=[(30, +1)], first_price=55)                         # too few rows
    started = match(4, goals=[(1, +1)])                                       # in play after a goal: prices far off
    short = match(5, goals=[(30, +1)])
    short["event"][0]["ft_pt"] = KO + 80 * MIN                                # closed after 80 minutes
    unsettled = match(6, goals=[(30, +1)])
    unsettled["event"][0]["result"] = None
    _, t = run(whole, gap, late, started, short, unsettled)
    t = t.set_index("event_id")
    assert t["usable"].tolist() == [True, False, False, False, False, False]
    assert t.loc[2, "max_gap_min"] >= 14 and t.loc[3, "n_rows"] < gg.USABLE_ROWS
    assert t.loc[4, "start_jump"] > gg.USABLE_START and t.loc[1, "start_jump"] < 0.1
    assert np.isnan(t.loc[2, "reliability"]) and t.loc[1, "reliability"] == gg.RELIABILITY["agreed"][0]


def test_a_goal_after_the_last_price_is_placed_by_the_settlement_and_flagged():
    g, t = run(match(goals=[(30, +1)], settle=(1, 1), last_price=SPAN - 6))
    assert g["evidence"].tolist() == ["ou+ah", "settle"] and g["side"].tolist() == ["home", "away"]
    late = g.iloc[1]
    assert not late["timed"] and late["by_settlement"] and late["confidence"] == gg.CONFIDENCE["settle"][0]
    assert late["pt"].value // 10**6 >= KO + (SPAN - 6) * MIN and late["match_min"] == 90.0
    assert (late["home_score"], late["away_score"]) == (1, 1)
    m = t.iloc[0]
    assert not m["consistent"] and not m["agrees_result"] and m["agrees_over25"] and not m["agrees_margin"]
    assert m["reconciled"] and (m["n_edits"], m["n_unplaced"], m["n_goals"]) == (1, 1, 2)
    assert m["reliability"] == gg.RELIABILITY["unplaced"][0]


def test_the_cheapest_edit_the_settlement_allows():
    keep, cand = float(gg._logit(0.95)), float(-gg._logit(0.3))
    # agreed already: nothing changes
    assert gg._solve([(True, 1, keep)], 1, 1, 0) == ((1,), 0, 0)
    # one home goal short, a home candidate on offer: it is cheaper than a goal no price shows
    assert cand < gg.UNSEEN_COST and gg._solve([(True, 1, keep), (False, 1, cand)], 2, 1, 0) == ((1, 1), 0, 0)
    # the candidate is on the wrong side: an unseen home goal
    assert gg._solve([(True, 1, keep), (False, -1, cand)], 2, 1, 0) == ((1, 0), 1, 0)
    # a weak goal against a draw: out it goes, cheaper than an unseen equaliser when over 2.5 forbids a third
    weak = float(gg._logit(0.53))
    assert gg._solve([(True, 1, keep), (True, -1, keep), (True, 1, weak)], 0, 0, 0) == ((1, -1, 0), 0, 0)
    # a goal with no side takes the side the margin needs
    assert gg._solve([(True, 0, keep)], -1, -1, 0) == ((-1,), 0, 0)
    # the lines stop at four: five nil satisfies a margin of four or more, three nil does not
    assert gg._solve([(True, 1, keep)] * 5, 4, 1, 1) == ((1,) * 5, 0, 0)
    assert gg._solve([(True, 1, keep)] * 3, 4, 1, 1) == ((1,) * 3, 1, 0)
    # out of reach: more unseen goals than allowed
    assert gg._solve([], 4, 1, 1) is None


def test_a_reconciliation_drops_the_weakest_goal_promotes_a_candidate_or_turns_a_side():
    import pandas as pd
    goals = pd.DataFrame({"e": [0, 1, 1, 2], "t": [1000, 1000, 2000, 1000], "side": [1.0, 1.0, 1.0, 1.0],
                          "kind": ["ou+ah", "ou+ah", "ah", "ou+ah"], "edit": ""})
    cands = pd.DataFrame({"e": [0, 0], "t": [2000, 3000], "side": [-1.0, 1.0], "kind": ["ou", "mo"],
                          "chance": [0.5, 0.08]})
    ev = pd.DataFrame({"final_margin": [0.0, 1.0, -1.0, np.nan], "result": ["D", "H", "A", None],
                       "over25": [0, 0, 0, None]})
    need = np.array([True, True, True, False])
    out, done, edits, unplaced = gg._reconcile(goals, cands, ev, need, np.full(4, 9000.0))
    out = out.sort_values(["e", "t"]).reset_index(drop=True)
    # match 0 settled 1-1: the away candidate comes in; match 1 settled 1-0: the handicap-only goal goes;
    # match 2 settled 0-1 with one sure home goal: cheaper turned than dropped and replaced by an unseen one
    assert out[["e", "t", "side", "kind", "edit"]].to_numpy().tolist() == [
        [0, 1000, 1.0, "ou+ah", ""], [0, 2000, -1.0, "ou", "added"], [1, 1000, 1.0, "ou+ah", ""],
        [2, 1000, -1.0, "ou+ah", "turned"]]
    assert done.tolist() == [True, True, True, False] and edits.tolist() == [1, 1, 1, 0]
    assert unplaced.tolist() == [0, 0, 0, 0]


def test_keys_are_unique_required_columns_are_filled_and_every_column_is_documented():
    g, t = run(match(1, goals=[(20, +1), (22, -1), (70, +1)]), match(2, goals=[]), match(3, goals=[(88, -1)]))
    assert not g.duplicated(["event_id", "goal_no"]).any() and not t.duplicated(["event_id"]).any()
    assert t["event_id"].tolist() == [1, 2, 3] and t.set_index("event_id").loc[2, "n_goals"] == 0
    for spec, frame in ((gg.GOAL, g), (gg.TIMELINE, t)):
        assert list(frame.columns) == spec.names
        for c in spec.columns:
            assert c.doc and (c.nullable or not frame[c.name].isna().any())
    nil = t.set_index("event_id").loc[2]
    assert nil["consistent"] and nil["min_level"] == pytest.approx(nil["ht_start_min"] + SPAN - nil["ht_end_min"])
    assert np.isnan(nil["first_goal_min"]) and np.isnan(nil["min_confidence"])


def test_the_build_is_a_pure_function_of_its_inputs_whatever_their_order():
    ms = [match(1, goals=[(20, +1), (64, -1)]), match(2, goals=[(10, -1), (30, -1), (50, +1), (80, +1)]),
          match(3, goals=[(30, +1)], settle=(1, 1), last_price=SPAN - 6)]
    first = gg.build(inputs(*ms))
    rng = np.random.default_rng(7)
    mixed = {k: v.take(pa.array(rng.permutation(v.num_rows))) for k, v in inputs(*reversed(ms)).items()}
    again = gg.build(mixed)
    for name in ("gold/goal", "gold/timeline"):
        assert first[name].equals(again[name])


def test_an_empty_partition_and_one_with_nothing_in_play_give_empty_tables():
    out = gg.build(inputs())
    assert out["gold/goal"].num_rows == 0 and out["gold/timeline"].num_rows == 0
    assert conform(gg.GOAL, out["gold/goal"]).schema == gg.GOAL.schema
    assert conform(gg.TIMELINE, out["gold/timeline"]).schema == gg.TIMELINE.schema
    never = match(1, goals=[(30, +1)])
    never["event"][0]["went_in_play"] = False
    out = gg.build(inputs(never))
    assert out["gold/goal"].num_rows == 0 and out["gold/timeline"].num_rows == 0
    silent = match(2, goals=[])
    silent["tick"] = []                                        # in play, and not one price: a row, no goals
    once = match(3, goals=[], first_price=10, last_price=10)   # one price message
    g, t = run(silent, once)
    assert len(g) == 0 and t["n_rows"].tolist() == [0, 1] and not t["usable"].any()
    assert t["consistent"].tolist() == [True, True] and t["n_goals"].tolist() == [0, 0]
