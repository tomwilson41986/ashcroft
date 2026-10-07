"""The football model: names matched across sources, the canonical matches and their versions, the benchmark prices,
the data-quality checks, the Betfair market map, the ratings and their prices, the value rules, and the trader's
decisions on paper. Small frames written by hand exercise the code; the model is evaluated on the real data only
(reports/football_model.md)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from football import data, live, model, ratings, reference
from sources.common import Store


# --------------------------------------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------------------------------------

def test_names_normalise_across_sources():
    n = reference.normalise
    assert n("Man Utd") == n("Man United") == n("Manchester United FC")
    assert n("Nottm Forest") == n("Nott'm Forest")
    assert n("Sheff Wed") == n("Sheffield Weds")
    assert n("Brighton & Hove Albion FC") == "brighton hove albion"
    assert reference.team_id("ENG", "Liverpool") != reference.team_id("URU", "Liverpool")


def test_resolve_takes_the_alias_then_the_name_then_a_unique_fuzzy_best():
    cands = {"ESP:ath madrid": "Ath Madrid", "ESP:real madrid": "Real Madrid", "ESP:betis": "Betis"}
    al = reference.load_aliases()
    assert reference.resolve("Atletico Madrid", "ESP", cands, "betfair", al)[0] == "ESP:ath madrid"
    assert reference.resolve("Real Madrid", "ESP", cands, "betfair", al)[2] == "exact"
    tid, _, how = reference.resolve("Real Betis", "ESP", cands, "betfair", al)
    assert tid == "ESP:betis" and how == "fuzzy"
    assert reference.resolve("Madrid", "ESP", cands, "betfair", al)[0] is None     # two equally good: no match


def test_pair_event_needs_both_sides_and_a_clear_winner():
    fx = [("m1", "Man United", "Fulham"), ("m2", "Man City", "Chelsea"), ("m3", "Wolves", "Sheffield Weds")]
    assert reference.pair_event("Man Utd", "Fulham", fx, "ENG", "betfair")[0] == 0
    assert reference.pair_event("Wolves", "Sheff Wed", fx, "ENG", "betfair")[0] == 2
    assert reference.pair_event("Arsenal", "Spurs", fx, "ENG", "betfair")[0] is None


# --------------------------------------------------------------------------------------------------------------------
# Canonical matches
# --------------------------------------------------------------------------------------------------------------------

def _silver(rows):
    base = {"Div": "E0", "season": "2025", "Time": "15:00", "FTHG": 1, "FTAG": 0, "HTHG": 0, "HTAG": 0,
            "source_file": "2526/E0.csv"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_odds_columns_are_read_by_bookmaker_phase_market_and_outcome():
    p = data.parse_odds_column
    assert p("PSCH") == ("PS", "close", "1X2", "H")
    assert p("BFECD") == ("BFE", "close", "1X2", "D")
    assert p("BFDH") == ("BFD", "open", "1X2", "H")          # Betfred, not Betfair
    assert p("BFE>2.5") == ("BFE", "open", "OU25", "O")
    assert p("PC<2.5") == ("P", "close", "OU25", "U")
    assert p("AvgCAHA") == ("Avg", "close", "AH", "A")
    assert p("HST") is None and p("AHh") is None and p("Referee") is None


def test_kickoff_is_uk_local_time_turned_to_utc():
    k = data.kickoff_utc(pd.Series(["2025-08-16", "2025-12-20"]), pd.Series(["15:00", "15:00"]))
    assert k.iloc[0].hour == 14 and k.iloc[1].hour == 15        # BST, then GMT


def test_match_id_survives_a_reschedule_and_counts_repeat_meetings():
    a = data.canonical(_silver([
        {"HomeTeam": "Celtic", "AwayTeam": "Rangers", "match_date": "2025-09-01", "Div": "SC0"},
        {"HomeTeam": "Celtic", "AwayTeam": "Rangers", "match_date": "2026-04-01", "Div": "SC0"}]))
    assert a.match_id.nunique() == 2 and sorted(a.meeting) == [1, 2]
    b = data.canonical(_silver([{"HomeTeam": "Celtic", "AwayTeam": "Rangers", "match_date": "2025-09-03",
                                 "Div": "SC0"}]))
    assert b.match_id.iloc[0] == a.match_id.iloc[0]


def test_benchmarks_take_pinnacle_to_2024_and_betfair_after():
    rows = _silver([
        {"HomeTeam": "A", "AwayTeam": "B", "match_date": "2024-09-01", "season": "2024", "PSCH": 2.0, "PSCD": 3.5,
         "PSCA": 4.0, "BFECH": 2.1, "BFECD": 3.6, "BFECA": 4.2},
        {"HomeTeam": "A", "AwayTeam": "B", "match_date": "2025-09-01", "season": "2025", "PSCH": 2.0, "PSCD": 3.5,
         "PSCA": 4.0, "BFECH": 2.1, "BFECD": 3.6, "BFECA": 4.2}])
    g = data.canonical(rows).sort_values("match_date")
    assert list(g.close_1x2_src) == ["PSC", "BFEC"]
    assert list(g.close_h) == [2.0, 2.1]


def test_a_changed_score_raises_the_version_and_keeps_the_old_row():
    s = _silver([{"HomeTeam": "A", "AwayTeam": "B", "match_date": "2025-09-01"}])
    g1, h1 = data.version(None, data.canonical(s), now="t1")
    assert g1.record_version.iloc[0] == 1 and h1.empty
    g2, h2 = data.version(g1, data.canonical(s), now="t2")
    assert g2.record_version.iloc[0] == 1 and g2.last_updated_ts.iloc[0] == "t1" and h2.empty
    s.loc[0, "FTHG"] = 2
    g3, h3 = data.version(g2, data.canonical(s), now="t3")
    assert g3.record_version.iloc[0] == 2 and g3.first_seen_ts.iloc[0] == "t1"
    assert len(h3) == 1 and h3.ft_home.iloc[0] == 1


def test_data_quality_flags_impossible_rows_and_short_seasons():
    s = _silver([{"HomeTeam": "A", "AwayTeam": "B", "match_date": "2023-09-01", "season": "2023", "FTHG": 1,
                  "HTHG": 2},
                 {"HomeTeam": "B", "AwayTeam": "A", "match_date": "2023-10-01", "season": "2023"},
                 {"HomeTeam": "C", "AwayTeam": "A", "match_date": "2023-10-01", "season": "2023"},
                 {"HomeTeam": "A", "AwayTeam": "B", "match_date": "2025-09-01", "season": "2025"}])
    issues = data.dq_checks(data.canonical(s), today="2025-09-02")
    names = set(issues.check_name)
    assert "ht_above_ft" in names and "season_match_count" in names


def test_betfair_events_tie_to_their_match_and_teach_the_names():
    s = _silver([{"HomeTeam": "Man United", "AwayTeam": "Fulham", "match_date": "2025-08-16", "Time": "20:00"},
                 {"HomeTeam": "Wolves", "AwayTeam": "Sheffield Weds", "match_date": "2025-08-16", "Time": "15:00"}])
    g = data.canonical(s)
    bf = pd.DataFrame([
        {"event_id": 1, "event_name": "Man Utd v Fulham", "country": "GB", "market_time": "2025-08-16T19:00:00Z",
         "market_id": "1.1", "market_type": "MATCH_ODDS", "selection_id": s_, "runner_name": n, "won": w,
         "ltp_close": c, "ltp_first": c, "ltp_t_60": c, "ltp_t_1": c, "handicap": 0, "bsp": None}
        for s_, n, w, c in ((10, "Man Utd", 1, 1.8), (11, "Fulham", 0, 5.0), (12, "The Draw", 0, 3.9))] + [
        {"event_id": 2, "event_name": "Arsenal v Spurs", "country": "GB", "market_time": "2025-08-16T11:30:00Z",
         "market_id": "1.2", "market_type": "MATCH_ODDS", "selection_id": 20, "runner_name": "Arsenal", "won": 1,
         "ltp_close": 2.0, "ltp_first": 2.0, "ltp_t_60": 2.0, "ltp_t_1": 2.0, "handicap": 0, "bsp": None}])
    mp, learned, review = data.betfair_map(g, bf)
    mid = g[g.home_team == "Man United"].match_id.iloc[0]
    assert set(mp.match_id) == {mid} and sorted(mp.outcome) == ["A", "D", "H"]
    assert list(review.betfair_event_id) == [2]
    assert ("Man Utd", "ENG:manchester united") in set(zip(learned.source_name, learned.team_id))
    closes = data.betfair_closes(mp)
    assert closes.set_index("match_id").loc[mid, "bf_close_h"] == 1.8 and closes.bf_result.iloc[0] == "H"


def test_other_sources_link_and_disagreeing_scores_are_flagged():
    g = data.canonical(_silver([{"HomeTeam": "Man United", "AwayTeam": "Fulham", "match_date": "2025-08-16"}]))
    other = pd.DataFrame([{"competition_id": "ENG1", "match_date": "2025-08-16", "home": "Manchester United FC",
                           "away": "Fulham FC", "ft_home": 2, "ft_away": 0}])
    linked = data.link_other(g, other, "openfootball")
    assert len(linked) == 1
    bad = data.cross_check(g, linked, "openfootball")
    assert list(bad.check_name) == ["score_vs_openfootball"]


def test_api_football_owns_a_fixtures_status():
    s = _silver([{"HomeTeam": "A", "AwayTeam": "B", "match_date": "2025-08-01"}])
    fx = pd.DataFrame([{"Div": "E0", "Date": "20/09/2025", "Time": "15:00", "HomeTeam": "B", "AwayTeam": "A",
                        "match_date": "2025-09-20", "season": None}])
    g = data.canonical(s, fx)
    linked = pd.DataFrame([{"match_id": g[g.status == "SCHEDULED"].match_id.iloc[0], "status": "POSTPONED",
                            "kickoff_utc": pd.Timestamp("2025-09-21T14:00Z"), "apif_fixture_id": 9, "venue": "X"}])
    out = data.apply_api_football(g, linked)
    assert set(out.status) == {"FINISHED", "POSTPONED"}


# --------------------------------------------------------------------------------------------------------------------
# Ratings and prices
# --------------------------------------------------------------------------------------------------------------------

def test_score_matrix_prices_add_up():
    m = ratings.score_matrix(1.6, 1.1, -0.05)
    p = ratings.markets(m)
    assert abs(p["p_h"] + p["p_d"] + p["p_a"] - 1) < 1e-9 and p["p_h"] > p["p_a"]
    dd = ratings.diff_dist(m)
    # a half-goal handicap is the plain result; a whole line refunds the push
    assert abs(ratings.ah_fair(dd, -0.5, "H") - 1 / p["p_h"]) < 1e-3
    assert abs(ratings.ah_return(dd, 0.0, 2.0, "H") - (p["p_h"] - p["p_a"])) < 1e-9
    q = ratings.ah_return(dd, -0.25, 2.0, "H")                  # half on 0, half on -0.5
    assert abs(q - (ratings.ah_return(dd, 0.0, 2.0, "H") + ratings.ah_return(dd, -0.5, 2.0, "H")) / 2) < 1e-12


def test_the_fit_finds_the_stronger_side_and_the_home_edge():
    rng = np.random.default_rng(1)
    teams = [f"ENG:t{i}" for i in range(6)]
    att = np.linspace(-0.4, 0.4, 6)
    rows = []
    day = pd.Timestamp("2024-08-01")
    for k in range(40):
        for i in range(6):
            for j in range(6):
                if i != j and rng.random() < 0.5:
                    lh, la = np.exp(0.1 + 0.25 + att[i] - (-att[j])), np.exp(0.1 + att[j] - (-att[i]))
                    rows.append({"match_date": (day + pd.Timedelta(days=k * 7)).strftime("%Y-%m-%d"),
                                 "status": "FINISHED", "home_team_id": teams[i], "away_team_id": teams[j],
                                 "home_team": teams[i], "away_team": teams[j], "competition_id": "ENG1",
                                 "ft_home": rng.poisson(lh), "ft_away": rng.poisson(la)})
    r, _ = ratings.fit_pool(pd.DataFrame(rows), "ENG", "2025-08-01", l2=0.5, xi=0.0)
    assert 0.1 < r.home < 0.4
    assert r.teams["ENG:t5"]["att"] > r.teams["ENG:t0"]["att"]
    lh, la = r.rates("ENG:t5", "ENG:t0", "ENG1")
    assert lh > la
    assert r.strength("ENG:new", "ENG1") == (r.att_div[0], r.def_div[0])   # an unseen side: its division's mean


# --------------------------------------------------------------------------------------------------------------------
# Value rules
# --------------------------------------------------------------------------------------------------------------------

def test_asian_handicap_results_split_quarter_lines():
    r = model._ah_result([1, 0, 0, 2, 1], [-0.75, -0.25, 0.0, -1.0, -1.0], "h")
    assert list(r) == [0.5, -0.5, 0.0, 1.0, 0.0]
    assert list(model._ah_result([1], [-0.75], "a")) == [-0.5]


def test_value_rule_backs_only_where_the_edge_clears_and_scores_the_close():
    df = pd.DataFrame({"match_id": ["a", "b"], "status": "FINISHED", "season": 2024, "competition_id": "ENG1",
                       "tier": 1, "match_date": "2024-09-01", "ft_home": [2, 0], "ft_away": [0, 0],
                       "pp_h": [0.6, 0.3], "pp_d": [0.25, 0.3], "pp_a": [0.15, 0.4], "pp_o25": [0.5, 0.5],
                       "bfe_open_h": [2.0, 2.0], "bfe_open_d": [3.5, 3.5], "bfe_open_a": [6.0, 6.0],
                       "bfe_open_o25": [2.0, 2.0], "bfe_open_u25": [2.0, 2.0],
                       "y_h": [1.0, 0.0], "y_d": [0.0, 1.0], "y_a": [0.0, 0.0], "y_o25": [0.0, 0.0],
                       "qclose_h": [0.55, 0.3], "qclose_d": [0.25, 0.3], "qclose_a": [0.2, 0.4],
                       "qclose_o25": [0.5, 0.5]})
    b = model.bets(df, "pp", "bfe", 0.05)
    assert list(zip(b.match_id, b.outcome)) == [("a", "H"), ("b", "A")]
    a = b[b.match_id == "a"].iloc[0]
    assert a.pnl == pytest.approx(0.98) and a.clv == pytest.approx(0.10)


def test_pooling_recovers_the_market_when_the_model_adds_nothing():
    rng = np.random.default_rng(0)
    q = rng.dirichlet([4, 3, 3], 4000)
    y = np.array([np.eye(3)[rng.choice(3, p=p)] for p in q])
    noise = rng.dirichlet([2, 2, 2], 4000)
    th = model.fit_pooling(noise, q, y, "1x2")
    assert abs(th[0]) < 0.1 and 0.8 < th[1] < 1.2


# --------------------------------------------------------------------------------------------------------------------
# The trader, on paper
# --------------------------------------------------------------------------------------------------------------------

def _model():
    r = ratings.Ratings(pool="ENG", fitted_to="2026-10-07", mu=0.2, home=0.25, rho=-0.05, divisions=["ENG1"],
                        att_div=[0.0], def_div=[0.0], teams={
                            "ENG:arsenal": {"att": 0.5, "def": 0.4, "div": "ENG1", "n": 80, "name": "Arsenal"},
                            "ENG:fulham": {"att": -0.1, "def": -0.1, "div": "ENG1", "n": 80, "name": "Fulham"},
                            "ENG:luton": {"att": -0.3, "def": -0.3, "div": "ENG2", "n": 80, "name": "Luton"}})
    return {"ratings": {"ENG": r.to_dict()}, "pooling": {"1x2": [1.0, 0.0, 0.0], "ou25": [1.0, 0.0, 0.0]}}


def test_the_pricer_takes_league_matches_of_rated_sides_only():
    p = live.Pricer(_model())
    m = p.match("Arsenal v Fulham", "GB", "English Premier League")
    assert m["home"] == "ENG:arsenal" and m["away"] == "ENG:fulham"
    assert p.match("Arsenal v Luton", "GB", "English Premier League") is None      # different divisions
    assert p.match("Arsenal v Fulham", "GB", "English FA Cup") is None
    assert p.match("Arsenal Women v Fulham Women", "GB", "Womens Super League") is None
    assert p.match("Arsenal v Fulham", "DE", "") is None
    pr = p.probs(m)
    assert pr["p_h"] > pr["p_a"]


@dataclass
class Q:
    selection_id: int
    back: list
    lay: list
    status: str = "ACTIVE"
    last_traded: float | None = None


@dataclass
class B:
    market_id: str
    runners: dict
    status: str = "OPEN"
    inplay: bool = False
    total_matched: float = 0.0
    bsp_reconciled: bool = False


@dataclass
class FakeFill:
    matched: float
    avg_price: float
    status: str = "SUCCESS"
    bet_id: str = "p1"
    error: str = ""


@dataclass
class FakeExchange:
    books_: dict
    placed: list = field(default_factory=list)

    def books(self, ids, with_sp=False):
        return {i: self.books_[i] for i in ids if i in self.books_}

    def back(self, mid, sid, price, size, ref, min_fill=2.0):
        self.placed.append((mid, sid, price, size))
        return FakeFill(size, price)


def test_the_trader_backs_once_at_a_real_book_and_keeps_the_close():
    cfg = live.load_config()
    cfg = {**cfg, "prob": "model", "edge": 0.05}
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    mk = {"marketId": "1.5", "marketType": "MATCH_ODDS", "event": {"id": "e1", "name": "Arsenal v Fulham"},
          "runners": [{"selectionId": 1, "runnerName": "Arsenal", "sortPriority": 1},
                      {"selectionId": 2, "runnerName": "Fulham", "sortPriority": 2},
                      {"selectionId": 3, "runnerName": "The Draw", "sortPriority": 3}]}
    pricer = live.Pricer(_model())
    m = pricer.match("Arsenal v Fulham", "GB")
    p = pricer.probs(m)
    fair_h = 1 / p["p_h"]
    # Arsenal offered well over the model's fair price; Fulham and the draw under it
    book = B("1.5", {1: Q(1, [(round(fair_h * 1.3, 2), 50.0)], [(round(fair_h * 1.32, 2), 50.0)]),
                     2: Q(2, [(1.5, 50.0)], [(1.52, 50.0)]), 3: Q(3, [(2.0, 50.0)], [(2.02, 50.0)])})
    ex = FakeExchange({"1.5": book})
    t = live.Trader(ex, pricer, cfg, "paper")
    t.markets["1.5"] = {**mk, "match": m, "sel": live.selections(mk, m), "competition_name": "EPL",
                        "start": now + timedelta(minutes=60)}
    t.step(now)
    t.step(now + timedelta(minutes=1))
    assert ex.placed == [("1.5", 1, book.runners[1].back[0][0], cfg["limits"]["level_stake"])]
    assert {r["outcome"]: r["action"] for r in t.rows} == {"H": "back", "D": "skip:edge", "A": "skip:edge"}
    assert (("1.5", 1) in t.closes) and t.turnover == cfg["limits"]["level_stake"]
    # the settlement: Arsenal won 2-0; the close was at the price taken
    ledger = pd.DataFrame(t.rows)
    closes = pd.DataFrame(t.close_rows())
    gold = pd.DataFrame([{"match_date": "2026-10-10", "home_team_id": "ENG:arsenal", "away_team_id": "ENG:fulham",
                          "ft_home": 2, "ft_away": 0, "match_id": "x", "status": "FINISHED"}])
    s = live.settle_rows(ledger, closes, gold, 0.02)
    assert len(s) == 1 and s.won.iloc[0] == 1 and s.pnl.iloc[0] > 0 and s.clv.notna().all()


def test_the_trader_waits_for_a_real_book_and_respects_the_caps():
    cfg = {**live.load_config(), "prob": "model", "edge": -1.0}
    cfg["limits"] = {**cfg["limits"], "max_match_stake": 2.0}
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    mk = {"marketId": "1.6", "marketType": "MATCH_ODDS", "event": {"id": "e2", "name": "Arsenal v Fulham"},
          "runners": [{"selectionId": 1, "runnerName": "Arsenal", "sortPriority": 1},
                      {"selectionId": 2, "runnerName": "Fulham", "sortPriority": 2},
                      {"selectionId": 3, "runnerName": "The Draw", "sortPriority": 3}]}
    pricer = live.Pricer(_model())
    m = pricer.match("Arsenal v Fulham", "GB")
    wide = B("1.6", {1: Q(1, [(1.5, 50.0)], [(3.0, 50.0)]), 2: Q(2, [(4.0, 50.0)], [(9.0, 50.0)]),
                     3: Q(3, [(3.0, 50.0)], [(6.0, 50.0)])})
    ex = FakeExchange({"1.6": wide})
    t = live.Trader(ex, pricer, cfg, "paper")
    t.markets["1.6"] = {**mk, "match": m, "sel": live.selections(mk, m), "competition_name": "EPL",
                        "start": now + timedelta(minutes=60)}
    t.step(now)
    assert not ex.placed and not t.rows                     # a placeholder book: nothing decided yet
    ex.books_["1.6"] = B("1.6", {1: Q(1, [(1.5, 50.0)], [(1.52, 50.0)]), 2: Q(2, [(6.0, 50.0)], [(6.2, 50.0)]),
                                 3: Q(3, [(4.0, 50.0)], [(4.1, 50.0)])})
    t.step(now + timedelta(minutes=1))
    assert len(ex.placed) == 1                              # GBP2 a match: one back, the others capped
    assert sorted(r["action"] for r in t.rows) == ["back", "skip:match_cap", "skip:match_cap"]


def test_the_switch_needs_the_exact_word():
    assert live.live_switch_on({"FOOTBALL_LIVE": "yes"}) and not live.live_switch_on({"FOOTBALL_LIVE": "1"})
    assert not live.live_switch_on({})


def test_settle_writes_the_summary(tmp_path):
    store = Store(root=tmp_path)
    store.put_parquet("football/gold/matches.parquet", pd.DataFrame([{
        "match_date": "2026-10-10", "home_team_id": "ENG:arsenal", "away_team_id": "ENG:fulham", "ft_home": 0,
        "ft_away": 1, "match_id": "x", "status": "FINISHED"}]))
    ledger = pd.DataFrame([{"action": "back", "status": "SUCCESS", "market_id": "1.50", "selection_id": 1,
                            "market_type": "MATCH_ODDS", "outcome": "H", "kickoff_utc": "2026-10-10T14:00:00+00:00",
                            "home_team_id": "ENG:arsenal", "away_team_id": "ENG:fulham", "matched": 2.0,
                            "avg_price": 2.0, "back": 2.0}])
    store.put("football/live/paper/2026-10-10/ledger.csv", ledger.to_csv(index=False).encode())
    from datetime import date
    s = live.settle(store, 3, "paper", today=date(2026, 10, 11))
    assert s["settled"] == 1 and s["pnl"] == -2.0
