"""The Professor (the owner's tipster, 23 Oct 2026): the rule as config.json states it, the odds sources on the pages they
were built against, the publisher's record, and the workflow's switches."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from professor import selection
from professor.candidates import Candidate
from professor.scrapers import racingpost
from professor.scrapers.base import norm_name, norm_venue, to_decimal
from professor.writeup import compose, validate

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "professor"


def cand(runner="Horse", venue="Cheltenham", p_win=0.25, best_back=5.0, size=120.0, market="1.1", sel=1,
         day="2026-10-23", off_utc="2026-10-23T14:35:00+00:00", first="2026-10-23T05:40:00+00:00") -> Candidate:
    return Candidate(day=day, market_id=market, selection_id=sel, runner=runner, venue=venue,
                     off_utc=datetime.fromisoformat(off_utc), first_utc=datetime.fromisoformat(first),
                     first_price=best_back, best_back=best_back, back_size=size, p_win=p_win, edge=0.05,
                     exp_bsp=1 / p_win, predicted_bfsp=best_back * 0.9, stake=25.0)


@pytest.fixture
def cfg():
    c = selection.load_config()
    c["withheld_share"] = 0.0
    return c


# ----------------------------------------------------------------------------------------------- the rule
def test_the_advised_price_is_what_three_major_bookmakers_offer(cfg):
    quotes = {"bet365": 6.0, "William Hill": 5.5, "Paddy Power": 5.5, "Coral": 5.0, "Tote": 7.0}
    price, books = selection.books_price(quotes, cfg["major_bookmakers"], 3)
    assert price == 5.5 and set(books) == {"bet365", "William Hill", "Paddy Power"}   # the Tote is not major
    assert selection.books_price({"bet365": 6.0, "Coral": 5.0}, cfg["major_bookmakers"], 3) == (None, [])


def test_a_horse_qualifies_on_the_overlay_over_the_expected_bsp(cfg):
    c = cand(p_win=0.25)                                   # expected BSP 4.0
    tip, why = selection.qualify(c, cfg, {"bet365": 4.5, "William Hill": 4.5, "Coral": 4.5})
    assert tip is not None and tip.price_source == "books" and abs(tip.overlay - 0.125) < 1e-9
    tip, why = selection.qualify(c, cfg, {"bet365": 4.2, "William Hill": 4.2, "Coral": 4.2})
    assert tip is None and "overlay" in why


def test_bookmaker_prices_only_by_the_owners_decision(cfg):
    assert cfg["require_books"] is True                    # the owner, 25 Oct: "bookmaker only"
    c = cand(p_win=0.25, best_back=4.8)
    tip, why = selection.qualify(c, cfg, None)             # no bookmaker quotes the race: no tip
    assert tip is None and why == "no bookmaker quotes"
    tip, _ = selection.qualify(c, cfg, {"bet365": 4.8, "William Hill": 4.8, "Coral": 4.8})
    assert tip is not None and tip.price_source == "books"


def test_the_exchange_stands_in_only_when_no_bookmaker_quotes_the_race(cfg):
    cfg = dict(cfg, require_books=False)                    # the fallback, kept for an owner who turns it back on
    c = cand(p_win=0.25, best_back=4.8)
    tip, why = selection.qualify(c, cfg, None)             # the race was not read: the exchange price is advised (GBP120 at it)
    assert tip is not None and tip.price_source == "exchange" and tip.advised == 4.8
    tip, why = selection.qualify(c, cfg, {"bet365": 4.8})   # the race was read and the horse has one quote: not advised
    assert tip is None and "bookmakers" in why
    cfg2 = dict(cfg, require_books=True)
    assert selection.qualify(c, cfg2, None)[0] is None


def test_the_price_cap(cfg):
    c = cand(p_win=0.01, best_back=60.0)
    assert selection.qualify(c, cfg, None)[0] is None


def test_the_card_ranks_by_win_probability_one_a_race_and_caps(cfg):
    cfg = dict(cfg, max_tips=3, free_tips=2)
    cands = [cand("A", p_win=0.30, best_back=4.0, market="1.1", sel=1), cand("B", p_win=0.28, best_back=4.5, market="1.1", sel=2),
             cand("C", p_win=0.20, best_back=6.5, market="1.2", sel=3), cand("D", p_win=0.15, best_back=9.0, market="1.3", sel=4),
             cand("E", p_win=0.12, best_back=11.0, market="1.4", sel=5)]
    quotes = {(c.market_id, c.selection_id): {b: c.best_back for b in ("bet365", "William Hill", "Coral")} for c in cands}
    tips, fates = selection.card(cands, cfg, quotes)
    assert [t.cand.runner for t in tips] == ["A", "C", "D"]        # B is the same race as A; E is past the cap
    assert [t.tier for t in tips] == ["free", "free", "private"]
    assert len(fates) == 5 and all(f["fate"] == "qualifies" for f in fates)


def test_the_withheld_fifth_takes_no_rank_and_is_a_fixed_draw(cfg):
    cfg = dict(cfg, withheld_share=0.5, max_tips=10)
    cands = [cand(f"H{i}", p_win=0.3 - i * 0.01, best_back=4.0 + i, market=f"1.{i}", sel=i) for i in range(1, 21)]
    quotes = {(c.market_id, c.selection_id): {b: c.best_back for b in ("bet365", "William Hill", "Coral")} for c in cands}
    tips, _ = selection.card(cands, cfg, quotes)
    held = [t for t in tips if t.withheld]
    shown = [t for t in tips if not t.withheld]
    assert held and shown and all(t.rank == 0 and t.tier == "withheld" for t in held)
    assert [t.rank for t in shown] == list(range(1, len(shown) + 1))
    again, _ = selection.card(cands, cfg, quotes)
    assert [t.withheld for t in again] == [t.withheld for t in tips]


def test_the_alert_loop_takes_fills_in_arrival_order_with_the_floor_and_the_cap(cfg):
    cfg = dict(cfg, max_tips=2, min_p_win_alert=0.15, min_alert_size=50)
    cands = [cand("A", p_win=0.20, best_back=6.0, market="1.1", sel=1), cand("B", p_win=0.10, best_back=12.0, market="1.2", sel=2),
             cand("C", p_win=0.22, best_back=5.5, market="1.3", sel=3, size=10), cand("D", p_win=0.25, best_back=5.0, market="1.4", sel=4),
             cand("E", p_win=0.30, best_back=4.0, market="1.5", sel=5)]
    quotes = {(c.market_id, c.selection_id): {b: c.best_back for b in ("bet365", "William Hill", "Coral")} for c in cands}
    new, fates = selection.alerts(cands, [], cfg, quotes)
    assert [t.cand.runner for t in new] == ["A", "D"]              # B under the floor, C too little money, E past the cap
    assert [t.rank for t in new] == [1, 2]
    later, _ = selection.alerts(cands, new, cfg, quotes)
    assert later == []


# ----------------------------------------------------------------------------------------------- the words
def test_the_template_passes_its_own_validator(cfg):
    c = cand("Calico", p_win=0.24, best_back=4.5)
    tip, _ = selection.qualify(c, cfg, {"bet365": 5.0, "William Hill": 5.0, "Coral": 5.0, "Ladbrokes": 4.5})
    text = compose(tip, cfg)
    assert validate({k: v for k, v in text.items() if k != "sheet"}, text["sheet"], cfg) == []
    assert "guarantee" not in text["body"].lower() and len(text["x_text"]) <= 280
    assert "18+" in text["body"] and "syndicate" in text["body"].lower()


def test_the_validator_rejects_a_number_not_in_the_sheet_a_banned_claim_and_a_url(cfg):
    sheet = {"runner": "X", "advised": "5.00", "compliance": cfg["compliance"]}
    bad = {"body": "X at 5.00 is a guaranteed 7.00 winner, see www.example.com", "x_text": "x" * 281}
    reasons = validate(bad, sheet, cfg)
    assert any("7.00" in r for r in reasons) and any("banned" in r for r in reasons) and any("URL" in r for r in reasons)
    assert any("280" in r for r in reasons)


# ----------------------------------------------------------------------------------------------- the sources
def test_prices_parse():
    assert to_decimal("7/2") == 4.5 and to_decimal("EVS") == 2.0 and to_decimal("4.50") == 4.5
    assert to_decimal("SP") is None and to_decimal("") is None and to_decimal("1/5") == 1.2
    assert norm_name("JPR One (IRE)") == "jprone" and norm_venue("Wolverhampton (AW)") == "wolverhampton"
    assert norm_venue("Dundalk (A.W)") == "dundalk" and norm_venue("Wolverhampton (A.W)") == "wolverhampton"


def test_racing_post_index_lists_the_days_races():
    races = racingpost.races_from_index((FIX / "racingpost_index.html").read_text(), "2026-10-23")
    assert len(races) == 12 and all(r["url"].endswith("/betting-odds/") for r in races)
    assert {r["venue"] for r in races} >= {"Cheltenham"} and all(r["hhmm"].count(":") == 1 for r in races)


def test_racing_post_odds_table_gives_every_bookmakers_price_on_every_runner():
    quotes = racingpost.quotes_from_race((FIX / "racingpost_betting_odds.html").read_text(), "Cheltenham", "15:35", "929291")
    assert len(quotes) == 80
    by = {}
    for q in quotes:
        by.setdefault(q.horse, {})[q.bookmaker] = q.price
    assert by["JPR One"]["William Hill"] == 4.5 and by["JPR One"]["bet365"] == 4.5 and by["JPR One"]["Tote"] == 4.5
    assert by["Calico"]["Betfair Sportsbook"] == 5.5 and by["Calico"]["Ladbrokes"] == 4.5
    assert {q.bookmaker for q in quotes} == {"William Hill", "Ladbrokes", "bet365", "Coral", "Paddy Power",
                                             "Betfair Sportsbook", "Sky Bet", "BoyleSports", "Betfred", "Tote"}
    assert all(q.race_hhmm == "15:35" and q.venue == "Cheltenham" for q in quotes)


def test_the_quotes_match_our_horses_by_course_time_and_name(cfg):
    from professor import odds
    quotes = racingpost.quotes_from_race((FIX / "racingpost_betting_odds.html").read_text(), "Cheltenham", "15:35", "929291")
    df = odds.frame(quotes, cfg)
    ours = [cand("JPR One", "Cheltenham", sel=1, off_utc="2026-10-23T14:35:00+00:00"),
            cand("Calico (IRE)", "Cheltenham", sel=2, off_utc="2026-10-23T14:35:00+00:00"),
            cand("Nobody", "Cheltenham", sel=3, off_utc="2026-10-23T14:35:00+00:00")]
    q, bog = odds.match(df, ours)
    assert q[("1.1", 1)]["bet365"] == 4.5 and len(q) == 2
    price, books = selection.books_price(q[("1.1", 1)], cfg["major_bookmakers"], 3)
    assert price == 4.5 and len(books) >= 3


# ----------------------------------------------------------------------------------------------- the workflow
def test_the_workflow_posts_nothing_by_itself():
    spec = yaml.safe_load((ROOT / ".github" / "workflows" / "professor-publish.yml").read_text())
    on = spec.get("on", spec.get(True))
    assert [c["cron"] for c in on["schedule"]] == ["35 8 * * *", "35 9 * * *"]
    assert on["workflow_dispatch"]["inputs"]["dry_run"]["default"] is True
    choose = spec["jobs"]["choose"]["steps"][0]["run"]
    assert 'MODE=card; DRY=true' in choose and 'TZ=Europe/London' in choose
    run = next(s for s in spec["jobs"]["professor"]["steps"] if s.get("name") == "Run")
    assert run["env"]["PROFESSOR_LIVE"] == "${{ vars.PROFESSOR_LIVE }}"
    assert "BETFAIR_LIVE_APP_KEY" not in json.dumps(spec)                 # the tipster never holds the live key
    assert spec["concurrency"]["cancel-in-progress"] is False


def test_racing_posts_non_runners_are_read_and_never_advised(cfg):
    from professor import odds
    state = {"props": {"pageProps": {"initialState": {"racePage": {"data": {"runners": [
        {"horseName": "Palamon", "nonRunner": True}, {"horseName": "Sony Bill", "nonRunner": False}]}}}}}}
    html = ('<html><body><script id="__NEXT_DATA__" type="application/json">' + json.dumps(state) + "</script></body></html>")
    assert racingpost.non_runners_from_race(html) == ["Palamon"]
    from professor.scrapers.base import Quote
    df = odds.frame([Quote("racingpost", racingpost.NON_RUNNER, "Cheltenham", "15:35", "Palamon", 0.0)], cfg)
    ours = [cand("Palamon", "Cheltenham", sel=7, off_utc="2026-10-23T14:35:00+00:00")]
    nrs = odds.non_runners(df, ours)
    assert nrs == {("1.1", 7)}
    tips, fates = selection.card(ours, cfg, {}, {}, nrs)
    assert tips == [] and fates[0]["fate"] == "a non-runner"
    assert odds.match(df, ours)[0].get(("1.1", 7), {}) == {}          # a non-runner row is never a price


def test_a_tip_is_settled_at_the_advised_price_and_at_the_bsp():
    import pandas as pd
    from professor.settle import settle_tips, summary
    tips = pd.DataFrame([
        {"day": "2026-10-24", "market_id": "1.1", "selection_id": 1, "advised": 5.0, "withheld": False},
        {"day": "2026-10-24", "market_id": "1.2", "selection_id": 2, "advised": 8.0, "withheld": False},
        {"day": "2026-10-24", "market_id": "1.3", "selection_id": 3, "advised": 6.0, "withheld": False},
        {"day": "2026-10-24", "market_id": "1.4", "selection_id": 4, "advised": 9.0, "withheld": True}])
    results = pd.DataFrame([
        {"market_id": "1.1", "selection_id": 1, "result": "WINNER", "bsp": 4.0},
        {"market_id": "1.2", "selection_id": 2, "result": "LOSER", "bsp": 10.0},
        {"market_id": "1.3", "selection_id": 3, "result": "REMOVED", "bsp": None}])
    s = settle_tips(tips, results)
    assert list(s["pts_advised"]) == [4.0, -1.0, 0.0, 0.0]
    assert abs(s["pts_bsp"].iloc[0] - 3.0 * 0.98) < 1e-9 and s["pts_bsp"].iloc[1] == -1.0
    assert abs(s["clv"].iloc[0] - 0.25) < 1e-9 and abs(s["clv"].iloc[1] + 0.2) < 1e-9
    assert list(s["result"]) == ["WINNER", "LOSER", "REMOVED", "PENDING"]
    summ = summary(s)
    assert summ["published"]["settled"] == 2 and summ["published"]["pts_advised"] == 3.0
    assert summ["published"]["non_runners"] == 1 and summ["withheld"]["pending"] == 1


def test_an_abandoned_meeting_voids_our_horses_in_it(cfg):
    from professor import odds
    from professor.scrapers.base import Quote
    df = odds.frame([Quote("racingpost", "ABANDONED", "Aintree", "12:40", "*", 0.0)], cfg)
    ours = [cand("Anything", "Aintree", sel=9, off_utc="2026-10-25T12:40:00+00:00"),
            cand("Elsewhere", "Galway", sel=10, off_utc="2026-10-25T12:32:00+00:00")]
    ab = odds.abandoned(df, ours)
    assert ab == {("1.1", 9)}
    _, fates = selection.card(ours, cfg, {}, {}, set(), ab)
    assert fates[0]["fate"] == "the meeting is abandoned" and fates[1]["fate"] != "the meeting is abandoned"
