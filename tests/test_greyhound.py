"""The greyhound model's engine: comments read as GBGB writes them, and every metric lag-safe (a race's own result,
and its day's other results, never reach its features)."""

import io

import numpy as np
import pandas as pd

from greyhound.data import clean, grade_rank
from greyhound.metrics import GreyhoundMetricsEngine, parse_comment
from greyhound.model import against_price_files, race_norm


def test_comments_read_break_lead_trouble_path_and_finish():
    assert parse_comment("QAw,Rls,Led1,ALd") == {"brk": 1.0, "led": 1.0, "trouble": 0.0, "fin": 0.0, "path": 1.0}
    got = parse_comment("VSAw,BBmp1,Wide,FcdTCk2")
    assert got["brk"] == -2.0 and got["trouble"] == 2.0 and got["path"] == 3.0
    assert parse_comment("MsdBrk,RlsToMid,FinWell")["path"] == 1.5
    assert parse_comment("EP,MidToWide,LdRnIn") == {"brk": 1.0, "led": 0.0, "trouble": 0.0, "fin": 1.0, "path": 2.5}
    assert parse_comment("SAw,SltCrd1,Mid,Fdd")["trouble"] == 0.5
    assert np.isnan(parse_comment("")["brk"])


def test_grades_rank_lower_as_better():
    assert grade_rank("OR") == 0 and grade_rank("A1") == 1 and grade_rank("A11") == 11
    assert grade_rank("B2") == 4 and grade_rank("D3") == 3 and np.isnan(grade_rank("??"))


def _runs(n_days=12, tracks=("Romford", "Hove"), seed=0):
    """A small field of real-looking structure (six dogs, six traps, a few races a day) for the engine's mechanics;
    not data a model is trained or scored on."""
    rng = np.random.default_rng(seed)
    rows, race_id = [], 1000
    dogs = list(range(1, 25))
    for day in range(n_days):
        date = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        for k, track in enumerate(tracks):
            for r in range(2):
                field = rng.choice(dogs, 6, replace=False)
                order = rng.permutation(6)
                race_id += 1
                for trap, (dog, pos) in enumerate(zip(field, order), start=1):
                    t = 28.5 + 0.08 * pos + rng.normal(0, 0.05)
                    rows.append({"race_date": date, "race_time": f"1{8 + r}:{10 * k:02d}:00", "meeting_id": day * 10 + k,
                                 "track": track, "race_id": race_id, "race_class": "A5", "distance_m": 480.0,
                                 "going": 0.0, "runners": 6, "trap": trap, "dog_id": int(dog), "dog_name": f"Dog{dog}",
                                 "sire": f"S{dog % 3}", "trainer": f"T{dog % 4}", "born": "Jan-2024", "sex": "d",
                                 "sp_decimal": 6.0, "position": float(pos + 1), "beaten": "", "beaten_lengths": 1.0,
                                 "sectional": 4.6 + 0.02 * pos, "run_time": t, "adjusted_time": t, "weight_kg": 32.0,
                                 "comment": ["QAw,Rls,ALd", "SAw,Crd1,Mid,RanOn", "EP,Wide"][pos % 3]})
    return clean(pd.DataFrame(rows))


def test_every_metric_is_blind_to_its_own_race_and_its_day():
    runs = _runs()
    eng = GreyhoundMetricsEngine(blocks=("parity", "hrb", "lib"))
    base = eng.calculate_all(runs)
    feats = eng.features
    last_day = runs.race_date.max()
    target = runs[runs.race_date == last_day].race_id.max()
    # rewrite the last day's results: reverse every finishing order, change every time and comment
    alt = runs.copy()
    day = alt.race_date == last_day
    alt.loc[day, "position"] = 7 - alt.loc[day, "position"]
    alt.loc[day, "run_time"] += 1.0
    alt.loc[day, "adjusted_time"] += 1.0
    alt.loc[day, "sectional"] -= 0.3
    alt.loc[day, "comment"] = "VSAw,Crd1,Wide,Fdd"
    alt = clean(alt.drop(columns=["won", "placed2", "field", "grade", "grade_family", "sp_p", "sp_p_norm", "raceid",
                                  "t", "is_trial"]))
    other = GreyhoundMetricsEngine(blocks=("parity", "hrb", "lib")).calculate_all(alt)
    a = base[base.race_id == target].sort_values("trap")[feats].reset_index(drop=True)
    b = other[other.race_id == target].sort_values("trap")[feats].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_a_dogs_form_reads_only_its_earlier_days():
    runs = _runs()
    df = GreyhoundMetricsEngine().calculate_all(runs)
    first = df[df.race_date == df.groupby("dog_id").race_date.transform("min")]
    assert first.runs_before.eq(0).all() and first.gsr_mean1.isna().all()
    dog = df.dog_id.iloc[0]
    d = df[df.dog_id == dog]
    per_day = d.groupby("race_date").gsr.mean()                       # a dog's day: the mean of its runs that day
    want = per_day.shift(1).reindex(d.race_date).values
    np.testing.assert_allclose(d.gsr_mean1.values, want, rtol=1e-9, equal_nan=True)


def test_probabilities_sum_to_one_in_a_race_and_the_price_file_join_needs_the_whole_field():
    pred = pd.DataFrame({"race_id": [1, 1, 2, 2], "p_raw": [0.2, 0.6, 0.3, 0.3]})
    assert race_norm(pred.p_raw, pred.race_id).groupby(pred.race_id).sum().round(9).tolist() == [1.0, 1.0]
    pred = pd.DataFrame({"race_id": [1, 1], "race_date": ["2026-01-01"] * 2, "race_time": ["18:10"] * 2,
                         "track": ["Romford"] * 2, "trap": [1, 2], "field": [2, 2], "won": [1.0, 0.0],
                         "p_model": [0.6, 0.4], "t": pd.to_datetime(["2026-01-01"] * 2)})
    prices = pd.DataFrame({"market": ["win"] * 2, "country": ["GB"] * 2, "race_date": ["2026-01-01"] * 2,
                           "race_time": ["18:10"] * 2, "track": ["Romford"] * 2, "trap": [1, 2], "dog": ["A", "B"],
                           "event_id": [9, 9], "bsp": [1.8, 2.4], "ppwap": [1.9, 2.3], "morningwap": [2.0, 2.2],
                           "morning_vol": [50.0, 10.0], "pp_vol": [500.0, 300.0]})
    assert against_price_files(pred, prices.iloc[:1])["markets"] == 0   # one runner of two: no whole race to score
    got = against_price_files(pred, prices)
    assert got["markets"] == 1 and got["runners"] == 2
    # a morning WAP on one runner of two prices no whole field: no log-loss from it, and the BSP's still read
    part = prices.assign(morningwap=[2.0, float("nan")], win_lose=[1.0, 0.0])
    got = against_price_files(pred, part)
    assert got["logloss"]["morningwap"] is None and got["logloss"]["bsp"] is not None
    assert got["price_coverage"]["morningwap"] == {"runners_priced": 0.5, "markets_whole_field": 0.0}
    # the file's result disagrees with GBGB's: the join found another race, and the market is dropped
    got = against_price_files(pred, prices.assign(win_lose=[0.0, 1.0]))
    assert got["markets"] == 0 and got["markets_result_disagrees"] == 1


def test_a_greyhound_price_file_reads_track_trap_and_country_and_skips_a_broken_row(tmp_path):
    from sources import greyhound_prices as gp
    from sources.common import Store
    head = ("EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_ID,SELECTION_NAME,WIN_LOSE,BSP,PPWAP,MORNINGWAP,PPMAX,"
            "PPMIN,IPMAX,IPMIN,MORNINGTRADEDVOL,PPTRADEDVOL,IPTRADEDVOL\n")
    rows = ("1,Romford 3rd Oct,A5 400m,03-10-2026 18:33,11,1. Goldcash Warrior,1,3.2,3.4,3.6,4,3,3,1.01,25.5,800,90\n"
            "1,Romford 3rd Oct,A5 400m,03-10-2026 18:33,12,2. Bad, Name,0,5.1,5,5.2,6,4,5,4,10,300,20\n"
            "2,AUS / Albion Park (AUS) 3rd Oct,R1 331m,03-10-2026 09:10,21,3. Zoom,0,7,7,7,8,6,7,6,5,100,1\n")
    raw = tmp_path / "betfair_prices_raw"
    raw.mkdir()
    (raw / "dwbfgreyhoundwin04102026.csv").write_text(head + rows)
    store = Store(root=tmp_path / "sources")
    got = gp.build(store)
    assert got[2026]["files"] == 1 and got[2026]["unreadable"] == 0 and got[2026]["runners"] == 2
    t = store.get_parquet("greyhound_prices/prices_2026.parquet")
    gb = t[t.track == "Romford"].iloc[0]
    assert (gb.trap, gb.dog, gb.country, gb.race_time, gb.bsp, gb.morning_vol) == (1, "Goldcash Warrior", "GB",
                                                                                  "18:33", 3.2, 25.5)
    au = t[t.country == "AUS"].iloc[0]
    assert au.track == "Albion Park" and au.trap == 3
    assert gp.build(store)[2026] == {"files": 1, "unchanged": True}


def test_the_forward_test_backs_at_the_first_book_only_where_it_was_offered_and_settles_at_bsp():
    from greyhound.track import STAKE, book_marks, join_markets, paper_bets
    day = pd.Timestamp("2026-10-05").date()
    pred = pd.DataFrame({"race_id": [7, 7], "race_date": ["2026-10-05"] * 2, "race_time": ["19:04"] * 2,
                         "track": ["Romford"] * 2, "trap": [1, 2], "field": [2, 2], "won": [1.0, 0.0],
                         "p_model": [0.6, 0.4], "dog_id": [1, 2], "dog_name": ["A", "B"], "sp_decimal": [1.8, 2.5]})
    markets = pd.DataFrame({"market_id": ["1.5"] * 2, "market_type": ["WIN"] * 2, "country": ["GB"] * 2,
                            "venue": ["Romford"] * 2, "market_start_utc": ["2026-10-05T18:04:00Z"] * 2,
                            "selection_id": [11, 12], "runner_name": ["1. Alpha", "2. Beta"]})
    book = dict(source="recorder", market_id="1.5", status="OPEN", inplay=0, number_of_active_runners=2,
                sp_actual=None)
    books = pd.DataFrame([
        dict(book, polled_utc="2026-10-05T10:00:00Z", selection_id=11, back1=2.5, back1_size=10.0),   # edge 0.47
        dict(book, polled_utc="2026-10-05T10:00:00Z", selection_id=12, back1=4.0, back1_size=1.0),    # edge 0.58, GBP1
        dict(book, polled_utc="2026-10-05T18:03:00Z", selection_id=11, back1=1.7, back1_size=50.0),
        dict(book, polled_utc="2026-10-05T18:03:00Z", selection_id=12, back1=2.6, back1_size=50.0),
        dict(book, source="final", status="CLOSED", polled_utc="2026-10-06T07:00:00Z", selection_id=11, back1=None,
             back1_size=None, sp_actual=2.0),
        dict(book, source="final", status="CLOSED", polled_utc="2026-10-06T07:00:00Z", selection_id=12, back1=None,
             back1_size=None, sp_actual=2.2)])
    from betfair_recorder import BOOK_FIELDS
    books = books.reindex(columns=BOOK_FIELDS)                                 # the recorder's file, all its columns
    priced = join_markets(pred, markets)
    assert len(priced) == 2 and set(priced.selection_id) == {11, 12}
    assert join_markets(pred, markets.iloc[:1]).empty                          # one runner of two: no race
    bets = paper_bets(priced, book_marks(books, markets, day))
    first = bets[(bets.mark == "first") & (bets.threshold == 0.2)].set_index("selection_id")
    assert first.loc[11].filled and not first.loc[12].filled                  # GBP1 offered: not a fill at GBP2
    assert first.loc[11].pnl == round((2.5 - 1) * 0.98 * STAKE, 10)
    assert abs(first.loc[11].pnl_bsp - (2.0 - 1) * 0.98 * STAKE) < 1e-9 and abs(first.loc[11].clv - 0.25) < 1e-9
    assert first.loc[12].pnl == -STAKE
    assert not len(bets[(bets.mark == "off_1") & (bets.selection_id == 11)])  # 1.7 at the off: no edge left


def test_parity_measures_read_a_race_as_the_horse_engine_does_and_the_bsp_joins_by_trap():
    from greyhound.parity import attach_bsp, per_run
    race = pd.DataFrame({"race_id": [1] * 4, "raceid": ["1"] * 4, "race_date": ["2026-01-01"] * 4,
                         "race_time": ["18:10"] * 4, "track": ["Hove"] * 4, "distance_m": [515.0] * 4,
                         "trap": [1, 2, 3, 4], "field": [4] * 4, "position": [1.0, 2.0, 3.0, 4.0],
                         "won": [1.0, 0, 0, 0], "sectional": [4.40, 4.30, 4.50, 4.60], "lbw": [0.0, 1.0, 2.0, 5.0],
                         "esr": [0.0] * 4, "gsr": [0.0] * 4, "gsr_ewm": [1.0, 2.0, 3.0, np.nan],
                         "sp_p_norm": [0.4, 0.3, 0.2, 0.1], "market_pos": [2, 1, 3, 4]})
    race = attach_bsp(race, pd.DataFrame({"market": ["win"] * 4, "country": ["GB"] * 4,
                                          "race_date": ["2026-01-01"] * 4, "race_time": ["18:10"] * 4,
                                          "track": ["Hove"] * 4, "trap": [1.0, 2.0, 3.0, 4.0],
                                          "bsp": [2.0, 4.0, 5.0, 20.0]}))
    m = per_run(race)
    assert m.nfp.tolist() == [1.0, 2 / 3, 1 / 3, 0.0]                     # 1 the winner .. 0 last
    z = (5 - 2 * np.arange(1, 5)) / (3 * np.sqrt(5 / 9) * 3)               # the owner's formula, N = 4
    np.testing.assert_allclose(m.nfpz.values, z) and None
    assert abs(m.nfpz.mean()) < 1e-12 and abs(m.nfpz.std(ddof=0) - 1 / 3) < 1e-12
    assert m.ep.tolist() == [1 / 3, 0.0, 2 / 3, 1.0] and m.lead.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert m.gain.tolist()[0] == (2 - 1) / 3                               # second at the bend, won: one place made
    assert abs(m.nres[0] - (1.0 - 2 / 3)) < 1e-12                          # won from second in the market's order
    assert abs(m.rs[3] - 2.0) < 1e-12 and abs(m.rs[0] - 2.5) < 1e-12       # the others' pre-race ratings
    p = 1 / np.array([2.0, 4.0, 5.0, 20.0])
    assert abs(m.bsp_ae[0] - (1 - p[0] / p.sum())) < 1e-12


def test_the_day_so_far_settles_the_paper_bets_on_the_result_and_reads_the_move_against_the_last_book(tmp_path):
    import greyhound.track as T
    import betfair_recorder
    from betfair_recorder import BOOK_FIELDS
    from sources.common import Store
    pred = pd.DataFrame({"race_id": [7, 7], "race_date": ["2026-10-06"] * 2, "race_time": ["19:04"] * 2,
                         "track": ["Romford"] * 2, "trap": [1, 2], "field": [2, 2], "won": [1.0, 0.0],
                         "p_model": [0.6, 0.4], "dog_id": [1, 2], "dog_name": ["A", "B"], "sp_decimal": [1.8, 2.5]})
    markets = pd.DataFrame({"market_id": ["1.5"] * 2, "market_type": ["WIN"] * 2, "country": ["GB"] * 2,
                            "venue": ["Romford"] * 2, "market_start_utc": ["2026-10-06T18:04:00Z"] * 2,
                            "selection_id": [11, 12], "runner_name": ["1. Alpha", "2. Beta"]})
    b = dict(source="recorder", market_id="1.5", status="OPEN", inplay=0, number_of_active_runners=2)
    books = pd.DataFrame([dict(b, polled_utc="2026-10-06T10:30:00Z", selection_id=11, back1=2.5, back1_size=10.0),
                          dict(b, polled_utc="2026-10-06T10:30:00Z", selection_id=12, back1=4.0, back1_size=5.0),
                          dict(b, polled_utc="2026-10-06T18:03:30Z", selection_id=11, back1=2.0, back1_size=50.0),
                          dict(b, polled_utc="2026-10-06T18:03:30Z", selection_id=12, back1=2.2, back1_size=50.0)]
                         ).reindex(columns=BOOK_FIELDS)
    old_read, old_price = betfair_recorder._read_day_csv, T.price_day
    betfair_recorder._read_day_csv = lambda day, name, s3=None: books if name.startswith("books") else markets
    T.price_day = lambda df, bo, f, d: pred
    try:
        s = T.intraday(Store(root=tmp_path / "sources"), pd.Timestamp("2026-10-06").date(), None, None,
                       {"features": []})
    finally:
        betfair_recorder._read_day_csv, T.price_day = old_read, old_price
    p = s["primary"]
    assert p["filled"] == 2 and p["winners"] == 1
    assert abs(p["pnl"] - ((2.5 - 1) * 0.98 * T.STAKE - T.STAKE)) < 0.01                # one won at 2.5, one lost
    assert abs(p["clv_vs_last%"] - 100 * ((2.5 / 2.0 - 1) + (4.0 / 2.2 - 1)) / 2) < 0.01
    assert list(s["primary_by_hour_uk"]) == ["19:00"]
    assert abs(p["expected_winners_at_price"] - (1 / 2.5 + 1 / 4.0)) < 0.01           # the prices' own expectation
    assert abs(p["expected_winners_at_last"] - (1 / 2.0 + 1 / 2.2)) < 0.01
    assert "Expected" in T.intraday_markdown(s)


def test_head_to_head_counts_earlier_days_meetings_with_todays_field_only():
    from greyhound.hrb import h2h, season_date
    df = pd.DataFrame({"race_id": [1, 1, 1, 2, 2, 3, 3, 3], "race_date": ["2026-01-01"] * 3 + ["2026-01-02"] * 2
                       + ["2026-01-03"] * 3, "dog_id": [10, 20, 30, 10, 20, 10, 20, 30],
                       "position": [1, 2, 3, 2, 1, np.nan, np.nan, np.nan]})          # race 3: today's card
    h = h2h(df)
    a = h.loc[5]                                                                         # dog 10 in race 3
    assert a.hb_h2h_meetings == 3 and a.hb_h2h_ahead == 2 and a.hb_h2h_opponents_met == 2
    assert h.loc[7].hb_h2h_meetings == 2 and h.loc[7].hb_h2h_ahead == 0                  # dog 30: behind both, once
    assert h.loc[3].hb_h2h_meetings == 1 and h.loc[3].hb_h2h_ahead == 1                  # race 2 reads race 1 only
    assert h.loc[0].hb_h2h_meetings == 0                                                 # nothing before day 1
    assert season_date("15.Sp.25") == pd.Timestamp("2025-09-15") and season_date("Suppressed") is None


def test_a_card_reads_grade_trip_trap_and_dog_from_the_catalogue_and_finds_the_dogs_history():
    from greyhound.card import cards, parse_market_name
    assert parse_market_name("A5 480m") == ("A5", 480.0) and parse_market_name("OR3 500m") == ("OR3", 500.0)
    assert parse_market_name("To Be Placed") == (None, None)
    assert parse_market_name("HC 500m") == ("HP", 500.0)                        # Betfair's hurdle code, GBGB's grade
    hist = pd.DataFrame({"dog_id": [7, 8], "dog_name": ["Goldcash Warrior", "Other"], "race_date": ["2026-10-01"] * 2,
                         "race_time": ["18:00:00"] * 2, "sire": ["S", "T"], "dam": ["D", "E"], "trainer": ["X", "Y"],
                         "born": ["Jan-2024"] * 2, "sex": ["d", "b"]})
    m = pd.DataFrame({"market_id": ["1.9"] * 2, "market_type": ["WIN"] * 2, "country": ["GB"] * 2,
                      "market_name": ["A5 480m"] * 2, "market_start_utc": ["2026-10-06T17:04:00Z"] * 2,
                      "venue": ["Romford"] * 2, "selection_id": [11, 12],
                      "runner_name": ["1. Goldcash Warrior", "2. New Dog"]})
    c = cards(m, hist).set_index("trap")
    assert c.loc[1, "dog_id"] == 7 and c.loc[1, "trainer"] == "X" and bool(c.loc[1, "matched_history"])
    assert c.loc[2, "dog_id"] == -12 and not bool(c.loc[2, "matched_history"])
    assert c.loc[1, "race_time"] == "18:04:00" and c.loc[1, "race_class"] == "A5" and c.loc[1, "distance_m"] == 480.0
    assert c.is_card.all() and c.position.isna().all() and c.weight_kg.isna().all()
    m2 = pd.concat([m, m.assign(market_id="1.8", market_start_utc="2026-10-06T16:47:00Z")], ignore_index=True)
    c2 = cards(m2, hist)
    assert dict(zip(c2.market_id, c2.race_number)) == {"1.8": 1.0, "1.9": 2.0}     # numbered by the off
    from greyhound.live import CARD_UNSAFE, LIVE_BLOCKS, drop_removed, removed_runners
    books = pd.DataFrame({"polled_utc": ["a", "b", "a"], "market_id": ["1.9", "1.9", "1.9"],
                          "selection_id": [12, 12, 11], "runner_status": ["ACTIVE", "REMOVED", "ACTIVE"]})
    left = drop_removed(cards(m, hist), removed_runners(books))
    assert list(left.selection_id) == [11] and left.runners.tolist() == [1]           # the non-runner is off
    assert "hrb" in LIVE_BLOCKS and {"hb_weight_vs_max", "hb_win_going", "hb_prize_1st"} <= CARD_UNSAFE


class _Book:
    def __init__(self, runners, status="OPEN"):
        self.status, self.inplay, self.runners = status, False, runners


def test_the_live_rule_takes_what_is_offered_within_the_edge_and_tops_up_to_the_stake_until_the_off(tmp_path):
    import copy
    from greyhound.live import Trader, decide, load_config, settle_rows, stake_for
    from sources.common import Store
    from trading.exchange import PaperExchange, Quote
    cfg = load_config()                                                 # the live settings: GBP5 level, held
    S = cfg["limits"]["level_stake"]
    assert S == 5.0 and stake_for(4.0, cfg) == S and stake_for(13.0, cfg) == S and not cfg["trade_out"]
    assert cfg["limits"]["max_daily_turnover"] is None and cfg["limits"]["max_bets_per_day"] is None   # no daily cap
    assert cfg["tight"] == 1.25 and cfg["decide_window_minutes"] == 1.5 and not cfg["reconsider"]   # the T-1 rule
    assert cfg["limits"]["max_race_stake"] is None                      # no race limit (the owner, 7 Oct)
    cfg.update(tight=None, stop_before_off=0, decide_window_minutes=None, reconsider=True, top_up=True)  # anytime
    cfg["limits"]["max_race_stake"] = 10.0                              # the limit itself tested at GBP10
    out = copy.deepcopy(cfg)                                            # the trading variant: staked to win GBP12
    out.update(trade_out=True)
    out["limits"].update(min_stake=2.0, max_stake=10.0)
    assert stake_for(4.0, out) == 4.0 and stake_for(2.0, out) == 10.0 and stake_for(1.8, out) is None
    book = _Book({11: Quote(11, back=[(4.0, 50.0)], lay=[(4.2, 30.0)]), 12: Quote(12, back=[(1.6, 80.0)], lay=[(1.65, 9.0)]),
                  13: Quote(13, back=[(9.0, 1.5), (8.8, 1.0), (4.0, 10.0)], lay=[(9.6, 5.0)]),
                  15: Quote(15, back=[(1.01, 5.0)], lay=[(1000.0, 2.0)]), 14: Quote(14, status="REMOVED")})
    pred = pd.DataFrame({"selection_id": [11, 12, 13, 14, 15], "p_model": [0.3, 0.4, 0.15, 0.1, 0.05],
                         "track": ["Hove"] * 5, "race_time": ["18:04:00"] * 5, "trap": [1, 2, 3, 4, 5],
                         "dog_name": list("ABCDE")})
    rows = {r["selection_id"]: r for r in decide(pred, book, cfg)}
    assert rows[11]["action"] == "back" and rows[11]["stake"] == S      # edge +0.31 at 4.0: the whole stake
    assert rows[12]["action"] == "none" and 14 not in rows              # no edge; a non-runner is left out
    assert rows[13]["action"] == "back" and rows[13]["stake"] == 2.5 and rows[13]["price"] == 8.8   # what is offered
    assert rows[15]["action"] == "none"                                 # 1.01: under the smallest price
    assert decide(pred, book, cfg, have={11: 3.0})[0]["stake"] == 2.0   # topped up to the stake
    tight = dict(cfg, tight=1.25)
    assert {r["selection_id"]: r for r in decide(pred, book, tight)}[15]["action"] == "wait"
    assert decide(pred[pred.selection_id != 13], book, cfg)[0]["action"] == "retry"  # the field not all priced: again

    class X(PaperExchange):
        def __init__(self):
            super().__init__(None)
            self.sent = []

        def books(self, mids, with_sp=False):
            self._books.update({m: book for m in mids})
            return {m: book for m in mids}

        def lay_at_bsp(self, market_id, selection_id, liability, ref):
            self.sent.append((market_id, selection_id, liability))
            return super().lay_at_bsp(market_id, selection_id, liability, ref)

    store = Store(root=tmp_path / "sources")
    day = pd.Timestamp("2026-10-06").date()
    p = pred.assign(market_id="1.9", race_date="2026-10-06")
    buf = io.StringIO()
    p.to_csv(buf, index=False)
    store.put("greyhound/live/predictions/2026-10-06.csv", buf.getvalue().encode())
    now = {"t": pd.Timestamp("2026-10-06T08:00:00Z").to_pydatetime()}
    x = X()
    t = Trader(store, x, cfg, day, "paper", tmp_path / "ledger.csv", now=lambda: now["t"])
    assert t.step() == 0 and t.bets == 2 and t.turnover == S + 2.5      # 11 whole, 13 what was there
    assert x.sent == []                                                 # held to the result: no lay at SP
    assert ("1.9", 11) in t.decided_sel and ("1.9", 13) not in t.decided_sel
    book.runners[13] = Quote(13, back=[(4.0, 10.0)], lay=[(9.6, 5.0)])     # 13's money taken: no edge at 4.0
    assert t.step() == 0 and t.bets == 2                                # nothing more
    book.runners[13] = Quote(13, back=[(9.0, 6.0)], lay=[(9.6, 5.0)])     # more comes
    assert t.step() == 0 and t.bets == 3 and t.race_stake["1.9"] == 10.0   # 13 topped up, to the race's GBP10
    assert ("1.9", 13) in t.decided_sel                                 # stake complete
    book.runners[12] = Quote(12, back=[(3.0, 80.0)], lay=[(3.1, 9.0)])    # 12 drifts into an edge
    assert t.step() == 0 and t.bets == 3                                # but the race is full
    free = dict(cfg, limits=dict(cfg["limits"], max_race_stake=None))  # no race limit: 12 is backed too
    t3 = Trader(store, X(), free, day, "paper", tmp_path / "free.csv", now=lambda: now["t"])
    t3.step()
    assert t3.race_stake["1.9"] == 15.0 and ("1.9", 12) in t3.decided_sel
    ledger = pd.read_csv(tmp_path / "ledger.csv", dtype={"market_id": str})
    assert (ledger.action == "none").sum() == 3                        # 12, 15 and 13 (no edge) noted once each
    now["t"] = pd.Timestamp("2026-10-06T17:08:00Z").to_pydatetime()      # 18:08 UK: 4 minutes late, still open
    assert t._open(pd.Timestamp(now["t"])).all()
    now["t"] = pd.Timestamp("2026-10-06T17:10:00Z").to_pydatetime()      # 6 minutes late: done with it
    assert not t._open(pd.Timestamp(now["t"])).any()
    t2 = Trader(store, X(), cfg, day, "paper", tmp_path / "ledger.csv", now=lambda: now["t"])
    assert ("1.9", 11) in t2.decided_sel and ("1.9", 13) in t2.decided_sel and t2.turnover == 10.0   # carried on
    final = pd.DataFrame({"market_id": ["1.9"], "selection_id": [11], "runner_status": ["LOSER"], "sp_actual": [3.0],
                          "polled_utc": ["2026-10-07T07:00:00Z"]})
    s = settle_rows(ledger[ledger.selection_id == 11], final).iloc[0]
    assert abs(s.pnl_back + S) < 1e-9 and s.pnl_lay == 0.0             # lost, nothing laid
    assert abs(s.clv - (4.0 / 3.0 - 1)) < 1e-9


def test_a_back_refused_for_funds_is_tried_again_and_a_restart_does_not_count_it(tmp_path):
    from greyhound.live import Trader, load_config
    from sources.common import Store
    from trading.exchange import Fill, PaperExchange, Quote
    cfg = dict(load_config(), decide_window_minutes=None, reconsider=True)
    book = _Book({11: Quote(11, back=[(4.0, 50.0)], lay=[(4.2, 30.0)]), 12: Quote(12, back=[(1.6, 80.0)], lay=[(1.65, 9.0)])})
    pred = pd.DataFrame({"selection_id": [11, 12], "p_model": [0.4, 0.6], "track": ["Hove"] * 2,
                         "race_time": ["18:04:00"] * 2, "trap": [1, 2], "dog_name": list("AB"),
                         "market_id": "1.9", "race_date": "2026-10-06"})

    class X(PaperExchange):
        def __init__(self, broke):
            super().__init__(None)
            self.broke, self.tries = broke, 0

        def books(self, mids, with_sp=False):
            self._books.update({m: book for m in mids})
            return {m: book for m in mids}

        def back(self, market_id, selection_id, price, size, ref, min_fill=2.0):
            self.tries += 1
            if self.broke:
                return Fill(market_id, int(selection_id), "BACK", "LIMIT", price, size, status="FAILURE",
                            error="INSUFFICIENT_FUNDS (ERROR_IN_ORDER)")
            return super().back(market_id, selection_id, price, size, ref, min_fill)

    store = Store(root=tmp_path / "sources")
    buf = io.StringIO()
    pred.to_csv(buf, index=False)
    store.put("greyhound/live/predictions/2026-10-06.csv", buf.getvalue().encode())
    now = pd.Timestamp("2026-10-06T08:00:00Z").to_pydatetime()
    day = pd.Timestamp("2026-10-06").date()
    x = X(broke=True)
    t = Trader(store, x, cfg, day, "live", tmp_path / "ledger.csv", now=lambda: now)
    assert t.step() == 0 and t.step() == 0 and x.tries == 2 and t.bets == 0   # refused: tried again, not done
    t2 = Trader(store, X(broke=False), cfg, day, "live", tmp_path / "ledger.csv", now=lambda: now)
    assert ("1.9", 11) not in t2.decided_sel and t2.tries[("1.9", 11)] == 2  # a restart tries it again
    assert t2.step() == 0 and t2.bets == 1 and ("1.9", 11) in t2.decided_sel  # the money is back: backed
    cap = dict(cfg, max_tries=2)
    t3 = Trader(store, X(broke=True), cap, day, "live", tmp_path / "other.csv", now=lambda: now)
    t3.step(), t3.step(), t3.step()
    assert ("1.9", 11) in t3.decided_sel                                      # refused too often: given up


def test_the_paper_rule_decides_each_dog_once_in_the_last_minutes_before_the_off(tmp_path):
    from greyhound.live import Trader, load_config
    from sources.common import Store
    from trading.exchange import PaperExchange, Quote
    cfg = load_config()
    book = _Book({11: Quote(11, back=[(4.0, 50.0)], lay=[(4.2, 30.0)]), 12: Quote(12, back=[(1.6, 80.0)], lay=[(1.65, 9.0)])})
    pred = pd.DataFrame({"selection_id": [11, 12], "p_model": [0.4, 0.6], "track": ["Hove"] * 2,
                         "race_time": ["18:04:00"] * 2, "trap": [1, 2], "dog_name": list("AB"),
                         "market_id": "1.9", "race_date": "2026-10-06"})

    class X(PaperExchange):
        def books(self, mids, with_sp=False):
            self._books.update({m: book for m in mids})
            return {m: book for m in mids}

    store = Store(root=tmp_path / "sources")
    buf = io.StringIO()
    pred.to_csv(buf, index=False)
    store.put("greyhound/live/predictions/2026-10-06.csv", buf.getvalue().encode())
    now = {"t": pd.Timestamp("2026-10-06T16:30:00Z").to_pydatetime()}             # 17:30 UK: 34 minutes out
    t = Trader(store, X(None), cfg, pd.Timestamp("2026-10-06").date(), "paper", tmp_path / "l.csv", now=lambda: now["t"])
    assert t.step() == 0 and t.bets == 0 and not t.decided_sel                      # too early: not looked at
    now["t"] = pd.Timestamp("2026-10-06T17:03:00Z").to_pydatetime()                 # a minute before the off
    assert t.step() == 1 and t.bets == 1                                            # 11 backed, 12 no edge: done
    book.runners[12] = Quote(12, back=[(3.0, 80.0)], lay=[(3.1, 9.0)])               # 12 drifts after its look
    assert t.step() == 0 and t.bets == 1                                            # decided once: not backed
    now["t"] = pd.Timestamp("2026-10-06T17:03:45Z").to_pydatetime()                 # under 30 seconds: no bet
    assert not t._open(pd.Timestamp(now["t"])).any()


def test_the_racecard_recorder_writes_a_card_only_when_it_changes_and_counts_the_weights(tmp_path):
    from greyhound import today_cards as T
    from sources.common import Store
    card = {"races": [{"prizes": "1st £150", "runners": [{"trap": 1, "name": "A", "weight": {"weight": 312}},
                                                          {"trap": 2, "name": "B", "weight": None}]}]}

    class Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class Session:
        def get(self, url, headers=None, timeout=None):
            if url.endswith("/tracks"):
                return Resp(["Romford"])
            if "/content/" in url:
                return Resp({"content": [{"filename": "rom-0710.json"}]})
            return Resp(card)

    assert T.card_files({"content": ["a.json", {"name": "b.json"}]}) == ["a.json", "b.json"]
    assert T.weight_count(card) == (2, 1)
    store, seen = Store(root=tmp_path / "sources"), {}
    now = pd.Timestamp("2026-10-07T17:00:00Z").to_pydatetime()
    s = T.poll(store, Session(), seen, pause=0, now=now)
    assert s["tracks"]["Romford"] == {"cards": 1, "written": 1, "runners": 2, "weighted": 1}
    assert T.poll(store, Session(), seen, pause=0, now=now)["tracks"]["Romford"]["written"] == 0   # unchanged
    day = tmp_path / "sources" / "greyhounds_today" / "2026-10-07"
    assert len(list(day.glob("*Z_Romford_rom-0710.json"))) == 1 and len((day / "polls.jsonl").read_text().splitlines()) == 2


def test_the_race_softmax_objective_is_a_conditional_logit_over_each_race():
    from greyhound.model import _race_softmax, _starts, race_objective
    race = np.array([0, 0, 0, 1, 1])
    won = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    second = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    s = np.array([0.2, 1.0, -0.5, 0.0, 0.3])
    p = _race_softmax(s, _starts(race))
    assert np.allclose([p[:3].sum(), p[3:].sum()], 1.0)
    fobj, feval = race_objective(race, won)
    g, h = fobj(s, None)
    assert np.allclose([g[:3].sum(), g[3:].sum()], 0.0) and (h > 0).all()        # a softmax's gradient sums to 0
    assert abs(feval(s, None)[1] + (np.log(p[1]) + np.log(p[3])) / 2) < 1e-12      # mean -log p(winner)
    g2, _ = race_objective(race, won, second)[0](s, None)
    assert not np.allclose(g2, g) and g2[1] == g[1]                                 # the winner: first stage only
