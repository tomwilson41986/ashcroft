"""The greyhound model's engine: comments read as GBGB writes them, and every metric lag-safe (a race's own result,
and its day's other results, never reach its features)."""

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
    eng = GreyhoundMetricsEngine()
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
    other = GreyhoundMetricsEngine().calculate_all(alt)
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
    assert m.ep.tolist() == [1 / 3, 0.0, 2 / 3, 1.0] and m.lead.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert m.gain.tolist()[0] == (2 - 1) / 3                               # second at the bend, won: one place made
    assert abs(m.nres[0] - (1.0 - 2 / 3)) < 1e-12                          # won from second in the market's order
    assert abs(m.rs[3] - 2.0) < 1e-12 and abs(m.rs[0] - 2.5) < 1e-12       # the others' pre-race ratings
    p = 1 / np.array([2.0, 4.0, 5.0, 20.0])
    assert abs(m.bsp_ae[0] - (1 - p[0] / p.sum())) < 1e-12
