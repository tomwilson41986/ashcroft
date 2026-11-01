"""The open-to-close panel (sources/open_close.py): each reader gives the one schema from its source's own layout, the
build writes a parquet a year and can be rerun, a source not held is an empty frame, not an error. Local folder store,
a temporary SQLite database, synthetic numbers; no network, no S3."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

import betfair_prices as bp
import betfair_recorder as br
from sources import open_close as oc
from sources.common import Store


@pytest.fixture
def store(tmp_path):
    return Store(root=tmp_path / "sources")


def historic_frame(n_markets=4, market_type="WIN", country="GB", year=2026):
    """Rows as betfair_historic._MarketState.rows writes them (plus a removed runner in the first market)."""
    rows = []
    for m in range(n_markets):
        off = pd.Timestamp(f"{year}-03-0{1 + m}T12:00:00Z")
        for s in range(3):
            rows.append({"market_id": f"1.{m:03d}", "event_id": str(100 + m), "event_name": f"Race {m}",
                         "event_type_id": "4339", "market_type": market_type, "market_name": "Win",
                         "country": country, "venue": "Romford", "market_time": off.isoformat().replace("+00:00", ".000Z"),
                         "open_date": None, "turned_in_play": True, "went_in_play_utc": None, "number_of_winners": 1,
                         "settled": None, "bsp_market": True, "selection_id": 500 + s, "runner_name": f"{s + 1}. Dog",
                         "handicap": 0.0, "sort_priority": s + 1, "runner_status": "WINNER" if s == 0 else "LOSER",
                         "won": 1 if s == 0 else 0, "bsp": 2.5 + s, "ltp_first": 3.0 + s, "ltp_t_1440": None,
                         "ltp_t_60": 2.8 + s, "ltp_t_15": 2.7 + s, "ltp_t_1": 2.6 + s, "ltp_close": 2.5 + s})
    rows.append({**rows[0], "selection_id": 599, "runner_name": "9. Gone", "runner_status": "REMOVED", "won": None,
                 "bsp": None, "ltp_first": None, "ltp_close": None, "ltp_t_60": None, "ltp_t_15": None, "ltp_t_1": None})
    return pd.DataFrame(rows)


def test_the_historic_reader_maps_first_marks_close_bsp_and_result_and_counts_runners_without_the_removed(store):
    store.put_parquet("betfair_historic/markets_greyhound_racing_2026.parquet", historic_frame())
    store.put_parquet("betfair_historic/markets_tennis_2025.parquet", historic_frame(2, "MATCH_ODDS", "FR", 2025))
    p = oc.read_betfair_historic(store, "greyhound_racing")
    assert list(p.columns) == oc.SCHEMA
    assert len(p) == 13 and set(p["sport"]) == {"greyhound_racing"} and set(p["source"]) == {"betfair_historic"}
    first = p[(p["market_id"] == "1.000") & (p["selection_id"] == "500")].iloc[0]
    assert first["open_price"] == 3.0 and first["p_t60"] == 2.8 and first["p_t15"] == 2.7 and first["p_t1"] == 2.6
    assert first["close_price"] == 2.5 and first["bsp"] == 2.5 and first["result"] == 1
    assert pd.isna(first["p_t24h"]) and pd.isna(first["open_matched"]) and pd.isna(first["open_time_utc"])
    assert first["n_runners"] == 3                       # the removed runner is not counted
    assert str(first["off_time_utc"].tz) == "UTC" and first["off_time_utc"].hour == 12
    assert pd.isna(p[p["selection_id"] == "599"]["result"].iloc[0])
    t = oc.read_betfair_historic(store, "tennis")
    assert len(t) == 7 and set(t["country"]) == {"FR"} and set(t["market_type"]) == {"MATCH_ODDS"}
    assert oc.read_betfair_historic(store, "darts").empty


def _horse_db(path):
    conn = sqlite3.connect(path)
    bp.ensure_table(conn)
    rows = []
    for i in range(6):
        rows.append(("uk" if i < 4 else "ire", "win", "2026-07-04", "14:30", 900 if i < 4 else 901, "Ascot 4th Jul", "2m Hcap",
                     "04-07-2026 14:30", "Ascot", 10 + i, f"{i + 1}. Horse {i}", f"horse{i}", 1.0 if i == 0 else 0.0,
                     4.0 + i, 4.1 + i, 5.0 + i if i != 2 else 0.0, 5.0, 3.5, 6.0, 1.5, 120.0, 3000.0, 800.0, None, "f"))
    cols = ("country, market_type, race_date, race_time, event_id, menu_hint, event_name, event_dt, track_bf, selection_id, "
            "selection_name, horse_norm, win_lose, bsp, ppwap, morningwap, ppmax, ppmin, ipmax, ipmin, morning_vol, pp_vol, "
            "ip_vol, race_results_id, source_file")
    conn.executemany(f"INSERT INTO betfair_prices ({cols}) VALUES ({','.join('?' * 25)})", rows)
    br.ensure_tables(conn)
    cat = []
    for s in range(3):
        r = {f: None for f in br.LIVE_MARKET_FIELDS}
        r.update(race_date="2026-10-20", market_id="1.5000", market_type="WIN", market_start_utc="2026-10-20T13:30:00.000Z",
                 venue="Kempton", country="GB", event_id="33", event_name="Kempton 20th Oct", selection_id=700 + s,
                 runner_name=f"{s + 1}. Runner {s}", horse_norm=f"runner{s}")
        cat.append(tuple(r[f] for f in br.LIVE_MARKET_FIELDS))
    marks = []
    for s in range(3):
        for mark, mins, back, lay, matched in (("clock_08:00", 330, 5.0 + s, 5.4 + s, 2000.0), ("off_60", 60, 4.6 + s, 4.8 + s, 9000.0),
                                              ("off_15", 15, 4.4 + s, 4.6 + s, 20000.0), ("off_1", 1, 4.1 + s, 4.3 + s, 60000.0),
                                              ("last", 0.4, 4.0 + s, 4.2 + s, 70000.0), ("final", -6, None, None, 80000.0)):
            if s == 2 and mark == "clock_08:00":
                lay = None                               # one side only at 08:00: no open for this runner
            r = {f: None for f in br.MARK_FIELDS}
            r.update(race_date="2026-10-20", market_id="1.5000", selection_id=700 + s, mark=mark,
                     polled_utc="2026-10-20T07:01:00Z" if mark == "clock_08:00" else "2026-10-20T13:00:00Z",
                     minutes_to_off=mins, market_status="CLOSED" if mark == "final" else "OPEN",
                     runner_status=("WINNER" if s == 1 else "LOSER") if mark == "final" else "ACTIVE", back1=back,
                     lay1=lay, market_total_matched=matched, active_runners=3, bsp=4.05 + s if mark == "final" else None,
                     source="recorder")
            marks.append(tuple(r[f] for f in br.MARK_FIELDS))
    conn.executemany(f"INSERT INTO betfair_live_markets ({', '.join(br.LIVE_MARKET_FIELDS)}) "
                     f"VALUES ({', '.join('?' * len(br.LIVE_MARKET_FIELDS))})", cat)
    conn.executemany(f"INSERT INTO betfair_live_marks ({', '.join(br.MARK_FIELDS)}) "
                     f"VALUES ({', '.join('?' * len(br.MARK_FIELDS))})", marks)
    conn.commit()
    conn.close()


def test_the_price_file_reader_takes_the_morning_wap_as_the_open_and_the_bsp_as_the_close_in_utc(tmp_path):
    db = str(tmp_path / "h.db")
    _horse_db(db)
    p = oc.read_horse_prices(db)
    assert list(p.columns) == oc.SCHEMA and len(p) == 6
    uk = p[p["country"] == "GB"]
    assert len(uk) == 4 and set(p["country"]) == {"GB", "IE"} and set(p["market_type"]) == {"WIN"}
    r0 = p[p["selection_id"] == "10"].iloc[0]
    assert r0["open_price"] == 5.0 and r0["open_matched"] == 120.0 and r0["close_price"] == 4.0 and r0["bsp"] == 4.0
    assert r0["pre_play_matched"] == 3000.0 and r0["result"] == 1 and r0["n_runners"] == 4
    assert r0["market_id"] == "900:WIN" and r0["event_id"] == "900"
    # 14:30 UK in July is 13:30 UTC
    assert r0["off_time_utc"] == pd.Timestamp("2026-07-04T13:30:00Z")
    assert pd.isna(p[p["selection_id"] == "12"]["open_price"].iloc[0])      # a morning WAP of 0: no open
    assert oc.read_horse_prices(db, date_from="2026-08-01").empty


def test_the_recorder_reader_takes_the_0800_mid_as_the_open_with_its_time_and_the_books_at_the_marks(tmp_path):
    db = str(tmp_path / "h.db")
    _horse_db(db)
    p = oc.read_horse_marks(db)
    assert list(p.columns) == oc.SCHEMA and len(p) == 3 and set(p["source"]) == {"betfair_live_marks"}
    r = p[p["selection_id"] == "700"].iloc[0]
    assert r["open_price"] == pytest.approx(5.2) and r["open_matched"] == 2000.0
    assert r["open_time_utc"] == pd.Timestamp("2026-10-20T07:01:00Z")
    assert r["p_t60"] == pytest.approx(4.7) and r["p_t15"] == pytest.approx(4.5) and r["p_t1"] == pytest.approx(4.2)
    assert r["close_price"] == pytest.approx(4.1) and r["pre_play_matched"] == 70000.0
    assert r["bsp"] == pytest.approx(4.05) and r["result"] == 0 and r["n_runners"] == 3
    assert r["off_time_utc"] == pd.Timestamp("2026-10-20T13:30:00Z") and r["country"] == "GB"
    assert p[p["selection_id"] == "701"]["result"].iloc[0] == 1
    one_sided = p[p["selection_id"] == "702"].iloc[0]
    assert pd.isna(one_sided["open_price"]) and one_sided["close_price"] == pytest.approx(6.1)
    assert pd.isna(p["p_t24h"]).all()


def gold_frame():
    return pd.DataFrame({
        "match_id": ["a1", "a2", "a3"], "competition_id": "ENG1", "country": ["ENG", "ENG", "ESP"], "season": [2024, 2024, 2018],
        "match_date": ["2024-08-17", "2024-08-17", "2018-09-01"], "status": ["FINISHED", "FINISHED", "FINISHED"],
        "kickoff_utc": pd.to_datetime(["2024-08-17T14:00:00Z", "2024-08-17T16:30:00Z", None], utc=True),
        "home_team": ["Arsenal", "Leeds", "Betis"], "away_team": ["Wolves", "Hull", "Sevilla"],
        "ft_home": [2, 1, 0], "ft_away": [0, 1, 1],
        "open_h": [1.3, 2.2, np.nan], "open_d": [5.5, 3.3, np.nan], "open_a": [9.0, 3.4, np.nan], "open_1x2_src": ["PS", "BFE", None],
        "close_h": [1.25, 2.4, 2.0], "close_d": [6.0, 3.2, 3.3], "close_a": [10.0, 3.1, 4.0], "close_1x2_src": ["PSC", "BFEC", "PSC"],
        "open_o25": [1.6, 2.0, np.nan], "open_u25": [2.3, 1.85, np.nan], "open_ou_src": ["P", "P", None],
        "close_o25": [1.55, 2.1, np.nan], "close_u25": [2.4, 1.75, np.nan], "close_ou_src": ["PC", "PC", None],
        "open_ah_line": [-1.5, 0.0, np.nan], "open_ah_h": [1.9, 1.95, np.nan], "open_ah_a": [1.95, 1.9, np.nan], "open_ah_src": ["P", "P", None],
        "close_ah_line": [-1.5, 0.0, np.nan], "close_ah_h": [1.85, 2.0, np.nan], "close_ah_a": [2.0, 1.85, np.nan],
        "close_ah_src": ["PCAH", "PCAH", None],
    })


def test_the_football_reader_gives_one_row_an_outcome_with_the_books_named_and_the_result_settled(store):
    store.put_parquet("football/gold/matches.parquet", gold_frame())
    p = oc.read_football_gold(store)
    assert list(p.columns) == oc.SCHEMA
    assert p["market_type"].value_counts().to_dict() == {"MATCH_ODDS": 9, "OVER_UNDER_25": 4, "ASIAN_HANDICAP": 4}
    a1 = p[(p["event_id"] == "a1") & (p["market_type"] == "MATCH_ODDS")].set_index("selection_name")
    assert a1.loc["H", "open_price"] == 1.3 and a1.loc["H", "close_price"] == 1.25 and a1.loc["H", "result"] == 1
    assert a1.loc["D", "result"] == 0 and a1.loc["A", "result"] == 0 and a1.loc["H", "n_runners"] == 3
    assert a1.loc["H", "source"] == "football_gold:PS>PSC" and a1.loc["H", "event_name"] == "Arsenal v Wolves"
    assert a1.loc["H", "off_time_utc"] == pd.Timestamp("2024-08-17T14:00:00Z")
    ou = p[(p["event_id"] == "a2") & (p["market_type"] == "OVER_UNDER_25")].set_index("selection_name")
    assert ou.loc["Over 2.5", "result"] == 0 and ou.loc["Under 2.5", "result"] == 1
    ah = p[(p["event_id"] == "a1") & (p["market_type"] == "ASIAN_HANDICAP")]
    assert set(ah["selection_name"]) == {"AH home -1.5", "AH away -1.5"} and ah["result"].isna().all()
    # a close without an open stays (the move is simply not measurable there); no kick-off time -> midday UTC
    a3 = p[(p["event_id"] == "a3") & (p["market_type"] == "MATCH_ODDS")]
    assert len(a3) == 3 and a3["open_price"].isna().all() and a3["source"].iloc[0] == "football_gold:->PSC"
    assert a3["off_time_utc"].iloc[0] == pd.Timestamp("2018-09-01T12:00:00Z")
    assert pd.isna(p["p_t60"]).all() and pd.isna(p["bsp"]).all()
    assert oc.read_football_gold(Store(root=store.root / "nothing")).empty


def greyhound_prices_frame():
    return pd.DataFrame({
        "market": ["win"] * 3 + ["place"] * 3, "event_id": [7001] * 3 + [7002] * 3, "menu_hint": "Romford 2nd Jan",
        "event_name": "A1 400m", "event_dt": "02-01-2026 11:04", "selection_id": [1, 2, 3, 1, 2, 3],
        "selection_name": ["1. Fast", "2. Slow", "3. Mid"] * 2, "bsp": [2.0, 8.0, 4.0, 1.4, 3.0, 1.8],
        "ppwap": [2.1, 7.8, 4.1, 1.4, 3.1, 1.8], "morningwap": [2.6, 7.0, 0.0, 1.5, 3.4, 1.9], "ppmax": 3.0, "ppmin": 1.5,
        "ipmax": 5.0, "ipmin": 1.1, "morning_vol": [12.0, 3.0, 0.0, 2.0, 1.0, 1.0], "pp_vol": [400.0, 50.0, 120.0, 90.0, 20.0, 60.0],
        "ip_vol": 10.0, "win_lose": [1.0, 0.0, 0.0, 1.0, 0.0, 1.0], "race_date": "2026-01-02", "race_time": "11:04",
        "trap": [1, 2, 3] * 2, "dog": ["Fast", "Slow", "Mid"] * 2, "track": "Romford", "country": ["GB"] * 3 + ["AUS"] * 3,
        "source_file": "dwbfgreyhoundwin02012026.csv"})


def test_the_greyhound_price_reader_and_the_build_write_a_panel_a_year_that_a_rerun_overwrites(store):
    store.put_parquet("greyhound_prices/prices_2026.parquet", greyhound_prices_frame())
    store.put_parquet("betfair_historic/markets_greyhound_racing_2026.parquet", historic_frame(2))
    g = oc.read_greyhound_prices(store)
    assert len(g) == 6 and set(g["country"]) == {"GB", "AU"} and set(g["market_type"]) == {"WIN", "PLACE"}
    fast = g[(g["market_id"] == "7001:WIN") & (g["selection_id"] == "1")].iloc[0]
    assert fast["open_price"] == 2.6 and fast["open_matched"] == 12.0 and fast["close_price"] == 2.0 and fast["result"] == 1
    assert fast["off_time_utc"] == pd.Timestamp("2026-01-02T11:04:00Z") and fast["n_runners"] == 3
    assert pd.isna(g[g["selection_id"] == "3"]["open_price"].iloc[0])
    out = oc.build(store, "greyhound_racing")
    assert out["readers"] == {"betfair_historic": 7, "greyhound_prices": 6, "betfair_live_books": 0}
    assert list(out["years"]) == ["2026"] and out["years"]["2026"]["rows"] == 13
    key = "open_close/panel_greyhound_racing_2026.parquet"
    assert store.get_parquet(key).shape == (13, len(oc.SCHEMA))
    again = oc.build(store, "greyhound_racing")                    # the same rows, the same one file
    assert again["years"]["2026"]["rows"] == 13 and store.get_parquet(key).shape == (13, len(oc.SCHEMA))
    size_mb = store.listing("open_close/")[key] / 1e6
    assert oc.report(store) == {"greyhound_racing": {"2026": pytest.approx(size_mb, abs=0.01)}}
    assert set(store.get_parquet(key)["source"]) == {"betfair_historic", "greyhound_prices"}


def _greyhound_record(tmp_path, day="2026-10-06"):
    """One market's day in the recorder's own fields (betfair_recorder.BOOK_FIELDS / MARKET_FIELDS), as
    betfair_live/<day>/books_greyhound.csv.gz and markets_greyhound.csv.gz."""
    import gzip
    d = tmp_path / "betfair_live" / day
    d.mkdir(parents=True)
    cat = []
    for s in range(3):
        r = {f: None for f in br.MARKET_FIELDS}
        r.update(market_id="1.9000", market_type="WIN", market_name="A3 480m", market_start_utc=f"{day}T11:04:00.000Z",
                 venue="Romford", country="GB", event_id="55", event_name="Romford 6th Oct", selection_id=800 + s,
                 runner_name=f"{s + 1}. Dog {s}", sort_priority=s + 1, recorded_utc=f"{day}T06:55:00Z")
        cat.append(r)
    polls = [  # polled, source, status, inplay, matched, (back1, lay1) a runner, sp_actual a runner, runner statuses
        (f"{day}T08:30:00Z", "recorder", "OPEN", 0, 50.0, [(3.0, 3.4), (None, 5.0), (6.0, 7.0)], None, "ACTIVE"),
        (f"{day}T09:00:00Z", "recorder", "OPEN", 0, 120.0, [(2.8, 3.0), (4.5, 4.9), (6.0, 6.6)], None, "ACTIVE"),
        (f"{day}T10:04:00Z", "recorder", "OPEN", 0, 900.0, [(2.6, 2.7), (4.8, 5.0), (6.4, 6.8)], None, "ACTIVE"),
        (f"{day}T10:49:00Z", "recorder", "OPEN", 0, 1800.0, [(2.5, 2.6), (4.9, 5.1), (6.6, 7.0)], None, "ACTIVE"),
        (f"{day}T11:01:00Z", "recorder", "OPEN", 0, 2500.0, [(2.45, 2.55), (5.0, 5.1), (6.7, 7.1)], None, "ACTIVE"),
        (f"{day}T11:03:10Z", "recorder", "OPEN", 0, 3000.0, [(2.4, 2.5), (5.0, 5.2), (6.8, 7.2)], None, "ACTIVE"),
        (f"{day}T11:05:00Z", "bsp", "SUSPENDED", 1, 3400.0, [(None, None)] * 3, [2.42, 5.1, 7.0], "ACTIVE"),
        (f"{day}T11:10:00Z", "final", "CLOSED", 0, 3400.0, [(None, None)] * 3, None, ["WINNER", "LOSER", "LOSER"]),
    ]
    books = []
    for polled, source, status, inplay, matched, prices, sps, statuses in polls:
        for s in range(3):
            r = {f: None for f in br.BOOK_FIELDS}
            r.update(polled_utc=polled, source=source, market_id="1.9000", status=status, inplay=inplay, complete=0,
                     bsp_reconciled=int(sps is not None), number_of_runners=3, number_of_active_runners=3,
                     market_total_matched=matched, selection_id=800 + s,
                     runner_status=statuses if isinstance(statuses, str) else statuses[s],
                     back1=prices[s][0], back1_size=10.0 if prices[s][0] else None, lay1=prices[s][1],
                     lay1_size=12.0 if prices[s][1] else None, sp_actual=sps[s] if sps else None)
            books.append(r)
    (d / "books_greyhound.csv.gz").write_bytes(gzip.compress(pd.DataFrame(books)[br.BOOK_FIELDS].to_csv(index=False).encode()))
    (d / "markets_greyhound.csv.gz").write_bytes(gzip.compress(pd.DataFrame(cat)[br.MARKET_FIELDS].to_csv(index=False).encode()))


def test_the_recorded_books_reader_takes_the_first_two_sided_book_as_the_open_with_its_time_and_money(tmp_path):
    _greyhound_record(tmp_path)
    store = Store(root=tmp_path / "sources")
    p = oc.read_recorded_books(store, "greyhound_racing", "greyhound")
    assert list(p.columns) == oc.SCHEMA and len(p) == 3 and set(p["source"]) == {"betfair_live_books:greyhound"}
    r = p[p["selection_id"] == "800"].iloc[0]
    assert r["open_price"] == pytest.approx(3.2) and r["open_time_utc"] == pd.Timestamp("2026-10-06T08:30:00Z")
    assert r["open_matched"] == 50.0
    assert r["p_t60"] == pytest.approx(2.65) and r["p_t15"] == pytest.approx(2.55) and r["p_t1"] == pytest.approx(2.45)
    assert r["p_t3"] == pytest.approx(2.5)
    assert r["close_price"] == pytest.approx(2.45) and r["pre_play_matched"] == 3000.0
    assert r["bsp"] == pytest.approx(2.42) and r["result"] == 1 and r["n_runners"] == 3
    assert r["off_time_utc"] == pd.Timestamp("2026-10-06T11:04:00Z") and r["country"] == "GB"
    assert r["market_type"] == "WIN" and r["event_name"] == "Romford 6th Oct" and r["sport"] == "greyhound_racing"
    # one side only at the first poll: the open is the first book with both sides
    r1 = p[p["selection_id"] == "801"].iloc[0]
    assert r1["open_price"] == pytest.approx(4.7) and r1["open_time_utc"] == pd.Timestamp("2026-10-06T09:00:00Z")
    assert r1["open_matched"] == 120.0 and r1["result"] == 0 and r1["bsp"] == pytest.approx(5.1)
    assert oc.read_recorded_books(store, date_from="2026-10-07").empty
    assert oc.read_recorded_books(Store(root=tmp_path / "elsewhere" / "sources")).empty
    # the greyhound open the build uses: the T-3 book (the money arrives in the last three minutes), its time and money
    t3 = oc.read_recorded_books(store, open_mark=oc.GREYHOUND_OPEN_MARK)
    g0 = t3[t3["selection_id"] == "800"].iloc[0]
    assert g0["open_price"] == pytest.approx(2.5) and g0["open_time_utc"] == pd.Timestamp("2026-10-06T11:01:00Z")
    assert g0["open_matched"] == 2500.0 and g0["p_t3"] == pytest.approx(2.5) and g0["close_price"] == pytest.approx(2.45)
    out = oc.build(store, "greyhound_racing")
    assert out["readers"] == {"betfair_historic": 0, "greyhound_prices": 0, "betfair_live_books": 3}
    assert out["years"]["2026"]["rows"] == 3
    built = store.get_parquet("open_close/panel_greyhound_racing_2026.parquet")
    assert built[built["selection_id"] == "800"]["open_matched"].iloc[0] == 2500.0


def test_a_sport_with_nothing_held_builds_nothing_and_the_cli_runs_from_a_local_root(tmp_path, capsys):
    store = Store(root=tmp_path / "sources")
    assert oc.build(store, "tennis") == {"sport": "tennis", "readers": {"betfair_historic": 0}, "years": {}}
    assert store.listing("open_close/") == {}
    with pytest.raises(ValueError):
        oc.build(store, "curling")
    store.put_parquet("betfair_historic/markets_tennis_2025.parquet", historic_frame(2, "MATCH_ODDS", "FR", 2025))
    assert oc.main(["--build", "--sport", "tennis", "--root", str(tmp_path / "sources"), "--report"]) == 0
    text = capsys.readouterr().out
    assert "panel_tennis_2025.parquet" in text and '"tennis"' in text
    assert "open_close/panel_tennis_2025.parquet" in store.listing("open_close/")


def test_horse_racing_through_build_needs_the_database_and_writes_by_the_year_of_the_off(tmp_path):
    store = Store(root=tmp_path / "sources")
    db = str(tmp_path / "h.db")
    _horse_db(db)
    assert oc.build(store, "horse_racing") == {"sport": "horse_racing", "readers": {}, "years": {}}
    out = oc.build(store, "horse_racing", db_path=db)
    assert out["readers"] == {"betfair_prices": 6, "betfair_live_marks": 3} and out["years"]["2026"]["rows"] == 9
    p = store.get_parquet("open_close/panel_horse_racing_2026.parquet")
    assert p["source"].value_counts().to_dict() == {"betfair_prices": 6, "betfair_live_marks": 3}
    assert str(p["off_time_utc"].dt.tz) == "UTC" and p["result"].dtype == "Int64"
