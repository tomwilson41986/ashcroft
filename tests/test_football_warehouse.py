"""The football warehouse's foundation: paths, the stream-file parser, bronze ingest and the silver build
(hand-written stream files: a few messages each, shaped like Betfair's)."""

from __future__ import annotations

import bz2
import io
import json
import zipfile

import pandas as pd
import pyarrow as pa
import pytest

from football.warehouse import bronze, silver
from football.warehouse.parse import Batch, iso_ms, runner_roles, split_event_name
from football.warehouse.schema import SILVER_MARKET, SILVER_TICK, conform
from football.warehouse.store_io import read_manifest, read_part
from sources.common import Store

KO = "2026-01-29T20:00:00.000Z"
KO_MS = iso_ms(KO)
MO_RUNNERS = [{"status": "ACTIVE", "sortPriority": 1, "id": 71085, "name": "Sturm Graz"},
              {"status": "ACTIVE", "sortPriority": 2, "id": 31317, "name": "Brann"},
              {"status": "ACTIVE", "sortPriority": 3, "id": 58805, "name": "The Draw"}]


def _defn(status="OPEN", in_play=False, runners=None, market_type="MATCH_ODDS", market_time=KO, **extra):
    d = {"eventId": "35155405", "eventTypeId": "1", "marketType": market_type, "marketTime": market_time,
         "openDate": market_time, "suspendTime": market_time, "status": status, "inPlay": in_play, "betDelay": 0,
         "numberOfWinners": 1, "numberOfActiveRunners": 3, "bettingType": "ODDS", "name": "Match Odds",
         "eventName": "Sturm Graz v Brann", "timezone": "GMT", "regulators": ["MR_INT"], "version": 1,
         "runners": runners or MO_RUNNERS, "turnInPlayEnabled": True, "marketBaseRate": 5.0}
    d.update(extra)
    return d


def _msg(pt, market_id="1.252787713", defn=None, rc=None):
    mc = {"id": market_id}
    if defn is not None:
        mc["marketDefinition"] = defn
    if rc is not None:
        mc["rc"] = rc
    return {"op": "mcm", "clk": "1", "pt": pt, "mc": [mc]}


def _file(msgs) -> bytes:
    return bz2.compress("\n".join(json.dumps(m) for m in msgs).encode())


def match_odds_file() -> bytes:
    settled = [dict(r, status=s) for r, s in zip(MO_RUNNERS, ("WINNER", "LOSER", "LOSER"))]
    return _file([
        _msg(KO_MS - 86_400_000, defn=_defn()),                                              # 0: a day before
        _msg(KO_MS - 3_600_000, rc=[{"id": 71085, "ltp": 2.5}, {"id": 31317, "ltp": 3.0}]),   # 1: T-60
        _msg(KO_MS - 60_000, rc=[{"id": 71085, "ltp": 2.4}, {"id": 58805, "ltp": 3.4}]),      # 2: T-1
        _msg(KO_MS + 90_000, defn=_defn(in_play=True, betDelay=5)),                           # 3: kick-off, 90 s late
        _msg(KO_MS + 600_000, rc=[{"id": 71085, "ltp": 2.3}]),                                # 4: in play
        _msg(KO_MS + 1_200_000, defn=_defn("SUSPENDED", True), rc=[{"id": 71085, "ltp": 1.5}]),  # 5: a goal
        _msg(KO_MS + 1_260_000, defn=_defn("OPEN", True)),                                    # 6: back open
        _msg(KO_MS + 6_600_000, defn=_defn("CLOSED", True, runners=settled, numberOfActiveRunners=0,
                                           settledTime="2026-01-29T21:55:00.000Z")),          # 7: settled
    ])


def test_paths_are_read_with_or_without_the_plan_folder_and_inside_a_nested_zip():
    full = bronze.plan_path("xds/BASIC/2026/Jan/29/35155405/1.252787713.bz2")
    bare = bronze.plan_path("2026/Jan/29/35155405/1.252787713.bz2")
    nested = bronze.plan_path("Oct/1/36000000/1.262000001.bz2", container="2026/Oct.zip")
    assert full == bare and full["plan_path"] == "BASIC/2026/Jan/29/35155405/1.252787713.bz2"
    assert (full["year"], full["month"], full["day"], full["event_id"], full["market_id"]) == \
        (2026, 1, 29, 35155405, "1.252787713")
    assert nested["plan_path"] == "BASIC/2026/Oct/1/36000000/1.262000001.bz2" and nested["month"] == 10
    assert bronze.plan_path("2026/Oct.zip") is None and bronze.plan_path("Oct/1/36000000/1.2.bz2") is None


def test_event_names_and_roles():
    assert split_event_name("Sturm Graz v Brann") == ("Sturm Graz", "Brann")
    assert split_event_name("Some Outright") == (None, None)
    roles = runner_roles("MATCH_ODDS", MO_RUNNERS, "Sturm Graz", "Brann")
    assert roles == {(71085, 0.0): "home", (31317, 0.0): "away", (58805, 0.0): "draw"}
    # names that do not match the event's: Betfair's order decides (home first)
    assert runner_roles("MATCH_ODDS", MO_RUNNERS, None, None)[(71085, 0.0)] == "home"
    ou = [{"id": 47972, "name": "Under 2.5 Goals", "sortPriority": 1}, {"id": 47973, "name": "Over 2.5 Goals",
                                                                       "sortPriority": 2}]
    assert runner_roles("OVER_UNDER_25", ou, "A", "B") == {(47972, 0.0): "under", (47973, 0.0): "over"}


def test_the_parser_reads_a_market_and_stamps_each_price_with_the_state_it_was_published_in():
    b = Batch()
    assert b.add_file(match_odds_file(), "BASIC/2026/Jan/29/35155405/1.252787713.bz2", "abc") == 1
    t = {k: v.to_pandas() for k, v in b.tables().items()}
    m = t["silver/market"].iloc[0]
    assert (m["n_messages"], m["n_definitions"], m["n_ticks"], m["n_ticks_pre"], m["n_ticks_inplay"]) == (8, 5, 6, 4, 2)
    assert m["kickoff_delay_s"] == 90 and m["n_susp_inplay"] == 1 and m["n_winners"] == 1
    assert m["in_play_pt"].value // 10**6 == KO_MS + 90_000 and m["closed_pt"].value // 10**6 == KO_MS + 6_600_000
    assert (m["home_name"], m["away_name"], m["status_final"]) == ("Sturm Graz", "Brann", "CLOSED")
    r = t["silver/runner"].set_index("role")
    assert r.loc["home", "won"] == 1 and r.loc["away", "won"] == 0 and r.loc["draw", "won"] == 0
    # the close is the last price before the market went in play, never an in-play price
    assert (r.loc["home", "ltp_first"], r.loc["home", "ltp_last_pre"], r.loc["home", "ltp_last"]) == (2.5, 2.4, 1.5)
    assert r.loc["draw", "ltp_last_pre"] == 3.4 and r.loc["away", "n_ticks_pre"] == 1
    k = t["silver/tick"].sort_values(["pt", "selection_id"]).reset_index(drop=True)
    goal = k[k["ltp"] == 1.5].iloc[0]
    # the price published with the suspension carries the suspension
    assert bool(goal["in_play"]) and goal["status"] == "SUSPENDED" and goal["t_inplay_s"] == 1110
    assert k["t_off_s"].iloc[0] == 3600 and (k[~k["in_play"]]["t_inplay_s"].isna()).all()
    s = t["silver/spell"]
    assert s["status"].tolist() == ["OPEN", "OPEN", "SUSPENDED", "OPEN", "CLOSED"]
    assert s["in_play"].tolist() == [False, True, True, True, True]
    assert s["duration_s"].iloc[2] == 60.0 and s["duration_s"].iloc[4] == 0.0


def test_a_handicap_runner_is_a_selection_and_a_line():
    runners = [{"status": "ACTIVE", "sortPriority": i + 1, "id": sid, "hc": hc, "name": name}
               for i, (sid, hc, name) in enumerate([(71085, -0.5, "Sturm Graz"), (31317, 0.5, "Brann"),
                                                    (71085, -1.0, "Sturm Graz"), (31317, 1.0, "Brann")])]
    raw = _file([
        _msg(KO_MS - 7_200_000, "1.252787720", _defn(runners=runners, market_type="ASIAN_HANDICAP")),
        _msg(KO_MS - 3_600_000, "1.252787720", rc=[{"id": 71085, "hc": -0.5, "ltp": 1.9},
                                                   {"id": 71085, "hc": -1.0, "ltp": 2.6}]),
    ])
    b = Batch()
    b.add_file(raw, "BASIC/2026/Jan/29/35155405/1.252787720.bz2", "abc")
    r = b.tables()["silver/runner"].to_pandas()
    assert len(r) == 4 and set(zip(r["selection_id"], r["handicap"], r["role"])) == {
        (71085, -0.5, "home"), (71085, -1.0, "home"), (31317, 0.5, "away"), (31317, 1.0, "away")}
    assert r.set_index(["selection_id", "handicap"]).loc[(71085, -1.0), "ltp_last_pre"] == 2.6


def test_a_file_that_cannot_be_read_adds_nothing():
    b = Batch()
    assert b.add_file(b"BZh not a bz2 file", "BASIC/2026/Jan/1/1/1.1.bz2", "x") == 0
    assert b.add_file(_file([_msg(1, rc=[{"id": 1, "ltp": 2.0}])]), "BASIC/2026/Jan/1/1/1.2.bz2", "x") == 0
    assert [p for p, _ in b.failed] == ["BASIC/2026/Jan/1/1/1.1.bz2", "BASIC/2026/Jan/1/1/1.2.bz2"]
    assert all(v.num_rows == 0 for v in b.tables().values())


def test_a_price_outside_betfairs_range_is_counted_and_left_out():
    raw = _file([_msg(KO_MS - 3_600_000, defn=_defn(), rc=[{"id": 71085, "ltp": 2.5}]),
                 _msg(KO_MS - 60_000, defn=_defn("SUSPENDED"), rc=[{"id": 71085, "ltp": 1.0}, {"id": 31317, "ltp": 1.0},
                                                                   {"id": 58805, "ltp": 1.0}])])   # a void close
    b = Batch()
    b.add_file(raw, "BASIC/2026/Jan/29/35155405/1.252787713.bz2", "abc")
    t = {k: v.to_pandas() for k, v in b.tables().items()}
    assert t["silver/tick"]["ltp"].tolist() == [2.5]
    m = t["silver/market"].iloc[0]
    assert (m["n_ticks"], m["n_ltp_invalid"]) == (1, 3)
    assert t["silver/runner"].set_index("role").loc["home", "ltp_last_pre"] == 2.5


def test_a_required_column_may_not_hold_a_null():
    bad = pa.table({c.name: pa.array([None], type=c.type) for c in SILVER_TICK.columns})
    with pytest.raises(ValueError):
        conform(SILVER_TICK, bad)


def test_a_value_its_column_cannot_hold_is_an_error_not_a_quiet_change():
    import numpy as np
    from football.warehouse.schema import Column, TableSpec, from_frame
    spec = TableSpec("x/y", "a test table", (Column("k", pa.int64(), "the key", False),
                                             Column("n", pa.int16(), "a small whole number")), key=("k",), sort=("k",))
    assert conform(spec, pa.table({"k": [1], "n": pa.array([7.0])})).column("n").to_pylist() == [7]
    for bad in (pa.array([70000]), pa.array([1.9]), pa.array([5e9])):       # would wrap, truncate, overflow
        with pytest.raises(ValueError, match="does not fit"):
            conform(spec, pa.table({"k": [1], "n": bad}))
    ok = from_frame(spec, pd.DataFrame({"k": [1, 2], "n": [3.0, np.nan]}))
    assert ok.column("n").to_pylist() == [3, None]
    with pytest.raises(ValueError, match="fraction"):
        from_frame(spec, pd.DataFrame({"k": [1], "n": [1.9]}))
    with pytest.raises(ValueError, match="fraction"):
        from_frame(spec, pd.DataFrame({"k": [1], "n": [np.inf]}))


def _bundle(tmp_path, files: dict, nested: dict | None = None):
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        if nested:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as inner:
                for name, data in nested.items():
                    inner.writestr(name, data)
            z.writestr("2026/Jan.zip", buf.getvalue())
        z.writestr("2026/readme.txt", b"not a market file")
    return path


def test_ingest_is_idempotent_and_keeps_the_longer_copy(tmp_path):
    store = Store(root=tmp_path / "wh")
    mo = match_odds_file()
    short = _file([_msg(KO_MS - 86_400_000, defn=_defn())])
    rep = bronze.ingest_bundle(store, _bundle(tmp_path, {"2026/Jan/29/35155405/1.252787713.bz2": short},
                                              nested={"Jan/29/35155405/1.252787713.bz2": short}), "first")
    assert (rep["market_files"], rep["not_market_files"], rep["duplicates_in_bundle"], rep["new"]) == (2, 1, 1, 1)
    again = bronze.ingest_bundle(store, _bundle(tmp_path, {"2026/Jan/29/35155405/1.252787713.bz2": short}), "again")
    assert (again["new"], again["same"]) == (0, 1)
    before = read_manifest(store, "bronze/raw")["parts"]["2026-01"]["written_utc"]
    assert read_manifest(store, "bronze/raw")["parts"]["2026-01"]["sources"] == ["first"]
    longer = bronze.ingest_bundle(store, _bundle(tmp_path, {"2026/Jan/29/35155405/1.252787713.bz2": mo}), "later")
    assert longer["replaced"] == 1
    held = read_part(store, "bronze/raw", 2026, 1)
    assert held.num_rows == 1 and held.column("data")[0].as_py() == mo and held.column("source")[0].as_py() == "later"
    assert read_manifest(store, "bronze/raw")["parts"]["2026-01"]["written_utc"] >= before
    assert bronze.index(store).column_names[-2:] == ["year", "month"]


def test_silver_builds_once_reads_a_market_from_its_longest_file_and_reconciles(tmp_path):
    store = Store(root=tmp_path / "wh")
    mo = match_odds_file()
    stub = _file([_msg(KO_MS - 86_400_000, defn=_defn())])            # the same market filed under another day
    bronze.ingest_bundle(store, _bundle(tmp_path, {"2026/Jan/29/35155405/1.252787713.bz2": mo,
                                                   "2026/Jan/28/35155405/1.252787713.bz2": stub,
                                                   "2026/Jan/29/35155405/1.9.bz2": b"BZh broken"}), "b")
    rep = silver.build(store, workers=1)
    assert [b["part"] for b in rep["built"]] == ["2026-01"] and rep["failed_checks"] == []
    entry = read_manifest(store, "silver")["parts"]["2026-01"]
    assert (entry["files_in"], entry["superseded"], entry["files_read"], entry["files_failed"]) == (3, 1, 1, 1)
    assert entry["superseded_files"] == ["BASIC/2026/Jan/28/35155405/1.252787713.bz2"]
    assert entry["rows"][SILVER_MARKET.name] == 1 and entry["rows"][SILVER_TICK.name] == 6
    assert all(v for k, v in entry["checks"].items() if k.endswith("_ok"))
    tick = read_part(store, SILVER_TICK.name, 2026, 1).to_pandas()
    assert tick["pt"].is_monotonic_increasing or len(tick["selection_id"].unique()) > 1
    assert list(tick.columns) == SILVER_TICK.names
    # nothing changed: nothing is rebuilt; a forced build gives the same rows
    assert silver.build(store, workers=1)["built"] == []
    forced = silver.build(store, force=True, workers=1)
    assert forced["built"][0]["rows"][SILVER_TICK.name] == 6


# ---------------------------------------------------------------------------------------------------------------
# gold: the event, the checkpoint grid, the move panel, the team dimension
# ---------------------------------------------------------------------------------------------------------------

OU_RUNNERS = [{"status": "ACTIVE", "sortPriority": 1, "id": 47972, "name": "Under 2.5 Goals"},
              {"status": "ACTIVE", "sortPriority": 2, "id": 47973, "name": "Over 2.5 Goals"}]


def over_under_file() -> bytes:
    settled = [dict(r, status=s) for r, s in zip(OU_RUNNERS, ("LOSER", "WINNER"))]
    kw = {"market_type": "OVER_UNDER_25", "name": "Over/Under 2.5 Goals"}
    return _file([
        _msg(KO_MS - 86_400_000, "1.252787714", _defn(runners=OU_RUNNERS, **kw)),
        _msg(KO_MS - 7_200_000, "1.252787714", rc=[{"id": 47972, "ltp": 2.0}, {"id": 47973, "ltp": 2.0}]),
        _msg(KO_MS - 120_000, "1.252787714", rc=[{"id": 47973, "ltp": 1.8}]),
        _msg(KO_MS + 90_000, "1.252787714", _defn(in_play=True, runners=OU_RUNNERS, **kw)),
        _msg(KO_MS + 4_000_000, "1.252787714", _defn("CLOSED", True, runners=settled, **kw)),   # the third goal
    ])


@pytest.fixture()
def warehouse(tmp_path):
    from football.warehouse import gold
    store = Store(root=tmp_path / "wh")
    bronze.ingest_bundle(store, _bundle(tmp_path, {"2026/Jan/29/35155405/1.252787713.bz2": match_odds_file(),
                                                   "2026/Jan/29/35155405/1.252787714.bz2": over_under_file()}), "b")
    silver.build(store, workers=1)
    rep = gold.build(store, only=["gold_event", "gold_checkpoint", "gold_move", "gold_team"])
    assert rep["failed"] == []
    return store


def test_the_event_row_carries_sides_result_and_the_closing_book(warehouse):
    e = read_part(warehouse, "gold/event", 2026, 1).to_pandas().iloc[0]
    assert (e["home_id"], e["away_id"], e["result"], e["over25"]) == (71085, 31317, "H", 1)
    assert (e["mo_market_id"], e["ou_market_id"]) == ("1.252787713", "1.252787714") and pd_isna(e["ah_market_id"])
    assert e["went_in_play"] and not e["void"] and e["kickoff_delay_s"] == 90 and e["n_markets"] == 2
    book = 1 / 2.4 + 1 / 3.0 + 1 / 3.4
    assert e["close_legs"] == 3 and abs(e["close_book_sum"] - book) < 1e-12
    assert abs(e["p_home"] - (1 / 2.4) / book) < 1e-12
    assert e["close_age_s"] == 3690.0                       # the away price last traded at T-60; the off 90 s late


@pytest.mark.parametrize("home, away, want", [
    ("Sturm Graz", "Brann", "senior"), ("Willem II", "Ajax", "senior"), ("Boca Juniors", "River Plate", "senior"),
    ("Club 2 de Mayo", "Olimpia", "senior"), ("Schalke 04", "Mainz 05", "senior"),
    ("Leones FC", "Racing Club", "senior"),
    ("Arsenal (W)", "Chelsea (W)", "women"), ("England U21", "Spain U21", "youth"),
    ("Aberdeen Youth", "Hearts Youth", "youth"), ("Jong Ajax", "Jong PSV", "reserves"),
    ("Bayern Munich II", "Unterhaching", "reserves"), ("Ludogorets II Razgrad", "CSKA", "reserves"),
    ("Hamburg III", "Altona 93", "reserves"), ("Zenit 2", "Dynamo", "reserves"),
    ("Barcelona B", "Real Madrid C", "reserves"),
    ("Sochaux (Am)", "Dijon", "reserves"), ("Buxton Res", "Leek", "reserves"),
    ("Libertad Asuncion (Res)", "Club 2 de Mayo (Res", "reserves"),          # Betfair cut the name at 50 characters
])
def test_the_kind_of_football_is_read_from_the_sides_names(home, away, want):
    from football.warehouse.gold import sql
    from football.warehouse.gold_event import CATEGORY
    ev = pa.table({"event_name": [f"{home} v {away}"], "home_name": [home], "away_name": [away]})
    assert sql(f"SELECT {CATEGORY} AS category FROM ev", ev=ev).column("category").to_pylist() == [want]


def test_a_checkpoint_price_is_the_last_one_at_or_before_it_never_a_later_one(warehouse):
    c = read_part(warehouse, "gold/checkpoint", 2026, 1).to_pandas()
    mo = c[c["market_id"] == "1.252787713"].set_index(["cp_min", "role"])
    # T-60: home and away have just traded (age 0), the draw has no price yet: the book is not complete
    assert mo.loc[(60, "home"), "ltp"] == 2.5 and mo.loc[(60, "home"), "age_s"] == 0
    assert (60, "draw") not in mo.index and not mo.loc[(60, "home"), "book_complete"]
    assert pd_isna(mo.loc[(60, "home"), "p_norm"])
    # T-30: the same prices, 30 minutes older; T-1: the new home price and the draw's first
    assert mo.loc[(30, "home"), "ltp"] == 2.5 and mo.loc[(30, "home"), "age_s"] == 1800
    assert mo.loc[(1, "home"), "ltp"] == 2.4 and mo.loc[(1, "draw"), "ltp"] == 3.4
    assert mo.loc[(1, "away"), "age_s"] == 3540 and mo.loc[(1, "home"), "book_max_age_s"] == 3540
    assert mo.loc[(1, "home"), "book_complete"] and mo.loc[(1, "home"), "n_ticks_window"] == 1
    assert mo.loc[(1, "away"), "n_ticks_window"] == 0 and mo.loc[(1, "away"), "n_ticks_before"] == 1
    # the close is the last price before the market went in play (90 s after the scheduled off), not the goal's 1.5
    assert mo.loc[(0, "home"), "ltp"] == 2.4 and mo.loc[(0, "home"), "age_s"] == 150
    assert abs(mo.loc[(0, "home"), "p_norm"] - (1 / 2.4) / (1 / 2.4 + 1 / 3.0 + 1 / 3.4)) < 1e-12
    # nothing before the first trade: no row at T-24h
    assert 1440 not in c["cp_min"].tolist()
    ou = c[c["market_id"] == "1.252787714"].set_index(["cp_min", "role"])
    assert ou.loc[(120, "over"), "p_norm"] == 0.5 and ou.loc[(1, "over"), "ltp"] == 1.8


def pd_isna(v) -> bool:
    return v != v


def test_the_move_is_from_a_complete_book_to_the_close_and_says_how_stale_each_end_is(warehouse):
    m = read_part(warehouse, "gold/move", 2026, 1).to_pandas()
    mo = m[m["market_type"] == "MATCH_ODDS"]
    assert sorted(mo["cp_min"].unique().tolist()) == [1]            # the only checkpoint with a full 1X2 book
    home = mo[mo["role"] == "home"].iloc[0]
    assert home["y_log"] == 0 and home["price_ratio"] == 1 and home["won"] == 1 and home["tier"] == "D"
    assert not home["usable"] and not home["cp_fresh"] and home["cp_book_age_s"] == 3540   # the away price: an hour old
    assert home["category"] == "senior" and home["kickoff_shift_s"] == 0
    ou = m[(m["market_type"] == "OVER_UNDER_25") & (m["role"] == "over")].set_index("cp_min")
    assert abs(ou.loc[120, "y_log"] - __import__("math").log((1 / 1.8) / (1 / 1.8 + 1 / 2.0) / 0.5)) < 1e-12
    assert ou.loc[120, "price_ratio"] == 2.0 / 1.8 and ou.loc[120, "won"] == 1 and ou.loc[120, "n_ticks_after"] == 1
    assert bool(ou.loc[120, "is_main"])


# ---------------------------------------------------------------------------------------------------------------
# the market's clock: what Betfair's files do that must not be taken at face value
# ---------------------------------------------------------------------------------------------------------------

MIN = 60_000
SETTLED = [dict(r, status=s) for r, s in zip(MO_RUNNERS, ("WINNER", "LOSER", "LOSER"))]


def _iso(ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _all(h, d, a):
    return [{"id": 71085, "ltp": h}, {"id": 58805, "ltp": d}, {"id": 31317, "ltp": a}]


def _silver_of(raw: bytes) -> dict:
    b = Batch()
    assert b.add_file(raw, "BASIC/2026/Jan/29/35155405/1.252787713.bz2", "abc") == 1, b.failed
    return {k.split("/")[1]: v.to_pandas() for k, v in b.tables().items()}


def _gold_of(tmp_path, raw: bytes) -> dict:
    """The gold tables of a one-market warehouse built from a stream file."""
    from football.warehouse import gold
    store = Store(root=tmp_path / "wh")
    bronze.ingest_bundle(store, _bundle(tmp_path, {"2026/Jan/29/35155405/1.252787713.bz2": raw}), "b")
    silver.build(store, workers=1)
    assert gold.build(store, only=["gold_event", "gold_checkpoint", "gold_move"])["failed"] == []
    return {n: read_part(store, f"gold/{n}", 2026, 1).to_pandas() for n in ("event", "checkpoint", "move")}


def _ms_of(v) -> int:
    return v.value // 10**6


def rewritten_file() -> bytes:
    """The kick-off time is 20:00:00 until the message that turns the market in play, 203 s late, which rewrites
    it to 20:02:23 (as Betfair does on about one market in eleven)."""
    late = KO_MS + 143_000
    return _file([
        _msg(KO_MS - 26 * 60 * MIN, defn=_defn()),
        _msg(KO_MS - 10 * MIN, rc=_all(2.5, 3.4, 3.0)),
        _msg(KO_MS - 30_000, rc=_all(2.4, 3.4, 3.1)),
        _msg(KO_MS + 60_000, rc=_all(2.3, 3.5, 3.2)),                      # the match is late: still before the off
        _msg(KO_MS + 203_000, defn=_defn(in_play=True, market_time=_iso(late))),
        _msg(KO_MS + 20 * MIN, rc=_all(1.5, 4.0, 8.0)),
        _msg(KO_MS + 115 * MIN, defn=_defn("CLOSED", True, runners=SETTLED, market_time=_iso(late))),
    ])


def test_the_schedule_is_the_one_that_stood_before_the_off_not_betfairs_rewrite(tmp_path):
    m = _silver_of(rewritten_file())["market"].iloc[0]
    assert _ms_of(m["market_time"]) == KO_MS + 143_000 and _ms_of(m["market_time_pre"]) == KO_MS
    assert _ms_of(m["schedule_set_pt"]) == KO_MS - 26 * 60 * MIN            # never moved before the off
    assert _ms_of(m["in_play_pt"]) == _ms_of(m["off_pt"]) == KO_MS + 203_000
    assert m["kickoff_delay_s"] == 203 and m["n_inplay_turns"] == 1 and m["n_market_time_changes"] == 1
    g = _gold_of(tmp_path, rewritten_file())
    e = g["event"].iloc[0]
    assert _ms_of(e["kickoff"]) == KO_MS and e["kickoff_delay_s"] == 203 and e["kickoff_shift_s"] == 0
    home = g["checkpoint"][g["checkpoint"]["role"] == "home"].set_index("cp_min")
    # T-1 is a minute before 20:00, not a minute before the rewritten 20:02:23: it must not hold the 2.3 that was
    # published at 20:01, nor the 2.4 of 19:59:30
    assert _ms_of(home.loc[1, "cp_time"]) == KO_MS - MIN and home.loc[1, "ltp"] == 2.5
    assert home.loc[5, "ltp"] == 2.5 and home.loc[0, "ltp"] == 2.3 and home["schedule_known"].all()
    assert _ms_of(home.loc[0, "cp_time"]) == KO_MS + 203_000 and home.loc[0, "age_s"] == 143.0
    tick = _silver_of(rewritten_file())["tick"]
    assert tick.loc[tick["ltp"] == 2.4, "t_off_s"].iloc[0] == 30
    assert tick.loc[tick["ltp"] == 2.3, "t_off_s"].iloc[0] == -60


def test_a_checkpoint_before_a_reschedule_was_not_known_at_the_time(tmp_path):
    early = _iso(KO_MS - 2 * 60 * MIN)                                    # first listed for two hours earlier
    raw = _file([
        _msg(KO_MS - 30 * 60 * MIN, defn=_defn(market_time=early)),
        _msg(KO_MS - 8 * 60 * MIN, rc=_all(2.5, 3.4, 3.0)),
        _msg(KO_MS - 3 * 60 * MIN, defn=_defn()),                          # moved to 20:00 three hours before it
        _msg(KO_MS - 10 * MIN, rc=_all(2.4, 3.4, 3.1)),
        _msg(KO_MS + 60_000, defn=_defn(in_play=True)),
        _msg(KO_MS + 115 * MIN, defn=_defn("CLOSED", True, runners=SETTLED)),
    ])
    m = _silver_of(raw)["market"].iloc[0]
    assert _ms_of(m["market_time_pre"]) == KO_MS and _ms_of(m["schedule_set_pt"]) == KO_MS - 3 * 60 * MIN
    g = _gold_of(tmp_path, raw)
    assert g["event"].iloc[0]["kickoff_shift_s"] == 7200 and _ms_of(g["event"].iloc[0]["schedule_set"]) == \
        KO_MS - 3 * 60 * MIN
    known = g["checkpoint"][g["checkpoint"]["role"] == "home"].set_index("cp_min")["schedule_known"]
    assert not known[360] and known[180] and known[120] and known[0]      # T-360 was 14:00: the match was then at 18:00
    mv = g["move"][g["move"]["role"] == "home"].set_index("cp_min")
    assert not mv.loc[360, "schedule_known"] and mv.loc[60, "schedule_known"]


def test_a_turn_in_play_made_by_mistake_is_not_the_kick_off(tmp_path):
    day = 24 * 60 * MIN
    raw = _file([
        _msg(KO_MS - 2 * day, defn=_defn()),
        _msg(KO_MS - day - 10 * MIN, rc=_all(2.6, 3.4, 2.9)),
        _msg(KO_MS - day, defn=_defn(in_play=True)),                       # turned in play a day early ...
        _msg(KO_MS - day + 2 * MIN, rc=_all(2.5, 3.4, 3.0)),
        _msg(KO_MS - day + 5 * MIN, defn=_defn()),                         # ... and turned back
        _msg(KO_MS - 10 * MIN, rc=_all(2.4, 3.4, 3.1)),
        _msg(KO_MS + 60_000, defn=_defn(in_play=True)),                    # the kick-off
        _msg(KO_MS + 20 * MIN, rc=_all(1.5, 4.0, 8.0)),
        _msg(KO_MS + 115 * MIN, defn=_defn("CLOSED", True, runners=SETTLED)),
    ])
    sv = _silver_of(raw)
    m = sv["market"].iloc[0]
    assert _ms_of(m["in_play_pt"]) == KO_MS + 60_000 and m["n_inplay_turns"] == 2 and m["kickoff_delay_s"] == 60
    assert (m["n_ticks_pre"], m["n_ticks_inplay"]) == (9, 3)
    k = sv["tick"]
    assert not k.loc[k["ltp"] == 2.5, "in_play"].iloc[0] and k.loc[k["ltp"] == 2.5, "pre_off"].iloc[0]
    assert sv["runner"].set_index("role").loc["home", "ltp_last_pre"] == 2.4
    g = _gold_of(tmp_path, raw)
    home = g["checkpoint"][g["checkpoint"]["role"] == "home"].set_index("cp_min")
    assert home.loc[0, "ltp"] == 2.4 and _ms_of(home.loc[0, "cp_time"]) == KO_MS + 60_000
    assert home.loc[60, "ltp"] == 2.5 and g["event"].iloc[0]["inplay_turns"] == 2
    assert abs(g["event"].iloc[0]["p_home"] - (1 / 2.4) / (1 / 2.4 + 1 / 3.4 + 1 / 3.1)) < 1e-12


def test_a_drop_out_of_play_in_mid_match_does_not_make_in_running_prices_a_close():
    raw = _file([
        _msg(KO_MS - 26 * 60 * MIN, defn=_defn()),
        _msg(KO_MS - 10 * MIN, rc=_all(1.4, 5.9, 9.4)),
        _msg(KO_MS + 60_000, defn=_defn(in_play=True)),                    # the kick-off
        _msg(KO_MS + 55 * MIN, defn=_defn()),                              # out of play for nine minutes
        _msg(KO_MS + 58 * MIN, rc=_all(1.74, 4.0, 5.8)),                   # in running, whatever the flag says
        _msg(KO_MS + 64 * MIN, defn=_defn(in_play=True)),
        _msg(KO_MS + 115 * MIN, defn=_defn("CLOSED", True, runners=SETTLED)),
    ])
    sv = _silver_of(raw)
    m = sv["market"].iloc[0]
    assert _ms_of(m["in_play_pt"]) == KO_MS + 60_000 and m["n_inplay_turns"] == 2
    k = sv["tick"]
    assert k.loc[k["ltp"] == 1.74, "in_play"].iloc[0] and not k.loc[k["ltp"] == 1.74, "pre_off"].iloc[0]
    assert k.loc[k["ltp"] == 1.74, "t_inplay_s"].iloc[0] == 57 * 60
    assert sv["runner"].set_index("role").loc["home", "ltp_last_pre"] == 1.4   # the close, not 1.74


def test_a_market_that_never_turns_in_play_ends_its_pre_off_life_at_the_scheduled_kick_off(tmp_path):
    raw = _file([
        _msg(KO_MS - 26 * 60 * MIN, defn=_defn(turnInPlayEnabled=False)),
        _msg(KO_MS - 10 * MIN, rc=_all(2.0, 3.5, 4.0)),
        _msg(KO_MS + 40 * MIN, rc=_all(1.2, 6.0, 20.0)),                   # traded while the match was played
        _msg(KO_MS + 170 * MIN, defn=_defn("CLOSED", runners=SETTLED, turnInPlayEnabled=False)),
    ])
    sv = _silver_of(raw)
    m = sv["market"].iloc[0]
    assert pd_isna(m["in_play_pt"]) and _ms_of(m["off_pt"]) == KO_MS and pd_isna(m["kickoff_delay_s"])
    assert (m["n_ticks"], m["n_ticks_pre"], m["n_ticks_inplay"]) == (6, 3, 0)
    late = sv["tick"][sv["tick"]["ltp"] == 1.2].iloc[0]
    assert not late["pre_off"] and not late["in_play"] and late["t_off_s"] == -2400
    assert sv["runner"].set_index("role").loc["home", "ltp_last_pre"] == 2.0
    g = _gold_of(tmp_path, raw)
    e = g["event"].iloc[0]
    assert not e["went_in_play"] and e["result"] == "H" and e["close_age_s"] == 600.0
    assert abs(e["p_home"] - 0.5 / (0.5 + 1 / 3.5 + 0.25)) < 1e-12        # the 2.0 of before the off, not the 1.2
    close = g["checkpoint"][(g["checkpoint"]["role"] == "home") & (g["checkpoint"]["cp_min"] == 0)].iloc[0]
    assert close["ltp"] == 2.0 and _ms_of(close["cp_time"]) == KO_MS and close["age_s"] == 600.0


def test_a_match_called_off_before_its_kick_off_has_no_close_and_no_checkpoint_after_it_closed(tmp_path):
    raw = _file([
        _msg(KO_MS - 26 * 60 * MIN, defn=_defn()),
        _msg(KO_MS - 8 * 60 * MIN, rc=_all(2.0, 3.5, 4.0)),
        _msg(KO_MS - 3 * 60 * MIN, defn=_defn("CLOSED", runners=[dict(r, status="REMOVED") for r in MO_RUNNERS])),
    ])
    g = _gold_of(tmp_path, raw)
    assert sorted(g["checkpoint"]["cp_min"].unique().tolist()) == [180, 360]       # up to the moment it closed
    assert len(g["move"]) == 0 and g["event"].iloc[0]["void"]


def test_a_field_of_the_wrong_kind_is_a_blank_and_a_broken_file_fails_alone():
    odd_runners = [dict(MO_RUNNERS[0], sortPriority=40000, name=12)] + MO_RUNNERS[1:]
    odd = _file([_msg(KO_MS - 3_600_000, defn=_defn(betDelay=70000, version="12a", runners=odd_runners,
                                                     numberOfWinners="one", marketBaseRate="five", eventTypeId="1")),
                 _msg(KO_MS - 60_000, rc=[{"id": 71085, "ltp": 2.5}, {"id": 31317, "ltp": "3.0"}])])
    no_time = _file([_msg(10 ** 19, defn=_defn(), rc=[{"id": 71085, "ltp": 2.5}])])
    orphan = _file([{"op": "mcm", "pt": KO_MS - 60_000, "mc": [
        {"id": "1.252787713", "marketDefinition": _defn(), "rc": [{"id": 71085, "ltp": 2.5}]},
        {"id": "1.999", "rc": [{"id": 5, "ltp": 2.0}]}]}])
    b = Batch()
    assert b.add_file(match_odds_file(), "BASIC/2026/Jan/29/35155405/1.252787713.bz2", "a") == 1
    assert b.add_file(no_time, "BASIC/2026/Jan/29/1/1.1.bz2", "b") == 0
    assert b.add_file(orphan, "BASIC/2026/Jan/29/1/1.2.bz2", "c") == 0
    assert b.add_file(odd, "BASIC/2026/Jan/29/35155406/1.3.bz2", "d") == 1
    assert [p for p, _ in b.failed] == ["BASIC/2026/Jan/29/1/1.1.bz2", "BASIC/2026/Jan/29/1/1.2.bz2"]
    assert "not a time" in b.failed[0][1] and "no definition" in b.failed[1][1]
    t = {k.split("/")[1]: v.to_pandas() for k, v in b.tables().items()}      # the batch still makes its tables
    assert len(t["market"]) == 2
    m = t["market"].set_index("plan_path").loc["BASIC/2026/Jan/29/35155406/1.3.bz2"]
    assert m["event_type_id"] == 1 and pd_isna(m["number_of_winners"]) and m["n_ltp_invalid"] == 1
    assert pd_isna(m["market_base_rate"]) and m["n_ticks"] == 1               # '3.0' in text is not a price
    odd_defn = t["definition"][t["definition"]["pt"] == pd.Timestamp(KO_MS - 3_600_000, unit="ms", tz="UTC")]
    assert pd_isna(odd_defn["bet_delay"].iloc[-1]) and pd_isna(odd_defn["version"].iloc[-1])
    r = t["runner"][(t["runner"]["market_id"] == "1.252787713") & (t["runner"]["selection_id"] == 71085)]
    assert len(r) == 2 and set(r["runner_name"]) == {"Sturm Graz", "12"}     # the same market id, read from two files


def test_the_kick_off_is_the_turn_nearest_its_own_schedule():
    from football.warehouse.parse import kick_off_turn
    assert kick_off_turn([]) is None and kick_off_turn([(3, 1000, None)]) == (3, 1000, None)
    # a day early against on time; on time against an hour into the match; a tie goes to the first
    assert kick_off_turn([(3, 0, 86_400_000), (9, 86_460_000, 86_400_000)])[0] == 9
    assert kick_off_turn([(3, 60_000, 0), (9, 3_800_000, 0)])[0] == 3
    assert kick_off_turn([(3, 50_000, 0), (9, 950_000, 1_000_000)])[0] == 3


def test_the_team_dimension_and_the_build_is_idempotent(warehouse):
    from football.warehouse import gold
    from football.warehouse.store_io import read_single
    t = read_single(warehouse, "gold/team").to_pandas().set_index("team_id")
    assert t.loc[71085, "team_name"] == "Sturm Graz" and t.loc[71085, "n_home"] == 1 and t.loc[31317, "n_home"] == 0
    again = gold.build(warehouse, only=["gold_event", "gold_checkpoint", "gold_move", "gold_team"])
    assert all(v["built"] == [] for v in again["modules"].values())


def test_gold_modules_run_after_the_modules_they_read():
    from football.warehouse import gold
    mods = gold.order(gold.load(["football.warehouse.gold_move", "football.warehouse.gold_team",
                                 "football.warehouse.gold_checkpoint", "football.warehouse.gold_event"]))
    names = [m.__name__.rsplit(".", 1)[-1] for m in mods]
    assert names.index("gold_checkpoint") < names.index("gold_move")
    assert names.index("gold_event") < names.index("gold_move") and names.index("gold_event") < names.index("gold_team")


# ---------------------------------------------------------------------------------------------------------------
# the quality report, the catalog, the dictionary
# ---------------------------------------------------------------------------------------------------------------

def test_the_quality_report_passes_a_sound_warehouse_and_fails_a_broken_one(warehouse):
    from football.warehouse import quality
    from football.warehouse.gold_event import SPEC as EVENT
    from football.warehouse.store_io import write_part
    rep = quality.report(warehouse)
    assert rep["ok"] and rep["hard_failed"] == [] and rep["rows"]["gold/event"] == 1
    assert "PASS" in quality.markdown(rep)
    # the same event written into a second partition: event ids are no longer unique across partitions
    write_part(warehouse, EVENT, 2026, 2, read_part(warehouse, "gold/event", 2026, 1))
    bad = quality.report(warehouse)
    assert not bad["ok"] and "gold/event: event_id unique across partitions" in bad["hard_failed"]


def test_a_stale_silver_partition_is_a_hard_failure(warehouse, tmp_path):
    from football.warehouse import quality
    newer = _file([_msg(KO_MS - 86_400_000, "1.252787799", _defn())])
    bronze.ingest_bundle(warehouse, _bundle(tmp_path, {"2026/Jan/29/35155405/1.252787799.bz2": newer}), "more")
    from football.warehouse import gold
    stale_silver, stale_gold = "silver built from the current bronze by the current parser", \
        "gold built from its current inputs"
    rep = quality.report(warehouse)                      # bronze moved on; silver was not rebuilt
    assert not rep["ok"] and stale_silver in rep["hard_failed"]
    silver.build(warehouse, workers=1)
    rep = quality.report(warehouse)                      # silver caught up; gold is now the one left behind
    assert stale_silver not in rep["hard_failed"] and stale_gold in rep["hard_failed"] and not rep["ok"]
    assert "gold/event 2026-01" in gold.stale(warehouse)
    assert gold.build(warehouse, only=["gold_event", "gold_checkpoint", "gold_move", "gold_team"])["failed"] == []
    assert quality.report(warehouse)["ok"] and gold.stale(warehouse) == []


def test_a_silver_partition_is_rebuilt_when_another_month_comes_to_hold_a_longer_copy_of_its_market(tmp_path):
    store = Store(root=tmp_path / "wh")
    stub = _file([_msg(KO_MS - 86_400_000, defn=_defn()), _msg(KO_MS - 3_600_000, rc=[{"id": 71085, "ltp": 2.5}])])
    one, two = tmp_path / "a", tmp_path / "b"
    one.mkdir(), two.mkdir()
    bronze.ingest_bundle(store, _bundle(one, {"2026/Jan/29/35155405/1.252787713.bz2": stub}), "a")
    assert len(silver.build(store, workers=1)["built"]) == 1
    # the match is played in February: Betfair files the whole market again under its new day
    bronze.ingest_bundle(store, _bundle(two, {"2026/Feb/2/35155405/1.252787713.bz2": match_odds_file()}), "b")
    rep = silver.build(store, workers=1)
    assert [b["part"] for b in rep["built"]] == ["2026-01", "2026-02"]       # January leaves its stub out now
    jan, feb = read_part(store, "silver/market", 2026, 1), read_part(store, "silver/market", 2026, 2)
    assert jan.num_rows == 0 and feb.num_rows == 1
    assert read_manifest(store, "silver")["parts"]["2026-01"]["superseded"] == 1
    assert silver.build(store, workers=1)["built"] == []


def test_a_bronze_write_that_died_half_way_is_put_right_by_the_next_ingest(tmp_path):
    from football.warehouse.store_io import manifest_key, part_key
    store = Store(root=tmp_path / "wh")
    files = {"2026/Jan/29/35155405/1.252787713.bz2": match_odds_file()}
    bronze.ingest_bundle(store, _bundle(tmp_path, files), "a")
    good = store.get(manifest_key("bronze/raw"))
    more = dict(files, **{"2026/Jan/29/35155405/1.252787714.bz2": over_under_file()})
    other = tmp_path / "second"
    other.mkdir()
    bronze.ingest_bundle(store, _bundle(other, more), "b")
    store.put(manifest_key("bronze/raw"), good)          # as if the run had died before the manifest was written
    assert read_manifest(store, "bronze/raw")["parts"]["2026-01"]["rows"] == 1
    again = bronze.ingest_bundle(store, _bundle(other, more), "b")
    assert again["same"] == 2 and again["new"] == 0
    assert read_manifest(store, "bronze/raw")["parts"]["2026-01"]["rows"] == 2
    assert read_part(store, "bronze/index", 2026, 1).num_rows == 2 and store.get(part_key("bronze/raw", 2026, 1))


def test_the_bundle_queue_reads_only_the_bundle_it_was_promised(tmp_path, monkeypatch):
    import hashlib
    import json
    import shutil
    from pathlib import Path
    from football.warehouse import ROOT
    store = Store(root=tmp_path / "wh")
    good = _bundle(tmp_path, {"2026/Jan/29/35155405/1.252787713.bz2": match_odds_file()})
    raw = good.read_bytes()
    queue = tmp_path / "bundles.json"
    queue.write_text(json.dumps([{"name": "b1", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]))
    marker = f"{ROOT}/bronze/bundles/b1.json"
    # the list holds no link (the repository is public): without one in the environment the bundle waits
    monkeypatch.delenv(bronze.URLS_ENV, raising=False)
    assert "waiting" in bronze.ingest_queue(store, str(queue), str(tmp_path), fetch=None)["b1"]
    monkeypatch.setenv(bronze.URLS_ENV, json.dumps({"b1": "https://example.invalid/secret-link", "b2": "y"}))

    def page(url, path):                                  # a share link that serves a web page
        Path(path).write_bytes(b"<html>quota exceeded</html>")

    def cut(url, path):                                   # a download cut short
        Path(path).write_bytes(raw[:len(raw) // 2])

    def other(url, path):                                 # the right size, other bytes
        Path(path).write_bytes(bytes(len(raw)))

    for fetch, why in ((page, "bytes"), (cut, "bytes"), (other, "SHA-256")):
        out = bronze.ingest_queue(store, str(queue), str(tmp_path), fetch=fetch)
        assert why in out["b1"]["failed"] and store.get(marker) is None and "secret-link" not in str(out)
        assert not (tmp_path / "b1.bundle").exists() and bronze.partitions(store) == []
    out = bronze.ingest_queue(store, str(queue), str(tmp_path), fetch=lambda url, path: shutil.copy(good, path))
    assert out["b1"]["market_files"] == 1 and store.get(marker) is not None
    assert b"secret-link" not in store.get(marker)             # what is stored does not carry the link either
    assert bronze.ingest_queue(store, str(queue), str(tmp_path), fetch=page) == {"b1": "already ingested"}
    # an entry without a size or a hash: a download with no market file in it is still not a bundle
    queue.write_text(json.dumps([{"name": "b2", "url": "https://example.invalid/listed-link"}]))    # a listed link
    empty = _bundle(tmp_path / "wh", {"readme.txt": b"nothing here"})
    out = bronze.ingest_queue(store, str(queue), str(tmp_path), fetch=lambda url, path: shutil.copy(empty, path))
    assert "no market file" in out["b2"]["failed"] and store.get(f"{ROOT}/bronze/bundles/b2.json") is None
    monkeypatch.delenv(bronze.URLS_ENV)                          # the entry's own link is enough
    out = bronze.ingest_queue(store, str(queue), str(tmp_path), fetch=lambda url, path: shutil.copy(good, path))
    assert out["b2"]["market_files"] == 1 and b"listed-link" not in store.get(f"{ROOT}/bronze/bundles/b2.json")


def test_the_catalog_gives_views_and_macros_over_the_tables(warehouse):
    from football.warehouse import catalog
    con = catalog.connect(warehouse)
    assert con.sql("SELECT count(*) FROM tick").fetchone()[0] == 9          # 6 match-odds prices, 3 over/under
    assert con.sql("SELECT result FROM event").fetchone()[0] == "H"
    kick = "(SELECT kickoff FROM event)"
    assert con.sql(f"SELECT price_asof('1.252787713', 71085, 0.0, {kick} - INTERVAL 30 MINUTE)").fetchone()[0] == 2.5
    assert con.sql(f"SELECT price_asof('1.252787713', 71085, 0.0, {kick})").fetchone()[0] == 2.4
    rows = con.sql("SELECT role, ltp FROM match_ticks(35155405) WHERE status = 'SUSPENDED'").fetchall()
    assert rows == [("home", 1.5)]
    assert "bronze_raw" not in catalog.tables(warehouse)                    # the raw bytes are not a view


def test_the_facts_are_queries_over_what_is_built_and_skip_what_is_not(warehouse):
    from football.warehouse import facts
    f = facts.facts(warehouse)
    assert f["rows"]["event"] == 1 and f["rows"]["market"] == 2 and f["rows"]["tick"] == 9
    assert f["ticks"] == {"all": 9, "pre_off": 7, "in_play": 2, "after_the_off_not_in_play": 0}
    assert f["markets"]["turned_in_play"] == 2 and f["markets"]["kickoff_time_rewritten_at_or_after_the_off"] == 0
    assert f["events"]["with_a_result"] == 1 and f["events"]["kickoff_delay_s"]["median"] == 90
    assert f["match_odds_home_by_checkpoint"][1] == {"rows": 1, "book_complete": 1.0, "schedule_known": 1.0}
    assert "club" not in f and "ratings" not in f and "in_play" not in f      # not built here: left out, not guessed


def test_the_dictionary_documents_every_table_and_column():
    from football.warehouse import dictionary
    md = dictionary.markdown()
    for spec in dictionary.specs():
        assert f"## `{spec.name}`" in md
        for c in spec.columns:
            assert f"| `{c.name}` |" in md and c.doc.strip(), (spec.name, c.name)
    assert "\u2014" not in md
