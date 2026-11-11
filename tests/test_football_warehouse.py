"""The football warehouse's foundation: paths, the stream-file parser, bronze ingest and the silver build
(hand-written stream files: a few messages each, shaped like Betfair's)."""

from __future__ import annotations

import bz2
import io
import json
import zipfile

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


def test_a_required_column_may_not_hold_a_null():
    bad = pa.table({c.name: pa.array([None], type=c.type) for c in SILVER_TICK.columns})
    with pytest.raises(ValueError):
        conform(SILVER_TICK, bad)


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
    assert not home["usable"] and home["cp_book_age_s"] == 3540     # the away price is an hour old
    ou = m[(m["market_type"] == "OVER_UNDER_25") & (m["role"] == "over")].set_index("cp_min")
    assert abs(ou.loc[120, "y_log"] - __import__("math").log((1 / 1.8) / (1 / 1.8 + 1 / 2.0) / 0.5)) < 1e-12
    assert ou.loc[120, "price_ratio"] == 2.0 / 1.8 and ou.loc[120, "won"] == 1 and ou.loc[120, "n_ticks_after"] == 1
    assert bool(ou.loc[120, "is_main"])


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
