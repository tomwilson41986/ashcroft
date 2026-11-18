"""Phase 0 of the ante-post plan (antepost/phase0.py). The stream lines and the betfair_prices rows are format fixtures
in Betfair's layouts, written for the test; no figure from them is used anywhere."""

import bz2
import io
import json
import sqlite3
import tarfile

import pandas as pd

from antepost import phase0 as p0

OFF = "2026-03-13T15:20:00.000Z"
DAY = 86400000
T_OFF = 1773415200000                                   # 2026-03-13T15:20:00Z in ms


def _defn(status, runners, market_type="ANTEPOST_WIN", in_play=False):
    return {"eventId": "9", "eventTypeId": "7", "eventName": "Cheltenham Gold Cup", "marketType": market_type,
            "name": "Gold Cup", "countryCode": "GB", "venue": "Cheltenham", "marketTime": OFF, "bspMarket": False,
            "turnInPlayEnabled": False, "status": status, "inPlay": in_play, "bettingType": "ODDS", "runners": runners}


def _stream():
    a, b, c = ({"id": 1, "name": "Alpha", "status": "ACTIVE"}, {"id": 2, "name": "Beta", "status": "ACTIVE"},
               {"id": 3, "name": "Gamma", "status": "ACTIVE"})
    b_removed = dict(b, status="REMOVED", removalDate="2026-03-01T10:00:00.000Z", adjustmentFactor=12.5)
    msgs = [
        {"pt": T_OFF - 100 * DAY, "mc": [{"id": "1.1", "marketDefinition": _defn("OPEN", [a, b])}]},
        {"pt": T_OFF - 95 * DAY, "mc": [{"id": "1.1", "rc": [{"id": 1, "ltp": 10.0}, {"id": 2, "ltp": 20.0}]}]},
        {"pt": T_OFF - 40 * DAY, "mc": [{"id": "1.1", "marketDefinition": _defn("OPEN", [a, b, c])}]},
        {"pt": T_OFF - 35 * DAY, "mc": [{"id": "1.1", "rc": [{"id": 1, "ltp": 8.0}, {"id": 3, "ltp": 30.0}]}]},
        {"pt": T_OFF - 12 * DAY, "mc": [{"id": "1.1", "marketDefinition": _defn("OPEN", [a, b_removed, c])}]},
        {"pt": T_OFF - 2 * DAY, "mc": [{"id": "1.1", "rc": [{"id": 1, "ltp": 6.0}]}]},
        # suspended for good a day out; an unrelated WIN market in the same file is left out
        {"pt": T_OFF - DAY, "mc": [{"id": "1.1", "marketDefinition": _defn("SUSPENDED", [a, b_removed, c])},
                                   {"id": "1.2", "marketDefinition": _defn("OPEN", [a], market_type="WIN")}]},
        {"pt": T_OFF + DAY, "mc": [{"id": "1.1", "marketDefinition": _defn(
            "CLOSED", [dict(a, status="WINNER"), b_removed, dict(c, status="LOSER")])}]},
    ]
    return "\n".join(json.dumps(dict(m, op="mcm")) for m in msgs).encode()


def test_a_market_is_read_with_its_clock_removals_and_marks():
    [m] = p0.parse_stream(_stream())
    row = m.market_row()
    assert row["market_id"] == "1.1" and row["final_status"] == "CLOSED" and row["removed"] == 1
    assert abs(row["hours_closed_before_off"] - 24) < 1e-6 and row["went_in_play"] is None
    assert row["status_path"] == "OPEN > SUSPENDED > CLOSED" and row["def_bspMarket"] == "False"
    r = {x["selection_id"]: x for x in m.runner_rows()}
    assert r[1]["final_status"] == "WINNER" and r[2]["final_status"] == "REMOVED"
    assert abs(r[2]["removed_days_before_off"] - (12 + 5 / 24 + 20 / 1440)) < 1e-6   # 1 Mar 10:00 to 13 Mar 15:20
    assert r[2]["adjustment_factor"] == 12.5
    # 90 days out: Alpha and Beta listed at their first prices; Gamma not yet listed
    assert r[1]["listed_d90"] and r[1]["ltp_d90"] == 10.0 and r[2]["ltp_d90"] == 20.0 and not r[3]["listed_d90"]
    # 7 days out: Beta removed; Alpha at 8, Gamma at 30
    assert not r[2]["listed_d7"] and r[1]["ltp_d7"] == 8.0 and r[3]["ltp_d7"] == 30.0
    # 6 hours out the market had closed (suspended a day before): nobody is listed
    assert not any(x["listed_h6"] for x in r.values())


def test_the_bundle_is_streamed_and_unreadable_files_counted(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in (("BASIC/2026/Mar/1/9/1.1.bz2", bz2.compress(_stream())),
                           ("BASIC/2026/Mar/1/9/1.3.bz2", b"BZh9 corrupt")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    buf.seek(0)
    mk, rn, rep = p0.stream_bundle(buf, chunk=1)
    assert rep["files"] == 2 and rep["unreadable"] == 1 and rep["files_with_antepost"] == 1
    assert list(mk["market_id"]) == ["1.1"] and len(rn) == 3


def test_the_race_day_join_tells_ran_ran_elsewhere_and_no_run(tmp_path):
    [m] = p0.parse_stream(_stream())
    mk, rn = pd.DataFrame([m.market_row()]), pd.DataFrame(m.runner_rows())
    db = tmp_path / "h.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE betfair_prices (race_date TEXT, market_type TEXT, event_id INTEGER, event_name TEXT, "
                "track_bf TEXT, event_dt TEXT, selection_id INTEGER, bsp REAL, win_lose REAL)")
    con.executemany("INSERT INTO betfair_prices VALUES (?,?,?,?,?,?,?,?,?)", [
        ("2026-03-13", "win", 500, "Chelt 13th Mar", "Cheltenham", "", 1, 5.0, 1.0),     # Alpha in the race
        ("2026-03-13", "win", 500, "Chelt 13th Mar", "Cheltenham", "", 4, 9.0, 0.0),     # a horse not in the market
        ("2026-03-13", "win", 501, "Chelt 13th Mar", "Cheltenham", "", 3, 15.0, 0.0),    # Gamma in another race
        ("2026-03-13", "place", 500, "Chelt 13th Mar", "Cheltenham", "", 2, 3.0, 0.0),   # place rows ignored
    ])
    con.commit()
    con.close()
    j, rep = p0.race_day_join(mk, rn, str(db))
    lab = dict(zip(j["selection_id"], j["label"]))
    assert lab == {1: "ran", 2: "no_run", 3: "ran_elsewhere"}
    assert j.set_index("selection_id").loc[1, "bsp"] == 5.0 and pd.isna(j.set_index("selection_id").loc[3, "bsp"])
    s = p0.summarise_historic(mk, j, rep)
    assert s["races_joined"] == 1 and s["label"] == {"ran": 1, "no_run": 1, "ran_elsewhere": 1}
    d90 = s["by_mark"]["d90"]
    assert d90["listed"] == 2 and d90["ran_share"] == 0.5
    # Alpha backed at 10 laid at 5: 10/5; Beta lost: 0; mean return (2 + 0) / 2 - 1 = 0
    assert d90["ran_price_vs_bsp"]["n"] == 1 and d90["market_whole_back_to_bsp"]["mean_return"] == 0.0
    assert p0.race_day_join(mk, rn, str(tmp_path / "none.db"))[1]["skipped"].startswith("no database")


def test_the_bundle_join_reads_the_race_from_the_same_event():
    [m] = p0.parse_stream(_stream())
    mk, rn = pd.DataFrame([m.market_row()]), pd.DataFrame(m.runner_rows())
    win = pd.DataFrame([
        {"event_id": "9", "market_id": "1.50", "selection_id": 1, "runner_status": "WINNER", "won": 1, "bsp": 5.0},
        {"event_id": "9", "market_id": "1.50", "selection_id": 3, "runner_status": "REMOVED", "won": None, "bsp": None},
        {"event_id": "9", "market_id": "1.50", "selection_id": 7, "runner_status": "LOSER", "won": 0, "bsp": 4.0},
        {"event_id": "9", "market_id": "1.51", "selection_id": 2, "runner_status": "LOSER", "won": 0, "bsp": 9.0},
        {"event_id": "8", "market_id": "1.60", "selection_id": 2, "runner_status": "LOSER", "won": 0, "bsp": 9.0},
    ])
    j, rep = p0.bundle_join(mk, rn, win)
    lab = dict(zip(j["selection_id"], j["label_b"]))
    assert lab == {1: "ran", 2: "ran_elsewhere", 3: "declared_nr"} and rep["events"] == 1
    j, check = p0.combine_labels(j)                       # no betfair_prices labels: the bundle's stand alone
    assert set(j["label_source"]) == {"bundle"} and check["both_found"] == 0
    assert j.set_index("selection_id").loc[1, "bsp"] == 5.0 and j.set_index("selection_id").loc[1, "won"] == 1.0
    assert p0.bundle_join(mk, rn, pd.DataFrame())[1] == {"skipped": "no WIN tables"}
