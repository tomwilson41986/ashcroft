"""cricket/ingest/cricsheet.py and cricket/db/loaders.py: rows in equal rows out, and a rerun adds nothing."""

import gzip
import io
import zipfile
from pathlib import Path

from cricket.db.loaders import connect, upsert_tables
from cricket.ingest.cricsheet import build_tables, checks, manifest_counts

FIX = Path(__file__).parent / "fixtures" / "cricket"


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in sorted(FIX.glob("*.json.gz")):
            z.writestr(p.name[: -len(".gz")], gzip.decompress(p.read_bytes()))
        z.writestr("README.txt", "not a match")
    return buf.getvalue()


def test_rows_in_equal_rows_out_with_no_duplicates_or_nulls():
    zb = _zip()
    t = build_tables(zb, chunk=7)   # a small chunk exercises the chunked build
    res = checks(t, manifest_counts(zb))
    assert res["pass"], res
    assert len(t["matches"]) == 50
    assert res["reconciliation"].deliveries_in.sum() == len(t["deliveries"])


def test_rerun_adds_zero_rows(tmp_path):
    zb = _zip()
    t = build_tables(zb)
    con = connect(f"sqlite:///{tmp_path / 'c.db'}")
    added, held = upsert_tables(con, t)
    assert added["deliveries"] == len(t["deliveries"]) and added["matches"] == 50
    added2, held2 = upsert_tables(con, build_tables(zb))
    assert set(added2.values()) == {0}
    assert held2 == held
    res = checks(t, manifest_counts(zb), held2)
    assert res["pass"]


def test_fetch_stores_a_file_once_and_skips_an_unchanged_copy(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import cricket.ingest.cricsheet as cs
    from sources.common import Store

    body = {"all_json.zip": _zip(), "people.csv": b"identifier,name\n", "names.csv": b"identifier,name\n"}
    monkeypatch.setattr(cs, "get", lambda s, url, timeout=0: SimpleNamespace(
        content=body[next(k for k, u in cs.RAW_FILES.items() if u == url)]))
    store = Store(root=tmp_path)
    first = cs.fetch(store, "2026-11-03")
    assert all(v.endswith("bytes") for v in first.values())
    assert cs.fetch(store, "2026-11-03") == {k: "kept" for k in cs.DEFAULT_FILES}
    again = cs.fetch(store, "2026-11-04")
    assert again == {k: "unchanged since 2026-11-03" for k in cs.DEFAULT_FILES}
    assert not (tmp_path / cs.RAW_PREFIX / "2026-11-04").exists()
    assert cs.latest_raw_day(store) == "2026-11-03"
    # the build reads the newest copy held; the raw file is byte-identical to what was downloaded
    assert store.get(f"{cs.RAW_PREFIX}/2026-11-03/all_json.zip") == body["all_json.zip"]


def test_an_html_page_in_place_of_a_file_is_never_stored(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import cricket.ingest.cricsheet as cs
    from sources.common import Store

    html = b"<!DOCTYPE html><html><head>"
    monkeypatch.setattr(cs, "get", lambda s, url, timeout=0: SimpleNamespace(content=html))
    out = cs.fetch(Store(root=tmp_path), "2026-11-04", ["people.csv", "all_json.zip"], tries=2, pause=0)
    assert all(v.startswith("not stored") for v in out.values())
    assert not (tmp_path / cs.RAW_PREFIX / "2026-11-04").exists()
    assert cs.looks_valid("people.csv", b"identifier,name\n") and cs.looks_valid("all_json.zip", _zip())


def test_player_map_full_names_initials_and_ambiguity():
    import pandas as pd

    from cricket.ingest.player_map import PlayerMap

    # real Cricsheet register rows (people.csv, names.csv of 2026-11-04)
    register = pd.DataFrame({"identifier": ["ba607b88", "0a476045", "3a04b1c0"],
                             "name": ["V Kohli", "S Dhawan", "S Kohli"],
                             "unique_name": ["V Kohli", "S Dhawan", "S Kohli"]})
    names = pd.DataFrame({"identifier": ["ba607b88", "0a476045"], "name": ["Virat Kohli", "Shikhar Dhawan"]})
    players = pd.DataFrame({"match_id": ["m1"] * 3, "team": ["India"] * 3,
                            "player_id": ["ba607b88", "0a476045", "3a04b1c0"]})
    matches = pd.DataFrame({"match_id": ["m1"], "start_date": ["2024-01-01"]})
    pm = PlayerMap(register, names, players, matches)
    assert pm.find("Virat Kohli", "India") == "ba607b88"          # Cricsheet's full name
    assert pm.find("VIRAT KOHLI", "India") == "ba607b88"
    assert pm.find("Shikhar  Dhawan", "India") == "0a476045"
    assert pm.find("Sahil Kohli", "India") == "3a04b1c0"          # first initial and surname, one candidate
    assert pm.find("Virat Kohli", "Australia") is None             # never played for the side
    ids, missing = pm.xi(["Virat Kohli", "Nobody Here"], "India")
    assert ids == ["ba607b88", None] and missing == ["Nobody Here"]
