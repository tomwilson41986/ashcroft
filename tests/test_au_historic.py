"""Gate A's importer (au/historic.py) keeps WIN markets only. The stream lines are format fixtures in Betfair's
Historic Data layout, written for the test; no figure from them is used anywhere."""

import bz2
import json

from au.historic import _win_rows


def _file(market_type):
    md = {"eventId": "1", "eventTypeId": "7", "marketType": market_type, "name": "R1 1200m Hcap", "countryCode": "AU",
          "venue": "Flemington", "marketTime": "2025-11-04T04:00:00.000Z", "bspMarket": True, "status": "CLOSED",
          "runners": [{"id": 11, "name": "1. Alpha", "status": "WINNER", "bsp": 4.2, "sortPriority": 1},
                      {"id": 12, "name": "2. Beta", "status": "LOSER", "bsp": 3.1, "sortPriority": 2}]}
    lines = [{"op": "mcm", "pt": 1762228800000, "mc": [{"id": "1.250000001", "marketDefinition": md}]}]
    return bz2.compress("\n".join(json.dumps(x) for x in lines).encode())


def test_only_win_markets_are_kept():
    _, rows = _win_rows(("a.bz2", _file("WIN")))
    assert len(rows) == 2 and {r["runner_name"] for r in rows} == {"1. Alpha", "2. Beta"}
    assert rows[0]["bsp"] in (4.2, 3.1) and "ltp_t_60" in rows[0]
    assert _win_rows(("b.bz2", _file("PLACE")))[1] == []
    assert _win_rows(("c.bz2", b"BZh9 corrupt"))[1] is None and _win_rows(("d.bz2", b"no market here"))[1] == []


def test_a_bundle_cut_short_is_read_to_the_cut(tmp_path):
    import io
    import tarfile
    from au.historic import stream_tar
    whole = tmp_path / "whole.tar"
    with tarfile.open(whole, "w") as t:
        for i, mt in enumerate(["WIN", "WIN", "PLACE"]):
            data = _file(mt)
            info = tarfile.TarInfo(f"BASIC/2025/Nov/4/1/1.25000000{i}.bz2")
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    raw = whole.read_bytes()
    cut = tmp_path / "cut.tar"
    cut.write_bytes(raw[: 512 * 3 + 100])                 # inside the second file
    tables, rep = stream_tar(str(cut))
    assert rep["truncated"]["files_read"] == 1 and rep["win_rows"] == 2
    assert 2025 in tables and len(tables[2025]) == 2
