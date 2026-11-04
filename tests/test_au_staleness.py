"""The staleness check's age of the T-60 price (au/staleness.py). Format fixtures written for the test; no figure from
them is used anywhere."""

import bz2
import json

from au.staleness import _ages


def test_the_age_is_minutes_from_the_last_price_change_to_the_mark():
    off = 1762228800000                                   # the scheduled off, ms
    md = {"marketType": "WIN", "countryCode": "AU", "marketTime": "2025-11-04T04:00:00.000Z",
          "runners": [{"id": 1, "status": "ACTIVE"}, {"id": 2, "status": "ACTIVE"}]}
    lines = [{"op": "mcm", "pt": off - 3 * 3600_000, "mc": [{"id": "1.9", "marketDefinition": md,
                                                             "rc": [{"id": 1, "ltp": 5.0}, {"id": 2, "ltp": 3.0}]}]},
             {"op": "mcm", "pt": off - 63 * 60_000, "mc": [{"id": "1.9", "rc": [{"id": 1, "ltp": 4.6}]}]},
             {"op": "mcm", "pt": off - 30 * 60_000, "mc": [{"id": "1.9", "rc": [{"id": 1, "ltp": 4.2}]}]}]
    raw = bz2.compress("\n".join(json.dumps(x, separators=(",", ":")) for x in lines).encode())
    rows = {r["selection_id"]: r for r in _ages(("a.bz2", raw))}
    assert rows[1]["ltp_t_60_seen"] == 4.6 and rows[1]["age_t_60_min"] == 3.0
    assert rows[2]["ltp_t_60_seen"] == 3.0 and rows[2]["age_t_60_min"] == 120.0
