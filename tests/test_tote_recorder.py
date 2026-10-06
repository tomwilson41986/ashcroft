"""The Tote recorder on made-up answers (code paths only; nothing here is a Tote pool or reported)."""

import gzip
import json
from datetime import date, datetime, timedelta, timezone

import tote_recorder as tr

KEY = "PLACEHOLDERKEY123"
API = "https://api.production.racing.tote.co.uk/"
CONFIG = f'window.env = {{\n  REACT_APP_TOTE_API_KEY: "{KEY}",\n  REACT_APP_TOTE_API_URL: "{API}",\n}};'
OFF = datetime(2026, 10, 6, 13, 0, tzinfo=timezone.utc)


def _pool(pid, name, rid, country="GB", post=OFF, legs=1):
    return {"id": pid, "total": 1000 + pid, "poolType": {"name": name}, "meeting": {"name": "X", "countryCode": country},
            "betTypes": [{"specials": [1, 2, 3]}],
            "legs": [{"raceId": rid + i, "racePostTime": post.isoformat().replace("+00:00", "Z")} for i in range(legs)]}


POOLS = [_pool(1, "WIN", 7), _pool(2, "PLACE", 7), _pool(3, "EXACTA", 7), _pool(4, "PLACEPOT", 7, legs=6),
         _pool(5, "WIN", 8, country="US"), _pool(6, "Swinger", 7)]
RUNNER = {"id": 11, "name": "A", "programNumber": "1", "baseWinStake": 4.9, "winstake": 5.5, "decimalShowPrice": 550,
          "silk": "https://img/x", "spotlight": "long text", "extraData": {"draw": 3, "spotlight": "long"}}


def test_races_and_slim_pools():
    races = tr.races_from_pools(POOLS)
    assert list(races) == [7]
    assert races[7]["pools"] == {"WIN": 1, "PLACE": 2, "EXACTA": 3, "SWINGER": 6} and races[7]["post"] == OFF
    assert set(tr.races_from_pools(POOLS, countries=())) == {7, 8}
    p = dict(_pool(1, "WIN", 7))
    p["legs"][0]["race"] = {"id": 7, "runners": [RUNNER]}
    s = tr.slim_pool(p)
    assert "betTypes" not in s
    r = s["legs"][0]["race"]["runners"][0]
    assert r == {"id": 11, "name": "A", "programNumber": "1", "decimalShowPrice": 550, "winstake": 5.5,
                 "baseWinStake": 4.9, "draw": 3}


def _recorder(tmp_path, now):
    rec = tr.ToteRecorder(date(2026, 10, 6), tmp_path, OFF + timedelta(hours=1), clock=lambda: now[0],
                          sleep=lambda s: None, pause=0)
    rec.races = tr.races_from_pools(POOLS)
    return rec


def test_marks_fall_due_once_and_a_late_start_skips_the_old_ones(tmp_path):
    now = [OFF - timedelta(minutes=60)]
    rec = _recorder(tmp_path, now)
    pools, results = rec.due(now[0])
    assert sorted((n, m) for _, n, _, m in pools) == [("PLACE", -60), ("WIN", -60)] and results == []
    assert rec.due(now[0]) == ([], [])                       # once
    now[0] = OFF - timedelta(minutes=5)
    pools, _ = rec.due(now[0])
    assert sorted((n, m) for _, n, _, m in pools) == [("EXACTA", -5), ("PLACE", -5), ("SWINGER", -5), ("WIN", -5)]
    late = _recorder(tmp_path, [OFF - timedelta(minutes=10)])   # a restart at T-10: T-60, -30, -15 passed over
    pools, _ = late.due(OFF - timedelta(minutes=10))
    assert sorted({m for *_, m in pools}) == [-10]
    now[0] = OFF + timedelta(minutes=10)
    _, results = rec.due(now[0])
    assert results == [(7, 10)]


class _Resp:
    def __init__(self, url, body, status=200):
        self.url, self.status_code, self.ok = url, status, status < 400
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


class _Session:
    def __init__(self):
        self.headers, self.asked = {}, []

    def get(self, url, timeout=None):
        self.asked.append(url)
        if url == "https://tote.co.uk/config.js":
            return _Resp(url, CONFIG)
        if url == f"{API}race-card/pools/today?key={KEY}":
            return _Resp(url, {"pools": POOLS})
        if url.startswith(f"{API}race-card/pool/") and url.endswith(f"?key={KEY}"):
            pid = int(url.split("/")[-1].split("?")[0])
            p = dict(next(p for p in POOLS if p["id"] == pid))
            p["legs"] = [dict(p["legs"][0], race={"id": 7, "runners": [RUNNER]})]
            return _Resp(url, p)
        if url.startswith(f"{API}race-card/race-results/") and "key=" not in url:
            return _Resp(url, {"results": [{"raceId": 7, "dividends": [{"pool": "WIN", "amount": 4.8}]}]})
        return _Resp(url, "no", status=404)


def test_a_day_is_recorded_without_the_key(tmp_path):
    t = [OFF - timedelta(minutes=61)]

    def sleep(s):
        t[0] += timedelta(seconds=max(s, 1))
    session = _Session()
    rec = tr.ToteRecorder(date(2026, 10, 6), tmp_path, OFF + timedelta(minutes=35), session=session,
                          clock=lambda: t[0], sleep=sleep, pause=0, upload_every=10 ** 9)
    s = rec.run()
    assert s["started"] and s["races"] == 1
    assert s["detail"] == 2 * len(tr.WIN_PLACE_MARKS) + 2 * len(tr.EXOTIC_MARKS)
    assert s["results"] == len(tr.RESULT_MARKS) + 1          # T+10, T+30 and the day's end
    assert s["pools"] >= 9                                  # every 10 minutes for 96 minutes
    day = tmp_path / "2026-10-06"
    for name in ("pools", "detail", "results"):
        lines = gzip.open(day / f"{name}.jsonl.gz", "rt").read().splitlines()
        assert lines and all(KEY not in line for line in lines)
    detail = [json.loads(x) for x in gzip.open(day / "detail.jsonl.gz", "rt")]
    win = [d for d in detail if d["pool"] == "WIN"]
    assert [d["mark"] for d in win] == list(tr.WIN_PLACE_MARKS)
    assert win[0]["answer"]["legs"][0]["race"]["runners"][0]["baseWinStake"] == 4.9
    assert all("betTypes" not in d["answer"] for d in detail)
    results = [json.loads(x) for x in gzip.open(day / "results.jsonl.gz", "rt")]
    assert results[0]["answer"]["results"][0]["dividends"][0]["amount"] == 4.8
    assert all(KEY not in u or "race-card/pool" in u or "pools/today" in u for u in session.asked)


def test_a_sample_takes_the_next_races_pools_at_once(tmp_path):
    t = [OFF - timedelta(hours=2)]

    def sleep(s):
        t[0] += timedelta(seconds=max(s, 1))
    rec = tr.ToteRecorder(date(2026, 10, 6), tmp_path, t[0] + timedelta(minutes=2), session=_Session(),
                          clock=lambda: t[0], sleep=sleep, pause=0, upload_every=10 ** 9)
    s = rec.run(sample=2)
    detail = [json.loads(x) for x in gzip.open(tmp_path / "2026-10-06" / "detail.jsonl.gz", "rt")]
    assert [(d["pool"], d["mark"]) for d in detail] == [("WIN", "sample"), ("PLACE", "sample")]
    assert s["detail"] == 2 and s["results"] == 0
