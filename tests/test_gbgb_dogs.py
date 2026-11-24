"""sources/gbgb_dogs.py with GBGB mocked: a rerun skips the dogs saved that day, the requests keep to the rate limit, a
failed fetch is a line with its status and the run carries on, a 5xx is retried, the day's ids come from the live
predictions file, and the saved files read back as a point-in-time table. Planted records, no network."""
import gzip
import json
from datetime import date

import pandas as pd
import requests

from sources import gbgb_dogs as gd
from sources.common import Store

DAY = date(2026, 11, 14)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.headers, self.text = status, body, {}, json.dumps(body)

    def json(self):
        return self._body


class FakeGBGB:
    """Answers /api/dogs/<id> from ``plan`` (id -> list of statuses, one a call, the last repeated)."""

    def __init__(self, clock, plan=None, trainer="Planted Trainer"):
        self.clock, self.plan, self.calls, self.trainer = clock, plan or {}, [], trainer

    def get(self, url, timeout=None):
        dog = int(url.rstrip("/").split("/")[-1])
        assert url == f"https://api.gbgb.org.uk/api/dogs/{dog}"
        self.calls.append((self.clock(), dog))
        seq = self.plan.get(dog, [200])
        st = seq[min(len([c for c in self.calls if c[1] == dog]) - 1, len(seq) - 1)]
        if st == "raise":
            raise requests.ConnectionError("planted fault")
        if st != 200:
            return Resp(st, {"code": st, "message": "Not Found" if st == 404 else "Server Error"})
        return Resp(200, {"dogId": dog, "dogName": f"Planted {dog}", "trainerName": self.trainer,
                          "ownerName": "Planted Owner", "dogSeason": "24.My.26", "dogSex": "b", "dogBorn": "Jun-2024"})


def run(store, ids, fake, clock, day=DAY, **kw):
    return gd.save_day(store, day, ids, s=fake, limiter=gd.RateLimiter(2.0, clock=clock, sleep=clock.sleep),
                       sleep=clock.sleep, **kw)


def lines(tmp_path, day=DAY):
    return [json.loads(x) for x in gzip.decompress((tmp_path / gd.day_key(day)).read_bytes()).decode().splitlines()]


def test_saved_once_and_a_rerun_skips_them(tmp_path):
    store, clock = Store(root=tmp_path), Clock()
    fake = FakeGBGB(clock)
    r = run(store, [1, 2, 3], fake, clock)
    assert r["fetched"] == 3 and r["status"] == {"200": 3}
    got = lines(tmp_path)
    assert [x["dog_id"] for x in got] == [1, 2, 3]
    assert set(got[0]) == {"fetched_utc", "dog_id", "status", "body"} and got[0]["fetched_utc"].endswith("Z")
    r2 = run(store, [1, 2, 3, 4], fake, clock)
    assert r2["already_saved"] == 3 and r2["fetched"] == 1
    assert [c[1] for c in fake.calls] == [1, 2, 3, 4]
    assert len(lines(tmp_path)) == 4


def test_the_rate_limit_and_the_cap(tmp_path):
    store, clock = Store(root=tmp_path), Clock()
    fake = FakeGBGB(clock)
    r = run(store, list(range(10, 30)), fake, clock, max_dogs=12)
    assert r["fetched"] == 12 and r["left_for_next_run"] == 8
    t = [c[0] for c in fake.calls]
    assert all(b - a >= 0.5 - 1e-9 for a, b in zip(t, t[1:]))       # at most 2 requests a second
    assert len(fake.calls) == 12


def test_a_failed_fetch_is_recorded_and_the_run_carries_on(tmp_path):
    store, clock = Store(root=tmp_path), Clock()
    fake = FakeGBGB(clock, plan={5: [404], 6: ["raise"], 7: [503, 200], 8: [500]})
    r = run(store, [5, 6, 7, 8, 9], fake, clock)
    got = {x["dog_id"]: x for x in lines(tmp_path)}
    assert got[5]["status"] == 404 and got[6]["status"] == 0 and "planted fault" in got[6]["body"]
    assert got[7]["status"] == 200                                    # retried after the 503
    assert got[8]["status"] == 500 and got[9]["status"] == 200
    assert r["status"] == {"404": 1, "0": 1, "200": 2, "500": 1}
    assert sum(1 for c in fake.calls if c[1] == 8) == 4               # every try used, with waits between
    # the rerun asks again only for the dogs without an answer (the fault and the 500), not the 404 or the 200s
    fake2 = FakeGBGB(clock)
    run(store, [5, 6, 7, 8, 9], fake2, clock)
    assert sorted(c[1] for c in fake2.calls) == [6, 8]


def test_the_days_ids_from_the_live_predictions(tmp_path):
    store = Store(root=tmp_path)
    store.put(gd.PREDICTIONS.format(day="2026-11-14"),
              b"race_id,trap,dog_id,dog_name\n-1,1,660182,A\n-1,2,-555,B\n-2,1,660182,A\n-2,3,650167,C\n")
    assert gd.day_dog_ids(store, DAY) == ([650167, 660182], "predictions")


def test_the_catalogue_fallback_matches_names(tmp_path):
    store = Store(root=tmp_path)
    store.put_parquet("gbgb/runs_2026.parquet", pd.DataFrame({
        "dog_id": [11, 12, 13], "dog_name": ["Planted One", "Planted Two", "Planted One"],
        "race_date": ["2026-01-01", "2026-02-01", "2026-03-01"]}))
    markets = pd.DataFrame({"market_type": ["WIN", "WIN", "PLACE"], "country": ["GB", "GB", "GB"],
                            "runner_name": ["1. Planted One", "2. Unknown Dog", "3. Planted Two"]})
    assert gd.ids_from_catalogue(store, DAY, markets=markets) == [13]


def test_the_point_in_time_table(tmp_path):
    store, clock = Store(root=tmp_path), Clock()
    run(store, [1, 2], FakeGBGB(clock, trainer="Old Trainer"), clock, day=date(2026, 11, 14))
    run(store, [1], FakeGBGB(clock, plan={2: [404]}, trainer="New Trainer"), clock, day=date(2026, 11, 16))
    run(store, [2], FakeGBGB(clock, plan={2: [404]}), clock, day=date(2026, 11, 16))
    t = gd.table(store)
    assert list(t.columns[:3]) == ["dog_id", "day", "fetched_utc"]
    assert len(t) == 3 and {"trainer", "owner", "season"} <= set(t.columns)
    assert t.set_index(["dog_id", "day"]).loc[(1, "2026-11-16"), "trainer"] == "New Trainer"
    keys = pd.DataFrame({"dog_id": [1, 1, 1, 2], "race_date": ["2026-11-13", "2026-11-15", "2026-11-16", "2026-11-20"]})
    pit = gd.as_of(t, keys)
    assert pit.trainer.isna().iloc[0]                                 # nothing saved by then: nothing, no leak
    assert list(pit.trainer.iloc[1:]) == ["Old Trainer", "New Trainer", "Old Trainer"]
    assert list(pit.race_date) == list(keys.race_date)
