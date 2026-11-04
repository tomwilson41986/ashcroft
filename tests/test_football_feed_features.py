"""Order-book features from the football feed: the book now, the flow, the path, the label; never a later read."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from football import exchange_feed as ef
from football import feed_features as ff

T0 = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
START = T0 + timedelta(hours=3)


def raw(mid, back, lay, traded):
    return {"marketId": mid, "status": "OPEN", "inplay": False, "totalMatched": 100.0, "runners": [
        {"selectionId": 11, "status": "ACTIVE", "totalMatched": sum(traded.values()),
         "ex": {"availableToBack": [{"price": back, "size": 100.0}, {"price": round(back - 0.02, 2), "size": 50.0}],
                "availableToLay": [{"price": lay, "size": 40.0}],
                "tradedVolume": [{"price": p, "size": v} for p, v in traded.items()]}}]}


def book_frame():
    prev, rows = {}, []
    path = [(0, 2.10, 2.14, {2.1: 100}), (30, 2.06, 2.08, {2.1: 100, 2.06: 40}),
            (60, 2.02, 2.04, {2.1: 100, 2.06: 40, 2.02: 80}), (90, 2.00, 2.02, {2.1: 100, 2.06: 40, 2.02: 80, 2.0: 20}),
            (170, 1.96, 1.98, {2.1: 100, 2.06: 40, 2.02: 80, 2.0: 20, 1.96: 300})]
    for m, b, l, tv in path:
        rows += ef.book_rows([raw("1.1", b, l, tv)], T0 + timedelta(minutes=m), 3, prev)
    return pd.DataFrame(rows)


def test_the_book_flow_and_path_at_a_moment():
    b = ff.prepare_books(book_frame())
    f = ff.features_at(b, T0 + timedelta(minutes=95), START).iloc[0]
    assert f["back1"] == 2.00 and f["lay1"] == 2.02 and f["spread_ticks"] == 1
    assert f["depth_back"] == 150 and f["depth_lay"] == 40 and f["imbalance"] == pytest.approx(110 / 190)
    assert f["traded_60m"] == pytest.approx(80 + 20)                    # reads at 60 and 90 minutes
    assert f["mid_logret_60m"] == pytest.approx(np.log(2.07 / 2.01))    # mid at 30 min (2.07) to now (2.01)
    assert f["minutes_to_off"] == pytest.approx(85) and f["read_age_min"] == pytest.approx(5)


def test_features_never_read_a_later_book():
    b = ff.prepare_books(book_frame())
    t = T0 + timedelta(minutes=95)
    a = ff.features_at(b, t, START)
    b2 = b.copy()
    later = b2["polled"] > t
    b2.loc[later, ["back1", "lay1", "mid", "depth_back", "traded_new"]] = [9.0, 9.2, 9.1, 1.0, 1e6]
    pd.testing.assert_frame_equal(a, ff.features_at(b2, t, START))


def test_the_label_runs_to_the_last_pre_play_read():
    b = ff.prepare_books(book_frame())
    f = ff.labels(b, ff.features_at(b, T0 + timedelta(minutes=95), START)).iloc[0]
    assert f["y_mid_logret_to_close"] == pytest.approx(np.log(2.01 / 1.97))
    assert f["y_back_now_lay_close"] == pytest.approx(2.00 / 1.98 - 1)


def test_a_day_is_built_from_the_feeds_files(tmp_path):
    w = ef.DayWriter(store=None, root=tmp_path, depth=3)
    b = book_frame()
    day_of = {"1.1": "2026-11-07"}
    for _, g in b.groupby("polled_utc"):
        w.books(g.to_dict("records"), day_of, datetime.fromisoformat(g["polled_utc"].iloc[0].replace("Z", "+00:00")))
    w.markets([{"market_id": "1.1", "market_type": "MATCH_ODDS", "market_start_utc": START.isoformat(),
                "selection_id": 11, "runner_name": "Arsenal"}], day_of)
    out = ff.build_day(None, "2026-11-07", every_min=30, root=tmp_path)
    assert len(out) >= 5 and {"spread_ticks", "traded_60m", "y_mid_logret_to_close"} <= set(out.columns)
    assert out["moment_utc"].is_monotonic_increasing


def test_the_feed_throttle_holds_betfair_requests_under_its_ceilings():
    """The owner, 4 Nov 2026 ("use live key ... do not pass betfair thresholds"): no more than the per-second and
    per-hour ceilings, whatever the feed asks; it waits instead and counts every call."""
    from football.exchange_feed import MAX_CALLS_PER_SECOND, Throttle, batch_size
    t = {"now": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        t["now"] += s
    th = Throttle(lambda *a, **k: "ok", per_second=5.0, per_hour=20, clock=lambda: t["now"], sleep=sleep)
    stamps = []
    for _ in range(45):
        assert th("listMarketBook", {}) == "ok"
        stamps.append(t["now"])
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= 0.2 - 1e-9                                  # never above 5 a second
    for i, s in enumerate(stamps):                                   # never more than 20 in any rolling hour
        assert sum(1 for x in stamps if s - 3600.0 < x <= s) <= 20
    assert th.calls == 45 and th.report()["limit_per_hour"] == 20
    assert MAX_CALLS_PER_SECOND <= 5.0
    assert batch_size(5, True) * 20 * 5 / 3 <= 200                     # each request under Betfair's 200 points
