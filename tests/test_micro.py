"""research/micro (the high-frequency track's Stage 1) on a hand-built Exchange Stream: the ladder in ticks, the side
of each trade, the window before the off, the feature signs, no look-ahead in features or targets, and the
walk-forward scoring. Fixtures only: the models are evaluated on the recorded stream (research/queries)."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.micro import events, features, models  # noqa: E402

OFF = datetime(2026, 11, 5, 19, 0, tzinfo=timezone.utc)


def _line(t: datetime, mc: dict) -> str:
    return json.dumps({"rt": t.isoformat(), "m": json.dumps({"op": "mcm", "pt": int(t.timestamp() * 1000),
                                                             "mc": [mc]})})


def _defn(in_play=False, status="OPEN"):
    return {"marketTime": OFF.isoformat().replace("+00:00", "Z"), "marketType": "WIN", "status": status,
            "inPlay": in_play, "venue": "Romford",
            "runners": [{"id": 1, "status": "ACTIVE"}, {"id": 2, "status": "ACTIVE"}]}


def _stream():
    """Market 1.5: an image 400 s out (outside the window), then changes inside the last 300 s, then in play."""
    t0 = OFF - timedelta(seconds=400)
    L = [_line(t0, {"id": "1.5", "img": True, "marketDefinition": _defn(), "rc": [
        {"id": 1, "atb": [[3.0, 50]], "atl": [[3.05, 80]], "trd": [[3.0, 100]]},
        {"id": 2, "atb": [[2.0, 60]], "atl": [[2.02, 40]], "trd": [[2.0, 50]]}]})]
    t = OFF - timedelta(seconds=200)
    L.append(_line(t, {"id": "1.5", "rc": [{"id": 1, "trd": [[3.0, 130]], "atb": [[3.0, 20]]}]}))       # backers take 30
    L.append(_line(t + timedelta(seconds=1), {"id": "1.5", "rc": [{"id": 1, "trd": [[3.05, 25]]}]}))       # layers take 25
    L.append(_line(t + timedelta(seconds=2), {"id": "1.5", "rc": [{"id": 1, "atb": [[3.0, 0], [2.98, 70]],
                                                                    "atl": [[3.05, 0], [3.0, 30]]}]}))  # shortens
    L.append(_line(t + timedelta(seconds=8), {"id": "1.5", "rc": [{"id": 1, "atl": [[3.0, 60]]}]}))
    L.append(_line(t + timedelta(seconds=20), {"id": "1.5", "rc": [{"id": 1, "atb": [[3.05, 40], [2.98, 0]],
                                                                     "atl": [[3.0, 0], [3.1, 50]]}]}))  # drifts
    L.append(_line(OFF + timedelta(seconds=5), {"id": "1.5", "marketDefinition": _defn(in_play=True)}))
    L.append(_line(OFF + timedelta(seconds=6), {"id": "1.5", "rc": [{"id": 1, "atb": [[3.5, 10]]}]}))
    return L


def test_the_ladder_in_ticks():
    ti = events.tick_index
    assert ti(1.02) - ti(1.01) == pytest.approx(1)
    assert ti(2.02) - ti(2.0) == pytest.approx(1) and ti(2.0) - ti(1.99) == pytest.approx(1)
    assert ti(3.05) - ti(3.0) == pytest.approx(1) and ti(6.2) - ti(6.0) == pytest.approx(1)
    assert ti(20.0) - ti(10.0) == pytest.approx(20) and np.isnan(ti(None))          # 10 to 20 in 0.5s


def test_rows_keep_the_window_and_stop_at_the_off():
    rows = events.book_rows(_stream())
    assert len(rows) == 5 and (rows.to_off <= 300).all() and (rows.to_off > 0).all()
    assert set(rows.sid) == {1}


def test_the_side_of_each_trade():
    rows = events.book_rows(_stream())
    assert (rows.iloc[0].buy, rows.iloc[0].sell) == (30.0, 0.0)          # traded at the best back: backers took it
    assert (rows.iloc[1].buy, rows.iloc[1].sell) == (0.0, 25.0)          # at the best lay: layers took it


def test_feature_signs():
    f = features.build(events.book_rows(_stream()))
    r = f.iloc[0]                                                       # back 3.0 for 20, lay 3.05 for 80
    assert r.spread_ticks == pytest.approx(1)
    assert r.qi == pytest.approx((80 - 20) / 100)                       # the bid (lay side) is heavier: up pressure
    assert r.micro_dev > 0                                              # the microprice leans toward the ask
    assert f.iloc[0].flow_1s > 0 and f.iloc[1].flow_1s < f.iloc[0].flow_1s


def test_targets_look_forward_and_features_do_not():
    f = features.build(events.book_rows(_stream()))
    pm = f.pm.to_numpy()
    assert pm[2] > pm[1] and pm[4] < pm[3]                              # shortened, then drifted
    assert f.y_next.iloc[0] == 1 and f.y_next.iloc[2] == -1             # the next move from each row
    assert np.isnan(f.y_next.iloc[4])                                   # no later move
    assert f.y_1s.iloc[1] == 1 and f.y_5s.iloc[2] == 0                  # up within 1 s of row 1; flat 5 s after row 2
    assert np.isnan(f.y_10s.iloc[4])                                    # past the runner's last row
    rows = events.book_rows(_stream())
    cut = features.build(rows.iloc[:3])                                 # the same rows with the future removed
    for c in features.FEATURES:
        a, b = f[c].iloc[:3].to_numpy(float), cut[c].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True), c


def _synthetic(n_days=4, n=400, seed=0):
    """Rows where queue imbalance drives the next move: a model reading qi must beat the baseline."""
    rng = np.random.default_rng(seed)
    out = []
    for d in range(n_days):
        qi = rng.uniform(-1, 1, n)
        y = np.where(rng.uniform(size=n) < 0.5 + 0.4 * qi, 1, -1)
        df = pd.DataFrame({"market": f"1.{d}", "sid": np.arange(n) % 5, "t": pd.Timestamp("2026-11-05", tz="UTC")
                           + pd.to_timedelta(np.arange(n), unit="s"), "day": f"2026-11-0{d + 5}", "qi": qi,
                           "spread_ticks": 1.0, "to_off": rng.uniform(0, 300, n), "y_next": y.astype(float)})
        for c in features.FEATURES:
            if c not in df:
                df[c] = rng.normal(size=n)
        df["pm"], df["micro_dev"] = 0.3, qi / 2
        df["bb"], df["bl"] = 3.0, 3.05
        out.append(df)
    return pd.concat(out, ignore_index=True)


def test_walk_forward_scores_against_the_baseline():
    res = models.walk_forward(_synthetic(), "y_next")
    assert set(res.model) == set(models.MODELS) and res.day.nunique() == 2      # days 3 and 4, after 2 training days
    base = res[res.model == "baseline"]
    assert (base.gain_mnats == 0).all()
    for m in ("qi_logit", "markov", "gbm", "kalman"):
        assert res[res.model == m].gain_mnats.mean() > 0, m                      # qi carries the signal
    mean, lo, hi = models.day_interval(res[res.model == "qi_logit"].gain_mnats)
    assert lo <= mean <= hi
