"""gold/path: the price-path features before each decision checkpoint (hand-written stream files through the real
parser for the exact cases; a slow, plain reference over seeded random ticks for the vectorised arithmetic)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from football.warehouse import bronze, gold_path, silver
from football.warehouse.gold_checkpoint import CHECKPOINTS
from football.warehouse.schema import TS, conform
from football.warehouse.store_io import read_part
from sources.common import Store
from tests.test_football_warehouse import KO_MS, MO_RUNNERS, OU_RUNNERS, _bundle, _defn, _file, _msg

H, M = 3_600_000, 60_000
L2 = math.log(2.0)                 # prices 1.25, 1.5, 2, 3, 5 are x = 2, 1, 0, -1, -2 in units of ln 2
HOME, AWAY, DRAW, UNDER, OVER = 71085, 31317, 58805, 47972, 47973
MO, OU, AH = "1.252787713", "1.252787714", "1.252787720"

#: the home side's path up to T-60, then what comes after it (the part a T-60 decision never saw)
EARLY = [(-40 * H, None), (-30 * H, {HOME: 3.0}), (-20 * H, {HOME: 2.0}), (-8 * H, {HOME: 3.0}),
         (-6 * H, {DRAW: 4.0}), (-5 * H, {AWAY: 5.0}), (-4 * H, {HOME: 2.0}), (-150 * M, {HOME: 1.5}),
         (-90 * M, {HOME: 2.0}), (-70 * M, {HOME: 1.5}), (-65 * M, {AWAY: 4.0})]
LATE = [(-30 * M, {HOME: 3.0}), (-5 * M, {HOME: 5.0, DRAW: 3.0})]


def match_odds_file(late=LATE, in_play_price=1.2) -> bytes:
    settled = [dict(r, status=s) for r, s in zip(MO_RUNNERS, ("WINNER", "LOSER", "LOSER"))]
    msgs = [_msg(KO_MS + at, MO, defn=_defn()) if rc is None else
            _msg(KO_MS + at, MO, rc=[{"id": k, "ltp": v} for k, v in rc.items()]) for at, rc in EARLY + list(late)]
    msgs += [_msg(KO_MS + 90_000, MO, defn=_defn(in_play=True, betDelay=5)),
             _msg(KO_MS + 10 * M, MO, rc=[{"id": HOME, "ltp": in_play_price}]),
             _msg(KO_MS + 110 * M, MO, defn=_defn("CLOSED", True, runners=settled, numberOfActiveRunners=0))]
    return _file(msgs)


def over_under_file() -> bytes:
    kw = {"market_type": "OVER_UNDER_25", "name": "Over/Under 2.5 Goals", "runners": OU_RUNNERS}
    return _file([_msg(KO_MS - 40 * H, OU, _defn(**kw)),
                  _msg(KO_MS - 3 * H, OU, rc=[{"id": UNDER, "ltp": 2.0}, {"id": OVER, "ltp": 2.0}]),
                  _msg(KO_MS - 80 * M, OU, rc=[{"id": OVER, "ltp": 1.8}]),
                  _msg(KO_MS + 90_000, OU, _defn(in_play=True, **kw))])


def handicap_file() -> bytes:
    runners = [{"status": "ACTIVE", "sortPriority": 1, "id": HOME, "hc": -0.5, "name": "Sturm Graz"},
               {"status": "ACTIVE", "sortPriority": 2, "id": AWAY, "hc": 0.5, "name": "Brann"}]
    return _file([_msg(KO_MS - 40 * H, AH, _defn(runners=runners, market_type="ASIAN_HANDICAP")),
                  _msg(KO_MS - 3 * H, AH, rc=[{"id": HOME, "hc": -0.5, "ltp": 1.9},
                                              {"id": AWAY, "hc": 0.5, "ltp": 2.0}])])


def path_of(tmp_path, name: str, mo: bytes) -> pd.DataFrame:
    """gold/path of a one-match warehouse built from stream files, through the real parser and the real build."""
    from football.warehouse import gold
    root = tmp_path / name
    root.mkdir()
    store = Store(root=root / "wh")
    bronze.ingest_bundle(store, _bundle(root, {f"2026/Jan/29/35155405/{MO}.bz2": mo,
                                               f"2026/Jan/29/35155405/{OU}.bz2": over_under_file(),
                                               f"2026/Jan/29/35155405/{AH}.bz2": handicap_file()}), "b")
    silver.build(store, workers=1)
    rep = gold.build(store, only=["gold_checkpoint", "gold_path"])
    assert rep["failed"] == [] and rep["modules"]["gold_path"]["built"] == ["2026-01"]
    return read_part(store, "gold/path", 2026, 1).to_pandas()


@pytest.fixture()
def path(tmp_path):
    return path_of(tmp_path, "base", match_odds_file())


def blank(v) -> bool:
    return v != v


def near(v, want) -> bool:
    return abs(v - want) < 1e-12


def test_one_row_a_match_odds_or_over_under_runner_a_decision_checkpoint_it_has_a_price_at(path):
    assert list(path.columns) == gold_path.SPEC.names
    assert set(path["market_type"]) == {"MATCH_ODDS", "OVER_UNDER_25"}            # never the handicap lines
    assert not path.duplicated(["market_id", "selection_id", "handicap", "cp_min"]).any()
    got = {(r, c) for r, c in zip(path["role"], path["cp_min"])}
    # home traded 30 h out: every checkpoint; the draw from T-6h (its first trade is that moment); away from T-2h
    assert got == {("home", c) for c in (1440, 360, 120, 60, 15)} | {("draw", c) for c in (360, 120, 60, 15)} | \
        {("away", c) for c in (120, 60, 15)} | {(r, c) for r in ("over", "under") for c in (120, 60, 15)}
    assert set(gold_path.PATH_CHECKPOINTS) <= set(CHECKPOINTS)


def test_returns_volatility_and_activity_at_t_minus_60(path):
    h = path[(path["role"] == "home") & (path["cp_min"] == 60)].iloc[0]
    assert h["cp_time"].value // 10**6 == KO_MS - 60 * M and near(h["x"], L2)       # 1.5: a chance of two thirds
    # the price as of 75, 90 (a tick exactly there counts), 120, 180, 420 and 1500 minutes before the off
    assert [round(h[f"ret_{k}"] / L2, 12) for k in ("15m", "30m", "1h", "2h", "6h", "24h")] == [1, 1, 0, 1, 2, 2]
    assert near(h["ret_first"], 2 * L2) and h["hours_since_first"] == 29.0
    assert near(h["rv_1h"], math.sqrt(2) * L2) and near(h["rv_6h"], 2 * L2) and near(h["rv_24h"], math.sqrt(6) * L2)
    assert [h[f"n_ticks_{k}"] for k in ("15m", "1h", "6h", "24h")] == [1, 2, 4, 6]


def test_trend_range_jump_and_mean_reversion_at_t_minus_60(path):
    h = path[(path["role"] == "home") & (path["cp_min"] == 60)].iloc[0]
    # the last 6 h, in ln 2: -1 for 3 h, 0 for 1.5 h, 1 for 1 h, 0 for 20 min, 1 for 10 min
    assert near(h["slope_6h"], 503 / 1296 * L2) and near(h["slope_r2_6h"], 253009 / 336528)
    assert near(h["er_6h"], 0.5) and near(h["er_24h"], 2 / 6)                         # net 2 of 4, and of 6, moved
    assert h["range_pos_24h"] == 1.0 and h["drawdown_24h"] == 0.0 and near(h["runup_24h"], 2 * L2)
    # three changes of one ln 2 in the last 2 h: the latest, a shortening 10 minutes before the checkpoint
    assert near(h["jump_abs_2h"], L2) and h["jump_sign_2h"] == 1 and near(h["jump_min_2h"], 10.0)
    assert near(h["twap_dev_1h"], -L2 / 3) and near(h["twap_dev_6h"], -47 / 36 * L2)
    assert near(h["last_dx"], L2) and h["run_len"] == 1
    assert h["tick_imb_1h"] == 0.0 and h["tick_imb_6h"] == 0.5                     # 1 up 1 down; 3 up 1 down
    # changes +1 -1 +1 +1 -1 +1 over 24 h: five pairs, products -1 -1 +1 -1 -1; three pairs in 6 h are too few
    assert blank(h["ac1_6h"]) and near(h["ac1_24h"], -0.6) and near(h["roll_24h"], 2 * math.sqrt(0.6) * L2)
    # a day out: one price, 6 h old: nothing has moved, and there is nothing to form a ratio from
    d = path[(path["role"] == "home") & (path["cp_min"] == 1440)].iloc[0]
    assert d["ret_6h"] == 0 and blank(d["ret_24h"]) and d["slope_24h"] == 0 and blank(d["slope_r2_24h"])
    assert blank(d["er_24h"]) and blank(d["range_pos_24h"]) and d["rv_24h"] == 0 and d["n_ticks_24h"] == 1


def test_a_feature_that_cannot_be_measured_is_blank(path):
    d = path[(path["role"] == "draw") & (path["cp_min"] == 60)].iloc[0]             # one trade, at T-6h
    assert d["ret_1h"] == 0 and d["ret_2h"] == 0 and blank(d["ret_6h"]) and d["ret_first"] == 0
    assert d["hours_since_first"] == 5.0 and (d["n_ticks_1h"], d["n_ticks_6h"]) == (0, 1)
    assert d["slope_6h"] == 0 and blank(d["slope_r2_6h"]) and blank(d["er_6h"]) and blank(d["range_pos_24h"])
    assert (d["jump_abs_2h"], d["jump_sign_2h"]) == (0, 0) and blank(d["jump_min_2h"]) and d["twap_dev_6h"] == 0
    assert blank(d["last_dx"]) and d["run_len"] == 0 and blank(d["tick_imb_6h"]) and blank(d["ac1_24h"])
    first = path[(path["role"] == "draw") & (path["cp_min"] == 360)].iloc[0]         # its first trade is T-6h itself
    assert first["hours_since_first"] == 0 and blank(first["ret_15m"]) and blank(first["slope_6h"])
    assert blank(first["twap_dev_1h"]) and first["n_ticks_15m"] == 1 and first["rv_1h"] == 0
    a = path[(path["role"] == "away") & (path["cp_min"] == 60)].iloc[0]             # 5.0 at T-5h, 4.0 at T-65
    assert near(a["ret_15m"], math.log(4 / 3)) and blank(a["ret_6h"]) and a["hours_since_first"] == 4.0
    assert a["er_6h"] == 1.0 and a["tick_imb_1h"] == 1.0 and a["run_len"] == 1 and near(a["jump_min_2h"], 5.0)


def test_the_book_catch_up_is_what_the_other_runners_did_since_this_one_last_traded(path):
    at60 = path[path["cp_min"] == 60].set_index("role")
    # home last traded at T-70; away then went 5.0 -> 4.0: the book sum rose 0.05, so home's two thirds owes 0.05
    p = 2 / 3 - (1 / 4.0 - 1 / 5.0)
    assert near(at60.loc["home", "book_catchup"], math.log(p / (1 - p)) - L2)
    assert at60.loc["away", "book_catchup"] == 0.0                # the last of the book to trade: nothing to catch
    assert blank(at60.loc["draw", "book_catchup"])                # away had no price when the draw last traded
    # over/under: under last traded at T-3h at evens; over has since shortened from 2.0 to 1.8
    assert near(at60.loc["under", "book_catchup"], math.log((0.5 - (1 / 1.8 - 0.5)) / (0.5 + (1 / 1.8 - 0.5))))
    assert at60.loc["over", "book_catchup"] == 0.0


def test_nothing_published_after_a_checkpoint_changes_its_features(tmp_path, path):
    """The point-in-time proof: alter, add and remove every tick after T-60 (a different close, a different
    in-play price, a tick one millisecond after the checkpoint): the rows at T-60 and before are identical."""
    altered = path_of(tmp_path, "altered", match_odds_file(
        late=[(-60 * M + 1, {HOME: 9.0, AWAY: 1.25, DRAW: 1.5}), (-45 * M, {HOME: 1.25}), (-2 * M, {AWAY: 9.0})],
        in_play_price=50.0))
    removed = path_of(tmp_path, "removed", match_odds_file(late=[]))
    key = ["market_id", "selection_id", "handicap", "cp_min"]
    base = path[path["cp_min"] >= 60].sort_values(key).reset_index(drop=True)
    assert len(base) == 13
    for other in (altered, removed):
        pd.testing.assert_frame_equal(base, other[other["cp_min"] >= 60].sort_values(key).reset_index(drop=True),
                                      check_exact=True)
    # ... and the test has teeth: T-15, after the altered ticks, is a different row in each
    def late_home(f):
        return f[(f["role"] == "home") & (f["cp_min"] == 15)].iloc[0]
    assert near(late_home(path)["ret_30m"], -2 * L2) and near(late_home(altered)["ret_1h"], 2 * L2)
    assert late_home(removed)["ret_30m"] == 0 and late_home(removed)["n_ticks_1h"] == 1
    # a tick AT the checkpoint's moment is the checkpoint's price
    on_time = path_of(tmp_path, "on_time", match_odds_file(late=[(-60 * M, {HOME: 3.0})]))
    at = on_time[(on_time["role"] == "home") & (on_time["cp_min"] == 60)].iloc[0]
    assert near(at["x"], -L2) and near(at["last_dx"], -2 * L2) and at["n_ticks_15m"] == 2 and at["jump_min_2h"] == 0


# ---------------------------------------------------------------------------------------------------------------
# the vectorised arithmetic against a plain reference, feature by feature, over seeded random ticks
# ---------------------------------------------------------------------------------------------------------------

LADDER = [1.2, 1.35, 1.5, 1.8, 2.0, 2.2, 2.6, 3.0, 4.0, 6.0, 10.0, 34.0]


def reference(ticks, when: int, others, market_type: str) -> dict:
    """One row's features the slow way. ``ticks``: the runner's [(pt ms, price)] in order; ``others``: the same for
    the other runners of its book."""
    seen = [(pt, float(-np.log(p - 1.0))) for pt, p in ticks if pt <= when]
    x0 = seen[-1][1]
    f = {"x": x0, "ret_first": x0 - seen[0][1], "hours_since_first": (when - seen[0][0]) / H}

    def window(minutes):
        """(the price as of the window's start or None, the window's start, [(pt, x, change)] inside it)."""
        start = when - minutes * M
        before = [v for pt, v in seen if pt <= start]
        inside = [(seen[i][0], seen[i][1], seen[i][1] - seen[i - 1][1] if i else None)
                  for i in range(len(seen)) if seen[i][0] > start]
        return (before[-1] if before else None), start, inside

    def steps(minutes):
        """The step path over the covered window: [(hours from its start, hours to, x)], and its length."""
        was, start, inside = window(minutes)
        a = start if was is not None else seen[0][0]
        level, t0, out = (was if was is not None else seen[0][1]), 0.0, []
        for pt, v, _ in inside:
            t1 = (pt - a) / H
            out.append((t0, t1, level))
            level, t0 = v, t1
        out.append((t0, (when - a) / H, level))
        return out, (when - a) / H

    for tag, minutes in gold_path.LOOKBACKS.items():
        was, _, inside = window(minutes)
        f[f"ret_{tag}"] = math.nan if was is None else x0 - was
        f[f"n_ticks_{tag}"] = len(inside)
    for tag in ("1h", "6h", "24h"):
        f[f"rv_{tag}"] = math.sqrt(sum(d * d for _, _, d in window(gold_path.LOOKBACKS[tag])[2] if d is not None))
    for tag in ("6h", "24h"):
        was, _, inside = window(gold_path.LOOKBACKS[tag])
        ch = [d for _, _, d in inside if d is not None]
        path_, w = steps(gold_path.LOOKBACKS[tag])
        f[f"slope_{tag}"] = f[f"slope_r2_{tag}"] = math.nan
        if w >= 0.25:
            mean = sum(v * (t1 - t0) for t0, t1, v in path_) / w
            cov = sum(v * ((t1 * t1 - t0 * t0) / 2 - w / 2 * (t1 - t0)) for t0, t1, v in path_) / w
            var = sum((v - mean) ** 2 * (t1 - t0) for t0, t1, v in path_) / w
            f[f"slope_{tag}"] = cov / (w * w / 12) if ch else 0.0
            if ch and var >= 1e-12:
                f[f"slope_r2_{tag}"] = cov * cov / (w * w / 12 * var)
        net = x0 - (was if was is not None else seen[0][1])
        f[f"er_{tag}"] = abs(net) / sum(abs(d) for d in ch) if ch and sum(abs(d) for d in ch) > 0 else math.nan
        pairs = [(ch[i - 1], ch[i]) for i in range(1, len(ch))]
        f[f"ac1_{tag}"] = math.nan
        if len(pairs) >= 5:
            den = math.sqrt(sum(a * a for a, _ in pairs) * sum(b * b for _, b in pairs))
            if den > 0:
                f[f"ac1_{tag}"] = sum(a * b for a, b in pairs) / den
        if tag == "24h":
            f["roll_24h"] = (2 * math.sqrt(max(-sum(a * b for a, b in pairs) / len(pairs), 0.0))
                             if len(pairs) >= 5 else math.nan)
    for tag in ("1h", "6h"):
        path_, w = steps(gold_path.LOOKBACKS[tag])
        f[f"twap_dev_{tag}"] = sum(v * (t1 - t0) for t0, t1, v in path_) / w - x0 if w > 0 else math.nan
        ch = [d for _, _, d in window(gold_path.LOOKBACKS[tag])[2] if d]
        ups, downs = sum(d > 0 for d in ch), sum(d < 0 for d in ch)
        f[f"tick_imb_{tag}"] = (ups - downs) / (ups + downs) if ups + downs else math.nan
    was, _, inside = window(1440)
    levels = [was if was is not None else seen[0][1]] + [v for _, v, _ in inside]
    low, high = min(levels), max(levels)
    f["range_pos_24h"] = (x0 - low) / (high - low) if high > low else math.nan
    f["drawdown_24h"], f["runup_24h"] = high - x0, x0 - low
    ch = [(pt, d) for pt, _, d in window(120)[2] if d is not None]
    f["jump_abs_2h"], f["jump_sign_2h"], f["jump_min_2h"] = 0.0, 0, math.nan
    if ch:
        big = max(abs(d) for _, d in ch)
        pt, d = [c for c in ch if abs(c[1]) == big][-1]
        f["jump_abs_2h"], f["jump_sign_2h"], f["jump_min_2h"] = big, int(np.sign(d)), (when - pt) / M
    diffs = [seen[i][1] - seen[i - 1][1] for i in range(1, len(seen))]
    f["last_dx"] = diffs[-1] if diffs else math.nan
    run = 0
    for d in reversed(diffs):
        if d == 0 or (run and np.sign(d) != np.sign(diffs[-1])):
            break
        run += 1
    f["run_len"] = int(run * np.sign(diffs[-1])) if diffs else 0
    last_trade, moved, priced = seen[-1][0], 0.0, 0
    for other in others:
        then = [p for pt, p in other if pt <= last_trade]
        now = [p for pt, p in other if pt <= when]
        if then:
            priced += 1
            moved += 1 / now[-1] - 1 / then[-1]
    f["book_catchup"] = math.nan
    if priced == len(others) == gold_path.BOOK_RUNNERS[market_type] - 1:
        p_own = 1 / [price for pt, price in ticks if pt == last_trade][-1]
        p = min(max(p_own - moved, 0.001), 1 / 1.01)
        f["book_catchup"] = math.log(p / (1 - p)) - math.log(p_own / (1 - p_own))
    return f


def random_inputs(seed: int, n_markets: int = 60):
    """Random pre-off ticks on a 5-minute grid (so ticks fall exactly on window edges and checkpoints), with
    in-play ticks, a handicap market and the non-decision checkpoints beside them: all to be ignored."""
    rng = np.random.default_rng(seed)
    tick, cp, books = [], [], {}
    for m in range(n_markets):
        market_type = "MATCH_ODDS" if m % 3 else "OVER_UNDER_25"
        if m == 5:
            market_type = "ASIAN_HANDICAP"
        market_id, ko = f"1.{1000 + m}", KO_MS + int(rng.integers(0, 20)) * 24 * H + int(rng.integers(0, 288)) * 5 * M
        n_runners = 3 if market_type == "MATCH_ODDS" else 2
        if m % 11 == 0:
            n_runners -= 1                                       # a book with a runner that never traded
        for r in range(n_runners):
            n_ticks = int(rng.choice([1, 2, 3, 8, 30, 120]))
            horizon = int(rng.choice([12, 120, 720]))            # 5-minute steps back from 10 minutes after the off
            steps = np.sort(rng.choice(np.arange(-horizon, 3), size=min(n_ticks, horizon + 3), replace=False))
            level = int(rng.integers(0, len(LADDER)))
            series = []
            for s in steps:
                level = int(np.clip(level + rng.choice([-2, -1, -1, 1, 1, 2]), 0, len(LADDER) - 1))
                if series and LADDER[level] == series[-1][1]:
                    level = level - 1 if level else 1
                series.append((ko + int(s) * 5 * M, LADDER[level]))
            books.setdefault((market_id, market_type), {})[100 + r] = series
            for seq, (pt, price) in enumerate(series):
                tick.append((market_id, market_type, 100 + r, 0.0, pt, seq, price, False))
            tick.append((market_id, market_type, 100 + r, 0.0, ko + 30 * M, 999, 1.01, True))      # in play
            for c in CHECKPOINTS:
                when = ko - c * M
                if c and any(pt <= when for pt, _ in series):
                    cp.append((market_id, 100 + r, 0.0, c, 7000 + m, market_type, f"r{r}", when))
    order = rng.permutation(len(tick))                           # the build must not lean on the input's order
    t = pd.DataFrame([tick[i] for i in order], columns=gold_path.COLUMNS["silver/tick"])
    c = pd.DataFrame(cp, columns=gold_path.COLUMNS["gold/checkpoint"])
    tick_table = pa.table({"market_id": t["market_id"], "market_type": t["market_type"],
                           "selection_id": pa.array(t["selection_id"], pa.int64()), "handicap": t["handicap"],
                           "pt": pa.array(t["pt"], pa.int64()).cast(TS), "seq": pa.array(t["seq"], pa.int32()),
                           "ltp": t["ltp"], "in_play": t["in_play"]})
    cp_table = pa.table({"market_id": c["market_id"], "selection_id": pa.array(c["selection_id"], pa.int64()),
                         "handicap": c["handicap"], "cp_min": pa.array(c["cp_min"], pa.int32()),
                         "event_id": pa.array(c["event_id"], pa.int64()), "market_type": c["market_type"],
                         "role": c["role"], "cp_time": pa.array(c["cp_time"], pa.int64()).cast(TS)})
    return {"silver/tick": tick_table, "gold/checkpoint": cp_table}, books, c


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_feature_matches_a_plain_reference_on_random_ticks(seed):
    inputs, books, cp = random_inputs(seed)
    out = conform(gold_path.SPEC, gold_path.build(inputs)[gold_path.SPEC.name]).to_pandas()
    want = cp[cp["cp_min"].isin(gold_path.PATH_CHECKPOINTS) & (cp["market_type"] != "ASIAN_HANDICAP")]
    assert len(out) == len(want) > 300 and not out.duplicated(["market_id", "selection_id", "cp_min"]).any()
    features = gold_path.SPEC.names[gold_path.SPEC.names.index("x"):]
    seen_blank, seen_value = set(), set()
    for row in out.itertuples(index=False):
        book = books[(row.market_id, row.market_type)]
        ref = reference(book[row.selection_id], row.cp_time.value // 10**6,
                        [v for k, v in book.items() if k != row.selection_id], row.market_type)
        for name in features:
            got, exp = getattr(row, name), ref[name]
            if blank(exp):
                assert blank(got), (name, row.market_id, row.selection_id, row.cp_min, got)
                seen_blank.add(name)
            else:
                assert abs(got - exp) <= 1e-9 * max(1.0, abs(exp)), (name, row.market_id, row.selection_id,
                                                                    row.cp_min, got, exp)
                seen_value.add(name)
    assert seen_value == set(features)                           # every feature was exercised with a value ...
    nullable = {c.name for c in gold_path.SPEC.columns if c.nullable and c.name in features}
    assert seen_blank == nullable                                # ... and every blankable one with a blank


def test_the_build_is_a_pure_function_of_its_inputs_whatever_their_order_and_empty_inputs_give_an_empty_table():
    inputs, _, _ = random_inputs(4, n_markets=12)
    one = gold_path.build(inputs)[gold_path.SPEC.name]
    flipped = {k: v.take(pa.array(np.arange(v.num_rows)[::-1])) for k, v in inputs.items()}
    key = [(c, "ascending") for c in gold_path.SPEC.sort]
    assert conform(gold_path.SPEC, one).sort_by(key).equals(
        conform(gold_path.SPEC, gold_path.build(flipped)[gold_path.SPEC.name]).sort_by(key))
    none = {k: v.slice(0, 0) for k, v in inputs.items()}
    assert gold_path.build(none)[gold_path.SPEC.name].num_rows == 0
    assert gold_path.build({"silver/tick": inputs["silver/tick"], "gold/checkpoint": none["gold/checkpoint"]})[
        gold_path.SPEC.name].schema.names == gold_path.SPEC.names
