"""The greyhound trade-out (greyhound/tradeout.py): the mechanics on synthetic books, never a figure of record."""

from __future__ import annotations

import io
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from betfair_recorder import BOOK_FIELDS
from greyhound import tradeout as to

DAY = date(2026, 10, 20)


class MemStore:
    def __init__(self):
        self.d = {}

    def put(self, k, b):
        self.d[k] = b

    def get(self, k):
        return self.d.get(k)

    def put_parquet(self, k, df):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.d[k] = buf.getvalue()

    def get_parquet(self, k):
        return pd.read_parquet(io.BytesIO(self.d[k])) if k in self.d else None

    def listing(self, prefix):
        return [k for k in self.d if k.startswith(prefix)]


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def synthetic_day(day=DAY, n_races=40, seed=0, beta_model=0.4):
    """Races of six whose SP book is softmax((1-b) lq_market + b lq_model): the model carries information."""
    rng = np.random.default_rng(seed)
    books, markets, pred = [], [], []
    for r in range(n_races):
        off = datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=15 * r)
        mid = f"1.{r:06d}"
        truth = rng.normal(0, 0.7, 6)
        lq_mod = truth + rng.normal(0, 0.4, 6)
        lq_mkt = truth + rng.normal(0, 0.4, 6)
        pm = np.exp(lq_mod) / np.exp(lq_mod).sum()
        qm = np.exp(lq_mkt) / np.exp(lq_mkt).sum()
        u = (1 - beta_model) * np.log(qm) + beta_model * np.log(pm)
        qc = np.exp(u) / np.exp(u).sum()
        winner = rng.choice(6, p=qc)
        for i in range(6):
            sel = 1000 + 10 * r + i
            markets.append({"market_id": mid, "market_type": "WIN", "market_start_utc": _iso(off), "venue": "Romford",
                            "country": "GB", "selection_id": sel, "runner_name": f"{i + 1}. Dog {i}"})
            pred.append({"race_id": r, "race_date": f"{day}", "race_time": off.strftime("%H:%M"), "track": "Romford",
                         "trap": i + 1, "field": 6, "won": int(i == winner), "p_model": pm[i], "market_id": mid,
                         "selection_id": sel})
            fair = 1 / qm[i] * 1.02
            for m in (10, 5, 3, 1, 0.4):
                back = round(fair * 0.97, 2)
                books.append({**{f: None for f in BOOK_FIELDS}, "polled_utc": _iso(off - timedelta(minutes=m)),
                              "source": "recorder", "market_id": mid, "status": "OPEN", "inplay": False,
                              "selection_id": sel, "runner_status": "ACTIVE", "back1": back, "back1_size": 50.0,
                              "lay1": round(fair * 1.03, 2), "lay1_size": 40.0, "last_traded": round(fair, 2),
                              "runner_total_matched": 300.0 / m, "market_total_matched": 2000.0 / m,
                              "number_of_active_runners": 6})
            books.append({**{f: None for f in BOOK_FIELDS}, "polled_utc": _iso(off + timedelta(minutes=2)),
                          "source": "final", "market_id": mid, "status": "CLOSED", "inplay": False,
                          "selection_id": sel, "runner_status": "WINNER" if i == winner else "LOSER",
                          "sp_actual": round(1.0 / qc[i], 3), "number_of_active_runners": 6})
    return pd.DataFrame(pred), pd.DataFrame(books), pd.DataFrame(markets)


def _one(**kw):
    base = {"race_id": 1, "race_date": "2026-11-04", "market_id": "1.1", "mark": "off_1", "won": 0, "p_model": 0.3,
            "back1": 5.0, "back1_size": 100.0, "back2": 4.8, "back2_size": 100.0, "back3": 4.6, "back3_size": 100.0,
            "lay1": 5.2, "lay1_size": 100.0, "lay2": 5.4, "lay2_size": 100.0, "lay3": 5.6, "lay3_size": 100.0,
            "bsp": 4.0}
    base.update(kw)
    return pd.DataFrame([base])


def test_back_sized_to_win_and_settled_by_the_result_as_the_lay_at_sp_pays():
    d = _one(back1_size=10.0, back2_size=10.0, back3_size=100.0, won=0)
    ev = pd.DataFrame({"inv": [1 / 4.0]})                    # every price from 4.6 to 5 clears the bar at 1/BSP = 0.25
    p = to.positions(d, ev, "racebook", bars=(0.03,))
    back = p[p.side == "back"].iloc[0]
    target = to.TO_WIN / (5.0 - 1.0)                          # GBP25 to win GBP100 at 5
    assert back.target == pytest.approx(target)
    assert back.stake == pytest.approx(25.0)                  # down the ladder: 10 at 5, 10 at 4.8, 5 at 4.6
    m = (10 * 5.0 + 10 * 4.8 + 5 * 4.6) / 25.0
    assert back.price == pytest.approx(m)
    # the dog lost: the back's stake gone, the lay at SP (liability = winnings) wins liability / (BSP - 1)
    assert back.pnl == pytest.approx(-25.0 + 25.0 * (m - 1) / (4.0 - 1))
    won = to.positions(d.assign(won=1), ev, "racebook", bars=(0.03,))
    assert won[won.side == "back"].pnl.iloc[0] == pytest.approx(0.0)  # the dog won: back and lay cancel
    assert back.clv_value == pytest.approx(25.0 * (m / 4.0 - 1))      # the expectation, kept beside it


def test_the_ladder_stops_where_the_price_no_longer_clears_the_bar():
    d = _one(back1=5.0, back1_size=5.0, back2=4.0, back2_size=100.0)
    ev = pd.DataFrame({"inv": [1 / 4.5]})                    # 5 clears (+11%), 4 does not (-11%)
    p = to.positions(d, ev, "racebook", bars=(0.0,))
    assert p[p.side == "back"].stake.iloc[0] == pytest.approx(5.0)
    # no back whose winnings are under the GBP10 minimum lay at SP
    tiny = to.positions(_one(back1_size=2.0, back2=4.0), ev, "racebook", bars=(0.0,))
    assert tiny.empty or (tiny.side != "back").all()


def test_lay_closed_by_a_back_at_sp():
    d = _one(lay1=3.0, lay1_size=100.0, won=1, bsp=4.0)
    ev = pd.DataFrame({"inv": [1 / 4.0]})                    # laying 3 against an SP of 4: 1 - 3/4 = +25%
    p = to.positions(d, ev, "racebook", bars=(0.03,))
    lay = p[p.side == "lay"].iloc[0]
    assert lay.stake == pytest.approx(to.TO_WIN / 2.0)        # liability = TO_WIN
    assert lay.pnl == pytest.approx(lay.stake * (4.0 - lay.price))     # won: the SP back pays more than the lay costs
    lost = to.positions(d.assign(won=0), ev, "racebook", bars=(0.03,))
    assert lost[lost.side == "lay"].pnl.iloc[0] == pytest.approx(0.0)


def test_several_dogs_a_race_and_commission_on_the_market_net():
    d = pd.concat([_one(selection_id=1, won=0, bsp=4.0), _one(selection_id=2, won=1, bsp=4.0)], ignore_index=True)
    ev = pd.DataFrame({"inv": [0.25, 0.25]})
    p = to.positions(d, ev, "racebook", bars=(0.03,))
    backs = p[p.side == "back"]
    assert len(backs) == 2                                    # both dogs in the race backed
    m = to.settle(backs)
    assert len(m) == 1                                        # one market
    gross = backs.pnl.sum()
    assert m.net.iloc[0] == pytest.approx(gross - to.COMMISSION * max(gross, 0.0))


def test_no_position_over_the_price_cap():
    d = _one(back1=25.0, back2=24.0, back3=23.0, lay1=26.0, lay2=27.0, lay3=28.0, bsp=20.0, p_model=0.9)
    ev = pd.DataFrame({"inv": [0.5]})
    assert to.positions(d, ev, "racebook").empty
    assert to.positions(d, None, "win_edge").empty


def test_trades_a_day_and_the_interval_count_every_scored_day():
    d = _one(won=0)
    ev = pd.DataFrame({"inv": [0.25]})
    p = to.positions(d, ev, "racebook", bars=(0.03,))
    p = p[p.side == "back"]
    t = to.rule_table(p, days=["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-07"], n_boot=200)
    r = t.iloc[0]
    assert r.days == 4 and r.active_days == 1 and r.per_day == pytest.approx(0.2)   # 1 trade over 4 days, rounded


def test_a_blank_in_play_value_does_not_empty_the_day():
    pred, books, markets = synthetic_day(n_races=2)
    # a CSV with one blank in the column: pandas reads it as float, so the rest read "0.0", not "0" or "False"
    books["inplay"] = 0.0
    books.loc[books.index[0], "inplay"] = np.nan
    assert to._inplay(books.inplay).sum() == 0             # "0.0" and the blank both read as not in play
    assert len(to.day_panel(pred, books, markets, DAY))


def test_retimed_books_sit_at_the_market_last_match():
    pred, books, markets = synthetic_day(n_races=2)
    live = books[books.source == "recorder"].copy()
    polled = pd.to_datetime(live.polled_utc, utc=True)
    # a delayed feed: every snapshot shows the market 90 seconds earlier, the last match a few seconds before that
    live["last_match_utc"] = (polled - pd.Timedelta(seconds=93)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    pre = to.pre_off_books(live, markets, retimed=True)
    shift = (pd.to_datetime(live.loc[pre.index, "polled_utc"], utc=True) - pre.polled).dt.total_seconds()
    assert np.allclose(shift, 93)
    # the snapshot polled 0.4 minutes before the off now sits 1.95 minutes before it, so off_1 has no book there
    assert (pre.minutes_to_off > 0).all()


def test_a_test_day_before_the_first_is_refused():
    frozen = {"first_test_day": "2026-11-10", "models": {}}
    with pytest.raises(ValueError):
        to.track_day(MemStore(), frozen, date(2026, 11, 9))


def test_panel_marks_and_whole_races():
    pred, books, markets = synthetic_day(n_races=3)
    # one runner of race 0 offers no back price 1 minute out: race 0 is dropped at off_1 only
    sel0 = pred.selection_id.iloc[0]
    one = (books.selection_id == sel0) & (books.source == "recorder") & books.polled_utc.str.endswith("11:59:00Z")
    books.loc[one, "back1"] = None
    d = to.day_panel(pred, books, markets, DAY)
    assert set(d["mark"]) == {"first_traded", "off_5", "off_3", "off_1"}
    off1 = d[d["mark"] == "off_1"]
    assert sorted(off1.race_id.unique()) == [1, 2]
    assert (d.groupby(["race_id", "mark"]).size() == 6).all()
    # the book about two minutes before the 1-minute mark is the 3-minute one
    r = off1.iloc[0]
    assert (r.polled - r.polled_prev).total_seconds() == pytest.approx(120)
    # the SP's shares and our chances are books of one within each race and mark
    s = d.assign(q=np.exp(d.lq_close), m=np.exp(d.lq_model)).groupby(["race_id", "mark"])[["q", "m"]].sum()
    assert np.allclose(s.values, 1.0)


def test_closing_models_learn_what_the_model_adds():
    pred, books, markets = synthetic_day(n_races=300, seed=1, beta_model=0.4)
    d = to.day_panel(pred, books, markets, DAY)
    d = d[d["mark"] == "off_1"]
    tr, te = d[d.race_id < 200], d[d.race_id >= 200]
    f = to.fit_models(tr)
    g = to._codes(te)
    q = np.exp(te.lq_close.to_numpy())

    def xent(qh):
        return -float(np.sum(q * np.log(qh))) / (g.max() + 1)

    market = race_book_norm(1 / te["mid"].to_numpy(), g)
    for name in to.MODELS:
        qh, book = to.predict_share(te, f, name)
        assert xent(qh) < xent(market)                      # our price carries what the SP book took from it
        assert book == pytest.approx(1.0, abs=0.05)
    j = f["micro"]
    assert j["beta"][to.MICRO.index("lq_model")] > 0


def race_book_norm(x, g):
    from model.race_book import normalise_within
    return normalise_within(x, g)


def test_freeze_starts_the_day_after_and_never_before_start():
    store = MemStore()
    pred, books, markets = synthetic_day(n_races=30)
    store.put_parquet(f"{to.TRACK_PRED}/{DAY}.parquet", pred)

    one = to.day_panel(pred, books, markets, DAY)
    n_days = {"n": to.MIN_FREEZE_DAYS - 1}

    def fake_load(store_, first, last, s3=None):
        return pd.concat([one.assign(race_date=f"2026-10-{10 + k:02d}") for k in range(n_days["n"])],
                         ignore_index=True)

    orig = to.load_panels
    to.load_panels = fake_load
    try:
        waiting = to.freeze(store, date(2026, 11, 2))         # nine days of the trader's books: not yet
        assert "waiting" in waiting and to.FROZEN_KEY not in store.d
        n_days["n"] = to.MIN_FREEZE_DAYS
        out = to.freeze(store, date(2026, 11, 2))
        assert out["fit_days"] == to.MIN_FREEZE_DAYS
        assert out["first_test_day"] == "2026-11-04"
        assert set(out["models"]) == {"first_traded", "off_5", "off_3", "off_1"}
        assert json.loads(store.get(to.FROZEN_KEY))["rules"]["primary"]["mark"] == "off_1"
        assert to.freeze(store, date(2026, 11, 10))["first_test_day"] == "2026-11-11"
    finally:
        to.load_panels = orig


def test_a_test_day_writes_every_rule_and_the_summary_reads_it():
    store = MemStore()
    pred, books, markets = synthetic_day(n_races=60, seed=3)
    panel = to.day_panel(pred, books, markets, DAY)
    frozen = {"first_test_day": str(DAY), "models": {m: to.fit_models(g) for m, g in panel.groupby("mark")}}
    orig = to.load_panel
    to.load_panel = lambda store_, day, s3=None: panel
    try:
        out = to.track_day(store, frozen, DAY)
    finally:
        to.load_panel = orig
    assert out["races"] == 60
    s = to.summary(store, frozen)
    assert s["days"] == 1 and s["gate"].startswith("not yet judged")
    t = pd.DataFrame(s["rules_table"])
    assert {"racebook", "micro", "win_edge"} <= set(t.model)
    assert "| off_1 |" in to.markdown(s)


def test_entry_books_from_the_trader_and_the_sp_from_the_recorder():
    pred, books, markets = synthetic_day(n_races=5)
    trader = books[books.source == "recorder"].copy()
    trader["back1"] = trader.back1 + 0.5                       # the trader's own, fresher book
    trader["last_match_utc"] = trader.polled_utc
    d = to.day_panel(pred, trader, markets, DAY, closing_books=books)
    assert len(d) and (d.feed_lag_s == 0).all()
    r = d.iloc[0]
    src = books[(books.selection_id == r.selection_id) & (books.source == "final")]
    assert r.bsp == pytest.approx(float(src.sp_actual.iloc[0]))
    # without the recorder's settled books there is no SP in the trader's: no panel
    assert to.day_panel(pred, trader, markets, DAY).empty


def test_main_waits_for_the_trader_books_without_failing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(to, "load_panels", lambda *a, **k: pd.DataFrame())
    import sources.common as sc
    monkeypatch.setattr(sc, "Store", lambda root=None: MemStore())
    out = tmp_path / "t.json"
    assert to.main(["--freeze", "--summary", "--days", "4", "--out", str(out)]) == 0
    assert "0 of 10 days" in out.read_text()


def test_study_end_to_end_on_synthetic_days(tmp_path, monkeypatch):
    from research.greyhound import tradeout_study as st
    frames = []
    for k in range(18):
        day = DAY + timedelta(days=k)
        pred, books, markets = synthetic_day(day=day, n_races=40, seed=10 + k)
        pred["race_id"] = pred.race_id + 1000 * k
        frames.append(to.day_panel(pred, books, markets, day))
    panel = pd.concat(frames, ignore_index=True)
    monkeypatch.setattr(st, "load", lambda store, first, last, source="recorder", pred_dir=None: (panel, [{"day": "x", "races_priced": 1}]))
    monkeypatch.setattr(st, "WINDOWS", (3, None))
    out = tmp_path / "study"
    assert st.main(["--store", str(tmp_path), "--from", str(DAY), "--to", str(DAY + timedelta(days=17)),
                    "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text())
    assert len(res["holdout_days"]) == st.HOLDOUT_DAYS and not set(res["holdout_days"]) & set(res["dev_days"])
    acc = pd.DataFrame(res["accuracy_dev"]).set_index(["mark", "model"])
    # the model carries information in the synthetic SP: the forecasts with it beat the market; the shuffle does not
    assert acc.loc[("off_1", "micro"), "xent_gain_per_race"] > 0
    assert acc.loc[("off_1", "micro_shuffled"), "xent_gain_per_race"] < acc.loc[("off_1", "micro"),
                                                                              "xent_gain_per_race"]
    assert res["learning_accuracy"] and res["importance"]["off_1"]["gbm_gain_share"]
    assert "Holdout, read once" in out.with_suffix(".md").read_text()


def test_the_trader_keeps_its_books_only_when_asked(tmp_path, monkeypatch):
    from greyhound import live
    assert live.book_recorder(DAY, env={"BETFAIR_RECORD": "0"}) is None
    monkeypatch.setenv("BETFAIR_LIVE_DIR", str(tmp_path))
    import betfair_recorder
    monkeypatch.setattr(betfair_recorder, "ROOT", tmp_path)
    rec = live.book_recorder(DAY, env={"BETFAIR_RECORD": "1"})
    assert rec is not None and rec.books_path.name == "books_greyhound_trader.csv"


def test_full_model_days_join_their_markets_and_meet_the_base_and_the_sp(tmp_path, monkeypatch):
    from research.greyhound import score_full as sf
    import betfair_recorder
    pred, books, markets = synthetic_day(n_races=6)
    uk = pd.to_datetime(markets.drop_duplicates("market_id").set_index("market_id").market_start_utc, utc=True)
    pred["race_time"] = pred.market_id.map(uk.dt.tz_convert("Europe/London").dt.strftime("%H:%M"))
    store = MemStore()
    store.put_parquet(f"{to.TRACK_PRED}/{DAY}.parquet", pred)            # the 5 Oct model's day, joined already
    full = pred.drop(columns=["market_id", "selection_id"]).assign(dog_id=1, dog_name="d", sp_decimal=3.0)
    full["p_model"] = full.groupby("race_id").p_model.transform(lambda s: (s ** 0.8) / (s ** 0.8).sum())
    files = {"markets_greyhound.csv": markets, "books_greyhound.csv": books}
    monkeypatch.setattr(betfair_recorder, "_read_day_csv", lambda day, name, s3=None, root=None: files.get(name))
    logbsp = full.assign(pred_bsp=1.0 / full.p_model)
    out = sf.score_days({"full": full, "logbsp": logbsp}, DAY, DAY, store, tmp_path / "pred")
    assert out["days"][0]["races_on_betfair_full"] == 6 and out["days"][0]["races_on_betfair_logbsp"] == 6
    day = pd.read_parquet(tmp_path / "pred" / "full" / f"{DAY}.parquet")
    assert {"market_id", "selection_id", "p_model"} <= set(day.columns) and len(day) == 36
    assert "pred_bsp" in pd.read_parquet(tmp_path / "pred" / "logbsp" / f"{DAY}.parquet").columns
    h = out["head_to_head"]
    assert h["races"] == 6 and h["logloss"]["base_5oct"] != h["logloss"]["full"]
    assert set(h["logloss"]) == {"base_5oct", "full", "logbsp", "bsp"} and h["log_bsp_model_vs_sp"]["rmse_log"] > 0


def test_log_bsp_model_fits_before_the_cutoff_and_prices_after():
    from research.greyhound import score_full as sf
    rng = np.random.default_rng(0)
    n_races, field = 1500, 6
    days = pd.date_range("2026-08-01", periods=100, freq="D")
    race = np.repeat(np.arange(n_races), field)
    t = np.repeat(days[np.arange(n_races) % len(days)], field)
    x = rng.normal(size=len(race))
    bsp = np.exp(1.6 - 0.6 * x + rng.normal(0, 0.2, len(race)))
    df = pd.DataFrame({"race_id": race, "t": t, "race_date": t.strftime("%Y-%m-%d"), "race_time": "12:00",
                       "track": "X", "trap": np.tile(np.arange(1, 7), n_races), "dog_id": 1, "dog_name": "d",
                       "won": 0, "sp_decimal": 3.0, "field": field, "is_trial": False, "x": x.astype(np.float32),
                       "bsp": bsp})
    out, info = sf.fit_log_bsp(df, ["x"], "2026-10-20", date_from="2026-01-01", rounds=200)
    assert (pd.to_datetime(out.race_date) >= "2026-10-20").all() and len(out)
    assert np.allclose(out.groupby("race_id").p_model.sum(), 1.0)
    assert np.corrcoef(np.log(out.pred_bsp), -out.merge(df[["race_id", "trap", "x"]], on=["race_id", "trap"]).x)[0, 1] > 0.9
    assert info["fit_rows"] < len(df)
    assert info["cap"] == 200 and info["rounds"] <= 200 and info["es_rmse_by_round"]
    assert all(v > 0 for v in info["es_rmse_by_round"].values()) and info["hit_cap"] == (info["rounds_run"] >= 200)


def test_live_book_record_reads_densely_near_the_off_and_on_until_the_market_closes(tmp_path, monkeypatch):
    from greyhound import book_record as br
    import betfair_recorder
    off = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)
    clock = {"t": off - timedelta(minutes=20)}
    reads = []

    class FakeData:
        def _read(self, method, params):
            if method == "listMarketCatalogue":
                if params["filter"]["marketTypeCodes"] != ["WIN"]:
                    return []
                return [{"marketId": "1.9", "marketStartTime": off.strftime("%Y-%m-%dT%H:%M:%SZ"), "runners": []}]
            reads.append((clock["t"], params["priceProjection"]["priceData"]))
            closed = clock["t"] >= off + timedelta(seconds=40)        # the race went off 40 s late
            return [{"marketId": "1.9", "status": "CLOSED" if closed else "OPEN", "inplay": False, "runners": []}]

    class Rec:
        rows = 0
        def record_catalogue(self, cat): return len(cat)
        def record_books(self, raw, source, polled_at): return len(raw)
        def maybe_upload(self, force=False): return True

    def sleep(s):
        clock["t"] += timedelta(seconds=s)

    out = br.record(off.date(), off + timedelta(hours=1), FakeData(), Rec(), market_types=("WIN",),
                    now=lambda: clock["t"], sleep=sleep)
    times = [t for t, d in reads if d == br.PRICE_DATA]                     # the books (the results apart)
    assert reads[0][1] == br.PRICE_DATA and "EX_TRADED" in br.PRICE_DATA
    assert min(times) >= off - timedelta(minutes=br.START_MINUTES)          # nothing read before 15 minutes out
    gaps_near = [(b - a).total_seconds() for a, b in zip(times, times[1:]) if a >= off - timedelta(minutes=5)]
    assert max(gaps_near) <= br.EVERY_NEAR + 1.0                            # every 5 s in the last 5 minutes
    assert max(times) >= off + timedelta(seconds=40) and out["done"] == 1  # on past the off until it closed
    # the results: from the off until closed (the settled reads alone ask for SP_TRADED; the book reads ask for
    # SP_AVAILABLE too, for the projected SP)
    settles = [t for t, d in reads if "SP_TRADED" in d]
    assert settles and min(settles) >= off and out["settled"] == 1


def test_the_results_carry_the_bsp_read_at_the_off():
    from greyhound import book_record as br
    calls = {"n": 0}

    class FakeData:
        def _read(self, method, params):
            calls["n"] += 1
            if calls["n"] == 1:                    # in play: the BSP is there, no result yet
                return [{"marketId": "1.9", "status": "SUSPENDED", "runners": [
                    {"selectionId": 7, "status": "ACTIVE", "sp": {"actualSP": 4.2}}]}]
            return [{"marketId": "1.9", "status": "CLOSED", "runners": [   # closed: the winner, no BSP
                {"selectionId": 7, "status": "WINNER", "sp": {}}]}]

    written = []

    class Rec:
        def record_books(self, raw, source, polled_at):
            written.append((source, raw))
            return len(raw)
        def maybe_upload(self, force=False): return True

    bsp = {}
    t = datetime(2026, 11, 3, 14, 1, tzinfo=timezone.utc)
    assert br.settle_reads(FakeData(), Rec(), ["1.9"], bsp, t) == set() and not written
    assert br.settle_reads(FakeData(), Rec(), ["1.9"], bsp, t) == {"1.9"}
    source, raw = written[0]
    assert source == "final" and raw[0]["runners"][0]["sp"]["actualSP"] == 4.2 and raw[0]["runners"][0]["status"] == "WINNER"


def test_the_read_only_client_refuses_to_bet():
    from trading.exchange import BetfairData

    class Client:
        def _api_call(self, method, params):
            raise AssertionError("never reached")

    with pytest.raises(ValueError):
        BetfairData(client=Client())._read("placeOrders", {})
