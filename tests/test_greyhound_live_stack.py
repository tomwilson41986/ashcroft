"""greyhound/live.py on the stack (10 Nov 2026): the day's card priced end to end (the metrics tables built with the
card, the matrix blocks, the stack's price, the predictions file the trader and the early lay read), the fallback while
no stack is stored, and the early lay's forecast reading the stack's out-of-sample prices."""
from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from greyhound import live, nn, stack_model as sm  # noqa: E402
from sources.common import Store  # noqa: E402
from greyhound.metrics import GreyhoundMetricsEngine  # noqa: E402
from tests.test_greyhound_pit import synth  # noqa: E402

DAY = date(2025, 2, 9)          # synth(): 40 days from 1 Jan 2025


class FakeEngine(GreyhoundMetricsEngine):
    """The engine's place (its per-run figures kept, the metrics tables read them): two card-safe features from each
    dog's earlier days only."""

    def __init__(self, blocks=()):
        super().__init__(blocks=())
        self.block_features = {"base": ["x_prev_pos", "x_runs"]}
        self.features = ["x_prev_pos", "x_runs"]

    def calculate_all(self, runs, prices=None):
        d = runs.sort_values(["t", "race_id", "trap"], kind="stable").copy()
        from greyhound.pit import core                 # earlier days only, as the real engine (a dog can run twice a day)
        d["x_prev_pos"] = core.prev_day_values(d, "dog_id", ["position"])["position"].to_numpy()
        d["x_runs"] = core.prior_window(d, ["dog_id"], [], None)["n"].to_numpy()
        d["bsp"] = d.sp_decimal
        d["bsp_p_norm"] = d.sp_p_norm
        return d


def _card(runs):
    last = runs[runs.race_date == DAY.isoformat()].copy()
    for c in ("position", "beaten_lengths", "sectional", "run_time", "adjusted_time", "weight_kg", "sp_decimal",
              "market_pos"):
        last[c] = np.nan
    last["comment"] = ""
    last["is_card"] = True
    last["market_id"] = "1." + last.race_id.astype(str)
    last["selection_id"] = last.dog_id * 10
    last["matched_history"] = True
    return last.drop(columns=[c for c in ("won", "placed2", "sp_p", "sp_p_norm", "t", "field", "is_trial", "raceid",
                                          "grade", "grade_family") if c in last])


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    import betfair_recorder
    import greyhound.card as gcard
    import greyhound.metrics as gmetrics
    import greyhound.model as gmodel
    import greyhound.track as gtrack
    from greyhound import blocks as gblocks
    from greyhound.pit import ability, core, trouble
    monkeypatch.setattr(trouble, "MIN_FIT", 100)
    monkeypatch.setattr(ability, "FIT_BEFORE", "2025-01-15")
    monkeypatch.setattr(nn, "MAX_EPOCHS", 3)
    runs = synth()
    hist = runs[runs.race_date < DAY.isoformat()]
    card = _card(runs)
    monkeypatch.setattr(gtrack, "runs_through", lambda store, d, first_year=2018, trials_out=None: hist.copy())
    monkeypatch.setattr(betfair_recorder, "_read_day_csv",
                        lambda d, name, s3=None: pd.DataFrame({"x": [1]}) if name == "markets_greyhound.csv" else None)
    monkeypatch.setattr(gcard, "cards", lambda markets, h, trials=None: card.copy())
    monkeypatch.setattr(gmodel, "load_prices", lambda store, years=None: None)
    monkeypatch.setattr(gmetrics, "GreyhoundMetricsEngine", FakeEngine)
    monkeypatch.setattr(core, "raw_fetch_lag", lambda store: pd.Series(0, index=sorted(runs.race_date.unique())))
    real_attach = gblocks.attach

    def attach(df, name):                              # the simulator and the weight block stand in as one column
        if name.startswith("pit_"):
            return real_attach(df, name)
        df[f"{name}_x"] = 0.0
        return [f"{name}_x"]
    monkeypatch.setattr(gblocks, "attach", attach)
    monkeypatch.setattr(live, "model_frame", lambda df: df.sort_values(["t", "race_id", "trap"], kind="stable")
                        .reset_index(drop=True))
    store = Store(root=tmp_path / "src")
    return store, runs, hist, card


def _bundle(hist, feats):
    from greyhound import pit
    tabs = pit.build_tables(list(sm.PIT_TABLES), hist, None, {"fetch_lag": pd.Series(0, index=sorted(
        hist.race_date.unique()))}, keep=True)
    d = FakeEngine().calculate_all(hist)
    d = d[d.field >= 4].sort_values(["t", "race_id", "trap"], kind="stable").reset_index(drop=True)
    for name, tab in tabs.items():
        cols = [c for c in tab.columns if c.startswith(pit.module(name).PREFIX)]
        j = d[pit.KEY].merge(tab[pit.KEY + cols], on=pit.KEY, how="left")
        for c in cols:
            d[c] = j[c].to_numpy()
    d["sim_x"] = d["wpred_x"] = 0.0
    rc = {"learning_rate": 0.2, "early_stop": 5, "rounds": 30, "max_bin": 31, "seeds": 1, "nn_seeds": 1,
          "nn_seconds": 60.0}
    return sm.fit(d, [f for f in feats if f in d.columns], before=DAY.isoformat(), recipe=rc,
                  train_from="2025-01-01", log=lambda m: None)


def test_the_card_is_priced_by_the_stack(wired):
    store, runs, hist, card = wired
    feats = ["x_prev_pos", "x_runs", "sim_x", "wpred_x", "kf_gsr_mean", "fg_tot", "fm_level", "dr_win_vs_sp",
             "bd_p_lead", "tr_cost5", "cx_tr_ae365", "mk_sp_ae_all", "br_sire_gsr", "lf_runs_at_track"]
    bundle = _bundle(hist, feats)
    store.put(live.STACK_KEY, sm.to_bytes(bundle))
    store.put(live.STACK_META_KEY, b'{"trained_utc": "2025-02-09T04:00:00", "oos_prices": "greyhound/live/oos_x"}')
    out = live.predict(store, DAY)
    assert out["model"] == "stack_share" and out["runners"] == len(card)
    pred = pd.read_csv(io.BytesIO(store.get(f"{live.PREFIX}/predictions/{DAY}.csv")))
    assert set(pred.model_objective) == {"stack_share"} and set(pred.oos_prices) == {"greyhound/live/oos_x"}
    assert np.allclose(pred.groupby("race_id").p_model.sum(), 1.0)
    assert np.isfinite(pred.pred_bsp).all() and (pred.pred_bsp > 0).all() and pred.market_id.notna().all() and pred.selection_id.notna().all()
    # the card's tables are its run's: the same price from the nightly tables once the results are in
    from greyhound import pit
    full = FakeEngine().calculate_all(runs)
    full = full.sort_values(["t", "race_id", "trap"], kind="stable").reset_index(drop=True)
    tabs = pit.build_tables(list(sm.PIT_TABLES), runs, None, {"fetch_lag": pd.Series(0, index=sorted(
        runs.race_date.unique()))}, keep=True)
    for name, tab in tabs.items():
        cols = [c for c in tab.columns if c.startswith(pit.module(name).PREFIX)]
        j = full[pit.KEY].merge(tab[pit.KEY + cols], on=pit.KEY, how="left")
        for c in cols:
            full[c] = j[c].to_numpy()
    full["sim_x"] = full["wpred_x"] = 0.0
    day = full[full.race_date == DAY.isoformat()].reset_index(drop=True)
    again = sm.price(bundle, day)
    m = pred.merge(day[["race_id", "dog_id"]].assign(p_res=again.p_model.to_numpy()), on=["race_id", "dog_id"])
    assert len(m) == len(pred) and np.allclose(m.p_model, m.p_res, atol=1e-6)


def test_without_a_stack_the_log_bsp_model_prices(wired, monkeypatch):
    store, runs, hist, card = wired
    called = []
    monkeypatch.setattr(live, "load_model", lambda store: called.append(1) or (None, None))
    with pytest.raises(SystemExit):
        live.predict(store, DAY)
    assert called


def test_the_early_lay_reads_the_stacks_prices(tmp_path):
    from greyhound import laytest as lt
    store = Store(root=tmp_path)
    store.put(f"greyhound/live/predictions/{DAY}.csv", pd.DataFrame(
        {"market_id": ["1.1"], "model_objective": ["stack_share"], "oos_prices": [live.STACK_OOS_PRICES]})
        .to_csv(index=False).encode())
    assert lt.live_prices(store, DAY) == lt.LOGBSP_PRICES           # not published yet: the nearest prices
    store.put("greyhound/live/oos_prices_stack_all3/2025-02-01.parquet", b"x")
    assert lt.live_prices(store, DAY) == lt.LOGBSP_PRICES           # two models back is not the nearest
    store.put("greyhound/live/oos_prices_stack_m/2025-02-01.parquet", b"x")
    assert lt.live_prices(store, DAY) == "greyhound/live/oos_prices_stack_m"      # the stack before it, once there
    store.put(f"{live.STACK_OOS_PRICES}/2026-10-06.parquet", b"x")
    assert lt.live_prices(store, DAY) == live.STACK_OOS_PRICES


def test_the_live_engine_blocks_are_the_caches():
    from greyhound import fastloop
    assert tuple(live.LIVE_BLOCKS) == tuple(fastloop.BLOCKS)
    assert live.STACK_OBJECTIVE in fastloop.OBJECTIVES and live.PIT_FIRST_YEAR == 2014


def test_the_released_stack_reads_the_league_table_and_names_its_own_prices():
    """14 Nov 2026 (the owner: "Release the league table to the live price"): the league table in the stack, in the
    build order, under the recipe "stack_ml", which names its own out-of-sample prices; the early lay reads batch 32's
    stack's until they are published."""
    from greyhound import laytest as lt, pit
    assert sm.PIT_TABLES[-1] == "league" and "pit_league" in sm.MATRIX_BLOCKS
    assert pit.TABLES.index("league") > max(pit.TABLES.index(t) for t in sm.PIT_TABLES[:-1])   # built last
    assert sm.RECIPE["name"] == "stack_ml" and live.STACK_OOS_BY_RECIPE["stack_ml"] == live.STACK_OOS_PRICES
    assert live.STACK_OOS_PRICES == "greyhound/live/oos_prices_stack_ml"
    assert lt.NEAREST_PRICES[live.STACK_OOS_PRICES] == live.STACK_OOS_BY_RECIPE["stack_m"]
