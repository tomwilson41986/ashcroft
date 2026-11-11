"""The greyhound model live (the owner's ask, 6 Oct 2026: "try and bring a model live, either a trading model or a
straight back model").

Real orders only with the owner's switch, the repository variable ``GREYHOUND_LIVE=yes``, and only on the UK runner;
otherwise everything runs on paper (the same decisions, simulated against the live books). Four steps:

**The model from 10 Nov 2026** (the owner: "move the live greyhound pricing to the new model"): batch 18's stack
(greyhound/stack_model.py: the metrics database's tables, three seeds of race-softmax trees and three neural set
models, stretched by track; RMSE of log(BSP) 0.3366 against 0.4017 on 1 Jun - 5 Nov 2026). ``--train`` fits it on the
29 months to yesterday and saves it to ``sources/greyhound/live/stack/``; ``--predict`` builds the metrics tables with
the day's card as their last day (``card_tables``: a card row gets the features its run gets once the result is in,
tests/test_greyhound_pit.py) and prices the card with it. The log-BSP model below stays as the fallback while no stack
is stored.

``--train`` (GitHub's runners, weekly): the card-safe model. The research model reads the dog's weight on the day,
which GBGB records at the weigh-in and the morning does not know, so the live model leaves out every feature built
on it (``CARD_UNSAFE``); fitted on every GBGB race to yesterday, base, parity and HorseRaceBase features, saved to
``sources/greyhound/live/model.txt`` with its meta.

``--predict`` (GitHub's runners, hourly 07:21-19:21 UTC): today's cards from the recorder's greyhound catalogue in
S3 (``greyhound.card``), priced from the dogs' earlier days, to ``sources/greyhound/live/predictions/<day>.csv``.

``--trade`` (the UK runner, beside the recorder). Paper only since 7 Oct 14:53 UTC (the owner: GREYHOUND_LIVE=no),
on the tested rule (the owner, 8 Oct; ``live_config.json``): each runner looked at once, in the last 1.5 minutes before
the off and no later than 30 seconds before it, at a book whose back and lay are within ``tight`` (25%); backed when
its edge at the best back price clears 0.2, at a price from 1.5 to 20, GBP5 level, the stake taken down the ladder while
the edge holds. The model's chances are renormalised over the runners still in the market. The anytime rule of 7 Oct
(every poll to the off, no tightness, top-ups) is still there: ``decide_window_minutes`` null, ``reconsider`` and
``top_up`` true. No race limit and no daily cap (``null``); the S3 objects ``greyhound/STOP`` and ``trading/STOP`` stop
new bets within a minute.

``--settle`` (GitHub's runners, each day): the day's ledger against the recorder's settled books (BSP, the winner),
to ``sources/greyhound/live/<mode>/<day>/settled.csv`` and the running summary ``.../summary.json``.

    python -m greyhound.live --train
    python -m greyhound.live --predict
    GREYHOUND_LIVE=yes python -m greyhound.live --trade --live --until 21:30
    python -m greyhound.live --settle --days 3
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound.model import COM, PARAMS, modelling_rows, race_norm

log = logging.getLogger(__name__)

PREFIX = "greyhound/live"
MODEL_KEY, META_KEY = f"{PREFIX}/model.txt", f"{PREFIX}/meta.json"
CONFIG_PATH = Path(__file__).with_name("live_config.json")
#: features the morning card cannot know: the dog's weight on the day (GBGB's weigh-in), the going (declared at the
#: meeting), the prize money and the handicap marks (GBGB's result, not Betfair's catalogue)
CARD_UNSAFE = {"weight_change", "weight_vs_mean", "hb_weight_vs_max", "hb_weight_vs_min", "hb_win_going",
               "hb_plc_going", "hb_runs_going", "hb_prize_1st", "hb_handicap"}
#: the feature blocks the live model is fitted with (``greyhound.metrics``; HorseRaceBase's from 6 Oct 2026, report §8.1)
#: the owner, 4 Nov 2026 ("implement all recommendations in order"): the best pre-race model of report section 26.3
#: (every engine block, the race simulator, log(BSP) rescaled to the SP book: RMSE 0.4086 against core's 0.4418)
LIVE_BLOCKS = ("parity", "hrb", "lib", "times", "intent", "ability", "etime")
#: greyhound/blocks computed on the engine's matrix: the race simulator (roadmap F2) and, from 4 Nov 2026 (the owner: "add
#: the weight history to the live model"), the weight block (the last runs' weights, the weight history, recency, form;
#: batch 9, run 41834002314: RMSE of log(BSP) -0.0032, 90% -0.0038 to -0.0026, log-loss -0.00029)
LIVE_MATRIX_BLOCKS = ("sim", "wpred")
#: the out-of-sample prices of this recipe for the recorded days (greyhound.fastloop publish_pred,
#: research/greyhound/experiments/live_oos_wpred.json): the early lay's forecast is fitted on them (laytest.live_prices)
LIVE_OOS_PRICES = "greyhound/live/oos_prices_logbsp_wpred"
LIVE_OBJECTIVE = "logbsp"                         # 'binary' was the live model to 4 Nov 2026
LIVE_TRAIN_FROM = "2024-01-01"                    # the fast loop's quick recipe, the window section 26.3 was scored on
#: the stretch of the SP forecast within each race (greyhound.fastloop CALIBRATIONS), fitted on the fit's early-stopping
#: tenth, applied to pred_bsp only (the win chance is left alone). Batch 8 (run 41827648070, 1 Jun - 2 Nov 2026): per
#: track, RMSE of log(BSP) 0.4127 -> 0.4053 (-0.0074, 90% -0.0080 to -0.0067); it takes effect at the next retrain
LIVE_CALIBRATION: str | None = "stretch_track"
#: the live model from 10 Nov 2026: greyhound/stack_model.py, saved here; the log-BSP model above is the fallback
STACK_KEY, STACK_META_KEY = f"{PREFIX}/stack/bundle.pt", f"{PREFIX}/stack/meta.json"
#: its out-of-sample prices of the recorded days (fast-loop batch 19, research/greyhound/experiments/
#: batch19_live_prices.json): the early lay's forecast is fitted on them (greyhound.laytest live_prices)
STACK_OOS_PRICES = "greyhound/live/oos_prices_stack_all3"
STACK_OBJECTIVE = "stack_share"
#: the metrics tables are built from 2014, as the nightly build (greyhound-pit.yml) builds them: their constants are
#: fitted on the years before 2018
PIT_FIRST_YEAR = 2014
LEDGER_FIELDS = ["time_utc", "mode", "market_id", "selection_id", "track", "race_time", "trap", "dog", "p_model",
                 "price", "size_offered", "edge", "action", "stake", "matched", "avg_price", "bet_id", "status",
                 "error", "lay_liability", "lay_status", "minutes_to_off"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    """The tested rule's settings; the commission is always the account's rate on greyhounds (greyhound/costs.py)."""
    from greyhound import costs
    return {**json.loads(Path(path).read_text()), "commission": costs.commission()}


def live_switch_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("GREYHOUND_LIVE", "")).strip().lower() == "yes"


# --------------------------------------------------------------------------------------------------------------------
# Train and predict (GitHub's runners: S3, no Betfair)
# --------------------------------------------------------------------------------------------------------------------

def card_safe(features: list[str]) -> list[str]:
    return [f for f in features if f not in CARD_UNSAFE]


def fit(df: pd.DataFrame, features: list[str], before: str, date_from: str = "2019-01-01",
        objective: str = "binary", calibration: str | None = None):
    """The live model: every modelling race in [date_from, before), early-stopped on the last tenth. ``objective``
    'binary' on the winner, or 'logbsp': a regression on log(BSP) (the races with one), the fast loop's recipe
    (greyhound.fastloop BASE_PARAMS, learning rate 0.05), with the SP book's usual total from the fit's races so the
    prices can be rescaled to it (section 26.3's best_rescaled)."""
    import lightgbm as lgb
    d = modelling_rows(df[~df.is_card])
    d = d[(d.t >= date_from) & (d.t < before)]
    meta = {"before": before, "date_from": date_from, "objective": objective}
    if objective == "logbsp":
        from greyhound.fastloop import BASE_PARAMS
        d = d[pd.to_numeric(d.bsp, errors="coerce") > 1]
        y = np.log(np.clip(d.bsp.to_numpy(float), 1.01, 1000.0))
        params = dict(BASE_PARAMS, objective="regression", learning_rate=0.05)
        rounds, patience = 10000, 100
        race_sum = pd.Series(1.0 / d.bsp.to_numpy(float)).groupby(d.race_id.to_numpy()).sum()
        meta["book"] = float(np.median(race_sum))                 # the SP book's usual total (sum of 1/BSP a race)
    else:
        y, params, rounds, patience = d.won.to_numpy(float), PARAMS, 3000, 100
    cut = d.t.quantile(0.9)
    tr, es = d.t < cut, d.t >= cut
    m = lgb.train(params, lgb.Dataset(d.loc[tr, features], y[tr.to_numpy()]), rounds,
                  valid_sets=[lgb.Dataset(d.loc[es, features], y[es.to_numpy()])],
                  callbacks=[lgb.early_stopping(patience, verbose=False)])
    if objective == "logbsp" and calibration:
        # the stretch fitted on the early-stopping rows (the trees never trained on them), as the fast loop scores it
        from greyhound.fastloop import fit_stretch
        e = d[es]
        raw_es = m.predict(e[features], num_iteration=m.best_iteration)
        meta["stretch"] = fit_stretch(raw_es, y[es.to_numpy()], e.won.to_numpy(float),
                                      pd.factorize(e.race_id)[0], e.track.to_numpy(), calibration)
    return m, {**meta, "fit_rows": int(len(d)), "rounds": m.best_iteration}


def attach_blocks(df: pd.DataFrame, names) -> list[str]:
    """The matrix blocks (greyhound/blocks) added to the engine's output, in place; their feature names."""
    from greyhound import blocks as gblocks
    out = []
    for b in names or ():
        out += [f for f in gblocks.attach(df, b) if f not in out]
    return out


def price_cards(df: pd.DataFrame, booster, features: list[str], key: pd.DataFrame,
                meta: dict | None = None) -> pd.DataFrame:
    """The card rows of ``df`` priced and normalised within each race, with their Betfair market and selection
    (``key``: race_id, trap, market_id, selection_id, matched_history from ``greyhound.card.cards``)."""
    c = df[df.is_card]
    out = c[["race_id", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "race_class", "distance_m",
             "field"]].copy()
    out["p_raw"] = booster.predict(c[features]) if len(c) else []
    if (meta or {}).get("objective") == "logbsp":
        # the forecast SP: its chances normalised in the race, and the price rescaled to the SP book's usual total
        raw = out.p_raw.astype(float)
        out["p_model"] = race_norm(1.0 / np.exp(raw), out.race_id) if len(c) else []
        book = float((meta or {}).get("book") or 1.0)
        st = (meta or {}).get("stretch")
        if st and len(c):
            # batch 8 (4 Nov 2026): the BSP is more spread than the winners' chances, so the stretch (a share
            # proportional to pred_bsp^-b in each race, b by track) makes the SP forecast more accurate (RMSE of
            # log(BSP) 0.4127 -> 0.4053) and the win chance worse; it is applied to the SP forecast only, the chance
            # (what the early lay's forecast reads) left as the trees give it
            b = out.track.astype(str).map(st.get("by_track") or {}).fillna(st["b"]).to_numpy(float)
            centred = raw - raw.groupby(out.race_id).transform("mean")
            share = race_norm(pd.Series(np.exp(-b * centred.to_numpy()), index=out.index), out.race_id)
            out["pred_bsp"] = (1.0 / (book * share.astype(float))).round(3)
        else:
            out["pred_bsp"] = (1.0 / (book * out.p_model.astype(float))).round(3) if len(c) else []
    else:
        out["p_model"] = race_norm(out.p_raw, out.race_id) if len(c) else []
    out = out.merge(key, on=["race_id", "trap"], how="left")
    return out


def train(store, today: date) -> dict:
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices
    from greyhound.track import runs_through
    runs = runs_through(store, today - timedelta(days=1))
    eng = GreyhoundMetricsEngine(blocks=LIVE_BLOCKS)
    df = eng.calculate_all(runs, prices=load_prices(store, f"2014-{today.year}"))
    feats = card_safe(eng.features + attach_blocks(df, LIVE_MATRIX_BLOCKS))
    m, meta = fit(df, feats, before=f"{today:%Y-%m-%d}", date_from=LIVE_TRAIN_FROM, objective=LIVE_OBJECTIVE,
                  calibration=LIVE_CALIBRATION)
    imp = pd.Series(m.feature_importance("gain"), index=feats)
    meta.update({"features": feats, "blocks": list(eng.blocks), "matrix_blocks": list(LIVE_MATRIX_BLOCKS),
                 "oos_prices": LIVE_OOS_PRICES,
                 "card_unsafe_left_out": sorted(CARD_UNSAFE),
                 "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "top_features": {k: round(float(v), 4) for k, v in (imp / imp.sum()).sort_values(ascending=False)
                                  .head(20).items()}})
    store.put(MODEL_KEY, m.model_to_string(num_iteration=m.best_iteration).encode())
    store.put(META_KEY, json.dumps(meta, indent=1).encode())
    return {k: v for k, v in meta.items() if k != "features"}


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The rows the stack prices and is fitted on, as the fast loop's feature cache holds them (greyhound.fastloop
    ``build_cache``): the modelling races (four or more runners, no trials) from ``KEEP_FROM``, grouped by race in
    time order; the matrix blocks are computed on this frame, as on the cache."""
    from greyhound.fastloop import KEEP_FROM
    from greyhound.model import modelling_rows
    d = modelling_rows(df)
    d = d[d.t >= KEEP_FROM]
    return d.sort_values(["t", "race_id", "trap"], kind="stable").reset_index(drop=True)


def base_features(eng) -> list[str]:
    """The engine's card-safe features in the fast loop's order (greyhound.fastloop ``feature_set(..., 'card_safe')``)."""
    from greyhound.fastloop import feature_set
    return feature_set({"features": eng.block_features, "card_unsafe": sorted(CARD_UNSAFE)}, "card_safe")


def train_stack(store, today: date) -> dict:
    """The stack (greyhound/stack_model.py) on the 29 months to yesterday: the engine's features, the matrix blocks,
    the metrics tables as the nightly build left them (greyhound/pit, S3)."""
    from greyhound import stack_model as sm
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices
    from greyhound.track import runs_through
    runs = runs_through(store, today - timedelta(days=1))
    eng = GreyhoundMetricsEngine(blocks=LIVE_BLOCKS)
    df = eng.calculate_all(runs, prices=load_prices(store, f"2014-{today.year}"))
    del runs
    frame = model_frame(df[~df.is_card.fillna(False).astype(bool)])
    del df
    feats = sm.features(frame, base_features(eng))
    bundle = sm.fit(frame, feats, before=f"{today:%Y-%m-%d}", log=log.info)
    meta = {**bundle["meta"], "objective": STACK_OBJECTIVE, "features": feats, "blocks": list(eng.blocks),
            "oos_prices": STACK_OOS_PRICES, "card_unsafe_left_out": sorted(CARD_UNSAFE),
            "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    store.put(STACK_KEY, sm.to_bytes(bundle))
    store.put(STACK_META_KEY, json.dumps(meta, indent=1).encode())
    return {k: v for k, v in meta.items() if k != "features"}


def load_stack(store):
    from greyhound import stack_model as sm
    raw, meta = store.get(STACK_KEY), store.get(STACK_META_KEY)
    if not raw or not meta:
        return None, None
    return sm.from_bytes(raw), json.loads(meta)


def card_tables(store, day: date, card: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The metrics tables (greyhound/pit) built from 2014 with the day's card as their last day, each cut to the
    card's rows: what the nightly build gives a run once its result is in (tests/test_greyhound_pit.py
    ``test_a_card_gets_the_features_its_run_had``)."""
    from greyhound import pit, stack_model as sm
    from greyhound.card import with_cards
    from greyhound.data import clean
    from greyhound.model import load_prices
    from greyhound.pit import core
    from greyhound.track import runs_through
    hist = runs_through(store, day - timedelta(days=1), first_year=PIT_FIRST_YEAR)
    runs = clean(with_cards(hist, card))
    del hist
    ctx = {"fetch_lag": core.raw_fetch_lag(store), "prices": load_prices(store, f"{PIT_FIRST_YEAR}-{day.year}")}
    tabs = pit.build_tables(list(sm.PIT_TABLES), runs, None, ctx, keep=True)
    keys = runs.loc[runs.is_card, pit.KEY].drop_duplicates()
    return {n: t.merge(keys, on=pit.KEY) for n, t in tabs.items()}


def predict_stack(store, day: date, bundle: dict, meta: dict, s3=None) -> dict:
    """Today's card priced by the stack: the engine's features (history to yesterday and the card), the matrix blocks,
    the metrics tables built with the card (``card_tables``)."""
    from betfair_recorder import _read_day_csv
    from greyhound import blocks as gblocks, pit, stack_model as sm
    from greyhound.card import cards, with_cards
    from greyhound.data import clean
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices
    from greyhound.track import runs_through
    markets = _read_day_csv(day, "markets_greyhound.csv", s3=s3)
    if markets is None or not len(markets):
        return {"day": f"{day:%Y-%m-%d}", "note": "no greyhound catalogue recorded yet"}
    hist = runs_through(store, day - timedelta(days=1))
    card = cards(markets, hist)
    card = card[card.race_date == f"{day:%Y-%m-%d}"]
    card = drop_removed(card, removed_runners(_read_day_csv(day, "books_greyhound.csv", s3=s3)))
    if not len(card):
        return {"day": f"{day:%Y-%m-%d}", "note": "no GB win markets on the card"}
    t0 = time.monotonic()
    tabs = card_tables(store, day, card)
    t_tables = round(time.monotonic() - t0)
    eng = GreyhoundMetricsEngine(blocks=tuple(meta.get("blocks", LIVE_BLOCKS)))
    df = eng.calculate_all(clean(with_cards(hist, card)), prices=load_prices(store, f"2014-{day.year}"))
    del hist
    frame = model_frame(df)
    del df
    for b in sm.MATRIX_BLOCKS:
        if not b.startswith("pit_"):
            gblocks.attach(frame, b)
    for name, tab in tabs.items():                    # the card's rows only: the history is not priced
        cols = [c for c in tab.columns if c.startswith(pit.module(name).PREFIX)]
        j = frame[pit.KEY].merge(tab[pit.KEY + cols].drop_duplicates(pit.KEY), on=pit.KEY, how="left")
        for c in cols:
            frame[c] = j[c].to_numpy()
    missing = [f for f in bundle["feats"] if f not in frame.columns]
    if missing:
        raise SystemExit(f"the live model reads features this build no longer makes: {missing[:10]}")
    c = frame[frame.is_card.fillna(False).astype(bool)]
    priced = sm.price(bundle, c)
    out = c[["race_id", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "race_class", "distance_m",
             "field"]].copy()
    for k in ("p_raw", "p_model", "pred_bsp", "pred_bsp_raw"):
        out[k] = priced[k].to_numpy()
    out["pred_bsp"] = out.pred_bsp.round(3)
    key = card[["race_id", "trap", "market_id", "selection_id", "matched_history"]].drop_duplicates(["race_id", "trap"])
    pred = out.merge(key, on=["race_id", "trap"], how="left")
    pred["model_trained_utc"] = meta.get("trained_utc")
    pred["model_objective"] = STACK_OBJECTIVE                  # the early lay's forecast fit matches it
    pred["oos_prices"] = meta.get("oos_prices") or STACK_OOS_PRICES
    buf = io.StringIO()
    pred.to_csv(buf, index=False)
    store.put(f"{PREFIX}/predictions/{day:%Y-%m-%d}.csv", buf.getvalue().encode())
    unpriced = int(card.race_id.nunique() - pred.race_id.nunique())
    return {"day": f"{day:%Y-%m-%d}", "model": STACK_OBJECTIVE, "markets": int(pred.market_id.nunique()),
            "runners": len(pred), "races_not_priced": unpriced,      # under four runners: not a race the model prices
            "dogs_with_history": round(float(pred.matched_history.mean()), 3) if len(pred) else 0,
            "unknown_grade_or_trip": int(card.race_class.isna().sum()), "tables_seconds": t_tables}


def load_model(store):
    import lightgbm as lgb
    text, meta = store.get(MODEL_KEY), store.get(META_KEY)
    if not text or not meta:
        return None, None
    return lgb.Booster(model_str=text.decode()), json.loads(meta)


def predict(store, day: date, s3=None) -> dict:
    from betfair_recorder import _read_day_csv
    from greyhound.card import cards, with_cards
    from greyhound.data import clean
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices
    from greyhound.track import runs_through
    bundle, smeta = load_stack(store)
    if bundle is not None:
        return predict_stack(store, day, bundle, smeta, s3=s3)
    log.warning("no stack model stored yet (%s): the log-BSP model prices the card", STACK_KEY)
    booster, meta = load_model(store)
    if booster is None:
        raise SystemExit("no live model: run --train first")
    markets = _read_day_csv(day, "markets_greyhound.csv", s3=s3)
    if markets is None or not len(markets):
        return {"day": f"{day:%Y-%m-%d}", "note": "no greyhound catalogue recorded yet"}
    hist = runs_through(store, day - timedelta(days=1))
    card = cards(markets, hist)
    card = card[card.race_date == f"{day:%Y-%m-%d}"]
    card = drop_removed(card, removed_runners(_read_day_csv(day, "books_greyhound.csv", s3=s3)))
    eng = GreyhoundMetricsEngine(blocks=tuple(meta.get("blocks", ["parity"])))
    df = eng.calculate_all(clean(with_cards(hist, card)), prices=load_prices(store, f"2014-{day.year}"))
    attach_blocks(df, meta.get("matrix_blocks", []))
    missing = [f for f in meta["features"] if f not in df.columns]
    if missing:
        raise SystemExit(f"the live model reads features the engine no longer makes: {missing[:10]}")
    key = card[["race_id", "trap", "market_id", "selection_id", "matched_history"]].drop_duplicates(["race_id", "trap"])
    pred = price_cards(df, booster, meta["features"], key, meta)
    pred["model_trained_utc"] = meta.get("trained_utc")
    pred["model_objective"] = meta.get("objective", "binary")     # the early lay's forecast fit matches it
    pred["oos_prices"] = meta.get("oos_prices") or ""               # the prices that forecast is fitted on
    buf = io.StringIO()
    pred.to_csv(buf, index=False)
    store.put(f"{PREFIX}/predictions/{day:%Y-%m-%d}.csv", buf.getvalue().encode())
    return {"day": f"{day:%Y-%m-%d}", "markets": int(pred.market_id.nunique()), "runners": len(pred),
            "dogs_with_history": round(float(pred.matched_history.mean()), 3) if len(pred) else 0,
            "unknown_grade_or_trip": int(card.race_class.isna().sum())}


def removed_runners(books) -> set:
    """The (market, selection) pairs Betfair has marked removed (a non-runner, a reserve not in) in the recorded
    books: the card leaves them out, as the result will."""
    if books is None or not len(books) or "runner_status" not in books:
        return set()
    b = books.sort_values("polled_utc") if "polled_utc" in books else books
    last = b.drop_duplicates(["market_id", "selection_id"], keep="last")
    r = last[last.runner_status.astype(str).str.upper() == "REMOVED"]
    return set(zip(r.market_id.astype(str), pd.to_numeric(r.selection_id, errors="coerce")))


def drop_removed(card: pd.DataFrame, removed: set) -> pd.DataFrame:
    if not removed or not len(card):
        return card
    gone = [(m, s) in removed for m, s in zip(card.market_id.astype(str), card.selection_id)]
    out = card[~pd.Series(gone, index=card.index)].copy()
    out["runners"] = out.groupby("market_id").trap.transform("size")
    return out


def card_check(store, day: date, blocks=LIVE_BLOCKS, s3=None, matrix_blocks=LIVE_MATRIX_BLOCKS) -> dict:
    """Are the card's features the results' features? A past day priced twice: from the morning's card (the
    recorder's catalogue, history to the day before) and from GBGB's results for it. Each feature's share of runners
    where the two agree (both missing, or within 1e-6); a card-safe feature should agree for every dog with a GBGB
    history."""
    from betfair_recorder import _read_day_csv
    from greyhound.card import cards, with_cards
    from greyhound.data import clean
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices
    from greyhound.track import runs_through
    ds = f"{day:%Y-%m-%d}"
    markets = _read_day_csv(day, "markets_greyhound.csv", s3=s3)
    if markets is None or not len(markets):
        return {"day": ds, "note": "no greyhound catalogue recorded"}
    full = runs_through(store, day)
    hist = full[full.race_date.astype(str) < ds]
    card = cards(markets, hist)
    card = drop_removed(card[card.race_date == ds], removed_runners(_read_day_csv(day, "books_greyhound.csv", s3=s3)))
    # every runner is priced (a dog with no history still counts in its field); only the matched are compared
    m = card[card.matched_history]
    matched = set(zip(m.track, m.race_time.astype(str).str[:5], m.trap.astype(float)))
    prices = load_prices(store, f"2014-{day.year}")
    eng = GreyhoundMetricsEngine(blocks=tuple(blocks))
    a = eng.calculate_all(clean(with_cards(hist, card)), prices=prices)
    b = GreyhoundMetricsEngine(blocks=tuple(blocks)).calculate_all(clean(full), prices=prices)
    checked = list(eng.features) + attach_blocks(a, matrix_blocks)
    attach_blocks(b, matrix_blocks)
    k = ["track", "race_time", "trap"]
    a = a[a.is_card.fillna(False).astype(bool)].assign(race_time=lambda x: x.race_time.astype(str).str[:5])
    b = b[(b.race_date.astype(str) == ds)].assign(race_time=lambda x: x.race_time.astype(str).str[:5])
    j = a.merge(b, on=k, suffixes=("_card", "_res"))
    j = j[[(t, r, float(p)) in matched for t, r, p in zip(j.track, j.race_time, j.trap)]]
    agree = {}
    for f in checked:
        if f in k:                                           # a join key: equal by construction
            agree[f] = 1.0
            continue
        x, y = j[f"{f}_card"].astype(float), j[f"{f}_res"].astype(float)
        agree[f] = round(float(((x.isna() & y.isna()) | ((x - y).abs() <= 1e-6)).mean()), 4) if len(j) else None
    bad = {f: v for f, v in sorted(agree.items(), key=lambda kv: kv[1] or 0) if v is not None and v < 0.99}
    examples = {}
    for f in [f for f in ("field", "grade", "trainer_win_z", "gp_lead_share", "hb_age_vs_youngest") if f in bad]:
        x, y = j[f"{f}_card"].astype(float), j[f"{f}_res"].astype(float)
        off = j[~((x.isna() & y.isna()) | ((x - y).abs() <= 1e-6))]
        cols = [c for c in ("track", "race_time", "trap", "dog_name_card", "race_class_card", "race_class_res",
                            "trainer_card", "trainer_res") if c in j.columns]
        examples[f] = off[cols].assign(card=x[off.index], res=y[off.index]).head(6).astype(str).to_dict("records")
    return {"day": ds, "runners_compared": len(j), "features": len(agree),
            "agree_all": sum(1 for v in agree.values() if v is not None and v >= 0.99),
            "disagree": bad, "disagree_not_card_unsafe": [f for f in bad if f not in CARD_UNSAFE],
            "examples": examples}


# --------------------------------------------------------------------------------------------------------------------
# Trade (the UK runner)
# --------------------------------------------------------------------------------------------------------------------

def stake_for(price: float, cfg: dict) -> float | None:
    """The stake on a back at ``price``. Trading out, the lay at the Betfair SP needs the back's winnings to reach
    Betfair's smallest SP liability, so the stake is set to win ``target`` (at least ``min_stake``) and a price whose
    capped stake cannot win that much is not backed. Held, a level stake."""
    lim = cfg["limits"]
    if not cfg.get("trade_out", True):
        return float(lim["level_stake"])
    s = max(lim["min_stake"], math.ceil(100 * cfg["target"] / (price - 1.0)) / 100)
    s = min(s, lim["max_stake"])
    return s if s * (price - 1.0) + 1e-9 >= lim["min_bsp_liability"] else None


def decide(market_pred: pd.DataFrame, book, cfg: dict, done=frozenset(), have=None) -> list[dict]:
    """One market's decisions this poll: every runner still in it and not done (``done``: its stake is on, or it can
    never be backed), its edge at the best back price, and the back the rule makes (before the race's limit). Nothing
    is final but a stake complete: a runner without an edge now is looked at again each poll until the off, and a
    back takes whatever is offered within the edge (``have``: what is matched on it already), topped up later to the
    stake. With ``tight`` set, a runner waits until its back and lay are within ``tight`` of each other (an early
    greyhound book is placeholder offers, a back at 1.01 against a lay at 1000)."""
    have = have or {}
    lim = cfg["limits"]
    active = {sid: q for sid, q in book.runners.items() if q.status == "ACTIVE"}
    p = market_pred.set_index("selection_id")
    if not active or any(sid not in p.index for sid in active):
        # a runner the prices lack (a reserve in, a field priced before it was settled): the race is looked at again
        # each poll, as the prices are reloaded, until it can be priced whole or the off
        return [{"selection_id": None, "action": "retry", "error": "the field as it stands is not all priced"}]
    probs = p.loc[list(active), "p_model"].astype(float)
    probs = probs / probs.sum()                                 # renormalised over the runners still in
    smallest = float(lim.get("min_order", 1.0))
    out = []
    for sid, q in active.items():
        if sid in done:
            continue
        bb = q.best_back
        row = {"selection_id": sid, "p_model": round(float(probs[sid]), 4), "price": bb[0] if bb else None,
               "size_offered": bb[1] if bb else None, "track": p.loc[sid, "track"],
               "race_time": p.loc[sid, "race_time"], "trap": p.loc[sid, "trap"], "dog": p.loc[sid, "dog_name"]}
        if not bb:
            out.append({**row, "action": "none", "error": "no back price"})
            continue
        price, size = bb
        bl, tight = q.best_lay, cfg.get("tight")
        if tight and (not bl or bl[0] / price > tight):
            out.append({**row, "action": "wait"})
            continue
        edge = probs[sid] * (1 + (price - 1) * (1 - cfg["commission"])) - 1
        row["edge"] = round(float(edge), 4)
        stake = stake_for(price, cfg) if price > 1 else None
        if edge <= cfg["edge"] or price > lim["max_price"] or price < lim["min_price"]:
            out.append({**row, "action": "none"})
            continue
        if stake is None:
            out.append({**row, "action": "skip", "error": "the stake cannot reach the smallest SP lay"})
            continue
        want = math.floor(100 * (stake - have.get(sid, 0.0)) + 1e-6) / 100
        # down the ladder while the edge holds: the limit is the lowest price taken (the order fills at it or better)
        limit, got = None, 0.0
        for lv_price, lv_size in q.back:
            e = probs[sid] * (1 + (lv_price - 1) * (1 - cfg["commission"])) - 1
            if e <= cfg["edge"] or lv_price < lim["min_price"]:
                break
            got += lv_size
            limit = lv_price
            if got + 1e-9 >= want:
                break
        take = math.floor(100 * min(want, got) + 1e-6) / 100
        if limit is None or take + 1e-9 < smallest:
            out.append({**row, "action": "retry", "stake": take,
                        "error": f"GBP{got:.2f} offered within the edge, under the smallest order"})
        else:
            out.append({**row, "action": "back", "stake": take, "target": stake, "price": limit})
    return out


class Trader:
    def __init__(self, store, exchange, cfg: dict, day: date, mode: str, ledger_path: Path, kill=lambda: False,
                 now=lambda: datetime.now(timezone.utc)):
        self.store, self.x, self.cfg, self.day, self.mode = store, exchange, cfg, day, mode
        self.ledger_path, self.kill, self.now = ledger_path, kill, now
        self.decided: set[str] = set()                     # markets with every runner done
        self.decided_sel: set[tuple[str, int]] = set()      # (market, selection) done: its stake on, or never
        self.have: dict = {}                                # (market, selection) -> matched so far
        self.tries: dict = {}                               # (market, selection) -> orders sent
        self.turnover, self.bets, self.race_stake = 0.0, 0, {}
        self.noted: set = set()                             # (market, selection, reason) already written
        self.preds: pd.DataFrame | None = None
        self._loaded_at = None
        self._resume()

    def _target(self) -> float:
        return float(self.cfg["limits"]["level_stake"])

    def _resume(self) -> None:
        """A restarted session reads its own ledger: what is matched stays matched, a stake complete is not backed
        again, and the race's limit carries on."""
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open() as f:
            for r in csv.DictReader(f):
                if not r.get("selection_id"):
                    continue
                key = (r["market_id"], int(float(r["selection_id"])))
                if r["action"] == "back":
                    self.tries[key] = self.tries.get(key, 0) + 1
                    if r["status"] == "SUCCESS":
                        m = float(r["matched"] or 0)
                        self.turnover += m
                        self.bets += 1
                        self.race_stake[r["market_id"]] = self.race_stake.get(r["market_id"], 0.0) + m
                        self.have[key] = self.have.get(key, 0.0) + m
                    elif r["status"] == "UNKNOWN":
                        self.decided_sel.add(key)           # its fate unread: never sent twice
                elif r["action"] == "skip" or (r["action"] == "none" and not self.cfg.get("reconsider", True)):
                    self.decided_sel.add(key)
        for key, m in self.have.items():
            if m + 0.01 >= self._target() or not self.cfg.get("top_up", True):
                self.decided_sel.add(key)

    def _write(self, rows: list[dict]) -> None:
        new = not self.ledger_path.exists()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
        try:
            self.store.put(f"{PREFIX}/{self.mode}/{self.day:%Y-%m-%d}/ledger.csv", self.ledger_path.read_bytes())
        except Exception as exc:                                 # the local ledger is the record; S3 is its copy
            log.warning("ledger not copied to S3 (%s)", exc)

    def load_predictions(self) -> None:
        t = self.now()
        if self._loaded_at and (t - self._loaded_at).total_seconds() < 600:
            return
        raw = self.store.get(f"{PREFIX}/predictions/{self.day:%Y-%m-%d}.csv")
        self._loaded_at = t
        if raw:
            p = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
            p["off"] = pd.to_datetime(p.race_date + " " + p.race_time.astype(str).str[:5]).dt.tz_localize(
                "Europe/London").dt.tz_convert("UTC")
            self.preds = p

    def _open(self, t: pd.Timestamp) -> pd.Series:
        """The races still to bet on: up to ``stop_before_off`` minutes before the off; at 0, to the off itself and
        past it by ``late_start_minutes`` (a race goes off late; Betfair suspends the market at the off, and a
        suspended book is never bet on)."""
        stop = float(self.cfg.get("stop_before_off", 0))
        if stop > 0:
            return self.preds.off > t + pd.Timedelta(minutes=stop)
        return self.preds.off + pd.Timedelta(minutes=float(self.cfg.get("late_start_minutes", 5))) > t

    def step(self, horizon_minutes: float | None = None) -> int:
        """One poll: every race still open (``horizon_minutes``: only those off within it), every runner not done
        looked at in its book now. Returns how many races were done this poll (every runner's stake on or never)."""
        if self.kill():
            log.warning("kill switch on: no new bets")
            return 0
        self.load_predictions()
        if self.preds is None:
            return 0
        t = pd.Timestamp(self.now())
        lim = self.cfg["limits"]
        sel = (~self.preds.market_id.isin(self.decided)) & self._open(t)
        window = self.cfg.get("decide_window_minutes")           # only this close to the off (the tested T-1 rule)
        if window is not None:
            sel &= self.preds.off <= t + pd.Timedelta(minutes=float(window))
        if horizon_minutes is not None:
            sel &= self.preds.off <= t + pd.Timedelta(minutes=horizon_minutes)
        todo = self.preds[sel]
        mids = sorted(todo.market_id.dropna().unique())
        if not mids:
            return 0
        books = self.x.books(mids)
        done = 0
        for mid in mids:
            b = books.get(mid)
            if b is None or b.status != "OPEN" or b.inplay:
                continue
            done_here = {sid for m, sid in self.decided_sel if m == mid}
            have = {sid: v for (m, sid), v in self.have.items() if m == mid}
            rows = decide(todo[todo.market_id == mid], b, self.cfg, done=done_here, have=have)
            again = ("wait", "retry", "none") if self.cfg.get("reconsider", True) else ("wait", "retry")
            open_here = any(r["action"] in again for r in rows)
            rows = [r for r in rows if r["action"] != "wait"]
            mins = round(float((todo[todo.market_id == mid].off.iloc[0] - t).total_seconds() / 60), 1)
            out = []
            for r in rows:
                r.update({"time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": self.mode, "market_id": mid,
                          "minutes_to_off": mins})
                sid = r.get("selection_id")
                key = (mid, int(sid)) if sid is not None else (mid, None)
                if r["action"] == "back":
                    stake = r["stake"]
                    cap = lim.get("max_race_stake")                  # null: no race limit
                    room = math.inf if cap is None else \
                        math.floor(100 * (cap - self.race_stake.get(mid, 0.0)) + 1e-6) / 100
                    day_cap, bet_cap = lim.get("max_daily_turnover"), lim.get("max_bets_per_day")
                    if room + 1e-9 < float(lim.get("min_order", 1.0)) \
                            or (day_cap is not None and self.turnover + min(stake, room) > day_cap) \
                            or (bet_cap is not None and self.bets >= bet_cap):
                        r.update({"action": "skip", "error": "the day's or the race's limit"})
                    elif self.tries.get(key, 0) >= int(self.cfg.get("max_tries", 30)):
                        r.update({"action": "skip", "error": "too many orders refused"})
                    else:
                        stake = min(stake, room)
                        n = self.tries.get(key, 0) + 1
                        self.tries[key] = n
                        ref = f"gh-{self.day:%m%d}-{str(mid)[-8:]}-{sid}-{n}"
                        fill = self.x.back(mid, sid, r["price"], stake, ref, min_fill=stake)
                        r.update({"stake": stake, "matched": fill.matched, "avg_price": fill.avg_price,
                                  "bet_id": fill.bet_id, "status": fill.status, "error": fill.error})
                        if fill.status == "SUCCESS" and fill.matched > 0:
                            self.turnover += fill.matched
                            self.bets += 1
                            self.race_stake[mid] = self.race_stake.get(mid, 0.0) + fill.matched
                            self.have[key] = self.have.get(key, 0.0) + fill.matched
                            if self.have[key] + 0.01 < r.get("target", stake) and self.cfg.get("top_up", True):
                                open_here = True            # topped up later, while the edge holds
                            else:
                                self.decided_sel.add(key)
                            if self.cfg.get("trade_out", True):
                                liab = math.floor(100 * fill.matched * (fill.avg_price - 1.0)) / 100
                                if liab >= lim["min_bsp_liability"]:
                                    lay = self.x.lay_at_bsp(mid, sid, liab, ref + "-L")
                                    r.update({"lay_liability": liab, "lay_status": lay.status})
                                else:
                                    r.update({"lay_status": f"not laid: GBP{liab:.2f} under the smallest SP lay"})
                        elif fill.status == "UNKNOWN":
                            self.decided_sel.add(key)       # its fate unread: never sent twice
                        else:
                            open_here = True                # not filled (the money gone, funds): tried again
                if r["action"] == "skip" and sid is not None:
                    self.decided_sel.add(key)
                if r["action"] == "none" and sid is not None and not self.cfg.get("reconsider", True):
                    self.decided_sel.add(key)                   # decided once: no edge at its look is final
                if r["action"] in ("retry", "none"):        # noted once a runner and reason, not each poll
                    note = (mid, sid, r["action"], str(r.get("error", ""))[:24])
                    if note in self.noted:
                        continue
                    self.noted.add(note)
                if r["action"] != "none" or self.cfg.get("ledger_all", True):
                    out.append(r)
            if out:
                self._write(out)
            if not open_here:
                self.decided.add(mid)
                done += 1
        return done

    def run(self, until: datetime, poll_seconds: float = 60.0, sleep=time.sleep) -> dict:
        """Every race each ``poll_seconds``; between, every ``near_off_poll_seconds`` the races off within
        ``near_off_minutes`` (the last money before the off)."""
        fast = self.cfg.get("near_off_poll_seconds")
        near = float(self.cfg.get("near_off_minutes", 3))
        last_full = None
        while self.now() < until:
            try:
                if fast and last_full is not None and (self.now() - last_full).total_seconds() < poll_seconds:
                    self.step(horizon_minutes=near)
                else:
                    last_full = self.now()
                    self.step()
            except Exception as exc:                            # one failed poll: the next tries again
                log.warning("poll failed (%s)", exc)
            sleep(fast or poll_seconds)
        return {"day": f"{self.day:%Y-%m-%d}", "mode": self.mode, "markets_decided": len(self.decided),
                "bets": self.bets, "turnover": round(self.turnover, 2)}


def book_recorder(day: date, env=None):
    """Every book the trader reads (every 5 seconds in a race's last 3 minutes, where the greyhound market's money is),
    kept with BETFAIR_RECORD=1 (set on the UK runner) in its own day files, ``books_greyhound_trader.csv`` beside the
    recorder's 60-second ``books_greyhound.csv`` (the owner, 2 Nov 2026: the trade-out research needs the books the
    trader sees). Research data, never in the way of the trading: None when off or unavailable."""
    env = os.environ if env is None else env
    if str(env.get("BETFAIR_RECORD", "")).strip() != "1":
        return None
    try:
        from betfair_recorder import DayRecorder
        return DayRecorder(day, tag="greyhound_trader")
    except Exception as exc:                                    # the trading goes on without a record
        log.warning("greyhound book recorder unavailable (%s): trading without a record of the books", exc)
        return None


KILL_CHECK_SECONDS = 5.0      # how often the S3 STOP objects are read (the owner, 5 Nov 2026, the audit: was 60 s)


def kill_switch(store, every: float = KILL_CHECK_SECONDS):
    """True while S3 holds greyhound/STOP or trading/STOP. Fails safe: an S3 read that errors (anything but 'no such
    object') counts as STOP, so a trader that cannot see the switch stops laying."""
    state = {"at": None, "on": False}

    def check() -> bool:
        now = datetime.now(timezone.utc)
        if state["at"] and (now - state["at"]).total_seconds() < every:
            return state["on"]
        state["at"] = now
        state["on"] = any(store.s3_exists(k) if hasattr(store, "s3_exists") else _exists(store, k)
                          for k in ("greyhound/STOP", "trading/STOP"))
        return state["on"]
    return check


def _exists(store, key: str) -> bool:
    """An object at the bucket's root (the kill switches are outside the sources prefix)."""
    if store.root:
        return (Path(store.root).parent / key).exists()
    try:
        store.s3.head_object(Bucket=store.bucket, Key=key)
        return True
    except Exception as exc:
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        log.warning("kill switch %s unreadable (%s): treated as STOP", key, str(exc)[:200])
        return True


# --------------------------------------------------------------------------------------------------------------------
# Settle
# --------------------------------------------------------------------------------------------------------------------

def settle_rows(ledger: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Each matched back with its result and BSP (the recorder's settled books), its P&L held to the result and,
    with the lay at SP, traded out."""
    b = ledger[(ledger.action == "back") & (ledger.status == "SUCCESS")].copy()
    if not len(b):
        return b
    f = final.assign(market_id=final.market_id.astype(str),
                     selection_id=pd.to_numeric(final.selection_id, errors="coerce"))
    f = f.sort_values("polled_utc").drop_duplicates(["market_id", "selection_id"], keep="last")
    b["market_id"] = b.market_id.astype(str)
    b["selection_id"] = pd.to_numeric(b.selection_id, errors="coerce")
    b = b.merge(f[["market_id", "selection_id", "runner_status", "sp_actual"]], on=["market_id", "selection_id"],
                how="left")
    for c in ("matched", "avg_price", "lay_liability"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    won = b.runner_status == "WINNER"
    known = b.runner_status.isin(["WINNER", "LOSER"])
    b["bsp"] = pd.to_numeric(b.sp_actual, errors="coerce")
    b["pnl_back"] = np.where(won, b.matched * (b.avg_price - 1) * (1 - COM), -b.matched).astype(float)
    liab = b.lay_liability.fillna(0.0)
    b["pnl_lay"] = np.where(liab > 0, np.where(won, -liab, liab / (b.bsp - 1) * (1 - COM)), 0.0)
    b["pnl"] = b.pnl_back + b.pnl_lay
    b["clv"] = b.avg_price / b.bsp - 1
    b.loc[~known, ["pnl_back", "pnl_lay", "pnl"]] = np.nan              # removed or not settled yet
    b["result"] = b.runner_status
    return b


def settle(store, day: date, mode: str, s3=None) -> dict:
    from betfair_recorder import _read_day_csv
    raw = store.get(f"{PREFIX}/{mode}/{day:%Y-%m-%d}/ledger.csv")
    if not raw:
        return {"day": f"{day:%Y-%m-%d}", "mode": mode, "note": "not traded"}
    ledger = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    books = _read_day_csv(day, "books_greyhound.csv", s3=s3)
    final = books[books.source == "final"] if books is not None else pd.DataFrame(
        columns=["market_id", "selection_id", "runner_status", "sp_actual", "polled_utc"])
    s = settle_rows(ledger, final)
    buf = io.StringIO()
    s.to_csv(buf, index=False)
    store.put(f"{PREFIX}/{mode}/{day:%Y-%m-%d}/settled.csv", buf.getvalue().encode())
    out = {"day": f"{day:%Y-%m-%d}", "mode": mode, "markets_decided": int(ledger.market_id.nunique()),
           "signals": int((ledger.action.isin(["back", "skip"])).sum()), "bets": len(s)}
    if len(s):
        done = s.dropna(subset=["pnl"])
        out.update({"staked": round(float(s.matched.sum()), 2), "settled": len(done),
                    "pnl": round(float(done.pnl.sum()), 2), "pnl_held": round(float(done.pnl_back.sum()), 2),
                    "roi%": round(100 * float(done.pnl.sum() / done.matched.sum()), 2) if len(done) else None,
                    "clv%": round(100 * float(s.clv.mean()), 2) if s.clv.notna().any() else None})
    return out


def running(store, day: date, mode: str = "paper", s3=None) -> dict:
    """The day so far, a row per greyhound market the trader looked at (the owner, 3 Nov 2026: "keep a running table
    throughout the day on how the paper trader would have performed on each greyhound market"): its bets, the money
    staked, the average price taken, the BSP, the CLV, and, once the recorder has the market's settled book, the P&L
    traded out at the SP and held to the result, with the running total. Read from the trader's ledger as it is copied
    to S3 and the recorder's greyhound books (their settled reads, uploaded every 15 minutes)."""
    from betfair_recorder import _read_day_csv
    ds = f"{day:%Y-%m-%d}"
    raw = store.get(f"{PREFIX}/{mode}/{ds}/ledger.csv")
    if not raw:
        return {"day": ds, "mode": mode, "note": "no ledger yet today", "markets": []}
    ledger = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    finals = []
    for name in ("books_greyhound.csv", "books_greyhound_live.csv"):      # the recorder's, and the live-key record's
        books = _read_day_csv(day, name, s3=s3)                           # settled reads (greyhound/book_record.py)
        if books is not None and "source" in books:
            finals.append(books[books.source == "final"])
    final = pd.concat(finals, ignore_index=True) if finals else pd.DataFrame(
        columns=["market_id", "selection_id", "runner_status", "sp_actual", "polled_utc"])
    if len(final):                                  # one row a runner: one with a BSP over one without, then the latest
        final = final.assign(_has=pd.to_numeric(final.sp_actual, errors="coerce").notna(),
                             market_id=final.market_id.astype(str),
                             selection_id=pd.to_numeric(final.selection_id, errors="coerce"))
        final = final.sort_values(["_has", "polled_utc"]).drop_duplicates(
            ["market_id", "selection_id"], keep="last").drop(columns="_has")
    s = settle_rows(ledger, final)
    ledger["race_time"] = ledger.race_time.astype(str)
    looked = ledger.groupby("market_id").agg(track=("track", "first"), race_time=("race_time", "first"),
                                             runners=("selection_id", "nunique"))
    rows = []
    for mid, m in looked.sort_values("race_time").iterrows():
        b = s[s.market_id == mid] if len(s) else s
        bets = int(len(b))
        row = {"race_time": m.race_time, "track": m.track, "market_id": mid, "runners_seen": int(m.runners),
               "bets": bets}
        if bets:
            settled = b.pnl.notna().all()
            row.update({"staked": round(float(b.matched.sum()), 2),
                        "avg_price": round(float((b.avg_price * b.matched).sum() / b.matched.sum()), 2),
                        "bsp": round(float((b.bsp * b.matched).sum() / b.matched.sum()), 2) if b.bsp.notna().all()
                        else None,
                        "clv%": round(100 * float(b.clv.mean()), 2) if b.clv.notna().all() else None,
                        "result": ", ".join(sorted(set(b.result.dropna().astype(str)))) or "pending",
                        "pnl": round(float(b.pnl.sum()), 2) if settled else None,
                        "pnl_held": round(float(b.pnl_back.sum()), 2) if settled else None})
        rows.append(row)
    t = pd.DataFrame(rows)
    out = {"day": ds, "mode": mode, "updated_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "markets_seen": int(len(t)), "markets_bet": int((t.bets > 0).sum()) if len(t) else 0,
           "bets": int(t.bets.sum()) if len(t) else 0, "markets": rows}
    if len(t) and "pnl" in t:
        done = t[t.pnl.notna()]
        out.update({"markets_settled": int(len(done)), "staked_settled": round(float(done.staked.sum()), 2),
                    "pnl": round(float(done.pnl.sum()), 2), "pnl_held": round(float(done.pnl_held.sum()), 2),
                    "roi%": round(100 * float(done.pnl.sum() / done.staked.sum()), 2) if len(done) else None,
                    "clv%": round(100 * float(s.clv.mean()), 2) if s.clv.notna().any() else None,
                    "pending": int(((t.bets > 0) & t.pnl.isna()).sum())})
    return out


def running_markdown(r: dict) -> str:
    """The running table as markdown: the day's totals, then a row per market bet on (in race order) with the
    running total of the P&L traded out; the markets looked at and not bet on, counted."""
    lines = [f"## Greyhound {r['mode']} trader, {r['day']} (updated {r.get('updated_utc', '-')} UTC)", ""]
    if not r.get("markets"):
        return "\n".join(lines + [r.get("note", "no markets yet")]) + "\n"
    lines += [f"Markets looked at {r['markets_seen']}, bet on {r['markets_bet']}, bets {r['bets']}; settled "
              f"{r.get('markets_settled', 0)} (pending {r.get('pending', 0)}). Traded out at the SP: "
              f"**GBP{r.get('pnl', 0):+,.2f}** on GBP{r.get('staked_settled', 0):,.2f} staked "
              f"(ROI {r.get('roi%')}%); held to the result GBP{r.get('pnl_held', 0):+,.2f}; mean CLV {r.get('clv%')}%. "
              f"{100 * COM:g}% commission; paper only.", "",
              "| Race (UK) | Track | Bets | Staked | Avg price | BSP | CLV% | Result | P&L traded out | Held | Running |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    total = 0.0
    for m in r["markets"]:
        if not m.get("bets"):
            continue
        if m.get("pnl") is not None:
            total += m["pnl"]
        lines.append(f"| {m['race_time']} | {m['track']} | {m['bets']} | {m.get('staked')} | {m.get('avg_price')} | "
                     f"{m.get('bsp')} | {m.get('clv%')} | {m.get('result')} | {m.get('pnl')} | {m.get('pnl_held')} | "
                     f"{round(total, 2)} |")
    return "\n".join(lines) + "\n"


def summary(store, mode: str) -> dict:
    keys = sorted(k for k in store.listing(f"{PREFIX}/{mode}/") if k.endswith("/settled.csv"))
    parts = [pd.read_csv(io.BytesIO(store.get(k))) for k in keys]
    parts = [p for p in parts if len(p)]
    if not parts:
        return {"mode": mode, "days": len(keys), "bets": 0}
    s = pd.concat(parts, ignore_index=True).dropna(subset=["pnl"])
    r = s.pnl / s.matched
    se = float(np.sqrt(((r - r.mean()) ** 2 * s.matched).sum() / s.matched.sum() / max(1, len(s) - 1)))
    return {"mode": mode, "days": len(keys), "bets": len(s), "staked": round(float(s.matched.sum()), 2),
            "pnl": round(float(s.pnl.sum()), 2), "pnl_held": round(float(s.pnl_back.sum()), 2),
            "roi%": round(100 * float(s.pnl.sum() / s.matched.sum()), 2),
            "roi_90%": [round(100 * (float(s.pnl.sum() / s.matched.sum()) - 1.645 * se), 2),
                        round(100 * (float(s.pnl.sum() / s.matched.sum()) + 1.645 * se), 2)],
            "clv%": round(100 * float(s.clv.mean()), 2)}


def coverage(store, day: date, cfg: dict | None = None, s3=None, now=None) -> dict:
    """Does the trader see every race? The day's GB WIN markets in the recorder's catalogue, against the markets the
    card could read (a name "A5 480m"), the markets priced (``predictions/<day>.csv``) and each mode's ledger: what was
    decided and why not backed. A race whose off has passed and that no ledger decided is a race missed; for those,
    whether the recorder's books ever showed every runner tight (lay within ``tight`` of back) while the trader could
    still bet, and the runners the books had in that the prices did not (a whole field not priced is skipped)."""
    from betfair_recorder import _read_day_csv
    from greyhound.card import parse_market_name
    cfg = cfg or load_config()
    ds = f"{day:%Y-%m-%d}"
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    mk = _read_day_csv(day, "markets_greyhound.csv", s3=s3)
    if mk is None or not len(mk):
        return {"day": ds, "note": "no greyhound catalogue recorded"}
    w = mk[(mk.market_type.fillna("WIN") == "WIN") & (mk.country.fillna("GB") == "GB")].copy()
    w["off"] = pd.to_datetime(w.market_start_utc, utc=True, errors="coerce")
    w = w[w.off.dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d") == ds]
    races = w.drop_duplicates("market_id", keep="last").set_index("market_id")
    races["parsed"] = races.market_name.map(lambda n: parse_market_name(n)[1] is not None)
    raw = store.get(f"{PREFIX}/predictions/{ds}.csv")
    pred = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}) if raw else pd.DataFrame(
        columns=["market_id", "selection_id"])
    races["priced"] = races.index.isin(set(pred.market_id))
    races["gone"] = races.off <= now
    out = {"day": ds, "as_of_utc": now.strftime("%H:%M"), "gb_win_markets": len(races),
           "name_not_read": int((~races.parsed).sum()),
           "name_not_read_examples": races[~races.parsed].market_name.astype(str).value_counts().head(10).to_dict(),
           "priced": int(races.priced.sum()), "not_priced": int((~races.priced).sum()),
           "not_priced_examples": races[~races.priced].assign(off=lambda r: r.off.dt.strftime("%H:%M"))[
               ["venue", "market_name", "off"]].head(15).astype(str).to_dict("records"),
           "off_so_far": int(races.gone.sum())}
    books = _read_day_csv(day, "books_greyhound.csv", s3=s3)
    for mode in ("paper", "live"):
        lraw = store.get(f"{PREFIX}/{mode}/{ds}/ledger.csv")
        if not lraw:
            out[mode] = {"note": "no ledger"}
            continue
        led = pd.read_csv(io.BytesIO(lraw), dtype={"market_id": str})
        led["t"] = pd.to_datetime(led.time_utc, utc=True, errors="coerce")
        start = led.t.min()
        seen = set(led.market_id)
        window = races[(races.off > start + pd.Timedelta(minutes=cfg["stop_before_off"])) & races.gone]
        missed = window[~window.index.isin(seen)]
        sk = led[led.action == "skip"]
        bk = led[led.action == "back"]
        r = {"first_decision_utc": start.strftime("%H:%M") if pd.notna(start) else None,
             "markets_decided": len(seen), "runners_decided": int(led.selection_id.notna().sum()),
             "actions": led.action.value_counts().to_dict(),
             "skip_reasons": sk.error.fillna("").astype(str).str.replace(r"GBP[\d.]+", "GBPx", regex=True)
             .value_counts().to_dict(),
             "backs": {"sent": len(bk), "matched": int((bk.status == "SUCCESS").sum()),
                       "status": bk.status.fillna("").astype(str).value_counts().to_dict(),
                       "errors": bk.error.fillna("").astype(str).value_counts().head(8).to_dict()},
             "minutes_to_off_at_decision": led.dropna(subset=["selection_id"]).groupby("market_id")
             .minutes_to_off.max().describe(percentiles=[.1, .5, .9]).round(1).drop("count").to_dict(),
             "races_off_since_start": len(window), "races_off_not_decided": len(missed)}
        if len(missed) and books is not None and len(books):
            b = books[books.market_id.isin(missed.index)].copy()
            b["t"] = pd.to_datetime(b.polled_utc, utc=True, errors="coerce")
            b = b[(b.runner_status.astype(str).str.upper() == "ACTIVE") & (b.t >= start)]
            b = b.join(races.off, on="market_id")
            b = b[b.t < b.off - pd.Timedelta(minutes=cfg["stop_before_off"])]
            for c in ("back1", "lay1"):
                b[c] = pd.to_numeric(b[c], errors="coerce")
            b["tight"] = (b.back1 > 1) & (b.lay1 / b.back1 <= (cfg.get("tight") or np.inf))
            snap = b.groupby(["market_id", "t"]).tight.all()
            ever = snap.groupby(level=0).any()
            have = set(zip(pred.market_id.astype(str), pd.to_numeric(pred.selection_id, errors="coerce")))
            unpriced = b.loc[np.array([(m, float(s)) not in have for m, s in
                                       zip(b.market_id, pd.to_numeric(b.selection_id))], dtype=bool)]
            r["missed"] = {
                "with_books_seen": int(b.market_id.nunique()),
                "all_tight_at_some_poll": int(ever.sum()),
                "never_all_tight": int((~ever).sum()),
                "with_an_active_runner_not_priced": int(unpriced.market_id.nunique()),
                "not_priced_at_all": int((~missed.priced).sum()),
                "examples": missed[["venue", "market_name", "priced"]].assign(
                    off=missed.off.dt.strftime("%H:%M"),
                    ever_tight=[bool(ever.get(m, False)) for m in missed.index],
                    unpriced_runner=[m in set(unpriced.market_id) for m in missed.index]).head(20)
                .astype(str).to_dict("records")}
        if len(sk):
            r["skip_examples"] = sk[["track", "race_time", "trap", "price", "size_offered", "edge", "error",
                                     "minutes_to_off"]].head(12).astype(str).to_dict("records")
        out[mode] = r
    return out


# --------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    from betfair_recorder import uk_time_on, uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-if-missing", action="store_true", help="train only when no live model is stored yet")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--trade", action="store_true")
    ap.add_argument("--live", action="store_true", help="real orders (only with GREYHOUND_LIVE=yes)")
    ap.add_argument("--until", default="21:30", help="trade: stop at this UK time")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--check-card", action="store_true",
                    help="a past day (--date): the card's features against the results' features, feature by feature")
    ap.add_argument("--coverage", action="store_true",
                    help="the day's GB races against what was priced and what each mode's trader decided")
    ap.add_argument("--running", action="store_true",
                    help="the day so far, a row per market: bets, CLV and the P&L as each market settles")
    ap.add_argument("--out", default=None, help="--running: write the table's markdown here")
    ap.add_argument("--days", type=int, default=1, help="settle: the last N days")
    ap.add_argument("--date", default=None)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    out = {}
    if a.check_card:
        out["check_card"] = card_check(store, day)
    if a.coverage:
        out["coverage"] = coverage(store, day, load_config(Path(a.config)))
    if a.running:
        r = running(store, day, "paper")
        store.put(f"{PREFIX}/paper/{day:%Y-%m-%d}/running.json", json.dumps(r, indent=1, default=str).encode())
        md = running_markdown(r)
        try:                                        # the early lay beside it (greyhound/laytest.py); never breaks the table
            from greyhound import laytest
            lay = laytest.intraday(store, day)
            store.put(f"{laytest.PREFIX}/intraday/{day:%Y-%m-%d}.json", json.dumps(lay, indent=1, default=str).encode())
            md += "\n" + laytest.intraday_markdown(lay)
            r["early_lay"] = {k: v for k, v in lay.items() if k != "markets"}
        except Exception as exc:
            log.warning("early lay not scored (%s)", exc)
            md += f"\n### Early lay\n\nnot scored this time ({str(exc)[:200]})\n"
            r["early_lay"] = {"error": str(exc)[:200]}
        try:                                        # the live-key recorder's state (greyhound/book_record.py)
            from betfair_recorder import PREFIX as BL, bucket, s3_client
            st = json.loads(s3_client().get_object(Bucket=bucket(), Key=f"{BL}/{day:%Y-%m-%d}/status_greyhound_live.json")
                            ["Body"].read())
            md = (f"Live-key book recorder: last seen {st.get('time_utc')} UTC; {st.get('markets')} markets, "
                  f"{st.get('polls')} polls, {st.get('rows')} book rows, {st.get('traded_rows')} traded-by-price rows, "
                  f"{st.get('read_errors')} failed reads" + (f" (last: {st['last_error']})" if st.get("last_error")
                                                            else "") + ".\n\n") + md
        except Exception as exc:
            md = f"Live-key book recorder: no status yet ({str(exc)[:80]}).\n\n" + md
        try:                                        # the early lay's paper books, replayed on the books
            from greyhound import early_lay
            md = early_lay.paper_tables(store, day) + "\n" + md
        except Exception as exc:
            log.warning("paper early lays not read (%s)", exc)
        try:                                        # the live early lay's own orders (greyhound/early_lay.py)
            from greyhound import early_lay
            md = "\n".join(early_lay.table(store, day, m) for m in early_lay.modes_to_show(store, day)) + "\n" + md
        except Exception as exc:
            log.warning("live early lay not read (%s)", exc)
        try:                                        # the favourite back on paper (greyhound/fav_back.py)
            from greyhound import fav_back
            md = fav_back.table(store, day) + "\n" + md
        except Exception as exc:
            log.warning("favourite back not read (%s)", exc)
        try:                                        # the live quotes around the mid (greyhound/quote_mid.py)
            from greyhound import quote_mid
            md = quote_mid.table(store, day) + "\n" + md
        except Exception as exc:
            log.warning("live quotes around the mid not read (%s)", exc)
        store.put(f"{PREFIX}/paper/{day:%Y-%m-%d}/running.md", md.encode())
        if a.out:
            Path(a.out).write_text(md)
        out["running"] = {k: v for k, v in r.items() if k != "markets"}
    if a.train or (a.train_if_missing and store.get(STACK_META_KEY) is None and load_model(store)[0] is None):
        out["train"] = train_stack(store, day)        # the live model from 10 Nov 2026; ``train`` is the fallback's
    if a.predict:
        out["predict"] = predict(store, day)
    if a.trade:
        from trading.exchange import BetfairData, LiveExchange, PaperExchange
        cfg = load_config(Path(a.config))
        live = a.live and live_switch_on()
        if a.live and not live:
            log.warning("GREYHOUND_LIVE is not 'yes': trading on paper")
        data = BetfairData(source="greyhound", recorder=book_recorder(day))
        x = LiveExchange(data) if live else PaperExchange(data)
        x.login()
        mode = "live" if live else "paper"
        ledger = Path(os.getenv("GREYHOUND_LEDGER_DIR") or (Path.home() / ".ashcroft" / "greyhound")) / mode / \
            f"{day:%Y-%m-%d}" / "ledger.csv"
        start = uk_time_on(day, cfg["trade_from"])
        while datetime.now(timezone.utc) < start:
            time.sleep(30)
        try:
            out["trade"] = Trader(store, x, cfg, day, mode, ledger, kill=kill_switch(store)).run(
                uk_time_on(day, a.until), poll_seconds=cfg.get("poll_seconds", 60))
        finally:                                                # the day's books to S3 (copied every 15 min too)
            if data.recorder is not None:
                data.recorder.maybe_upload(force=True)
    if a.settle:
        out["settle"] = {}
        for mode in ("paper", "live"):
            days = [day - timedelta(days=i) for i in range(a.days, 0, -1)] if not a.date else [day]
            out["settle"][mode] = [settle(store, d, mode) for d in days]
            s = summary(store, mode)
            store.put(f"{PREFIX}/{mode}/summary.json", json.dumps(s, indent=1).encode())
            out["settle"][f"{mode}_summary"] = s
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
