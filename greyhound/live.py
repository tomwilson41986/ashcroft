"""The greyhound model live (the owner's ask, 6 Oct 2026: "try and bring a model live, either a trading model or a
straight back model").

Real orders only with the owner's switch, the repository variable ``GREYHOUND_LIVE=yes``, and only on the UK runner;
otherwise everything runs on paper (the same decisions, simulated against the live books). Four steps:

``--train`` (GitHub's runners, weekly): the card-safe model. The research model reads the dog's weight on the day,
which GBGB records at the weigh-in and the morning does not know, so the live model leaves out every feature built
on it (``CARD_UNSAFE``); fitted on every GBGB race to yesterday, base, parity and HorseRaceBase features, saved to
``sources/greyhound/live/model.txt`` with its meta.

``--predict`` (GitHub's runners, hourly 07:21-19:21 UTC): today's cards from the recorder's greyhound catalogue in
S3 (``greyhound.card``), priced from the dogs' earlier days, to ``sources/greyhound/live/predictions/<day>.csv``.

``--trade`` (the UK runner, beside the recorder; the owner, 7 Oct 2026: "take whatever profit / stake we can where we
believe we have an edge", "keep going right until the off"): every runner in every race is looked at each poll (20
seconds; every 5 within 3 minutes of the off) from 08:00 UK until Betfair suspends the market at the off. A dog is
backed whenever its edge at the best back price clears the bar (``live_config.json``: 0.2), at no more than the price
cap: the order takes what is offered down the ladder while the edge holds, at least GBP1, fill-or-kill (nothing
rests), and is topped up on later polls to the GBP5 level stake. Nothing is final but a stake complete: no edge now,
too little offered, a field not all priced, or a back refused (the account's funds) are looked at again on the next
poll. The model's chances are renormalised over the runners still in the market. Level backs held to the result
(``trade_out`` false); with ``trade_out`` each back is staked to win ``target`` and laid at the Betfair SP for its
winnings. GBP10 a race; no daily cap (``null`` in ``live_config.json``); the S3 objects ``greyhound/STOP`` and
``trading/STOP`` stop new bets within a minute.

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
LIVE_BLOCKS = ("parity", "hrb")
LEDGER_FIELDS = ["time_utc", "mode", "market_id", "selection_id", "track", "race_time", "trap", "dog", "p_model",
                 "price", "size_offered", "edge", "action", "stake", "matched", "avg_price", "bet_id", "status",
                 "error", "lay_liability", "lay_status", "minutes_to_off"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def live_switch_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("GREYHOUND_LIVE", "")).strip().lower() == "yes"


# --------------------------------------------------------------------------------------------------------------------
# Train and predict (GitHub's runners: S3, no Betfair)
# --------------------------------------------------------------------------------------------------------------------

def card_safe(features: list[str]) -> list[str]:
    return [f for f in features if f not in CARD_UNSAFE]


def fit(df: pd.DataFrame, features: list[str], before: str, date_from: str = "2019-01-01"):
    """The live model: every modelling race in [date_from, before), early-stopped on the last tenth."""
    import lightgbm as lgb
    d = modelling_rows(df[~df.is_card])
    d = d[(d.t >= date_from) & (d.t < before)]
    cut = d.t.quantile(0.9)
    tr, es = d[d.t < cut], d[d.t >= cut]
    m = lgb.train(PARAMS, lgb.Dataset(tr[features], tr.won), 3000, valid_sets=[lgb.Dataset(es[features], es.won)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, {"before": before, "date_from": date_from, "fit_rows": len(d), "rounds": m.best_iteration}


def price_cards(df: pd.DataFrame, booster, features: list[str], key: pd.DataFrame) -> pd.DataFrame:
    """The card rows of ``df`` priced and normalised within each race, with their Betfair market and selection
    (``key``: race_id, trap, market_id, selection_id, matched_history from ``greyhound.card.cards``)."""
    c = df[df.is_card]
    out = c[["race_id", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "race_class", "distance_m",
             "field"]].copy()
    out["p_raw"] = booster.predict(c[features]) if len(c) else []
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
    feats = card_safe(eng.features)
    m, meta = fit(df, feats, before=f"{today:%Y-%m-%d}")
    imp = pd.Series(m.feature_importance("gain"), index=feats)
    meta.update({"features": feats, "blocks": list(eng.blocks), "card_unsafe_left_out": sorted(CARD_UNSAFE),
                 "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "top_features": {k: round(float(v), 4) for k, v in (imp / imp.sum()).sort_values(ascending=False)
                                  .head(20).items()}})
    store.put(MODEL_KEY, m.model_to_string(num_iteration=m.best_iteration).encode())
    store.put(META_KEY, json.dumps(meta, indent=1).encode())
    return {k: v for k, v in meta.items() if k != "features"}


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
    missing = [f for f in meta["features"] if f not in df.columns]
    if missing:
        raise SystemExit(f"the live model reads features the engine no longer makes: {missing[:10]}")
    key = card[["race_id", "trap", "market_id", "selection_id", "matched_history"]].drop_duplicates(["race_id", "trap"])
    pred = price_cards(df, booster, meta["features"], key)
    pred["model_trained_utc"] = meta.get("trained_utc")
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


def card_check(store, day: date, blocks=LIVE_BLOCKS, s3=None) -> dict:
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
    k = ["track", "race_time", "trap"]
    a = a[a.is_card.fillna(False).astype(bool)].assign(race_time=lambda x: x.race_time.astype(str).str[:5])
    b = b[(b.race_date.astype(str) == ds)].assign(race_time=lambda x: x.race_time.astype(str).str[:5])
    j = a.merge(b, on=k, suffixes=("_card", "_res"))
    j = j[[(t, r, float(p)) in matched for t, r, p in zip(j.track, j.race_time, j.trap)]]
    agree = {}
    for f in eng.features:
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
                elif r["action"] == "skip":
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
            open_here = any(r["action"] in ("wait", "retry", "none") for r in rows)
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
                    room = math.floor(100 * (lim["max_race_stake"] - self.race_stake.get(mid, 0.0)) + 1e-6) / 100
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


def kill_switch(store):
    state = {"at": None, "on": False}

    def check() -> bool:
        now = datetime.now(timezone.utc)
        if state["at"] and (now - state["at"]).total_seconds() < 60:
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
    except Exception:
        return False


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
            unpriced = b[[(m, float(s)) not in have for m, s in zip(b.market_id, pd.to_numeric(b.selection_id))]]
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
    if a.train or (a.train_if_missing and load_model(store)[0] is None):
        out["train"] = train(store, day)
    if a.predict:
        out["predict"] = predict(store, day)
    if a.trade:
        from trading.exchange import BetfairData, LiveExchange, PaperExchange
        cfg = load_config(Path(a.config))
        live = a.live and live_switch_on()
        if a.live and not live:
            log.warning("GREYHOUND_LIVE is not 'yes': trading on paper")
        data = BetfairData(source="greyhound")
        x = LiveExchange(data) if live else PaperExchange(data)
        x.login()
        mode = "live" if live else "paper"
        ledger = Path(os.getenv("GREYHOUND_LEDGER_DIR") or (Path.home() / ".ashcroft" / "greyhound")) / mode / \
            f"{day:%Y-%m-%d}" / "ledger.csv"
        start = uk_time_on(day, cfg["trade_from"])
        while datetime.now(timezone.utc) < start:
            time.sleep(30)
        out["trade"] = Trader(store, x, cfg, day, mode, ledger, kill=kill_switch(store)).run(
            uk_time_on(day, a.until), poll_seconds=cfg.get("poll_seconds", 60))
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
