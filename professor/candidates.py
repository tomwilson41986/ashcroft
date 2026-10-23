"""The horses the syndicate holds this morning: the live trader's ledger (its copy in S3, written after every order)
read into one row a horse, with the race facts the write-up needs.

A candidate is a horse with a matched back. Its price is the first fill's (`avg_price`), the exchange's best back and
the money at it when the order went (`best_back`, `back_size`), the closing model's expected BSP (`1 / p_pool`) and
expected CLV (`edge`), and the model's own price (`best_back / exp(move)`). The facts come from the 04:00 predictions
file (market and selection ids, the runner's name, the model's price and win probability, the card's odds), the
market catalogue the recorder keeps beside the trader (cloth, draw, jockey, trainer, form, headgear, age, weight,
official rating, days since the last run) and the morning card kept by the predictions run (race name, class, type,
distance, going, runners). Every join is best effort: a horse without facts is still a candidate.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from betfair_prices import normalise_horse, normalise_track, race_time_to_24h
from professor import store

log = logging.getLogger(__name__)

K = ["market_id", "selection_id"]
FACTS_STATUS: dict = {}          # what the last facts_for() found: the sources' rows and columns, the matches
NUM = ["matched", "avg_price", "best_back", "back_size", "p_model", "p_market", "p_pool", "edge", "move", "bsp",
       "selection_id", "minutes_to_off"]
CATALOGUE_COLS = ["market_id", "selection_id", "runner_name", "cloth_number", "stall_draw", "jockey", "trainer", "age",
                  "weight_value", "official_rating", "form", "days_since_last_run", "wearing", "forecast_num",
                  "forecast_den", "market_start_utc", "venue", "race_type", "market_name", "event_name"]
CARD_COLS = ["race_time", "track", "race_name", "race_class", "race_type", "dist_furlongs", "going_description",
             "number_of_runners", "horse_name", "jockey_name", "trainer", "stall", "official_rating", "pounds",
             "horse_age", "days_since_lr", "headgear", "odds"]


@dataclass
class Candidate:
    day: str
    market_id: str
    selection_id: int
    runner: str
    venue: str
    off_utc: datetime | None
    first_utc: datetime | None
    first_price: float          # the first fill's average price
    best_back: float            # the exchange's best back when the first order went
    back_size: float            # the money at it
    p_win: float                # the closing model's expected 1/BSP
    edge: float                 # expected CLV at the first back
    exp_bsp: float              # 1 / p_win
    predicted_bfsp: float       # the model's own price
    stake: float                # the day's matched stake on the horse
    backs: int = 1
    facts: dict = field(default_factory=dict)

    @property
    def off_uk(self) -> str:
        if self.off_utc is None:
            return ""
        return self.off_utc.astimezone(_LONDON).strftime("%H:%M")

    @property
    def first_uk(self) -> str:
        if self.first_utc is None:
            return ""
        return self.first_utc.astimezone(_LONDON).strftime("%H:%M")

    @property
    def race(self) -> str:
        return f"{self.day}|{self.market_id}"

    def to_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "facts"}
        d["off_uk"], d["first_uk"] = self.off_uk, self.first_uk
        d["off_utc"] = self.off_utc.isoformat() if self.off_utc else None
        d["first_utc"] = self.first_utc.isoformat() if self.first_utc else None
        for k, v in self.facts.items():
            d[f"fact_{k}"] = v
        return d


try:
    from zoneinfo import ZoneInfo
    _LONDON = ZoneInfo("Europe/London")
except Exception:  # noqa: BLE001
    _LONDON = timezone.utc


# ---------------------------------------------------------------------------------- the ledger
def read_ledger(day: date | str) -> pd.DataFrame | None:
    """The live ledger's copy in S3, numeric where it should be; None when the day has none yet."""
    local = os.getenv("PROFESSOR_LEDGER_FILE")
    if local:                                     # a local ledger (a dry run on a downloaded copy, the tests)
        led = pd.read_csv(local, dtype={"market_id": str}, low_memory=False)
    else:
        led = store.read_csv(f"trading/live/{day}/ledger.csv", dtype={"market_id": str})
    if led is None or led.empty:
        return None
    for c in NUM:
        if c in led.columns:
            led[c] = pd.to_numeric(led[c], errors="coerce")
    led["t"] = pd.to_datetime(led["ts"], utc=True, errors="coerce")
    return led


def positions(led: pd.DataFrame, until: datetime | None = None) -> pd.DataFrame:
    """One row a horse: the first matched back (its price, the book read then, the closing model's view) and the
    day's stake. `until` keeps the backs filled by then (the tipster's cut)."""
    b = led[(led["event"] == "back") & (led["matched"] > 0)].sort_values("t")
    if until is not None:
        b = b[b["t"] <= until]
    if b.empty:
        return pd.DataFrame(columns=K)
    pos = b.groupby(K, sort=False).agg(
        runner=("runner", "first"), venue=("venue", "first"), off=("off", "first"), first_price=("avg_price", "first"),
        best_back=("best_back", "first"), back_size=("back_size", "first"), p_pool=("p_pool", "first"),
        edge=("edge", "first"), move=("move", "first"), first_t=("t", "first"), stake=("matched", "sum"),
        backs=("matched", "size")).reset_index()
    pos["off_utc"] = pd.to_datetime(pos["off"], utc=True, errors="coerce")
    pos["selection_id"] = pd.to_numeric(pos["selection_id"], errors="coerce").astype("Int64")
    return pos


# ---------------------------------------------------------------------------------- the facts
def _norm_track(x) -> str:
    try:
        return normalise_track(str(x))
    except Exception:  # noqa: BLE001
        return str(x).lower()


def predictions_file(day: date | str) -> pd.DataFrame | None:
    p = store.read_csv(f"predictions/{day}.csv", dtype={"market_id": str})
    if p is None or p.empty:
        return None
    p["selection_id"] = pd.to_numeric(p.get("selection_id"), errors="coerce").astype("Int64")
    return p


def catalogue(day: date | str) -> pd.DataFrame | None:
    """The recorder's market catalogue files of the day (betfair_live/<day>/markets*.csv.gz), the latest row a runner."""
    try:
        resp = store.s3().list_objects_v2(Bucket=store.BUCKET, Prefix=f"betfair_live/{day}/markets")
    except Exception as e:  # noqa: BLE001
        log.warning("catalogue listing: %s", str(e)[:100])
        return None
    keys = sorted(o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".csv.gz"))
    frames = []
    for k in keys[-4:]:
        df = store.read_csv(k, dtype={"market_id": str})
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return None
    cat = pd.concat(frames, ignore_index=True)
    keep = [c for c in CATALOGUE_COLS if c in cat.columns]
    cat = cat[keep].copy()
    cat["selection_id"] = pd.to_numeric(cat["selection_id"], errors="coerce").astype("Int64")
    return cat.drop_duplicates(K, keep="last")


def racecard(day: date | str) -> pd.DataFrame | None:
    """The morning card kept by the predictions run (racecards/<day>_<HHMM>.csv), the latest of the day."""
    try:
        resp = store.s3().list_objects_v2(Bucket=store.BUCKET, Prefix=f"racecards/{day}_")
    except Exception as e:  # noqa: BLE001
        log.warning("racecard listing: %s", str(e)[:100])
        return None
    keys = sorted(o["Key"] for o in resp.get("Contents", []))
    if not keys:
        return None
    card = store.read_csv(keys[-1])
    if card is None or card.empty:
        return None
    card = card[[c for c in CARD_COLS if c in card.columns]].copy()
    card["_track"] = card["track"].map(_norm_track) if "track" in card.columns else ""
    card["_hhmm"] = card["race_time"].map(race_time_to_24h).astype(str) if "race_time" in card.columns else ""
    card["_horse"] = card["horse_name"].map(normalise_horse) if "horse_name" in card.columns else ""
    return card


def facts_for(pos: pd.DataFrame, day: date | str) -> dict[tuple, dict]:
    """The facts of every position: the predictions file and the catalogue by market and selection, the card by
    track, time and horse."""
    out: dict[tuple, dict] = {}
    pred, cat, card = predictions_file(day), catalogue(day), racecard(day)
    FACTS_STATUS.clear()
    FACTS_STATUS.update({
        "predictions_rows": 0 if pred is None else int(len(pred)),
        "predictions_columns": [] if pred is None else list(pred.columns)[:30],
        "catalogue_rows": 0 if cat is None else int(len(cat)),
        "catalogue_columns": [] if cat is None else list(cat.columns),
        "catalogue_filled": {} if cat is None else {c: int(cat[c].notna().sum()) for c in cat.columns},
        "racecard_rows": 0 if card is None else int(len(card)),
        "racecard_columns": [] if card is None else list(card.columns),
        "racecard_sample": [] if card is None else card.head(3).astype(str).to_dict("records"),
        "positions_sample": pos.head(3).astype(str).to_dict("records"),
        "matched_predictions": 0, "matched_catalogue": 0, "matched_racecard": 0,
    })
    for r in pos.itertuples(index=False):
        key = (str(r.market_id), int(r.selection_id))
        f: dict = {}
        if pred is not None:
            m = pred[(pred["market_id"] == key[0]) & (pred["selection_id"] == key[1])]
            if len(m):
                row = m.iloc[0]
                FACTS_STATUS["matched_predictions"] += 1
                for c in ("runner_name", "venue", "race_time", "predicted_bfsp", "predicted_win_prob", "racecard_odds",
                          "bf_best_back", "bf_best_lay", "bf_total_matched"):
                    if c in m.columns and pd.notna(row[c]):
                        f[c] = row[c]
        if cat is not None:
            m = cat[(cat["market_id"] == key[0]) & (cat["selection_id"] == key[1])]
            if len(m):
                row = m.iloc[0]
                FACTS_STATUS["matched_catalogue"] += 1
                for c in CATALOGUE_COLS[2:]:
                    if c in m.columns and pd.notna(row[c]):
                        f[c] = row[c]
        if card is not None:
            off = pd.to_datetime(r.off_utc, utc=True, errors="coerce") if not isinstance(r.off_utc, pd.Timestamp) else r.off_utc
            hhmm = off.tz_convert(_LONDON).strftime("%H:%M") if pd.notna(off) else ""
            m = card[(card["_track"] == _norm_track(r.venue)) & (card["_hhmm"] == hhmm)
                     & (card["_horse"] == normalise_horse(str(r.runner)))]
            if len(m):
                row = m.iloc[0]
                FACTS_STATUS["matched_racecard"] += 1
                for c in CARD_COLS:
                    if c in m.columns and pd.notna(row[c]) and c not in f:
                        f[c] = row[c]
        out[key] = f
    return out


# ---------------------------------------------------------------------------------- the candidates
def build(day: date | str, until: datetime | None = None, with_facts: bool = True) -> list[Candidate]:
    """Every horse the ledger shows backed (by `until`), one Candidate each, the day's facts attached."""
    led = read_ledger(day)
    if led is None:
        log.warning("no ledger for %s", day)
        return []
    pos = positions(led, until)
    if pos.empty:
        return []
    facts = facts_for(pos, day) if with_facts else {}
    out = []
    for r in pos.itertuples(index=False):
        p_pool = float(r.p_pool) if pd.notna(r.p_pool) and r.p_pool > 0 else np.nan
        best_back = float(r.best_back) if pd.notna(r.best_back) and r.best_back > 1 else float(r.first_price)
        move = float(r.move) if pd.notna(r.move) else 0.0
        c = Candidate(
            day=str(day), market_id=str(r.market_id), selection_id=int(r.selection_id), runner=str(r.runner),
            venue=str(r.venue), off_utc=r.off_utc.to_pydatetime() if pd.notna(r.off_utc) else None,
            first_utc=r.first_t.to_pydatetime() if pd.notna(r.first_t) else None,
            first_price=float(r.first_price), best_back=best_back,
            back_size=float(r.back_size) if pd.notna(r.back_size) else 0.0,
            p_win=p_pool, edge=float(r.edge) if pd.notna(r.edge) else np.nan,
            exp_bsp=(1.0 / p_pool) if p_pool == p_pool else np.nan,
            predicted_bfsp=best_back / np.exp(move), stake=float(r.stake), backs=int(r.backs),
            facts=facts.get((str(r.market_id), int(r.selection_id)), {}))
        out.append(c)
    out.sort(key=lambda c: (c.first_utc or datetime.max.replace(tzinfo=timezone.utc)))
    return out


def frame(cands: list[Candidate]) -> pd.DataFrame:
    return pd.DataFrame([c.to_row() for c in cands])
