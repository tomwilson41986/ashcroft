"""Today's greyhound race cards, from Betfair's catalogue: the rows the metrics engine prices before the off.

The GBGB results are the model's history, but they arrive after racing; a live price needs the field beforehand. The
recorder keeps the day's GB greyhound catalogue (``betfair_live/<day>/markets_greyhound.csv``, live-record.yml):
each WIN market's venue, off time and name ("A5 480m": the grade and the trip), and its runners as "1. Dog Name"
(the trap and the dog). This module turns that into card rows in the GBGB table's shape, ``is_card`` set and no
result, each dog matched to its GBGB history by name (the most recent dog of that name; a dog never seen is a
debutant with a new id), with its sire, trainer, birth month and sex carried from its last run.

Nothing of the day's results is used: a card row has no position, time, weight or SP, and the engine's features for
it read the dogs' earlier days only (``greyhound.metrics``, lag-safe).
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd

NAME_RE = re.compile(r"^\s*([A-Z]{1,2}\d{0,2})\s+(\d{3,4})m\b", re.I)


#: Betfair's class codes where GBGB writes the grade differently (the card check of 6 Oct 2026: Betfair's "HC" hurdle
#: races are GBGB's "HP")
BETFAIR_TO_GBGB = {"HC": "HP"}


def parse_market_name(name) -> tuple[str | None, float | None]:
    """"A5 480m" -> ("A5", 480.0); "OR3 500m" -> ("OR3", 500.0); anything else -> (None, None)."""
    m = NAME_RE.match(str(name or ""))
    if not m:
        return (None, None)
    cls = m.group(1).upper()
    return (BETFAIR_TO_GBGB.get(cls, cls), float(m.group(2)))


def norm_name(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _race_id(market_id: str) -> int:
    """A card's race id, negative (GBGB's are positive) and stable for its market."""
    return -int(zlib.crc32(str(market_id).encode()))


def cards(markets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """The day's GB WIN markets (the recorder's catalogue rows) as card rows matched to ``history`` (the GBGB runs,
    before ``data.clean``)."""
    w = markets[(markets.market_type.fillna("WIN") == "WIN") & (markets.country.fillna("GB") == "GB")].copy()
    w = w.drop_duplicates(["market_id", "selection_id"], keep="last")
    uk = pd.to_datetime(w.market_start_utc, utc=True, errors="coerce").dt.tz_convert("Europe/London")
    parsed = w.market_name.map(parse_market_name)
    out = pd.DataFrame({
        "market_id": w.market_id.astype(str), "selection_id": pd.to_numeric(w.selection_id, errors="coerce"),
        "race_date": uk.dt.strftime("%Y-%m-%d"), "race_time": uk.dt.strftime("%H:%M:%S"),
        "track": w.venue.astype(str).str.strip(), "race_class": parsed.map(lambda x: x[0]),
        "distance_m": parsed.map(lambda x: x[1]),
        "trap": pd.to_numeric(w.runner_name.astype(str).str.extract(r"^(\d+)\.")[0], errors="coerce"),
        "dog_name": w.runner_name.astype(str).str.replace(r"^\d+\.\s*", "", regex=True).str.strip(),
        "_bf_trainer": w["trainer"] if "trainer" in w else np.nan,
    })
    out = out.dropna(subset=["trap", "distance_m"])
    out["race_id"] = out.market_id.map(_race_id)
    out["meeting_id"] = -out.groupby(["track", "race_date"]).ngroup() - 1
    out["runners"] = out.groupby("market_id").trap.transform("size")
    # the race's number at its meeting, as GBGB numbers a card: its order by the off
    out["race_number"] = out.groupby(["track", "race_date"]).race_time.rank(method="dense").astype(float)
    # each dog's GBGB history by name: the most recent dog of the name, with its static fields from its last run
    h = history.dropna(subset=["dog_id"]).sort_values(["race_date", "race_time"], kind="stable")
    last = h.assign(_n=h.dog_name.map(norm_name)).drop_duplicates("_n", keep="last").set_index("_n")
    key = out.dog_name.map(norm_name)
    found = key.isin(last.index)
    for col in ("dog_id", "sire", "dam", "trainer", "born", "sex", "season"):
        out[col] = key.map(last[col]) if col in last else np.nan
    new = ~found
    # a dog GBGB has no run for: its trainer from Betfair's catalogue, where the catalogue gives one
    out["trainer"] = out.trainer.astype(object)
    out.loc[new, "trainer"] = out.loc[new, "_bf_trainer"].astype(object)
    out = out.drop(columns="_bf_trainer")
    out.loc[new, "dog_id"] = -(out.loc[new, "selection_id"].astype("int64"))       # a debutant: a new id of its own
    out["dog_id"] = out.dog_id.astype("int64")
    for col in ("position", "beaten_lengths", "sectional", "run_time", "adjusted_time", "weight_kg", "sp_decimal",
                "going", "market_pos"):
        out[col] = np.nan
    out["comment"] = ""
    out["is_card"] = True
    out["matched_history"] = found.values
    return out


def with_cards(history: pd.DataFrame, card_rows: pd.DataFrame) -> pd.DataFrame:
    """The GBGB runs and the cards together, the cards' days' results (if any are already in) dropped: a card is
    priced from the days before it."""
    days = set(card_rows.race_date)
    h = history[~history.race_date.isin(days)]
    keep = [c for c in card_rows.columns if c not in ("market_id", "selection_id", "matched_history")]
    return pd.concat([h, card_rows[keep]], ignore_index=True)
