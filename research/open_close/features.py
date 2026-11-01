"""The open-close panel: its schema, the targets and the features read from the opening book only.

The panel is the data agent's standard shape (one row a selection in a market): sport, country, market_type,
market_id, event_id, event_name, selection_id, selection_name, off_time_utc, n_runners, open_price, open_time_utc,
open_matched, p_t24h, p_t60, p_t15, p_t1, close_price, bsp, pre_play_matched, result, source. Any extra column
(a model's forecast price, ``model_price``) is carried through.

Targets:

- ``y``: log(close / open) per selection, the move the trade catches;
- ``y_share``: the change in the selection's implied-probability share of its market (close share - open share),
  the race-normalised version, which a coherent book must forecast with the shares summing to zero.

Features (``FEATURES``), all from the opening book, no look-ahead: log open, implied probability, the book's
overround at the open, rank in the market, gap to the favourite, field size, hours to the off at the open (where the
open's time is held), log open matched (where held), the sport, country and market-type categoricals, the day of the
week and the hour of the off; and, in the "with model" arm, the model's forecast against the open
(``MODEL_FEATURES``, only where ``model_price`` is given).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PANEL_COLUMNS = ["sport", "country", "market_type", "market_id", "event_id", "event_name", "selection_id",
                 "selection_name", "off_time_utc", "n_runners", "open_price", "open_time_utc", "open_matched",
                 "p_t24h", "p_t60", "p_t15", "p_t1", "close_price", "bsp", "pre_play_matched", "result", "source"]
MODEL_COL = "model_price"

#: the price bands of the reports (research/queries/done/market_only_drift.py BANDS)
BANDS = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 1001.0]
BAND_LABELS = ["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-51", "51+"]

NUMERIC_FEATURES = ["log_open", "p_open", "overround_open", "p_share", "rank_open", "gap_fav", "n_runners",
                    "hours_to_off", "log_open_matched", "dow", "off_hour"]
CATEGORICAL_FEATURES = ["sport_code", "country_code", "market_type_code"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_FEATURES = ["log_model_vs_open", "model_share_vs_open"]
TARGETS = ["y", "y_share"]
_EPS = 1e-12


def price_band(price) -> pd.Categorical:
    """Each price's band label (the open's band places a trade in the reports)."""
    return pd.cut(pd.Series(np.asarray(price, float)), BANDS, labels=BAND_LABELS, right=False)


def empty_panel() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in PANEL_COLUMNS})


def coerce_panel(d: pd.DataFrame) -> pd.DataFrame:
    """The panel with every standard column present, the prices numeric and the times UTC."""
    d = d.copy()
    for c in PANEL_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    for c in ("n_runners", "open_price", "open_matched", "p_t24h", "p_t60", "p_t15", "p_t1", "close_price", "bsp",
              "pre_play_matched", "result"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if MODEL_COL in d.columns:
        d[MODEL_COL] = pd.to_numeric(d[MODEL_COL], errors="coerce")
    for c in ("off_time_utc", "open_time_utc"):
        d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
    for c in ("sport", "country", "market_type"):
        d[c] = d[c].fillna("unknown").astype(str)
    d["market_id"] = d["market_id"].astype(str)
    return d


def clean(d: pd.DataFrame, min_price: float = 1.01, max_price: float = 1000.0) -> pd.DataFrame:
    """Rows with an open and a close both on Betfair's ladder, in markets of two or more such rows."""
    d = coerce_panel(d)
    ok = (d["open_price"].between(min_price, max_price) & d["close_price"].between(min_price, max_price)
          & d["off_time_utc"].notna())
    d = d[ok].copy()
    size = d.groupby("market_id")["open_price"].transform("size")
    return d[size >= 2].reset_index(drop=True)


def add_targets(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    g = d["market_id"]
    p_open, p_close = 1.0 / d["open_price"], 1.0 / d["close_price"]
    d["y"] = np.log(d["close_price"] / d["open_price"])
    d["y_share"] = p_close / p_close.groupby(g).transform("sum") - p_open / p_open.groupby(g).transform("sum")
    return d


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """The opening-book features, by market. ``n_runners`` is the panel's where held, else the rows in the market."""
    d = d.copy()
    g = d["market_id"]
    d["log_open"] = np.log(d["open_price"])
    d["p_open"] = 1.0 / d["open_price"]
    d["overround_open"] = d["p_open"].groupby(g).transform("sum")
    d["p_share"] = d["p_open"] / d["overround_open"]
    d["rank_open"] = d.groupby("market_id")["open_price"].rank(method="first")
    d["gap_fav"] = d["log_open"] - d.groupby("market_id")["log_open"].transform("min")
    rows = d.groupby("market_id")["open_price"].transform("size")
    d["n_runners"] = pd.to_numeric(d["n_runners"], errors="coerce").fillna(rows)
    d["hours_to_off"] = (d["off_time_utc"] - d["open_time_utc"]).dt.total_seconds() / 3600.0
    d["log_open_matched"] = np.log1p(pd.to_numeric(d["open_matched"], errors="coerce").clip(lower=0))
    d["dow"] = d["off_time_utc"].dt.dayofweek.astype(float)
    d["off_hour"] = (d["off_time_utc"].dt.hour + d["off_time_utc"].dt.minute / 60.0).astype(float)
    for c in ("sport", "country", "market_type"):
        d[f"{c}_code"] = pd.Categorical(d[c].astype(str))
    d["month"] = d["off_time_utc"].dt.strftime("%Y-%m")
    d["day"] = d["off_time_utc"].dt.strftime("%Y-%m-%d")
    d["band"] = price_band(d["open_price"].to_numpy()).to_numpy()
    if MODEL_COL in d.columns:
        m = pd.to_numeric(d[MODEL_COL], errors="coerce")
        ok = m > 1.0
        d["log_model_vs_open"] = np.where(ok, np.log(d["open_price"] / m.where(ok, 2.0)), np.nan)
        pm = (1.0 / m.where(ok)).fillna(0.0)
        pm_share = pm / pm.groupby(g).transform("sum").replace(0.0, np.nan)
        d["model_share_vs_open"] = np.where(ok, pm_share - d["p_share"], np.nan)
    return d


def fair_prices(d: pd.DataFrame) -> pd.DataFrame:
    """Each market's open and close prices scaled so the implied probabilities sum to one at both ends (the
    bookmaker's margin taken out). A bookmaker's margin shrinks from the open to the close, which lengthens every
    price a little and reads as a drift that no exchange trade can catch: a Betfair book sums to about one at both
    ends. On a proxy panel this is the comparison that matches the exchange; ``overround_open_raw`` and
    ``overround_close_raw`` keep the margins."""
    d = d.copy()
    g = d["market_id"]
    oo = (1.0 / d["open_price"]).groupby(g).transform("sum")
    oc = (1.0 / d["close_price"]).groupby(g).transform("sum")
    d["overround_open_raw"], d["overround_close_raw"] = oo, oc
    d["open_price"] = d["open_price"] * oo
    d["close_price"] = d["close_price"] * oc
    return d


def prepare(d: pd.DataFrame, normalise_book: bool = False) -> pd.DataFrame:
    """clean -> (fair prices) -> targets -> features, the frame every model and the backtest read."""
    d = clean(d)
    if normalise_book:
        d = fair_prices(d)
    return add_features(add_targets(d))


def feature_columns(d: pd.DataFrame, with_model: bool) -> list[str]:
    cols = list(FEATURES)
    if with_model:
        cols += [c for c in MODEL_FEATURES if c in d.columns]
    return cols


def market_codes(d: pd.DataFrame) -> np.ndarray:
    """Integer codes 0..M-1 for the market, in order of first appearance (model/race_book.race_codes)."""
    return pd.factorize(d["market_id"].to_numpy())[0]
