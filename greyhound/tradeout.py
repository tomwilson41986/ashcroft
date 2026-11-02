"""The greyhound trade-out (the owner, 2 Nov 2026): "focus on taking a price then trading out at the off, like the
horse racing model". A paper forward test from a fresh date, and the panel the research reads.

The trade. Back a dog at the best back price offered at an entry mark before the off, and close it at the off by
laying the same money back at the Betfair SP (or the reverse: lay at the best lay, back at SP). Closed out, the
position is worth ``stake * (back / BSP - 1)`` (a lay ``stake * (1 - lay / BSP)``) whatever wins
(``model.race_book`` section 2), less commission on the market's net. It pays only where the price taken beats the
SP, so the rule is a forecast of the SP, as the horses' live rule is (TRADING.md: never at a price under the model's
predicted BFSP, each back laid at the SP).

The forecast of the SP (``model.race_book.ClosingModel``, the horses' closing model, unchanged): a conditional logit
on the SP book, from the market's price at the entry mark (``race_book.market_now``: the mid of a tight book, else the
back), our price (the frozen greyhound model of 5 Oct, ``greyhound.track``, out of sample on every day from 6 Oct)
and the runner's and market's matched money. A position's expected CLV is ``back * O * q_hat - 1`` backing and
``1 - lay * O * q_hat`` laying, q_hat the forecast SP share and O the SP book's median sum of 1/BSP in training.

Why the old rules were retired (reports/greyhound_model.md section 12): the first recorded book is an empty
placeholder book (back median 1.10, hours before the off), so the primary rule of 5 Oct never fired in 27 days; the
rules that did fire chose by the chance to win and took prices 7-12% under the SP on average.

The forward test, fixed here before any test day (``RULES``):

- **The books:** the greyhound trader's own (``books_greyhound_trader.csv``: the live key, every 5 seconds in a
  race's last 3 minutes, kept from 2 Nov 2026), not the recorder's. The recorder's are the delayed key's, about 90
  seconds old at the median in the last 3 minutes (reports/greyhound_model.md section 13), so a fill read off them is
  at a price already gone. The SP still comes from the recorder's settled books. (The owner, 2 Nov 2026: gather the
  live books first; the paper trader keeps the live key.)
- **Frozen once** (``--freeze``), when ``MIN_FREEZE_DAYS`` (10) days of the trader's books are held: the closing
  models, fitted on those days at each entry mark; written with the rules to ``greyhound/tradeout/frozen.json``.
  Until then ``--freeze`` reports how many days it has and writes nothing. The win model is the forward test's model
  of 5 Oct, not refitted, so the closing model's training prices are out of sample too.
- **Start:** the later of ``START`` (4 Nov 2026) and the day after the freeze. Paper only; nothing here can place an
  order.
- **Primary rule:** entry 1 minute before the off (the recorder's ``off_1`` mark), the horses' closing model, back
  where the expected CLV clears 0.03, a price up to 20 with at least the stake offered, GBP2 a position, closed at the
  SP; 2% commission on each market's net.
- **Reported, not judged:** entry at the dog's first traded book (``first_traded``) and 3 and 5 minutes before the
  off; the lay side; bars of 0 and 0.05; the microstructure closing model (``MICRO``: the market, our price, the last
  traded price against the mid, the price's move over the last two minutes, the queue imbalance and the spread); and
  the old selection (edge to win over 0.2) closed at the SP, for comparison.
- **Gate:** judged after 30 test days and 1,000 primary trades, whichever is later: a pass only if the return per
  GBP staked after commission has a 90% lower bound above 0 (a bootstrap by day). A pass goes to the owner, who
  decides on small real stakes.

The limits: the fill is taken on trust when the book offered the stake (each day's feed lag is reported, so a day
whose books turn out stale can be seen); the SP leg is assumed matched; the race's field is the one that ran (GBGB), so a late non-runner is not seen as the market saw it.

    python -m greyhound.tradeout --freeze                # once: fit and freeze the closing models and the rules
    python -m greyhound.tradeout --days 4                # the nightly run (after greyhound.track's): settle, summarise
    python -m greyhound.tradeout --summary
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from model import race_book

log = logging.getLogger(__name__)

PREFIX = "greyhound/tradeout"
TRACK_PRED = "greyhound/track/pred"          # the forward test of 5 Oct's priced runners, a day a file
FROZEN_KEY = f"{PREFIX}/frozen.json"
FIRST_BOOK_DAY = date(2026, 10, 6)            # the first day with recorded greyhound books
START = "2026-11-04"
#: whose books the test reads at the entry marks; the SP is always the recorder's settled book
ENTRY_BOOKS = {"trader": "books_greyhound_trader.csv", "recorder": "books_greyhound.csv"}
SOURCE = "trader"
MIN_FREEZE_DAYS = 10
STAKE = 2.0
COMMISSION = 0.02
MAX_PRICE = 20.0
WIN_EDGE = 0.2                                # the old selection: the chance to win at the back price
#: the entry marks: minutes before the off, as betfair_recorder.marks reads them (within a quarter, at least a
#: minute), and the dog's first book after its first matched bet
OFF_ENTRY = (5, 3, 1)
ENTRY_MARKS = ("first_traded", "off_5", "off_3", "off_1")
BARS = (0.0, 0.03, 0.05)
SIDES = ("back", "lay")
MODELS = ("racebook", "micro")
BOOK_COLS = ["back1", "back1_size", "back2", "back2_size", "back3", "back3_size", "lay1", "lay1_size", "lay2",
             "lay2_size", "lay3", "lay3_size", "last_traded", "runner_total_matched", "market_total_matched"]
#: the microstructure closing model's columns (``micro_design``), each standardised on the training rows
MICRO = ("lq_market", "lq_model", "last_vs_mid", "move", "imbalance", "log_spread", "lq_market_x_vol",
         "lq_model_x_vol")
RULES = {
    "start": f"the later of {START} and the day after the freeze",
    "books": "the greyhound trader's live-key books (books_greyhound_trader.csv); the SP from the recorder's settled "
             f"books; frozen once {MIN_FREEZE_DAYS} days of the trader's books are held",
    "primary": {"mark": "off_1", "model": "racebook", "side": "back", "bar": 0.03},
    "reported": {"marks": list(ENTRY_MARKS), "models": list(MODELS) + ["win_edge"], "sides": list(SIDES),
                 "bars": list(BARS)},
    "stake": STAKE, "commission": COMMISSION, "max_price": MAX_PRICE, "fill": "the best price offered at least the stake",
    "close": "the Betfair SP; commission on each market's net",
    "gate": "judged after 30 test days and 1,000 primary trades, whichever is later: a pass only if the return per "
            "GBP staked after commission has a 90% lower bound above 0 (bootstrap by day); a pass goes to the owner",
}


# --------------------------------------------------------------------------------------------------------------------
# The books at the entry marks
# --------------------------------------------------------------------------------------------------------------------

def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pre_off_books(books: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Every open, pre-play book of an active runner, with its minutes to the scheduled off."""
    offs = markets.drop_duplicates("market_id").set_index("market_id").market_start_utc
    b = books.copy()
    b["market_id"] = b.market_id.astype(str)
    b["selection_id"] = _num(b.selection_id)
    b["polled"] = pd.to_datetime(b.polled_utc, utc=True, errors="coerce")
    b["off"] = pd.to_datetime(b.market_id.map(offs.set_axis(offs.index.astype(str))), utc=True, errors="coerce")
    b["minutes_to_off"] = (b.off - b.polled).dt.total_seconds() / 60.0
    ok = ((b.source.astype(str) != "final") & (b.status.astype(str) == "OPEN")
          & b.inplay.astype(str).isin(["0", "False", "false"]) & (b.minutes_to_off > 0)
          & (b.runner_status.fillna("ACTIVE").astype(str) == "ACTIVE"))
    b = b[ok].dropna(subset=["polled", "selection_id"])
    for c in BOOK_COLS:
        b[c] = _num(b[c]) if c in b else np.nan
    return b


def entry_books(pre: pd.DataFrame) -> pd.DataFrame:
    """Each runner's book at each entry mark, and the book about two minutes before it (``*_prev``), for the price's
    move into the mark."""
    picks = []
    for m in OFF_ENTRY:
        tol = max(1.0, 0.25 * m)
        d = pre.assign(gap=(pre.minutes_to_off - m).abs())
        d = d[d.gap <= tol]
        picks.append(d.sort_values("gap").drop_duplicates(["market_id", "selection_id"]).assign(mark=f"off_{m}"))
    traded = pre[pre.last_traded > 1]
    picks.append(traded.sort_values("polled").drop_duplicates(["market_id", "selection_id"]).assign(mark="first_traded"))
    e = pd.concat(picks, ignore_index=True)
    keep = ["market_id", "selection_id", "mark", "polled", "minutes_to_off"] + BOOK_COLS
    e = e[keep]
    # the book at least 90 seconds before the mark's (the recorder polls a minute apart in the last hour)
    prev = pre[["market_id", "selection_id", "polled", "back1", "lay1"]].rename(
        columns={"polled": "polled_prev", "back1": "back1_prev", "lay1": "lay1_prev"}).sort_values("polled_prev")
    e = e.assign(_t=e.polled - pd.Timedelta(seconds=90)).sort_values("_t")
    e = pd.merge_asof(e, prev, left_on="_t", right_on="polled_prev", by=["market_id", "selection_id"],
                      direction="backward").drop(columns="_t")
    return e


def closing_prices(books: pd.DataFrame, markets: pd.DataFrame, day: date) -> pd.DataFrame:
    """Each runner's BSP as the recorder settled it (``betfair_recorder.marks``' final mark)."""
    from betfair_recorder import marks
    mk = marks(books, markets, day)
    f = mk[mk["mark"] == "final"][["market_id", "selection_id", "bsp"]].copy()
    f["market_id"] = f.market_id.astype(str)
    f["selection_id"] = _num(f.selection_id)
    f["bsp"] = _num(f.bsp)
    return f.dropna(subset=["bsp"]).drop_duplicates(["market_id", "selection_id"])


def feed_lag(pre: pd.DataFrame, minutes: float = 3.0) -> pd.Series:
    """Each market's median age of its books in its last ``minutes``: the poll's time less the market's last match
    (money matches every few seconds there, so this is the feed's delay)."""
    if "last_match_utc" not in pre:
        return pd.Series(dtype=float)
    x = pre[pre.minutes_to_off <= minutes].drop_duplicates(["market_id", "polled"])
    lag = (x.polled - pd.to_datetime(x.last_match_utc, utc=True, errors="coerce")).dt.total_seconds()
    return lag.groupby(x.market_id).median()


def day_panel(pred: pd.DataFrame, books: pd.DataFrame, markets: pd.DataFrame, day: date,
              closing_books: pd.DataFrame | None = None) -> pd.DataFrame:
    """One day's runners at each entry mark: our price, the book (``books``), the SP (from ``closing_books``, the
    recorder's settled books, when the entry books are another writer's), the result. A race is kept at a mark only
    when every runner that ran has a back price there and an SP."""
    if pred is None or not len(pred) or books is None or markets is None or not len(books):
        return pd.DataFrame()
    p = pred.copy()
    p["market_id"] = p.market_id.astype(str)
    p["selection_id"] = _num(p.selection_id)
    p = p[["race_id", "race_date", "race_time", "track", "trap", "field", "won", "p_model", "market_id",
           "selection_id"]]
    pre = pre_off_books(books, markets)
    if not len(pre):
        return pd.DataFrame()
    e = entry_books(pre)
    bsp = closing_prices(books if closing_books is None else closing_books, markets, day)
    d = p.merge(e, on=["market_id", "selection_id"], how="inner").merge(bsp, on=["market_id", "selection_id"],
                                                                       how="inner")
    d = d[(d.back1 > 1) & (d.bsp > 1)]
    d = d.assign(feed_lag_s=d.market_id.map(feed_lag(pre)))
    whole = d.groupby(["race_id", "mark"]).selection_id.transform("size") == d.field
    return add_features(d[whole].reset_index(drop=True))


# --------------------------------------------------------------------------------------------------------------------
# Features and the closing models
# --------------------------------------------------------------------------------------------------------------------

def _codes(d: pd.DataFrame) -> np.ndarray:
    return race_book.race_codes(d.race_id.astype(str) + "|" + d.mark.astype(str))


def _lq(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.log(np.clip(race_book.normalise_within(x, g), 1e-9, 1.0))


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """The race-normalised log chances of the market now, of our model, of the last traded price and of the book two
    minutes earlier; the queue imbalance, the spread, the money, the SP's share (the target)."""
    if not len(d):
        return d
    d = d.copy()
    g = _codes(d)
    d["mid"] = race_book.market_now(d.back1, d.lay1)
    d["lq_market"] = _lq(1.0 / d.mid.to_numpy(float), g)
    d["lq_model"] = _lq(d.p_model.to_numpy(float), g)
    last = d.last_traded.where(d.last_traded > 1, d.mid)
    d["last_vs_mid"] = _lq(1.0 / last.to_numpy(float), g) - d.lq_market
    prev = pd.Series(race_book.market_now(d.back1_prev, d.lay1_prev), index=d.index)
    prev = prev.where(d.back1_prev > 1, d.mid)
    d["move"] = d.lq_market - _lq(1.0 / prev.to_numpy(float), g)
    d["imbalance"] = np.log1p(d.back1_size.fillna(0).clip(lower=0)) - np.log1p(d.lay1_size.fillna(0).clip(lower=0))
    d["log_spread"] = np.log((d.lay1.where(d.lay1 > 1, d.back1) / d.back1).clip(lower=1.0))
    d["log_vol"] = np.log1p(d.runner_total_matched.fillna(0).clip(lower=0))
    d["log_depth"] = np.log1p(d.market_total_matched.fillna(0).clip(lower=0))
    d["overround"] = pd.Series(1.0 / d.mid.to_numpy(float)).groupby(g).transform("sum").to_numpy()
    d["rank"] = d.groupby(["race_id", "mark"]).mid.rank(method="first")
    d["lq_close"] = _lq(1.0 / d.bsp.to_numpy(float), g)
    d["edge_win"] = d.p_model * (1 + (d.back1 - 1) * (1 - COMMISSION)) - 1
    return d


def racebook_inputs(d: pd.DataFrame) -> pd.DataFrame:
    """The horses' closing model's inputs (race_book.closing_inputs) from the panel."""
    x = pd.DataFrame({"race": d.race_id.astype(str) + "|" + d.mark.astype(str), "morningwap": d.mid,
                      "predicted_bfsp": 1.0 / d.p_model.clip(lower=1e-6), "morning_vol": d.runner_total_matched})
    return race_book.closing_inputs(x)


def micro_raw(d: pd.DataFrame) -> np.ndarray:
    """The microstructure model's raw columns (standardised by the training rows' means and spreads in the fit)."""
    vol = d.log_vol
    cols = {"lq_market": d.lq_market, "lq_model": d.lq_model, "last_vs_mid": d.last_vs_mid, "move": d.move,
            "imbalance": d.imbalance, "log_spread": d.log_spread, "lq_market_x_vol": d.lq_market * vol,
            "lq_model_x_vol": d.lq_model * vol}
    return np.column_stack([np.nan_to_num(np.asarray(cols[c], float)) for c in MICRO])


def fit_clogit(X: np.ndarray, q: np.ndarray, g: np.ndarray, l2: float = 1e-3, start=None) -> np.ndarray:
    """A conditional logit on soft labels (the SP book's shares) by cross-entropy, the race_book fit for any design."""
    n_races = g.max() + 1

    def loss(beta):
        lqh = race_book.group_log_softmax(X @ beta, g)
        val = -float(np.sum(q * lqh)) / n_races + 0.5 * l2 * float(beta @ beta)
        grad = -((q - np.exp(lqh))[:, None] * X).sum(axis=0) / n_races + l2 * beta
        return val, grad

    b0 = np.zeros(X.shape[1]) if start is None else np.asarray(start, float)
    return minimize(loss, b0, jac=True, method="L-BFGS-B").x


def fit_models(d: pd.DataFrame) -> dict:
    """Both closing models on a panel of one entry mark (training days only), as JSON-ready dicts."""
    g = _codes(d)
    inv = 1.0 / d.bsp.to_numpy(float)
    q = race_book.normalise_within(inv, g)
    book = float(np.median(np.bincount(g, weights=inv)))
    rb = race_book.fit_closing_model(racebook_inputs(d), d.bsp.to_numpy(float), g)
    X = micro_raw(d)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    start = np.zeros(len(MICRO))
    start[0] = sd[0]                                         # the market as it stands
    beta = fit_clogit((X - mu) / sd, q, g, start=start)
    return {"racebook": {"beta": rb.beta.tolist(), "vol_mean": rb.vol_mean, "vol_sd": rb.vol_sd,
                         "depth_mean": rb.depth_mean, "depth_sd": rb.depth_sd,
                         "band_edges": rb.band_edges.tolist(), "band_sigma": rb.band_sigma.tolist(),
                         "book": rb.book, "n_train": rb.n_train},
            "micro": {"columns": list(MICRO), "beta": beta.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
                      "book": book},
            "rows": int(len(d)), "races": int(g.max() + 1), "days": int(d.race_date.nunique())}


def predict_share(d: pd.DataFrame, fitted: dict, name: str) -> tuple[np.ndarray, float]:
    """The forecast SP share of every runner (a book of 1 a race) and the SP book to scale it by."""
    g = _codes(d)
    if name == "racebook":
        j = fitted["racebook"]
        m = race_book.ClosingModel(beta=np.asarray(j["beta"]), vol_mean=j["vol_mean"], vol_sd=j["vol_sd"],
                                   depth_mean=j["depth_mean"], depth_sd=j["depth_sd"],
                                   band_edges=np.asarray(j["band_edges"]), band_sigma=np.asarray(j["band_sigma"]),
                                   book=j["book"], n_train=j["n_train"])
        return m.predict(racebook_inputs(d), g), float(j["book"])
    j = fitted["micro"]
    X = (micro_raw(d) - np.asarray(j["mu"])) / np.asarray(j["sd"])
    return np.exp(race_book.group_log_softmax(X @ np.asarray(j["beta"]), g)), float(j["book"])


def expected_clv(d: pd.DataFrame, q_hat: np.ndarray, book: float) -> pd.DataFrame:
    """Each position's expected value closed at the SP: backing at the best back, laying at the best lay."""
    inv = book * np.asarray(q_hat, float)
    lay = d.lay1.where(d.lay1 > 1)
    return pd.DataFrame({"ev_back": d.back1.to_numpy(float) * inv - 1.0,
                         "ev_lay": 1.0 - lay.to_numpy(float) * inv}, index=d.index)


# --------------------------------------------------------------------------------------------------------------------
# Positions and their settlement
# --------------------------------------------------------------------------------------------------------------------

def positions(d: pd.DataFrame, ev: pd.DataFrame, model: str, bars=BARS, stake: float = STAKE) -> pd.DataFrame:
    """Every rule's paper positions: a side, a bar on the expected CLV, the fill test (at least the stake offered at
    the price, a price up to MAX_PRICE). ``model`` 'win_edge' selects backs by the chance to win instead."""
    rows = []
    for side in SIDES:
        if model == "win_edge" and side == "lay":
            continue
        price = d.back1 if side == "back" else d.lay1
        size = d.back1_size if side == "back" else d.lay1_size
        ok = (price > 1) & (price <= MAX_PRICE) & (size.fillna(0) >= stake)
        score = d.edge_win if model == "win_edge" else ev[f"ev_{side}"]
        for bar in ([WIN_EDGE] if model == "win_edge" else bars):
            sel = ok & (score > bar)
            if sel.any():
                rows.append(d.loc[sel].assign(model=model, side=side, bar=bar, price=price[sel], ev=score[sel]))
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows, ignore_index=True)
    p["stake"] = stake
    p["value"] = np.where(p.side == "back", p.price / p.bsp - 1.0, 1.0 - p.price / p.bsp) * stake
    p["liability"] = np.where(p.side == "back", stake, (p.price - 1.0) * stake)
    # held to the result instead (for reference): commission on the winnings
    p["held"] = np.where(p.side == "back", np.where(p.won == 1, (p.price - 1) * (1 - COMMISSION), -1.0),
                         np.where(p.won == 1, -(p.price - 1), 1 - COMMISSION)) * stake
    return p


def settle(p: pd.DataFrame, keys=("mark", "model", "side", "bar")) -> pd.DataFrame:
    """Each rule's net per market after commission on the market's net (the exchange charges a market once)."""
    k = list(keys)
    m = p.groupby(k + ["race_date", "market_id"], dropna=False).agg(
        value=("value", "sum"), stake=("stake", "sum"), liability=("liability", "sum"), held=("held", "sum"),
        trades=("value", "size"))
    m["net"] = m.value - COMMISSION * m.value.clip(lower=0)
    return m.reset_index()


def rule_table(p: pd.DataFrame, keys=("mark", "model", "side", "bar"), n_boot: int = 2000) -> pd.DataFrame:
    """Each rule's trades, trades a day, ROI on the stake closed at the SP (90% by day), on the liability, and held to
    the result; the mean CLV of the price taken against the SP."""
    if p is None or not len(p):
        return pd.DataFrame()
    m = settle(p, keys)
    out = []
    for key, g in m.groupby(list(keys), dropna=False):
        day = g.groupby("race_date").agg(net=("net", "sum"), stake=("stake", "sum"), held=("held", "sum"))
        lo, hi = race_book.ratio_interval(day.net.to_numpy(), day.stake.to_numpy(), n=n_boot)
        tr = p.loc[(p[list(keys)] == pd.Series(dict(zip(keys, key)))).all(axis=1)]
        clv = np.where(tr.side == "back", tr.price / tr.bsp - 1, 1 - tr.price / tr.bsp)
        out.append({**dict(zip(keys, key)), "trades": int(g.trades.sum()), "days": int(len(day)),
                    "per_day": round(g.trades.sum() / max(1, len(day)), 1),
                    "staked": round(float(g.stake.sum()), 2), "net": round(float(g.net.sum()), 2),
                    "roi%": round(100 * float(g.net.sum() / g.stake.sum()), 2),
                    "roi_90%": [round(100 * lo, 2), round(100 * hi, 2)] if np.isfinite(lo) else None,
                    "roi_liability%": round(100 * float(g.net.sum() / g.liability.sum()), 2),
                    "held_roi%": round(100 * float(g.held.sum() / g.stake.sum()), 2),
                    "clv_mean%": round(100 * float(np.mean(clv)), 2),
                    "clv_positive%": round(100 * float(np.mean(clv > 0)), 1),
                    "price_median": round(float(tr.price.median()), 2)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------------------------------
# The store: panels, the freeze, the nightly run
# --------------------------------------------------------------------------------------------------------------------

def load_panel(store, day: date, s3=None, source: str = SOURCE) -> pd.DataFrame:
    """A day's panel from the forward test's priced runners and the greyhound books at the entry marks (``source``:
    the trader's or the recorder's), the SP from the recorder's settled books."""
    from betfair_recorder import _read_day_csv
    pred = store.get_parquet(f"{TRACK_PRED}/{day:%Y-%m-%d}.parquet")
    if pred is None or not len(pred):
        return pd.DataFrame()
    recorder = _read_day_csv(day, ENTRY_BOOKS["recorder"], s3=s3)
    books = recorder if source == "recorder" else _read_day_csv(day, ENTRY_BOOKS[source], s3=s3)
    markets = _read_day_csv(day, "markets_greyhound.csv", s3=s3)
    if books is None or recorder is None:
        return pd.DataFrame()
    return day_panel(pred, books, markets, day, closing_books=recorder)


def load_panels(store, first: date, last: date, s3=None, source: str = SOURCE) -> pd.DataFrame:
    frames, d = [], first
    while d <= last:
        try:
            f = load_panel(store, d, s3=s3, source=source)
        except Exception as exc:                                  # one bad day file: the rest are still read
            log.warning("%s: panel not built (%s)", d, exc)
            f = pd.DataFrame()
        if len(f):
            frames.append(f)
        d += timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def freeze(store, today: date, s3=None) -> dict:
    """Fit both closing models at every entry mark on every day of the trader's books before ``today`` and freeze them
    with the rules, once ``MIN_FREEZE_DAYS`` such days are held (else nothing is written: ``waiting``); the forward
    test starts the later of START and the day after."""
    d = load_panels(store, FIRST_BOOK_DAY, today - timedelta(days=1), s3=s3)
    n_days = int(d.race_date.nunique()) if len(d) else 0
    if n_days < MIN_FREEZE_DAYS:
        return {"waiting": f"{n_days} of {MIN_FREEZE_DAYS} days of the trader's books with priced runners held"}
    fitted = {m: fit_models(g) for m, g in d.groupby("mark")}
    first = max(date.fromisoformat(START), today + timedelta(days=1))
    out = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fit_from": str(d.race_date.min()),
           "fit_to": str(d.race_date.max()), "fit_days": n_days,
           "feed_lag_s_p50": round(float(d.drop_duplicates("market_id").feed_lag_s.median()), 1), "first_test_day": str(first), "rules": RULES, "models": fitted}
    store.put(FROZEN_KEY, json.dumps(out, indent=1).encode())
    return out


def track_day(store, frozen: dict, day: date, s3=None) -> dict:
    """One test day: every rule's positions, written to the ledger; None-final days are done again next night."""
    ds = f"{day:%Y-%m-%d}"
    d = load_panel(store, day, s3=s3)
    if not len(d):
        return {"day": ds, "note": "no panel yet (the forward test of 5 Oct has not priced the day, or no trader or "
                                   "settled books)"}
    parts = []
    for mark, g in d.groupby("mark"):
        if mark not in frozen["models"]:
            continue
        for name in MODELS:
            q, book = predict_share(g, frozen["models"][mark], name)
            parts.append(positions(g, expected_clv(g, q, book), name))
        parts.append(positions(g, None, "win_edge"))
    p = pd.concat([x for x in parts if len(x)], ignore_index=True) if any(len(x) for x in parts) else pd.DataFrame()
    store.put_parquet(f"{PREFIX}/days/{ds}.parquet", p)
    pr = RULES["primary"]
    prim = p[(p["mark"] == pr["mark"]) & (p.model == pr["model"]) & (p.side == pr["side"]) & (p.bar == pr["bar"])] \
        if len(p) else p
    lag = d.drop_duplicates("market_id").feed_lag_s
    return {"day": ds, "races": int(d.race_id.nunique()), "positions": int(len(p)), "primary_trades": int(len(prim)),
            "feed_lag_s_p50": round(float(lag.median()), 1) if lag.notna().any() else None,
            "primary_net": round(float(settle(prim).net.sum()), 2) if len(prim) else 0.0}


def summary(store, frozen: dict) -> dict:
    keys = sorted(k for k in store.listing(f"{PREFIX}/days/") if k.endswith(".parquet"))
    p = [store.get_parquet(k) for k in keys]
    p = pd.concat([x for x in p if x is not None and len(x)], ignore_index=True) if p else pd.DataFrame()
    out = {"first_test_day": frozen["first_test_day"], "days": len(keys), "rules": RULES}
    if not len(p):
        return out
    t = rule_table(p)
    out["rules_table"] = t.to_dict("records")
    pr = RULES["primary"]
    row = t[(t["mark"] == pr["mark"]) & (t.model == pr["model"]) & (t.side == pr["side"]) & (t.bar == pr["bar"])]
    out["primary"] = row.iloc[0].to_dict() if len(row) else None
    n_days, n_trades = len(keys), int(row.trades.iloc[0]) if len(row) else 0
    lo = row["roi_90%"].iloc[0][0] if len(row) and row["roi_90%"].iloc[0] else None
    out["gate"] = ("not yet judged: {} of 30 days, {} of 1,000 primary trades".format(n_days, n_trades)
                   if n_days < 30 or n_trades < 1000 else ("PASS" if lo is not None and lo > 0 else "FAIL"))
    return out


def markdown(s: dict) -> str:
    lines = [f"## Greyhound trade-out forward test from {s.get('first_test_day')} ({s.get('days', 0)} days)", "",
             f"Gate: {s.get('gate', 'no days yet')}", ""]
    if s.get("rules_table"):
        lines += ["| Mark | Model | Side | Bar | Trades (a day) | ROI closed at SP (90%) | Held to result | CLV |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in s["rules_table"]:
            lines.append(f"| {r['mark']} | {r['model']} | {r['side']} | {r['bar']} | {r['trades']} ({r['per_day']}) | "
                         f"{r['roi%']}% {r['roi_90%']} | {r['held_roi%']}% | {r['clv_mean%']}% |")
    return "\n".join(lines)


def main(argv=None) -> int:
    from betfair_recorder import uk_today
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="the sources store root (default S3)")
    ap.add_argument("--freeze", action="store_true", help="fit and freeze the closing models, once")
    ap.add_argument("--days", type=int, default=0, help="settle the last N days (to yesterday, UK) not yet done")
    ap.add_argument("--date", default=None, help="settle this one day again")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=None, help="write the report JSON here (and markdown beside it)")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    today = uk_today()
    report = {}
    raw = store.get(FROZEN_KEY)
    frozen = json.loads(raw) if raw else None
    if a.freeze:
        if frozen:
            report["frozen"] = "already frozen: " + frozen["frozen_utc"]
        else:
            frozen = freeze(store, today)
            if "waiting" in frozen:                            # not enough of the trader's books yet: nothing written
                report["frozen"] = frozen
                frozen = None
            else:
                report["frozen"] = {k: v for k, v in frozen.items() if k != "models"}
                report["frozen"]["fitted"] = {m: {k: v for k, v in f.items() if k in ("rows", "races", "days")}
                                              for m, f in frozen["models"].items()}
    if not frozen:
        if not a.freeze and (a.days or a.date or a.summary):
            ap.error("not frozen: run --freeze first")
        txt = json.dumps(report, indent=1, default=str)
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
            Path(a.out).with_suffix(".md").write_text(
                f"## Greyhound trade-out forward test: not started\n\n{(report.get('frozen') or {}).get('waiting', '')}\n")
        return 0
    days = []
    if a.date:
        days = [date.fromisoformat(a.date)]
    elif a.days:
        first = date.fromisoformat(frozen["first_test_day"])
        days = [d for d in (today - timedelta(days=i) for i in range(a.days, 0, -1)) if d >= first]
        days = [d for d in days if store.get_parquet(f"{PREFIX}/days/{d:%Y-%m-%d}.parquet") is None]
    if days:
        report["days"] = [track_day(store, frozen, d) for d in days]
    if frozen and (a.summary or days or a.freeze):
        s = summary(store, frozen)
        store.put(f"{PREFIX}/summary.json", json.dumps(s, indent=1, default=str).encode())
        report["summary"] = s
        if a.out:
            Path(a.out).with_suffix(".md").write_text(markdown(s))
    txt = json.dumps(report, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
