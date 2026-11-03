"""The greyhound trade-out (the owner, 2 Nov 2026): "focus on taking a price then trading out at the off, like the
horse racing model". A paper forward test from a fresh date, and the panel the research reads.

The trade, as the horse trader makes it (``trading/session.py``; the owner, 2 Nov 2026: "use any stake level, like
the horse racing model, fixed to win a certain amount; back multiple dogs per race"):

- **Back** a dog at the prices on offer at an entry mark, the stake sized to win ``TO_WIN`` (``TO_WIN / (price - 1)``,
  between ``MIN_STAKE`` and ``MAX_STAKE``), taken down the three best backs while each price still clears the bar
  (the horse trader's ladder), never a back whose winnings are under Betfair's GBP10 minimum for a lay at the SP.
  Several dogs in a race can be backed: each is its own position.
- **Close at the off** by laying the winnings at the Betfair SP (``MARKET_ON_CLOSE``, liability = stake x (price - 1)):
  if the dog wins the back and the lay cancel; if it loses the back's stake is lost and the lay wins
  ``liability / (BSP - 1)``. The position pays ``stake x (price - BSP) / (BSP - 1)`` when the dog loses and 0 when it
  wins: in expectation ``stake x (price / BSP - 1)`` when the BSP is calibrated, but settled here by the result, as
  the exchange settles it, with 2% commission on each market's net. (The first version booked the expectation as if
  it were certain; reports/greyhound_roadmap.md QA C2.)
- **Lay** the mirror image, reported only: lay at the best lays (stake sized so the liability is about ``TO_WIN``),
  closed by backing the same stake at the SP: ``stake x (BSP - lay)`` if the dog wins, 0 if it loses.

The forecast of the SP (``model.race_book.ClosingModel``, the horses' closing model, unchanged): a conditional logit
on the SP book, from the market's price at the entry mark (``race_book.market_now``), our price (the frozen
greyhound model of 5 Oct, ``greyhound.track``, out of sample on every day from 6 Oct) and the money matched. A
position's expected CLV at price m is ``m x E[1/BSP] - 1`` backing, ``1 - m x E[1/BSP]`` laying,
``E[1/BSP] = O x q_hat``.

The books (the owner, 2 Nov 2026: "use the live key, just don't execute bets yet"): ``greyhound/book_record.py``,
the live key, read-only, every 5 seconds in a race's last 5 minutes and on until the market closes, with each
runner's traded volume (``books_greyhound_live.csv``). The recorder's 60-second books are the delayed key's, about
90 seconds old near the off (reports/greyhound_model.md section 13): ``source="retimed"`` places each of them at the
market's last match instead of the poll, for research on the days before the live books.

The forward test (``RULES``):

- **Frozen once** (``--freeze``), when ``MIN_FREEZE_DAYS`` (10) days of the live books are held: the closing models
  fitted on those days; until then ``--freeze`` reports the days held and writes nothing.
- **Start:** the later of ``START`` and the day after the freeze. Paper only; nothing here can place an order.
- **Primary rule:** entry 1 minute before the off, the horses' closing model, back where the expected CLV clears 0.03,
  sized to win ``TO_WIN``, closed by the lay at the SP. Reported, not judged: the other marks, the lay side, the
  other bars, the microstructure model, the old edge-to-win selection.
- **Gate:** after 30 test days and 1,000 primary trades: a pass only if the return per GBP staked, settled by the
  result after commission, has a 90% lower bound above 0 (bootstrap over every test day, the days without a trade
  included) and the mean CLV against the SP is above 0. A pass goes to the owner.

The limits: a fill is taken on trust when the book offered the size at the price (the book at the moment of the
decision, read on the live key; queue position and our own impact are not modelled); the SP leg is assumed matched
once its liability reaches the minimum; a race with a late non-runner is dropped (no reduction factors applied).

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
#: whose books the test reads at the entry marks; the SP is always the recorder's settled book. "retimed": the
#: recorder's delayed books placed at the market's last match (research on the days before the live books)
ENTRY_BOOKS = {"live": "books_greyhound_live.csv", "recorder": "books_greyhound.csv",
               "retimed": "books_greyhound.csv", "trader": "books_greyhound_trader.csv"}
SOURCE = "live"
MIN_FREEZE_DAYS = 10
COMMISSION = 0.02
#: the stake, as the horse trader sizes it: to win TO_WIN at the price taken, between MIN_STAKE and MAX_STAKE; no
#: back whose winnings are under MIN_SP_LIABILITY (Betfair takes no lay at SP under GBP10 of liability)
TO_WIN = 100.0
MIN_STAKE = 2.0
MAX_STAKE = 50.0
MIN_SP_LIABILITY = 10.0
MIN_SP_BACK = 2.0                             # the smallest back at SP that closes a lay
LADDER = 3                                    # the best prices taken, as the horse trader's ladder
MAX_PRICE = 20.0
WIN_EDGE = 0.2                                # the old selection: the chance to win at the back price
#: the entry marks: minutes before the off, each a window of +-half a minute (no book serves two marks), and the
#: dog's first book after its first matched bet
OFF_ENTRY = (5, 3, 1)
MARK_HALF_WIDTH = 0.5
ENTRY_MARKS = ("first_traded", "off_5", "off_3", "off_1")
BARS = (0.0, 0.03, 0.05)
SIDES = ("back", "lay")
MODELS = ("racebook", "micro")
BOOK_COLS = ["back1", "back1_size", "back2", "back2_size", "back3", "back3_size", "lay1", "lay1_size", "lay2",
             "lay2_size", "lay3", "lay3_size", "last_traded", "runner_total_matched", "market_total_matched"]
#: the microstructure closing model's columns (``micro_design``), each standardised on the training rows
MICRO = ("lq_market", "lq_model", "last_vs_mid", "move", "imbalance", "log_spread", "lq_market_x_vol",
         "lq_model_x_vol")
#: a day whose panel holds under this share of the day's priced races is settled again the next night (up to
#: REDO_DAYS days back): its BSPs or books may still be arriving
MIN_COVERAGE = 0.9
REDO_DAYS = 3
RULES = {
    "start": f"the later of {START} and the day after the freeze",
    "books": "the live-key greyhound books (greyhound/book_record.py, books_greyhound_live.csv), read-only; the SP "
             f"from the recorder's settled books; frozen once {MIN_FREEZE_DAYS} days of the live books are held",
    "primary": {"mark": "off_1", "model": "racebook", "side": "back", "bar": 0.03},
    "reported": {"marks": list(ENTRY_MARKS), "models": list(MODELS) + ["win_edge"], "sides": list(SIDES),
                 "bars": list(BARS)},
    "stake": {"to_win": TO_WIN, "min": MIN_STAKE, "max": MAX_STAKE, "ladder": LADDER,
              "min_sp_liability": MIN_SP_LIABILITY, "several_dogs_a_race": True},
    "commission": COMMISSION, "max_price": MAX_PRICE,
    "close": "each back's winnings laid at the Betfair SP (liability = stake x (price - 1)), settled by the result; "
             "commission on each market's net",
    "gate": "judged after 30 test days and 1,000 primary trades, whichever is later: a pass only if the return per "
            "GBP staked, settled by the result after commission, has a 90% lower bound above 0 (bootstrap over every "
            "test day) and the mean CLV against the SP is above 0; a pass goes to the owner",
}


# --------------------------------------------------------------------------------------------------------------------
# The books at the entry marks
# --------------------------------------------------------------------------------------------------------------------

def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _inplay(x: pd.Series) -> pd.Series:
    """True where the book says in play: "True"/"1"/1.0 (a blank in the column makes pandas read it as float, and
    "0.0" is not "0": reports/greyhound_roadmap.md QA m3)."""
    t = x.astype(str).str.strip().str.lower()
    return t.isin(["true", "1", "1.0"])


def retime(b: pd.DataFrame) -> pd.Series:
    """Each delayed snapshot's estimated time in the market: the market's last match when it is within four minutes
    of the poll (near the off money matches every few seconds, so the last match is the snapshot's own time),
    otherwise the poll less the market's median lag over those snapshots, otherwise the poll."""
    lm = pd.to_datetime(b.last_match_utc, utc=True, errors="coerce") if "last_match_utc" in b else \
        pd.Series(pd.NaT, index=b.index, dtype="datetime64[ns, UTC]")
    lag = (b.polled - lm).dt.total_seconds()
    near = lag.where((lag >= 0) & (lag <= 240))
    market_lag = near.groupby(b.market_id).transform("median")
    est = b.polled - pd.to_timedelta(market_lag.fillna(0.0), unit="s")
    return lm.where(near.notna(), est)


def pre_off_books(books: pd.DataFrame, markets: pd.DataFrame, retimed: bool = False) -> pd.DataFrame:
    """Every open, pre-play book of an active runner, with its minutes to the scheduled off. ``retimed``: each
    snapshot at its estimated time in the market (``retime``), so a delayed snapshot polled after the scheduled off
    that still shows the market open counts for the minute it shows."""
    offs = markets.drop_duplicates("market_id").set_index("market_id").market_start_utc
    b = books.copy()
    b["market_id"] = b.market_id.astype(str)
    b["selection_id"] = _num(b.selection_id)
    b["polled"] = pd.to_datetime(b.polled_utc, utc=True, errors="coerce")
    b = b.dropna(subset=["polled", "selection_id"])
    b["off"] = pd.to_datetime(b.market_id.map(offs.set_axis(offs.index.astype(str))), utc=True, errors="coerce")
    if retimed:
        b["polled"] = retime(b)
    b["minutes_to_off"] = (b.off - b.polled).dt.total_seconds() / 60.0
    ok = ((b.source.astype(str) != "final") & (b.status.astype(str) == "OPEN") & ~_inplay(b.inplay)
          & (b.minutes_to_off > 0) & (b.runner_status.fillna("ACTIVE").astype(str) == "ACTIVE"))
    b = b[ok].dropna(subset=["polled"])
    for c in BOOK_COLS:
        b[c] = _num(b[c]) if c in b else np.nan
    return b


def entry_books(pre: pd.DataFrame) -> pd.DataFrame:
    """Each runner's book at each entry mark (the book nearest the mark within +-``MARK_HALF_WIDTH`` minutes, so no
    book serves two marks), and its book 90 seconds or more before (``*_prev``), for the price's move into the mark."""
    picks = []
    for m in OFF_ENTRY:
        d = pre.assign(gap=(pre.minutes_to_off - m).abs())
        d = d[d.gap <= MARK_HALF_WIDTH]
        picks.append(d.sort_values("gap").drop_duplicates(["market_id", "selection_id"]).assign(mark=f"off_{m}"))
    traded = pre[pre.last_traded > 1]
    picks.append(traded.sort_values("polled").drop_duplicates(["market_id", "selection_id"]).assign(mark="first_traded"))
    e = pd.concat(picks, ignore_index=True)
    keep = ["market_id", "selection_id", "mark", "polled", "minutes_to_off"] + BOOK_COLS
    e = e[keep]
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
              closing_books: pd.DataFrame | None = None, retimed: bool = False) -> pd.DataFrame:
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
    pre = pre_off_books(books, markets, retimed=retimed)
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
    """E[1/BSP] for each runner (``inv``), and each position's expected CLV at the best price: ``m x inv - 1``
    backing at the best back, ``1 - m x inv`` laying at the best lay."""
    inv = book * np.asarray(q_hat, float)
    lay = d.lay1.where(d.lay1 > 1)
    return pd.DataFrame({"inv": inv, "ev_back": d.back1.to_numpy(float) * inv - 1.0,
                         "ev_lay": 1.0 - lay.to_numpy(float) * inv}, index=d.index)


# --------------------------------------------------------------------------------------------------------------------
# Positions and their settlement
# --------------------------------------------------------------------------------------------------------------------

def target_stake(price, to_win: float = TO_WIN) -> np.ndarray:
    """The stake that wins ``to_win`` at ``price`` (a lay: whose liability is ``to_win``), between MIN_STAKE and
    MAX_STAKE."""
    price = np.asarray(price, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(to_win / (price - 1.0), MIN_STAKE, MAX_STAKE)


def ladder_fill(prices: np.ndarray, sizes: np.ndarray, ok: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Take each runner's best ``LADDER`` prices in order while ``ok`` (the price still clears the bar and the cap),
    up to the target stake: the matched stake and its average price (stake-weighted)."""
    n, k = prices.shape
    left = target.copy()
    matched = np.zeros(n)
    paid = np.zeros(n)
    going = np.ones(n, bool)
    for j in range(k):
        going &= ok[:, j] & np.isfinite(prices[:, j]) & (prices[:, j] > 1)
        take = np.where(going, np.minimum(left, np.nan_to_num(sizes[:, j]).clip(min=0)), 0.0)
        matched += take
        paid += take * np.nan_to_num(prices[:, j])
        left -= take
    with np.errstate(divide="ignore", invalid="ignore"):
        return matched, np.where(matched > 0, paid / np.where(matched > 0, matched, 1.0), np.nan)


def positions(d: pd.DataFrame, ev: pd.DataFrame | None, model: str, bars=BARS, to_win: float = TO_WIN) -> pd.DataFrame:
    """Every rule's paper positions, as the horse trader takes them: for each side and bar, each runner whose best
    price clears the bar is taken down the ladder to win ``to_win`` (several dogs a race allowed), a back only when
    its winnings reach the minimum lay at SP, a lay only when its stake reaches the minimum back at SP. ``model``
    'win_edge' selects backs by the chance to win at each price instead."""
    rows = []
    for side in SIDES:
        if model == "win_edge" and side == "lay":
            continue
        cols = [f"{side}{j}" for j in range(1, LADDER + 1)]
        prices = np.column_stack([_num(d[c]).to_numpy(float) if c in d else np.full(len(d), np.nan) for c in cols])
        sizes = np.column_stack([_num(d[f"{c}_size"]).to_numpy(float) if f"{c}_size" in d else np.full(len(d), np.nan)
                                 for c in cols])
        if model == "win_edge":
            score = d.p_model.to_numpy(float)[:, None] * (1 + (prices - 1) * (1 - COMMISSION)) - 1
        else:
            inv = ev["inv"].to_numpy(float)[:, None]
            score = prices * inv - 1.0 if side == "back" else 1.0 - prices * inv
        target = target_stake(prices[:, 0], to_win)
        for bar in ([WIN_EDGE] if model == "win_edge" else bars):
            ok = (score > bar) & (prices <= MAX_PRICE)
            matched, avg = ladder_fill(prices, sizes, ok, target)
            if side == "back":
                feasible = matched * (avg - 1.0) >= MIN_SP_LIABILITY - 1e-9
            else:
                feasible = matched >= MIN_SP_BACK - 1e-9
            sel = (matched > 0) & feasible
            if sel.any():
                rows.append(d.loc[sel].assign(model=model, side=side, bar=bar, stake=matched[sel], price=avg[sel],
                                              target=target[sel], ev=score[sel, 0]))
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows, ignore_index=True)
    s, m, bsp, won = p.stake.to_numpy(), p.price.to_numpy(), p.bsp.to_numpy(float), p.won.to_numpy() == 1
    back = (p.side == "back").to_numpy()
    # settled by the result, as the exchange settles the back and its lay at SP (liability = the winnings), or the
    # lay and its back at SP (the same stake)
    p["pnl"] = np.where(back, np.where(won, 0.0, -s + s * (m - 1.0) / (bsp - 1.0)),
                        np.where(won, s * (bsp - m), 0.0))
    p["risk"] = np.where(back, s, s * (m - 1.0))                       # the stake backed, the liability laid
    p["clv"] = np.where(back, m / bsp - 1.0, 1.0 - m / bsp)            # the price taken against the SP
    p["clv_value"] = p.clv * s                                         # the expectation, if the BSP is calibrated
    # held to the result instead (for reference): no close
    p["held"] = np.where(back, np.where(won, s * (m - 1.0), -s), np.where(won, -s * (m - 1.0), s))
    return p


def settle(p: pd.DataFrame, keys=("mark", "model", "side", "bar")) -> pd.DataFrame:
    """Each rule's net per market after commission on the market's net (the exchange charges a market once), for the
    close at SP and for holding to the result."""
    k = list(keys)
    m = p.groupby(k + ["race_date", "market_id"], dropna=False).agg(
        pnl=("pnl", "sum"), risk=("risk", "sum"), stake=("stake", "sum"), held=("held", "sum"),
        clv_value=("clv_value", "sum"), trades=("pnl", "size"))
    m["net"] = m.pnl - COMMISSION * m.pnl.clip(lower=0)
    m["held_net"] = m.held - COMMISSION * m.held.clip(lower=0)
    return m.reset_index()


def rule_table(p: pd.DataFrame, keys=("mark", "model", "side", "bar"), days=None, n_boot: int = 2000) -> pd.DataFrame:
    """Each rule's trades, trades a day over every scored day (``days``; the days with positions when not given), the
    return on the money at risk settled by the result (90%: a bootstrap over every scored day, those without a trade
    included), held to the result, and the CLV of the prices taken against the SP."""
    if p is None or not len(p):
        return pd.DataFrame()
    all_days = sorted(set(days) if days is not None else set(p.race_date.astype(str)))
    m = settle(p, keys)
    out = []
    for key, g in m.groupby(list(keys), dropna=False):
        day = g.groupby(g.race_date.astype(str)).agg(net=("net", "sum"), risk=("risk", "sum"))
        day = day.reindex(all_days, fill_value=0.0)
        lo, hi = race_book.ratio_interval(day.net.to_numpy(), day.risk.to_numpy(), n=n_boot)
        tr = p.loc[(p[list(keys)] == pd.Series(dict(zip(keys, key)))).all(axis=1)]
        out.append({**dict(zip(keys, key)), "trades": int(g.trades.sum()), "days": len(all_days),
                    "active_days": int((day.risk > 0).sum()),
                    "per_day": round(g.trades.sum() / max(1, len(all_days)), 1),
                    "staked": round(float(g.stake.sum()), 2), "risk": round(float(g.risk.sum()), 2),
                    "net": round(float(g.net.sum()), 2),
                    "roi%": round(100 * float(g.net.sum() / g.risk.sum()), 2),
                    "roi_90%": [round(100 * lo, 2), round(100 * hi, 2)] if np.isfinite(lo) else None,
                    "held_roi%": round(100 * float(g.held_net.sum() / g.risk.sum()), 2),
                    "clv_mean%": round(100 * float(tr.clv.mean()), 2),
                    "clv_positive%": round(100 * float((tr.clv > 0).mean()), 1),
                    "fill%": round(100 * float((tr.stake / tr.target).mean()), 1),
                    "price_median": round(float(tr.price.median()), 2)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------------------------------
# The store: panels, the freeze, the nightly run
# --------------------------------------------------------------------------------------------------------------------

def load_panel(store, day: date, s3=None, source: str = SOURCE) -> pd.DataFrame:
    """A day's panel from the forward test's priced runners and the greyhound books at the entry marks (``source``:
    ``live``, the live-key recorder's; ``recorder``, the delayed 60-second books as polled; ``retimed``, the same placed
    at the market's last match), the SP from the recorder's settled books."""
    from betfair_recorder import _read_day_csv
    pred = store.get_parquet(f"{TRACK_PRED}/{day:%Y-%m-%d}.parquet")
    if pred is None or not len(pred):
        return pd.DataFrame()
    recorder = _read_day_csv(day, ENTRY_BOOKS["recorder"], s3=s3)
    books = recorder if source in ("recorder", "retimed") else _read_day_csv(day, ENTRY_BOOKS[source], s3=s3)
    markets = _read_day_csv(day, "markets_greyhound.csv", s3=s3)
    if books is None or recorder is None or markets is None:
        return pd.DataFrame()
    return day_panel(pred, books, markets, day, closing_books=recorder, retimed=source == "retimed")


def load_panels(store, first: date, last: date, s3=None, source: str = SOURCE) -> pd.DataFrame:
    """Every day's panel; a day whose files fail to read is skipped, but if most days fail the run fails (a schema
    change must not read as "no books yet": QA m4)."""
    frames, failed, tried, d = [], [], 0, first
    while d <= last:
        tried += 1
        try:
            f = load_panel(store, d, s3=s3, source=source)
        except Exception as exc:
            log.warning("%s: panel not built (%s)", d, exc)
            failed.append(f"{d}: {exc}"[:200])
            f = pd.DataFrame()
        if len(f):
            frames.append(f)
        d += timedelta(days=1)
    if tried >= 4 and len(failed) > tried / 2:
        raise RuntimeError(f"{len(failed)} of {tried} days failed to build: " + "; ".join(failed[:3]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _clean(x):
    """A float for JSON: None for NaN or infinity (QA m7)."""
    return None if x is None or not np.isfinite(x) else x


def freeze(store, today: date, s3=None) -> dict:
    """Fit both closing models at every entry mark on every day of the live books before ``today`` and freeze them
    with the rules, once ``MIN_FREEZE_DAYS`` such days are held (else nothing is written: ``waiting``); the forward
    test starts the later of START and the day after."""
    d = load_panels(store, FIRST_BOOK_DAY, today - timedelta(days=1), s3=s3)
    n_days = int(d.race_date.nunique()) if len(d) else 0
    if n_days < MIN_FREEZE_DAYS:
        return {"waiting": f"{n_days} of {MIN_FREEZE_DAYS} days of the live greyhound books with priced runners held"}
    fitted = {m: fit_models(g) for m, g in d.groupby("mark")}
    first = max(date.fromisoformat(START), today + timedelta(days=1))
    lag = d.drop_duplicates("market_id").feed_lag_s
    out = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fit_from": str(d.race_date.min()),
           "fit_to": str(d.race_date.max()), "fit_days": n_days,
           "feed_lag_s_p50": _clean(round(float(lag.median()), 1)) if lag.notna().any() else None,
           "first_test_day": str(first), "rules": RULES, "models": fitted}
    store.put(FROZEN_KEY, json.dumps(out, indent=1).encode())
    return out


def track_day(store, frozen: dict, day: date, s3=None) -> dict:
    """One test day: every rule's positions to the ledger, and the day's coverage (the share of its priced races in
    the panel) beside it; a day under MIN_COVERAGE is settled again on the next nights (QA J6)."""
    ds = f"{day:%Y-%m-%d}"
    if day < date.fromisoformat(frozen["first_test_day"]):
        raise ValueError(f"{ds} is before the first test day {frozen['first_test_day']}: the frozen models saw it")
    d = load_panel(store, day, s3=s3)
    pred = store.get_parquet(f"{TRACK_PRED}/{ds}.parquet")
    priced = int(pred.dropna(subset=["market_id"]).race_id.nunique()) if pred is not None and len(pred) else 0
    if not len(d):
        return {"day": ds, "note": "no panel yet (the forward test of 5 Oct has not priced the day, or no live or "
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
    races = int(d[d["mark"] == RULES["primary"]["mark"]].race_id.nunique())
    coverage = round(races / priced, 3) if priced else 0.0
    store.put(f"{PREFIX}/days/{ds}.json", json.dumps({"day": ds, "races_priced": priced, "races_in_panel": races,
                                                      "coverage": coverage}).encode())
    pr = RULES["primary"]
    prim = p[(p["mark"] == pr["mark"]) & (p.model == pr["model"]) & (p.side == pr["side"]) & (p.bar == pr["bar"])] \
        if len(p) else p
    lag = d.drop_duplicates("market_id").feed_lag_s
    return {"day": ds, "races": int(d.race_id.nunique()), "coverage": coverage, "positions": int(len(p)),
            "primary_trades": int(len(prim)),
            "feed_lag_s_p50": _clean(round(float(lag.median()), 1)) if lag.notna().any() else None,
            "primary_net": round(float(settle(prim).net.sum()), 2) if len(prim) else 0.0}


def _settled(store, day: date, today: date) -> bool:
    """A test day done for good: its ledger is written and either its coverage reached MIN_COVERAGE or it is more
    than REDO_DAYS days old."""
    ds = f"{day:%Y-%m-%d}"
    if store.get_parquet(f"{PREFIX}/days/{ds}.parquet") is None:
        return False
    raw = store.get(f"{PREFIX}/days/{ds}.json")
    cov = json.loads(raw).get("coverage", 0.0) if raw else 0.0
    return cov >= MIN_COVERAGE or (today - day).days > REDO_DAYS


def summary(store, frozen: dict) -> dict:
    """The test so far: only days on or after the first test day count (QA J5), trades a day and the bootstrap over
    every one of them."""
    first = frozen["first_test_day"]
    keys = sorted(k for k in store.listing(f"{PREFIX}/days/") if k.endswith(".parquet") and k[-18:-8] >= first)
    days = [k[-18:-8] for k in keys]
    p = [store.get_parquet(k) for k in keys]
    p = pd.concat([x for x in p if x is not None and len(x)], ignore_index=True) if p else pd.DataFrame()
    out = {"first_test_day": first, "days": len(keys), "rules": RULES}
    if not len(p):
        return out
    t = rule_table(p, days=days)
    out["rules_table"] = t.to_dict("records")
    pr = RULES["primary"]
    row = t[(t["mark"] == pr["mark"]) & (t.model == pr["model"]) & (t.side == pr["side"]) & (t.bar == pr["bar"])]
    out["primary"] = row.iloc[0].to_dict() if len(row) else None
    n_days, n_trades = len(keys), int(row.trades.iloc[0]) if len(row) else 0
    lo = row["roi_90%"].iloc[0][0] if len(row) and row["roi_90%"].iloc[0] else None
    clv = float(row["clv_mean%"].iloc[0]) if len(row) else None
    out["gate"] = ("not yet judged: {} of 30 days, {} of 1,000 primary trades".format(n_days, n_trades)
                   if n_days < 30 or n_trades < 1000
                   else ("PASS" if lo is not None and lo > 0 and clv is not None and clv > 0 else "FAIL"))
    return out


def markdown(s: dict) -> str:
    lines = [f"## Greyhound trade-out forward test from {s.get('first_test_day')} ({s.get('days', 0)} days)", "",
             f"Gate: {s.get('gate', 'no days yet')}", ""]
    if s.get("rules_table"):
        lines += ["| Mark | Model | Side | Bar | Trades (a day) | Return closed at SP, by result (90%) | Held to result |"
                  " CLV | Filled |", "|---|---|---|---|---|---|---|---|---|"]
        for r in s["rules_table"]:
            lines.append(f"| {r['mark']} | {r['model']} | {r['side']} | {r['bar']} | {r['trades']} ({r['per_day']}) | "
                         f"{r['roi%']}% {r['roi_90%']} | {r['held_roi%']}% | {r['clv_mean%']}% | {r['fill%']}% |")
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
    first = date.fromisoformat(frozen["first_test_day"])
    if a.date:
        days = [date.fromisoformat(a.date)]
        if days[0] < first:                               # the frozen models saw it: never a test day (QA J5)
            ap.error(f"{a.date} is before the first test day {first}")
    elif a.days:
        days = [d for d in (today - timedelta(days=i) for i in range(a.days, 0, -1)) if d >= first]
        days = [d for d in days if not _settled(store, d, today)]
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
