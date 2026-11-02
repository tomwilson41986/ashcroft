"""The football paper trial's scorer: the pre-registered early-to-close rule, read at the recorded Betfair books.

The rule, the gate and the window are the pre-registration's (reports/preregistration_football_exchange.md, its
JSON block, read here and its SHA-256 kept beside the result). This module only scores: it places nothing, it
changes nothing in the live record, and it reads the football book record that paper-models.yml writes
(betfair_live/<day>/books_football_<C>.csv and books_football_eve_<C>.csv, with their markets_*.csv) through
betfair_recorder's own day-file reader (the server's copy, else S3).

For each match of the 22 main divisions, from the first Friday collection after the recorder is live:

1. **The early moment.** football-data's collection (matches.php): Friday 17:00 UK for Friday to Monday matches,
   Tuesday 13:00 UK for Tuesday to Thursday. The features come from the fixtures file of that collection, which
   source-data.yml fetches at 03:20 UTC the next day (bronze ``football/raw/fixtures/<day>/fixtures.csv``): the
   file of the first fetch day after the collection day is used, never the collection day's own (fetched before
   the collection). The Dixon-Coles probability is fitted on the matches before the collection day only.
2. **The forecast.** The registered model (``gbm_nopin``, football_exchange.exchange_models) fitted on every month
   of the exchange panel before the match's month, on the rule's market type, as the holdout was; its forecast of
   log(close / early) on fair prices from football-data's Betfair early book and the books of the same moment.
3. **The signal.** |forecast| over the rule's threshold: shorter, back at the early book and lay at the close;
   longer, lay then back. Greened, so the result does not matter.
4. **The book-width filter, on the recorder's own book.** The first recorded snapshot at or after the collection
   moment (within ``tolerance_min``), pre-play and open, with every selection's best back: its best-back book (the
   sum of 1/back) must sit in the rule's widths (1.05 or under). The share of signals whose recorded book was wide
   is reported.
5. **The fill.** A back at the recorded best back, a lay at the recorded best lay, only where the size shown at that
   price covers the notional stake (GBP10); else the trade is skipped and logged.
6. **The close.** The last recorded two-sided book before the market turns in play: a back closed by a lay at the
   best lay, a lay closed by a back at the best back. Return a stake unit, 2% commission on each market's net.

The window and the gate (the pre-registration's paper trial): at least 300 rule trades and at least 90 days,
whichever is later, capped at 12 months; passes only if the 90% lower bound of the return at the taken prices is
above zero (bootstrap by day, 2,000 draws), the mean CLV at the taken price (a back o/c - 1, a lay 1 - o/c, c the
recorded close mid) is above zero, and both halves of the window, split at its middle date, return above zero.
Until the window is complete the figures are interim and the gate reads "running", never a verdict.

    python -m research.open_close.football_paper_trial --panel data/football_exchange_panel.parquet \\
        --out out/football_paper_trial.json --trades-out out/football_paper_trial_trades.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sources.football import MAIN

from . import football_exchange as fx
from .backtest import summarise
from .features import add_features, coerce_panel, price_band

log = logging.getLogger(__name__)

PREREG = "reports/preregistration_football_exchange.md"
#: the countries paper-models.yml records (one recorder process each, tags football_<C> and football_eve_<C>)
COUNTRIES = ("GB", "ES", "IT", "DE", "FR", "NL", "BE", "PT", "TR", "GR")
#: football-data's collections (matches.php): weekday (Mon 0) -> UK clock time
COLLECTION_TIME = {4: "17:00", 1: "13:00"}
#: the earliest the recorder can be live (paper-models.yml's football book record, commit f16dcdc, 2 Nov 2026)
RECORDER_FROM = "2026-11-02"
NOTIONAL_GBP = 10.0
TOLERANCE_MIN = 60.0
CAP_DAYS = 365
#: each market type's selections: football-data's leg -> how the recorder's runner is told apart
SELECTIONS = {"MATCH_ODDS": ("H", "D", "A"), "OVER_UNDER_25": ("Over 2.5", "Under 2.5")}
#: the 22 main divisions football-data publishes Betfair early prices for (the rule's "leagues": "all")
MAIN_DIVISIONS = tuple(MAIN)

ReadDay = Callable[[date, str], "pd.DataFrame | None"]


# ---------------------------------------------------------------------------
# the calendar
# ---------------------------------------------------------------------------

def collection_of(match_date) -> tuple[date, pd.Timestamp]:
    """A match day's football-data collection: its day and its moment in UTC (Friday 17:00 UK for Friday to Monday
    matches, Tuesday 13:00 UK for Tuesday to Thursday)."""
    d = pd.Timestamp(match_date).normalize()
    dow = d.dayofweek
    back = dow - 1 if dow in (1, 2, 3) else (dow - 4) % 7
    c = d - pd.Timedelta(days=back)
    moment = pd.Timestamp(f"{c:%Y-%m-%d} {COLLECTION_TIME[c.dayofweek]}").tz_localize("Europe/London")
    return c.date(), moment.tz_convert("UTC")


def next_collection(day: date) -> date:
    """The first collection day (Tuesday or Friday) strictly after ``day``."""
    d = day + timedelta(days=1)
    while d.weekday() not in COLLECTION_TIME:
        d += timedelta(days=1)
    return d


def trial_start(first_recorded: date) -> date:
    """The first Friday collection on or after the recorder's first day (its job starts at 05:45 UTC, before 17:00)."""
    d = first_recorded
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# football-data's side: the fixtures file of the collection, the ids, the Dixon-Coles fit
# ---------------------------------------------------------------------------

def fixtures_stamp(stamps: list[str], collection: date) -> str | None:
    """The bronze day folder holding ``collection``'s fixtures file: the first fetch after the collection day and
    before the next collection (fetches run at 03:20 UTC, so the collection day's own folder holds the one before)."""
    lo, hi = collection + timedelta(days=1), next_collection(collection)
    ok = sorted(s for s in stamps if lo.isoformat() <= s <= hi.isoformat())
    return ok[0] if ok else None


def read_fixtures_file(store, stamp: str) -> pd.DataFrame | None:
    from sources.football import read_csv
    raw = store.get(f"football/raw/fixtures/{stamp}/fixtures.csv")
    if not raw:
        return None
    f = read_csv(raw)
    if f.empty or "Div" not in f.columns or "HomeTeam" not in f.columns:
        return None
    f["match_date"] = pd.to_datetime(f["Date"], dayfirst=True, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    return f


def cover_days(collection: date) -> list[str]:
    """The match days a collection prices: Friday to Monday, or Tuesday to Thursday."""
    n = 4 if collection.weekday() == 4 else 3
    return [(collection + timedelta(days=i)).isoformat() for i in range(n)]


def dc_probs(gold: pd.DataFrame, rows: pd.DataFrame, as_of: str, params: dict | None = None) -> pd.DataFrame:
    """The Dixon-Coles chances of ``rows`` (gold columns) from each pool's fit on the finished matches before
    ``as_of`` only (football.ratings.fit_pool: an exclusive bound), as the weekly walk-forward prices them."""
    from football.model import DEFAULTS, _price_rows
    from football.ratings import fit_pool
    p = {**DEFAULTS, **(params or {})}
    out = []
    r = rows.copy()
    for c in ("open_ah_line", "close_ah_line"):
        if c not in r.columns:
            r[c] = np.nan
    for pool, part in r.groupby("country"):
        fit, _ = fit_pool(gold[gold["country"] == pool], pool, as_of, xi=p["xi"], l2=p["l2"],
                          window_days=p["window_days"], sot_mix=p["sot_mix"])
        if fit is not None:
            out += _price_rows(fit, part)
    cols = ["match_id", "p_h", "p_d", "p_a", "p_o25"]
    return pd.DataFrame(out)[cols] if out else pd.DataFrame(columns=cols)


def trial_rows(fx_ids: pd.DataFrame, gold: pd.DataFrame, dc: pd.DataFrame | None, market_type: str) -> pd.DataFrame:
    """The registered model's input rows for one market type: football-data's Betfair early price (BFE*) as the
    open, the same moment's bookmakers' books, the Dixon-Coles chance; no close (it is not known yet). The open-side
    preparation is the panel's (features.prepare with normalise_book), the close left out."""
    meta = gold.drop_duplicates("match_id").set_index("match_id")
    ids = fx_ids.drop_duplicates("match_id")
    kick = pd.to_datetime(ids["match_id"].map(meta["kickoff_utc"]), utc=True, errors="coerce")
    day = pd.to_datetime(ids["match_date"], errors="coerce").dt.tz_localize("UTC") + pd.Timedelta(hours=15)
    base = pd.DataFrame({"sport": "football", "country": ids["match_id"].map(meta["country"]).astype(str),
                         "event_id": ids["match_id"].to_numpy(),
                         "event_name": (ids["HomeTeam"].astype(str) + " v " + ids["AwayTeam"].astype(str)).to_numpy(),
                         "off_time_utc": kick.fillna(day).to_numpy(), "open_time_utc": pd.NaT,
                         "season": ids["match_id"].map(meta["season"]).to_numpy(),
                         "competition_id": ids["match_id"].map(meta["competition_id"]).to_numpy()})
    parts = []
    for sel in SELECTIONS[market_type]:
        col = "BFE" + fx.LEG_SUFFIX[(market_type, sel)]
        p = base.copy()
        p["market_type"], p["selection_id"], p["selection_name"] = market_type, sel, sel
        p["market_id"] = p["event_id"].astype(str) + ":" + market_type
        p["open_price"] = (pd.to_numeric(ids[col], errors="coerce").to_numpy() if col in ids.columns else np.nan)
        p["close_price"] = np.nan
        p["n_runners"] = len(SELECTIONS[market_type])
        parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    d = coerce_panel(d)
    d = d[d["open_price"].between(1.01, 1000.0) & d["off_time_utc"].notna()]
    whole = d.groupby("market_id")["open_price"].transform("size") == len(SELECTIONS[market_type])
    d = d[whole].copy()
    if d.empty:
        return d
    d = fx.add_exchange_features(d, fx.book_frame(ids), gold, dc)
    oo = (1.0 / d["open_price"]).groupby(d["market_id"]).transform("sum")
    d["overround_open_raw"] = oo
    d["open_price"] = d["open_price"] * oo
    d = add_features(d)
    d["league_code"] = pd.Categorical(d["fd_code"].astype(str))
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# the recorder's side: markets to matches, the early book, the close
# ---------------------------------------------------------------------------

def recorder_reader(root: str | None = None) -> ReadDay:
    """betfair_recorder's own day-file reader (the server's copy under ``root``, else the gzipped S3 copy)."""
    from betfair_recorder import _read_day_csv
    return lambda day, name: _read_day_csv(day, name, None, Path(root) if root else None)


def recorded_day(read_day: ReadDay, day: date, stem: str) -> pd.DataFrame | None:
    """Every football tag's file of one stem (books or markets) for the day's matches, as one frame."""
    parts = []
    for c in COUNTRIES:
        for tag in (f"football_{c}", f"football_eve_{c}"):
            f = read_day(day, f"{stem}_{tag}.csv")
            if f is not None and len(f):
                parts.append(f.assign(_tag=tag))
    return pd.concat(parts, ignore_index=True) if parts else None


def first_recorded_day(read_day: ReadDay, start: date, end: date) -> date | None:
    d = start
    while d <= end:
        if recorded_day(read_day, d, "markets") is not None:
            return d
        d += timedelta(days=1)
    return None


def selection_key(market_type: str, runner: str, home: str, away: str) -> str | None:
    r = str(runner or "").strip()
    if market_type == "OVER_UNDER_25":
        low = r.lower()
        if "2.5" not in low:
            return None
        return "Over 2.5" if low.startswith("over") else ("Under 2.5" if low.startswith("under") else None)
    if r.lower() == "the draw":
        return "D"
    return "H" if r == home else ("A" if r == away else None)


def map_markets(markets: pd.DataFrame, fixtures: pd.DataFrame, aliases=None) -> pd.DataFrame:
    """The recorder's markets of a day -> (market_id, selection_id) rows carrying the match_id and football-data's
    leg, by both team names against that day's fixtures of the market's country (football.reference.pair_event,
    unique or nothing, as the Betfair map of football.data pairs them)."""
    from football.data import BF_COUNTRY, split_event
    from football.reference import pair_event
    cols = ["market_id", "selection_id", "market_type", "match_id", "leg", "event_name", "market_start_utc"]
    if markets is None or markets.empty or fixtures.empty:
        return pd.DataFrame(columns=cols)
    m = markets[markets["market_type"].isin(list(SELECTIONS))].copy()
    m["market_id"] = m["market_id"].astype(str)
    m["selection_id"] = _sid(m["selection_id"])
    out = []
    for mid, g in m.groupby("market_id", sort=False):
        first = g.iloc[0]
        teams = split_event(first["event_name"])
        if teams is None:
            continue
        cand = fixtures[fixtures["country"].isin(BF_COUNTRY.get(str(first["country"]), set()))]
        best, score = None, 0.0
        for country, sub in cand.groupby("country"):
            fl = list(zip(sub["match_id"], sub["HomeTeam"].astype(str), sub["AwayTeam"].astype(str)))
            i, sc = pair_event(teams[0], teams[1], fl, country, "betfair", aliases)
            if i is not None and sc > score:
                best, score = fl[i][0], sc
        if best is None:
            continue
        for r in g.drop_duplicates("selection_id").itertuples(index=False):
            leg = selection_key(first["market_type"], r.runner_name, teams[0], teams[1])
            if leg is not None:
                out.append({"market_id": mid, "selection_id": r.selection_id, "market_type": first["market_type"],
                            "match_id": best, "leg": leg, "event_name": first["event_name"],
                            "market_start_utc": first.get("market_start_utc")})
    return pd.DataFrame(out, columns=cols)


def _sid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    b = books.copy()
    b["market_id"] = b["market_id"].astype(str)
    b["selection_id"] = _sid(b["selection_id"])
    b["polled"] = pd.to_datetime(b["polled_utc"], utc=True, errors="coerce")
    for c in ("back1", "back1_size", "lay1", "lay1_size"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["inplay"] = b["inplay"].astype(str).str.lower().isin(("true", "1", "yes"))
    b["status"] = b["status"].astype(str).str.upper()
    return b[b["polled"].notna()].drop_duplicates(["market_id", "selection_id", "polled"]).sort_values("polled")


def early_book(b: pd.DataFrame, selections: list[str], moment: pd.Timestamp, tolerance_min: float
               ) -> tuple[pd.Timestamp | None, pd.DataFrame | None]:
    """One market's first snapshot at or after ``moment`` and within ``tolerance_min`` that is open, pre-play and
    carries a best back for every selection; (None, None) when there is none."""
    hi = moment + pd.Timedelta(minutes=tolerance_min)
    w = b[(b["polled"] >= moment) & (b["polled"] <= hi) & (b["status"] == "OPEN") & ~b["inplay"]]
    for t, snap in w.groupby("polled", sort=True):
        snap = snap.drop_duplicates("selection_id").set_index("selection_id")
        if all(s in snap.index and snap.loc[s, "back1"] > 1.0 for s in selections):
            return t, snap.loc[selections]
    return None, None


def close_book(b: pd.DataFrame) -> pd.DataFrame:
    """Each selection's last recorded two-sided book (best back and best lay) while open and before the market
    first shows in play."""
    first_inplay = b.loc[b["inplay"], "polled"].min()
    pre = b[(b["status"] == "OPEN") & ~b["inplay"] & b["back1"].gt(1.0) & b["lay1"].gt(1.0)]
    if pd.notna(first_inplay):
        pre = pre[pre["polled"] < first_inplay]
    return pre.groupby("selection_id").tail(1).set_index("selection_id")


def width_label(over: float) -> str:
    return str(fx.width_label([over])[0])


def fill_trade(side: int, early: pd.Series, close: pd.Series | None, notional: float) -> dict:
    """One signal at the recorded books: the fill (a back at the best back, a lay at the best lay, only where the
    size shown covers the notional), the close at the opposite side's real price, the greened return a stake unit
    before commission and the CLV against the close mid. ``skip`` names why a signal did not trade."""
    o_price, o_size = (early["back1"], early["back1_size"]) if side > 0 else (early["lay1"], early["lay1_size"])
    if not (o_price > 1.0):
        return {"skip": "no early price on the side"}
    if not (o_size >= notional):
        return {"skip": "size short at the early book", "fill_open": float(o_price), "size_open": float(o_size)}
    if close is None:
        return {"skip": "no two-sided close", "fill_open": float(o_price)}
    c = close["lay1"] if side > 0 else close["back1"]
    c_size = close["lay1_size"] if side > 0 else close["back1_size"]
    mid_c = (close["back1"] + close["lay1"]) / 2.0
    mid_o = ((early["back1"] + early["lay1"]) / 2.0) if early["lay1"] > 1.0 else o_price
    profit = o_price / c - 1.0 if side > 0 else 1.0 - o_price / c
    clv = o_price / mid_c - 1.0 if side > 0 else 1.0 - o_price / mid_c
    return {"skip": None, "fill_open": float(o_price), "size_open": float(o_size), "fill_close": float(c),
            "close_mid": float(mid_c), "profit": float(profit), "clv": float(clv),
            "right_way": bool(mid_c < mid_o) if side > 0 else bool(mid_c > mid_o),
            "close_size_short": bool(not (c_size >= notional * o_price / c))}


# ---------------------------------------------------------------------------
# the window, the halves and the gate
# ---------------------------------------------------------------------------

def window(trades: pd.DataFrame, start: date, as_of: date, min_bets: int, min_days: int,
           cap_days: int = CAP_DAYS) -> dict:
    """The trial's window: from ``start`` to the later of the day of the ``min_bets``-th trade and ``start`` +
    ``min_days``, capped at ``cap_days``. 'running' while that end is after ``as_of``."""
    cap_end = start + timedelta(days=cap_days)
    if as_of < start:
        return {"start": start.isoformat(), "end": start.isoformat(), "status": "not started", "calendar_days": 0,
                "cap_end": cap_end.isoformat()}
    days = sorted(pd.to_datetime(trades["day"]).dt.date) if len(trades) else []
    nth = days[min_bets - 1] if len(days) >= min_bets else None
    if nth is not None:
        end = max(nth, start + timedelta(days=min_days))
        status = "complete" if end <= as_of else "running"
        if end > cap_end:
            end, status = cap_end, ("capped (underpowered)" if cap_end <= as_of else "running")
    elif as_of >= cap_end:
        end, status = cap_end, "capped (underpowered)"
    else:
        end, status = as_of, ("not started" if not days else "running")
    if status == "running":
        end = min(as_of, cap_end)
    return {"start": start.isoformat(), "end": end.isoformat(), "status": status,
            "calendar_days": (end - start).days, "cap_end": cap_end.isoformat()}


def scored(trades: pd.DataFrame, commission: float, n_boot: int) -> dict:
    if trades.empty:
        return {"bets": 0}
    s = summarise(trades, commission=commission, n_boot=n_boot)
    s["clv_mean"] = float(trades["clv"].mean())
    s["close_size_short"] = int(trades["close_size_short"].sum())
    return s


def by_key(trades: pd.DataFrame, key: str, commission: float) -> dict:
    return {str(k): scored(g, commission, 0) for k, g in trades.groupby(key, observed=True)} if len(trades) else {}


def halves(trades: pd.DataFrame, start: date, end: date, commission: float) -> dict:
    mid = start + (end - start) / 2
    d = pd.to_datetime(trades["day"]).dt.date
    return {"split": mid.isoformat(), "first": scored(trades[d < mid], commission, 0),
            "second": scored(trades[d >= mid], commission, 0)}


def _finite(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def gate(result: dict, gt: dict) -> dict:
    """The paper trial's gate: the window complete (``min_bets`` trades and ``min_days`` days), the 90% lower bound
    of the return at the taken prices above zero, the mean CLV above zero, both halves above zero. A window still
    running is reported as running, never as a pass or a fail."""
    s, w, h = result["headline"], result["window"], result["halves"]
    checks = {"bets": s.get("bets", 0) >= gt["min_bets"], "days": w["calendar_days"] >= gt["min_days"],
              "roi_lo90_above_0": _finite(s.get("roi_lo90")) and s["roi_lo90"] > gt.get("roi_lo90_above", 0),
              "clv_above_0": (s.get("clv_mean") or -1) > gt.get("clv_mean_above", 0),
              "both_halves_above_0": (h["first"].get("roi") or -1) > 0 and (h["second"].get("roi") or -1) > 0}
    if w["status"] in ("running", "not started"):
        return {"checks": checks, "verdict": "running", "passed": None}
    return {"checks": checks, "verdict": "pass" if all(checks.values()) else "fail", "passed": bool(all(checks.values()))}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """The inputs of a scoring run. ``dc_fn`` and ``read_day`` stand in for the Dixon-Coles fit and the recorder's
    files (the tests' seam); by default the real ones."""

    gold: pd.DataFrame
    panel: pd.DataFrame
    store: object
    read_day: ReadDay
    rule: dict
    start: date
    as_of: date
    notional: float = NOTIONAL_GBP
    tolerance_min: float = TOLERANCE_MIN
    n_jobs: int = 4
    n_boot: int = 2000
    aliases: dict | None = None
    dc_fn: Callable | None = None
    _fits: dict = field(default_factory=dict)

    def model_for(self, market_type: str, month: str, model_name: str):
        """The registered model fitted on the panel's months before ``month`` (cached a market and month)."""
        key = (market_type, month, model_name)
        if key not in self._fits:
            train = self.prep[(self.prep["market_type"] == market_type) & (self.prep["month"] < month)]
            if train["month"].nunique() < fx.MIN_TRAIN_MONTHS:
                self._fits[key] = (None, None)
            else:
                mod = [m for m in fx.exchange_models(self.n_jobs) if m.name == model_name][0].fit(train)
                self._fits[key] = (mod, str(train["month"].max()))
        return self._fits[key]

    def __post_init__(self):
        self.prep = fx.prepared(self.panel)


def signals(t: Trial, market_types: list[str], model_name: str, threshold: float, funnel: dict) -> pd.DataFrame:
    """Every forecast of the trial's matches, with its side under ``threshold`` (0: no signal)."""
    from football.data import _with_ids
    stamps = sorted({k.split("/")[3] for k in t.store.listing("football/raw/fixtures/") if k.endswith(".csv")})
    out = []
    coll = t.start
    if coll.weekday() not in COLLECTION_TIME:
        coll = next_collection(coll)
    while coll <= t.as_of:
        days = [d for d in cover_days(coll) if d <= t.as_of.isoformat()]
        stamp = fixtures_stamp(stamps, coll)
        f = read_fixtures_file(t.store, stamp) if stamp else None
        funnel["collections"] += 1
        if f is None:
            funnel["collections_without_fixtures_file"] += 1
            coll = next_collection(coll)
            continue
        f = f[f["Div"].isin(MAIN_DIVISIONS) & f["match_date"].isin(days)]
        funnel["fixture_rows"] += int(len(f))
        ids = _with_ids(f, t.gold) if len(f) else f.assign(match_id=pd.Series(dtype=str))
        funnel["fixture_rows_with_match_id"] += int(len(ids))
        if ids.empty:
            coll = next_collection(coll)
            continue
        rows_g = t.gold[t.gold["match_id"].isin(ids["match_id"])]
        dc = (t.dc_fn or dc_probs)(t.gold, rows_g, coll.isoformat())
        _, moment = collection_of(days[0])
        for mt in market_types:
            r = trial_rows(ids, t.gold, dc, mt)
            funnel[f"{mt}:markets_with_bfe_book"] += int(r["market_id"].nunique()) if len(r) else 0
            for month, part in (r.groupby("month") if len(r) else []):
                mod, fit_month = t.model_for(mt, month, model_name)
                if mod is None:
                    funnel[f"{mt}:markets_without_a_fit"] += int(part["market_id"].nunique())
                    continue
                part = part.copy()
                part["yhat"] = mod.predict(part)
                part["fit_month"] = fit_month
                part["collection"], part["moment_utc"], part["fixtures_stamp"] = coll.isoformat(), moment, stamp
                out.append(part)
        coll = next_collection(coll)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    d["side"] = np.where(d["yhat"] < -threshold, 1, np.where(d["yhat"] > threshold, -1, 0))
    d["match_day"] = pd.to_datetime(d["off_time_utc"], utc=True).dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    return d


def trade_signals(t: Trial, sig: pd.DataFrame, widths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The signals at the recorded books -> (trades, skipped), the skipped with their reason."""
    trades, skipped = [], []
    meta = t.gold.drop_duplicates("match_id").set_index("match_id")
    for day, part in sig.groupby("match_day"):
        dd = date.fromisoformat(day)
        markets = recorded_day(t.read_day, dd, "markets")
        books = recorded_day(t.read_day, dd, "books")
        fix = pd.DataFrame({"match_id": part["event_id"].unique()})
        fix["HomeTeam"] = fix["match_id"].map(meta["home_team"]).astype(str)
        fix["AwayTeam"] = fix["match_id"].map(meta["away_team"]).astype(str)
        fix["country"] = fix["match_id"].map(meta["country"]).astype(str)
        mp = map_markets(markets, fix, t.aliases) if markets is not None else map_markets(None, fix)
        bk = clean_books(books) if books is not None and len(books) else None
        for (mid, mt), sel in part.groupby(["event_id", "market_type"]):
            legs = mp[(mp["match_id"] == mid) & (mp["market_type"] == mt)]
            base = {"day": day, "match_id": mid, "market_type": mt, "fd_code": sel["fd_code"].iloc[0],
                    "tier": sel["tier"].iloc[0], "collection": sel["collection"].iloc[0],
                    "weekday": pd.Timestamp(day).day_name()}
            todo = sel[sel["side"] != 0]
            if todo.empty:
                continue
            if legs.empty or bk is None or set(legs["leg"]) != set(SELECTIONS[mt]):
                skipped += [{**base, "selection": s, "side": int(sd), "reason": "no recorded market"}
                            for s, sd in zip(todo["selection_id"], todo["side"])]
                continue
            bf_mid = legs["market_id"].iloc[0]
            leg_to_sel = dict(zip(legs["leg"], legs["selection_id"]))
            b = bk[bk["market_id"] == bf_mid]
            moment = pd.Timestamp(sel["moment_utc"].iloc[0])
            t_early, early = early_book(b, [leg_to_sel[x] for x in SELECTIONS[mt]], moment, t.tolerance_min)
            if early is None:
                skipped += [{**base, "selection": s, "side": int(sd), "reason": "no recorded book at the early moment"}
                            for s, sd in zip(todo["selection_id"], todo["side"])]
                continue
            over = float((1.0 / early["back1"]).sum())
            wl = width_label(over)
            close = close_book(b)
            for r in todo.itertuples(index=False):
                sid = leg_to_sel[r.selection_id]
                row = {**base, "selection": r.selection_id, "side": int(r.side), "yhat": float(r.yhat),
                       "market_id": bf_mid, "early_utc": t_early.isoformat(), "book_over": over, "book_width": wl,
                       "fd_book_width": r.book_width, "fit_month": r.fit_month}
                if wl not in widths:
                    skipped.append({**row, "reason": "recorded book wide"})
                    continue
                res = fill_trade(int(r.side), early.loc[sid], close.loc[sid] if sid in close.index else None,
                                 t.notional)
                skip = res.pop("skip")
                if skip is not None:
                    skipped.append({**row, **res, "reason": skip})
                    continue
                trades.append({**row, **res})
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["stake"] = 1.0
        tr["band"] = price_band(tr["fill_open"].to_numpy()).astype(str).to_numpy()
        tr["month"] = tr["day"].str[:7]
    return tr, pd.DataFrame(skipped)


def score_trial(t: Trial, prereg_sha: str, trades_out: str | None = None) -> dict:
    """The registered rule (and the secondary read) at the recorded books, with the window and the gate."""
    rule = t.rule
    res = {"prereg": PREREG, "prereg_sha256": prereg_sha, "start": t.start.isoformat(), "as_of": t.as_of.isoformat(),
           "notional_gbp": t.notional, "tolerance_min": t.tolerance_min, "rules": []}
    for i, r in enumerate([rule] + list(rule.get("secondary", []))):
        funnel = _Counter()
        sig = signals(t, r["market_types"], r["model"], r["threshold"], funnel)
        n_sig = int((sig["side"] != 0).sum()) if len(sig) else 0
        tr, sk = trade_signals(t, sig, r["book_width"]) if n_sig else (pd.DataFrame(), pd.DataFrame())
        gt = {**rule["gate"], **r.get("gate", {})}
        w = window(tr, t.start, t.as_of, gt["min_bets"], gt["min_days"])
        inw = tr[pd.to_datetime(tr["day"]).dt.date <= date.fromisoformat(w["end"])] if len(tr) else tr
        com = rule.get("commission", 0.02)
        out = {"rule": r, "role": "primary" if i == 0 else "secondary", "funnel": dict(funnel),
               "forecasts": int(len(sig)), "signals": n_sig, "window": w,
               "trades_after_window": int(len(tr) - len(inw)),
               "headline": scored(inw, com, t.n_boot),
               "skipped": sk["reason"].value_counts().to_dict() if len(sk) else {},
               "wide_share_of_signals_at_a_recorded_book": _wide_share(inw, sk),
               "fits": sorted({str(x) for x in sig.get("fit_month", pd.Series(dtype=str)).dropna()})}
        end = date.fromisoformat(w["end"])
        out["halves"] = halves(inw, t.start, end, com) if len(inw) else {"first": {"bets": 0}, "second": {"bets": 0}}
        for key in ("book_width", "band", "tier", "fd_code", "month", "weekday"):
            out[f"by_{key}"] = by_key(inw, key, com)
        if len(sk):
            out["skipped_by_weekday"] = sk.groupby(["reason", "weekday"]).size().rename("n").reset_index().to_dict(
                "records")
        out["gate"] = gate(out, gt)
        res["rules"].append(out)
        if i == 0:
            res["gate"] = out["gate"]
        if trades_out and (len(tr) or len(sk)):
            stem = Path(trades_out)
            path = stem if i == 0 else stem.with_name(f"{stem.stem}_secondary{i}{stem.suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([tr.assign(traded=True), sk.assign(traded=False)], ignore_index=True).to_csv(path, index=False)
    return res


def _wide_share(tr: pd.DataFrame, sk: pd.DataFrame) -> float | None:
    """Of the signals that met a recorded early book, the share whose book was wide (over 1.05)."""
    at_book = len(tr) + (int(sk["book_width"].notna().sum()) if len(sk) and "book_width" in sk else 0)
    wide = int((sk["reason"] == "recorded book wide").sum()) if len(sk) else 0
    return wide / at_book if at_book else None


class _Counter(dict):
    def __missing__(self, key):
        return 0


def _json(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (pd.Timestamp, datetime, date)):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, indent=1, default=default)


def main(argv=None) -> int:
    from football.data import load_gold
    from football.reference import load_aliases
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="a local sources folder in place of S3 (football-data's bronze, gold)")
    ap.add_argument("--live-root", default=None, help="a local betfair_live folder in place of the server's / S3")
    ap.add_argument("--panel", required=True, help="the exchange panel (football_exchange --build-panel)")
    ap.add_argument("--build-panel", action="store_true", help="build the panel first (from gold, ~3 minutes)")
    ap.add_argument("--dc-cache", default=None)
    ap.add_argument("--prereg", default=PREREG)
    ap.add_argument("--start", default=None, help="the trial's first collection (default: the first Friday on or "
                                                  "after the recorder's first football day)")
    ap.add_argument("--as-of", default=None, help="the last match day scored (default: yesterday, UK)")
    ap.add_argument("--notional", type=float, default=NOTIONAL_GBP)
    ap.add_argument("--tolerance-min", type=float, default=TOLERANCE_MIN)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trades-out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    rule, sha = fx.read_prereg(a.prereg)
    store = Store(root=a.root)
    read_day = recorder_reader(a.live_root)
    uk_today = datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/London")).date()
    as_of = date.fromisoformat(a.as_of) if a.as_of else uk_today - timedelta(days=1)
    if a.start:
        start = date.fromisoformat(a.start)
    else:
        first = first_recorded_day(read_day, date.fromisoformat(RECORDER_FROM), as_of)
        if first is None:
            print(_json({"status": "no football book record found", "searched": [RECORDER_FROM, as_of]}))
            return 0
        start = trial_start(first)
    if a.build_panel or not Path(a.panel).exists():
        Path(a.panel).parent.mkdir(parents=True, exist_ok=True)
        fx.build(a.root, a.panel, a.dc_cache, a.n_jobs)
    t = Trial(gold=load_gold(store), panel=pd.read_parquet(a.panel), store=store, read_day=read_day, rule=rule,
              start=start, as_of=as_of, notional=a.notional, tolerance_min=a.tolerance_min, n_jobs=a.n_jobs,
              n_boot=a.n_boot, aliases=load_aliases(extra=store.get_parquet("football/gold/team_aliases.parquet")))
    res = score_trial(t, sha, trades_out=a.trades_out)
    text = _json(res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
