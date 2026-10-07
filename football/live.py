"""The football model on Betfair: paper by default, real orders only with the owner's switch.

Real orders only when the repository variable ``FOOTBALL_LIVE=yes`` and ``--live`` are both given, and only on the UK
runner; otherwise every decision is the same and simulated against the live books (``trading.exchange.PaperExchange``).
Four steps:

``--fit`` (GitHub's runners, S3 only): the ratings of every country as of today (``football.ratings.fit_pool``) and
the stage-2 pooling weights, fitted on the walk-forward prices of the last seasons, to
``sources/football/live/model.json``.

``--trade`` (the UK runner): every few minutes, Betfair's soccer MATCH_ODDS and OVER_UNDER_25 markets of the next
hours in the countries in scope; each event's two sides matched to rated teams of one country (the stored aliases
first, then the names; both sides must resolve and be in the same division: league matches only, no cups, no women's,
youth or reserve football). Each selection is decided once, at its first real book (back and lay within ``tight``)
inside the decision window before kick-off: the model's chances are pooled with the book's own (the pooling weights
of ``--fit``), and a selection is backed when its expected return at the best back price, after commission, clears
the edge, at no more than the price cap; fill-or-kill, level stakes, capped a match and a day. The book is read until
the market turns in play, and the last pre-play book is kept as the close (the trader's own CLV benchmark). The S3
objects ``football/STOP`` and ``trading/STOP`` stop new bets within a minute.

``--settle`` (GitHub's runners, S3 only): each day's ledger against the gold results (football.data), P&L at the
price taken and CLV against the recorded close, to ``.../<mode>/<day>/settled.csv`` and the running
``.../<mode>/summary.json``.

    python -m football.live --fit
    python -m football.live --trade --until 21:30                         # paper
    FOOTBALL_LIVE=yes python -m football.live --trade --live --until 21:30
    python -m football.live --settle --days 5
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from football.model import _pool_1x2, _pool_bin
from football.ratings import Ratings, markets, score_matrix
from football.reference import BY_ID, COMPETITIONS, load_aliases, normalise, resolve

log = logging.getLogger(__name__)

PREFIX = "football/live"
MODEL_KEY = f"{PREFIX}/model.json"
CONFIG_PATH = Path(__file__).with_name("live_config.json")
LOCAL = Path(os.path.expanduser("~/.ashcroft/football"))
EXCLUDE = ("cup", "trophy", "friendl", "women", "u21", "u23", "u19", "u18", "reserve", "youth", "play-off",
           "playoff", "shield", "super cup", "supercopa", "pokal", "coppa", "copa", "coupe", "beker", "taca")
LEDGER_FIELDS = ["time_utc", "mode", "uk_day", "market_id", "market_type", "event", "competition", "kickoff_utc",
                 "minutes_to_off", "country", "home_team_id", "away_team_id", "selection_id", "outcome", "p_model",
                 "q_book", "p", "back", "back_size", "lay", "ev", "action", "stake", "matched", "avg_price", "bet_id",
                 "status", "error"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text())


def live_switch_on(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("FOOTBALL_LIVE", "")).strip().lower() == "yes"


# --------------------------------------------------------------------------------------------------------------------
# Fit (S3)
# --------------------------------------------------------------------------------------------------------------------

def fit_live(g: pd.DataFrame, today: date, params: dict | None = None, pooling_seasons: int = 3,
             workers: int = 4) -> dict:
    """The ratings of every pool as of ``today`` and the pooling weights fitted on the walk-forward prices of the
    last ``pooling_seasons`` seasons (the same stage 1 the report scores)."""
    from football.model import DEFAULTS, fit_pooling, market_probs, outcomes, walk_forward
    from football.ratings import fit_pool
    p = {**DEFAULTS, **(params or {})}
    as_of = (today + timedelta(days=1)).isoformat()
    ratings = {}
    for pool in sorted(g.country.unique()):
        r, _ = fit_pool(g[g.country == pool], pool, as_of, xi=p["xi"], l2=p["l2"], window_days=p["window_days"],
                        sot_mix=p["sot_mix"])
        if r is not None:
            ratings[pool] = r.to_dict()
    start = date(today.year - pooling_seasons, 7, 1).isoformat()
    wf = walk_forward(g, start, None, {**p, "refit_days": 14}, workers)
    wf = wf[wf.status == "FINISHED"]
    wf = pd.concat([wf, outcomes(wf), market_probs(wf)], axis=1)
    ok = wf[["p_h", "p_d", "p_a", "qopen_h", "qopen_d", "qopen_a"]].notna().all(axis=1)
    th1 = fit_pooling(wf.loc[ok, ["p_h", "p_d", "p_a"]].to_numpy(), wf.loc[ok, ["qopen_h", "qopen_d", "qopen_a"]]
                      .to_numpy(), wf.loc[ok, ["y_h", "y_d", "y_a"]].to_numpy(), "1x2")
    ok2 = wf[["p_o25", "qopen_o25"]].notna().all(axis=1)
    th2 = fit_pooling(wf.loc[ok2, "p_o25"].to_numpy(), wf.loc[ok2, "qopen_o25"].to_numpy(),
                      wf.loc[ok2, "y_o25"].to_numpy(), "bin")
    return {"fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "as_of": as_of, "params": p,
            "ratings": ratings, "pooling": {"1x2": [float(x) for x in th1], "ou25": [float(x) for x in th2],
                                            "fit_rows_1x2": int(ok.sum()), "fit_rows_ou25": int(ok2.sum()),
                                            "from": start}}


# --------------------------------------------------------------------------------------------------------------------
# Pricing a Betfair event
# --------------------------------------------------------------------------------------------------------------------

POOLS_BY_BF = {}
for _c in COMPETITIONS:
    POOLS_BY_BF.setdefault(_c.betfair_country, [])
    if _c.country not in POOLS_BY_BF[_c.betfair_country]:
        POOLS_BY_BF[_c.betfair_country].append(_c.country)


class Pricer:
    def __init__(self, model: dict, aliases=None, min_matches: int = 10):
        self.model = model
        self.ratings = {k: Ratings.from_dict(v) for k, v in model["ratings"].items()}
        self.aliases = aliases or {}
        self.min_matches = min_matches
        self._cache: dict = {}

    def match(self, event_name: str, bf_country: str, competition: str = "") -> dict | None:
        """A Betfair event ('Man Utd v Fulham') -> the pool, the two team ids and the division, or None (not a
        league match of two rated sides of one country)."""
        key = (event_name, bf_country, competition)
        if key in self._cache:
            return self._cache[key]
        out = None
        comp = (competition or "").lower()
        parts = [x.strip() for x in str(event_name).split(" v ")]
        if len(parts) == 2 and not any(w in comp or w in str(event_name).lower() for w in EXCLUDE):
            for pool in POOLS_BY_BF.get(bf_country, []):
                r = self.ratings.get(pool)
                if r is None:
                    continue
                cand = {t: v["name"] for t, v in r.teams.items() if v.get("n", 0) >= self.min_matches}
                h, sh, _ = resolve(parts[0], pool, cand, "betfair", self.aliases)
                a, sa, _ = resolve(parts[1], pool, cand, "betfair", self.aliases)
                if h and a and h != a and r.teams[h]["div"] == r.teams[a]["div"]:
                    out = {"pool": pool, "home": h, "away": a, "division": r.teams[h]["div"],
                           "score": min(sh, sa)}
                    break
        self._cache[key] = out
        return out

    def probs(self, m: dict) -> dict:
        r = self.ratings[m["pool"]]
        lh, la = r.rates(m["home"], m["away"], m["division"])
        p = markets(score_matrix(lh, la, r.rho))
        p["p_u25"] = 1 - p["p_o25"]
        return p


def book_probs(book, ids: list[int]) -> list[float] | None:
    """The book's own chances: 1 / the mid of the best back and lay, normalised; None without both sides."""
    mids = []
    for sid in ids:
        q = book.runners.get(int(sid))
        if q is None or not q.back or not q.lay:
            return None
        mids.append((q.back[0][0] + q.lay[0][0]) / 2)
    inv = np.array([1 / m for m in mids])
    return list(inv / inv.sum())


def pooled(model: dict, kind: str, pm: list[float], qb: list[float]) -> list[float]:
    th = np.array(model["pooling"]["1x2" if kind == "MATCH_ODDS" else "ou25"])
    pm_, qb_ = np.clip(np.array([pm]), 1e-6, 1 - 1e-6), np.clip(np.array([qb]), 1e-6, 1 - 1e-6)
    if kind == "MATCH_ODDS":
        return list(_pool_1x2(th, pm_, qb_)[0])
    p = float(_pool_bin(th, pm_[:, 0], qb_[:, 0])[0])
    return [p, 1 - p]


def selections(market: dict, m: dict) -> list[tuple[int, str]]:
    """(selection id, outcome) in the order H, D, A for match odds (by Betfair's sort priority and the draw's name),
    O, U for over/under 2.5."""
    runners = sorted(market.get("runners", []), key=lambda r: r.get("sortPriority", 9))
    if market["marketType"] == "MATCH_ODDS":
        draw = [r for r in runners if str(r.get("runnerName", "")).lower() == "the draw"]
        sides = [r for r in runners if r not in draw]
        if len(draw) != 1 or len(sides) != 2:
            return []
        return [(int(sides[0]["selectionId"]), "H"), (int(draw[0]["selectionId"]), "D"),
                (int(sides[1]["selectionId"]), "A")]
    over = [r for r in runners if str(r.get("runnerName", "")).lower().startswith("over")]
    under = [r for r in runners if str(r.get("runnerName", "")).lower().startswith("under")]
    if len(over) != 1 or len(under) != 1:
        return []
    return [(int(over[0]["selectionId"]), "O"), (int(under[0]["selectionId"]), "U")]


def decide(cfg: dict, p: float, back: float | None, back_size: float, lay: float | None) -> tuple[str, float]:
    """('back', expected return) or ('skip:<why>', expected return) for one selection at its book."""
    com = cfg["commission"]
    if back is None or lay is None:
        return "skip:no_book", float("nan")
    ev = p * (back - 1) * (1 - com) - (1 - p)
    lim = cfg["limits"]
    if lay / back > cfg["tight"]:
        return "skip:wide", ev
    if back < lim["min_price"] or back > lim["max_price"]:
        return "skip:price", ev
    if ev <= cfg["edge"]:
        return "skip:edge", ev
    if back_size < lim["level_stake"]:
        return "skip:size", ev
    return "back", ev


# --------------------------------------------------------------------------------------------------------------------
# Trade (the UK runner)
# --------------------------------------------------------------------------------------------------------------------

def _uk_now() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Europe/London"))


def catalogue(data, countries: list[str], hours: float, types=("MATCH_ODDS", "OVER_UNDER_25")) -> list[dict]:
    now = datetime.now(timezone.utc)
    out = []
    for mt in types:
        for i in range(0, len(countries), 5):
            raw = data._read("listMarketCatalogue", {
                "filter": {"eventTypeIds": ["1"], "marketCountries": countries[i:i + 5], "marketTypeCodes": [mt],
                           "inPlayOnly": False,
                           "marketStartTime": {"from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                               "to": (now + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")}},
                "marketProjection": ["EVENT", "COMPETITION", "MARKET_START_TIME", "RUNNER_DESCRIPTION"],
                "maxResults": 1000, "sort": "FIRST_TO_START"}) or []
            data._keep("catalogue", raw)
            for m in raw:
                m["marketType"] = mt
            out += raw
    return out


class Trader:
    def __init__(self, exchange, pricer: Pricer, cfg: dict, mode: str, store=None, stop=None):
        self.ex, self.pricer, self.cfg, self.mode = exchange, pricer, cfg, mode
        self.store, self.stop = store, stop or (lambda: False)
        self.decided: set = set()
        self.markets: dict[str, dict] = {}         # market_id -> catalogue entry with the match
        self.closes: dict[tuple, dict] = {}        # (market_id, selection_id) -> the last pre-play book
        self.rows: list[dict] = []
        self.turnover = 0.0
        self.match_stake: dict[str, float] = {}
        self.bets = 0

    def refresh(self, data) -> None:
        for m in catalogue(data, self.cfg["countries"], self.cfg["hours_ahead"], tuple(self.cfg["markets"])):
            if m["marketId"] in self.markets:
                continue
            ev, comp = m.get("event") or {}, (m.get("competition") or {}).get("name", "")
            got = self.pricer.match(ev.get("name", ""), ev.get("countryCode", ""), comp)
            if got is None:
                continue
            sel = selections(m, got)
            if not sel:
                continue
            self.markets[m["marketId"]] = {**m, "match": got, "sel": sel, "competition_name": comp,
                                           "start": datetime.strptime(m["marketStartTime"][:19],
                                                                      "%Y-%m-%dT%H:%M:%S").replace(
                                               tzinfo=timezone.utc)}

    def step(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        live = {k: v for k, v in self.markets.items() if v["start"] > now - timedelta(minutes=5)}
        if not live:
            return
        books = self.ex.books(list(live))
        for mid, mk in live.items():
            book = books.get(mid)
            if book is None:
                continue
            ids = [s for s, _ in mk["sel"]]
            if not book.inplay and book.status == "OPEN":     # the close: the last book before the turn in play
                for sid in ids:
                    q = book.runners.get(sid)
                    if q is not None:
                        self.closes[(mid, sid)] = {"back": q.back[0][0] if q.back else None,
                                                   "lay": q.lay[0][0] if q.lay else None,
                                                   "ltp": q.last_traded, "time_utc": now.isoformat(timespec="seconds")}
            mins = (mk["start"] - now).total_seconds() / 60
            if book.inplay or book.status != "OPEN" or not (self.cfg["stop_before_off"] <= mins <=
                                                             self.cfg["decide_from_minutes"]):
                continue
            if all((mid, s) in self.decided for s in ids):
                continue
            qb = book_probs(book, ids)
            if qb is None:
                continue
            pm = self.pricer.probs(mk["match"])
            kind = mk["marketType"]
            pmv = [pm["p_h"], pm["p_d"], pm["p_a"]] if kind == "MATCH_ODDS" else [pm["p_o25"], pm["p_u25"]]
            pp = pooled(self.pricer.model, kind, pmv, qb) if self.cfg["prob"] == "pooled" else pmv
            for (sid, oc), p, p0, q0 in zip(mk["sel"], pp, pmv, qb):
                if (mid, sid) in self.decided:
                    continue
                q = book.runners.get(sid)
                back = q.back[0] if q and q.back else (None, 0.0)
                lay = q.lay[0][0] if q and q.lay else None
                action, ev = decide(self.cfg, p, back[0], back[1], lay)
                if action.startswith("skip:wide"):
                    continue                          # not a real book yet: decide at a later one
                self.decided.add((mid, sid))
                row = {"time_utc": now.isoformat(timespec="seconds"), "mode": self.mode,
                       "uk_day": _uk_now().date().isoformat(), "market_id": mid, "market_type": kind,
                       "event": (mk.get("event") or {}).get("name"), "competition": mk["competition_name"],
                       "kickoff_utc": mk["start"].isoformat(), "minutes_to_off": round(mins, 1),
                       "country": mk["match"]["pool"], "home_team_id": mk["match"]["home"],
                       "away_team_id": mk["match"]["away"], "selection_id": sid, "outcome": oc,
                       "p_model": round(p0, 4), "q_book": round(q0, 4), "p": round(p, 4), "back": back[0],
                       "back_size": back[1], "lay": lay, "ev": round(ev, 4) if ev == ev else None, "action": action}
                if action == "back":
                    row.update(self._back(mid, sid, back[0], mk))
                if action == "back" or self.cfg.get("ledger_all", True):
                    self.rows.append(row)

    def _back(self, mid: str, sid: int, price: float, mk: dict) -> dict:
        lim = self.cfg["limits"]
        stake = lim["level_stake"]
        key = mk["event"]["id"] if mk.get("event") else mid
        if self.stop():
            return {"action": "skip:stop"}
        if lim.get("max_daily_turnover") is not None and self.turnover + stake > lim["max_daily_turnover"]:
            return {"action": "skip:daily_cap"}
        if self.match_stake.get(key, 0.0) + stake > lim["max_match_stake"]:
            return {"action": "skip:match_cap"}
        if lim.get("max_bets_per_day") is not None and self.bets >= lim["max_bets_per_day"]:
            return {"action": "skip:bet_cap"}
        fill = self.ex.back(mid, sid, price, stake, ref=f"fb-{mid}-{sid}"[:32], min_fill=stake)
        out = {"stake": stake, "matched": fill.matched, "avg_price": fill.avg_price, "bet_id": fill.bet_id,
               "status": fill.status, "error": fill.error}
        if fill.status == "SUCCESS" and fill.matched:
            self.turnover += fill.matched
            self.match_stake[key] = self.match_stake.get(key, 0.0) + fill.matched
            self.bets += 1
        return out

    def close_rows(self) -> list[dict]:
        out = []
        for (mid, sid), c in self.closes.items():
            mk = self.markets.get(mid, {})
            oc = dict(mk.get("sel", [])).get(sid)
            out.append({"market_id": mid, "selection_id": sid, "outcome": oc, "market_type": mk.get("marketType"),
                        "home_team_id": (mk.get("match") or {}).get("home"),
                        "away_team_id": (mk.get("match") or {}).get("away"),
                        "kickoff_utc": mk["start"].isoformat() if mk.get("start") else None, **c})
        return out


def _csv(rows: list[dict], fields: list[str] | None = None) -> bytes:
    buf = io.StringIO()
    fields = fields or (list(rows[0]) if rows else [])
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()


def save_day(trader: Trader, store, day: str) -> None:
    """The ledger and the closes, locally (the run's artifact) and to S3."""
    base = f"{PREFIX}/{trader.mode}/{day}"
    files = {"ledger.csv": _csv(trader.rows, LEDGER_FIELDS), "closes.csv": _csv(trader.close_rows())}
    for name, data in files.items():
        p = LOCAL / trader.mode / day / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        if store is not None:
            try:
                store.put(f"{base}/{name}", data)
            except Exception as exc:                      # the local copy is the artifact; S3 retries next save
                log.warning("S3: %s not saved (%s)", name, exc)


def kill_switch(store):
    from greyhound.live import _exists
    state = {"at": None, "on": False}

    def check() -> bool:
        now = datetime.now(timezone.utc)
        if state["at"] and (now - state["at"]).total_seconds() < 60:
            return state["on"]
        state["at"] = now
        state["on"] = any(_exists(store, k) for k in ("football/STOP", "trading/STOP"))
        return state["on"]
    return check


def trade(until: str, live: bool, store, cfg: dict | None = None) -> dict:
    from trading.exchange import BetfairData, LiveExchange, PaperExchange
    cfg = cfg or load_config()
    raw = store.get(MODEL_KEY)
    if not raw:
        raise SystemExit("no football model in S3: run python -m football.live --fit first")
    model = json.loads(raw)
    aliases = load_aliases(extra=store.get_parquet("football/gold/team_aliases.parquet"))
    data = BetfairData(source="football")
    mode = "live" if live and live_switch_on() else "paper"
    if live and mode != "live":
        log.warning("--live without FOOTBALL_LIVE=yes: paper")
    ex = LiveExchange(data) if mode == "live" else PaperExchange(data)
    ex.login()
    trader = Trader(ex, Pricer(model, aliases, cfg["limits"].get("min_team_matches", 10)), cfg, mode, store,
                    kill_switch(store))
    day = _uk_now().date().isoformat()
    end_h, end_m = (int(x) for x in until.split(":"))
    last_cat, last_save = 0.0, 0.0
    log.info("football %s trading to %s UK; model fitted %s", mode, until, model.get("fitted_utc"))
    while True:
        uk = _uk_now()
        if (uk.hour, uk.minute) >= (end_h, end_m):
            break
        try:
            if time.monotonic() - last_cat > cfg["catalogue_seconds"]:
                trader.refresh(data)
                last_cat = time.monotonic()
                log.info("%d markets tracked", len(trader.markets))
            trader.step()
        except Exception as exc:                          # a bad read must not end the day
            log.warning("step failed: %s", str(exc)[:300])
        if time.monotonic() - last_save > 600:
            save_day(trader, store, day)
            last_save = time.monotonic()
        time.sleep(cfg["poll_seconds"])
    save_day(trader, store, day)
    out = {"mode": mode, "day": day, "markets": len(trader.markets), "decisions": len(trader.rows),
           "bets": trader.bets, "turnover": round(trader.turnover, 2)}
    log.info("football: %s", out)
    return out


# --------------------------------------------------------------------------------------------------------------------
# Settle (S3)
# --------------------------------------------------------------------------------------------------------------------

def settle_rows(ledger: pd.DataFrame, closes: pd.DataFrame, gold: pd.DataFrame, com: float) -> pd.DataFrame:
    """Each matched back against the gold result (same UK day of kick-off, both teams) and its recorded close."""
    b = ledger[(ledger.action == "back") & (ledger.status == "SUCCESS")].copy()
    if b.empty:
        return b
    b["match_date"] = pd.to_datetime(b.kickoff_utc, utc=True).dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")
    res = gold[gold.status == "FINISHED"][["match_date", "home_team_id", "away_team_id", "ft_home", "ft_away",
                                           "match_id"]]
    b = b.merge(res, on=["match_date", "home_team_id", "away_team_id"], how="left")
    d = b.ft_home - b.ft_away
    tot = b.ft_home + b.ft_away
    won = np.select([b.outcome == "H", b.outcome == "D", b.outcome == "A", b.outcome == "O", b.outcome == "U"],
                    [d > 0, d == 0, d < 0, tot >= 3, tot < 3], default=False)
    b["settled"] = b.ft_home.notna()
    b["won"] = np.where(b.settled, won.astype(float), np.nan)
    price = b.avg_price.astype(float).fillna(b.back.astype(float))
    b["pnl"] = np.where(b.settled, np.where(b.won == 1, b.matched * (price - 1) * (1 - com), -b.matched), np.nan)
    if closes is not None and len(closes):
        c = closes.copy()
        c["mid"] = (c.back.astype(float) + c.lay.astype(float)) / 2
        c["q_close"] = 1 / c.mid
        c["q_close"] = c.q_close / c.groupby("market_id").q_close.transform("sum")
        b = b.merge(c[["market_id", "selection_id", "q_close"]], on=["market_id", "selection_id"], how="left")
        b["clv"] = price * b.q_close - 1
    else:
        b["q_close"], b["clv"] = np.nan, np.nan
    return b


def settle(store, days: int, mode: str, cfg: dict | None = None, today: date | None = None) -> dict:
    from football.data import load_gold
    cfg = cfg or load_config()
    gold = load_gold(store)
    today = today or date.today()
    for i in range(days, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        base = f"{PREFIX}/{mode}/{day}"
        raw = store.get(f"{base}/ledger.csv")
        if not raw:
            continue
        ledger = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
        cl = store.get(f"{base}/closes.csv")
        closes = pd.read_csv(io.BytesIO(cl), dtype={"market_id": str}) if cl and cl.strip() else None
        s = settle_rows(ledger, closes, gold, cfg["commission"])
        store.put(f"{base}/settled.csv", s.to_csv(index=False).encode())
    # the running summary over every settled day
    parts = []
    for k in sorted(store.listing(f"{PREFIX}/{mode}/")):
        if k.endswith("/settled.csv"):
            raw = store.get(k)
            if raw and raw.strip():
                try:
                    parts.append(pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}))
                except pd.errors.EmptyDataError:
                    continue
    allb = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    done = allb[allb.settled == True] if len(allb) else allb  # noqa: E712
    summ = {"mode": mode, "bets": int(len(allb)), "settled": int(len(done)),
            "staked": float(done.matched.sum()) if len(done) else 0.0,
            "pnl": float(done.pnl.sum()) if len(done) else 0.0,
            "roi": float(done.pnl.sum() / done.matched.sum()) if len(done) and done.matched.sum() else None,
            "clv": float(done.clv.mean()) if len(done) and done.clv.notna().any() else None,
            "by_market": {k: {"n": int(len(g)), "pnl": float(g.pnl.sum()),
                              "clv": float(g.clv.mean()) if g.clv.notna().any() else None}
                          for k, g in done.groupby("market_type")} if len(done) else {},
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    store.put(f"{PREFIX}/{mode}/summary.json", json.dumps(summ, indent=1).encode())
    return summ


def main(argv=None) -> int:
    from football.data import load_gold
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--trade", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--live", action="store_true", help="real orders (with FOOTBALL_LIVE=yes)")
    ap.add_argument("--until", default="21:30", help="last UK time to trade, HH:MM")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--mode", default="paper", help="settle: paper or live")
    ap.add_argument("--root", default=None, help="a local sources folder in place of S3")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    store = Store(root=a.root)
    out = {}
    if a.fit:
        cfg = load_config()
        m = fit_live(load_gold(store), date.today(), cfg.get("model_params"))
        store.put(MODEL_KEY, json.dumps(m).encode())
        out["fit"] = {"as_of": m["as_of"], "pools": len(m["ratings"]), "pooling": m["pooling"],
                      "teams": sum(len(r["teams"]) for r in m["ratings"].values())}
    if a.trade:
        out["trade"] = trade(a.until, a.live, store)
    if a.settle:
        out["settle"] = {mode: settle(store, a.days, mode) for mode in ("paper", "live")}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
