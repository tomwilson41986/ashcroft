"""The morning models, live (the owner, 4 Nov 2026: "Create two live models; Steamers, Quick Returners. Set these up
and run to both win and place", then, asked: the place leg at the place SP has no edge in the research, so WIN only;
"Live now, half stakes"; and "leave the live trader horses out"). Spec: docs/morning_models_live.md.

Two models, each its own ledger, its own references and its own day limit:
  steamer       the CLEAN steamer model (frozen 4 Nov 2026, s3://$BUCKET/paper/morning_edges/model/): its predicted
                log(morning price / BSP) at or above its threshold (the training set's top 5%), 5+ runners
  quick_return  back within 1-4 days, flat / AW / hurdle / chase, 5+ runners
Selections come from the 06:00 card and earlier days only (morning/select.py), written before the trade by the
selection job (s3://$BUCKET/morning/selections/<day>.csv).

The live trader comes first:
  * NO ORDER BEFORE 10:00 UK (ENTRY_UK): the session does nothing at an earlier minute.
  * A horse the live trader has backed today (its ledger, s3://$BUCKET/trading/live/<day>/ledger.csv, read before
    every round of orders) is left alone: `skipped_live`.
  * Every order carries its own reference, "mes<mmdd><selection>" (steamer) or "meq<mmdd><selection>" (quick
    return), never the live trader's "ash<mmdd>": the live trader takes up only its own orders from Betfair's record
    (trading/session.py sync), so it never adopts, lays or settles these.

Staking (half the research's): to win GBP50 (stake = 50 / (price - 1)), GBP100 a bet at most, GBP1 at least (a
smaller stake is not placed), GBP2,000 staked a day per model. One order a selection, at 10:00 UK: a limit back,
FILL_OR_KILL, at the Betfair tick at or above 95% of the best back price, so it takes the best price and walks down
the ladder to 95% of it, and whatever is not matched at once is cancelled (nothing rests in the book). A selection
with no book at 10:00 is tried each minute to 10:15, then dropped. A race off within 2 minutes is not bet.

Real orders only with --live AND the repository variable MORNING_LIVE=yes; otherwise the same session runs on paper
against the live books (PaperExchange). The S3 object morning/STOP (or trading/STOP) stops new orders within a minute.

    python -m morning.live --select                  # the day's selections, from the 06:00 card (GitHub's runner)
    MORNING_LIVE=yes python -m morning.live --trade --live     # 10:00 UK (the morning runner, the live key)
    python -m morning.live --settle [--live]         # the evening: Betfair's record of the settled bets
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time as _time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "strat"))

log = logging.getLogger("morning.live")
UK = ZoneInfo("Europe/London")

ENTRY_UK = dtime(10, 0)          # no order before this, UK clock (the live trader's morning is its own)
RETRY_UNTIL_UK = dtime(10, 15)   # a selection with no book at 10:00 is tried each minute to here, then dropped
EARLIEST_START_UK = dtime(7, 30)  # a job started earlier than this waits for nothing: it is the wrong firing
TO_WIN = 50.0                    # half the research's GBP100 (the owner, 4 Nov 2026: "half stakes")
MAX_STAKE = 100.0
MIN_STAKE = 1.0                  # Betfair's smallest stake: a to-win stake under it is not placed
DAY_LIMIT = 2000.0               # staked a day, per model
WALK = 0.95                      # the limit: the tick at or above 95% of the best back price
MIN_AHEAD = timedelta(minutes=2)
COMMISSION = (0.02, 0.05)        # the desk's rate and the research's
MODELS = {"steamer": "mes", "quick_return": "meq"}   # model -> order reference prefix (never "ash")
ORDER = ("steamer", "quick_return")                  # a horse both select: the steamer's order first
PREFIX = "morning"
LIVE_TRADER_PREFIX = "ash"
LEDGER_COLS = ["ts", "mode", "model", "event", "market_id", "venue", "off", "selection_id", "runner", "pred",
               "best_back", "limit", "asked", "matched", "avg_price", "bet_id", "status", "error", "result", "pnl",
               "bsp", "clv"]                   # bsp, clv: written at settle (clv = average price / BSP - 1)


def bucket() -> str:
    return os.environ.get("ULTRA_BETTING_S3_BUCKET") or os.environ.get("PRED_BUCKET") or "ashcroft"


def _s3():
    import boto3
    return boto3.client("s3")


def s3_get(key: str) -> bytes | None:
    try:
        return _s3().get_object(Bucket=bucket(), Key=key)["Body"].read()
    except Exception:
        return None


def s3_put(key: str, body: bytes) -> None:
    _s3().put_object(Bucket=bucket(), Key=key, Body=body)


def uk_at(day: date, t: dtime) -> datetime:
    return datetime.combine(day, t, tzinfo=UK).astimezone(timezone.utc)


def uk_today(now: datetime | None = None) -> date:
    return (now or datetime.now(timezone.utc)).astimezone(UK).date()


def live_switch_on(env=None) -> bool:
    """The owner's switch for real orders: MORNING_LIVE=yes (a repository variable in the workflow)."""
    env = os.environ if env is None else env
    return str(env.get("MORNING_LIVE", "")).strip().lower() == "yes"


def ref(model: str, day: date, selection_id: int) -> str:
    return f"{MODELS[model]}{day:%m%d}{int(selection_id)}"[:32]


def ours(order: dict, model: str, day: date) -> bool:
    return str(order.get("customerOrderRef") or "").startswith(f"{MODELS[model]}{day:%m%d}")


def to_win_stake(price: float) -> float:
    """GBP50 to win at this price, at most GBP100; floored to the penny."""
    import math
    return math.floor(min(TO_WIN / (float(price) - 1.0), MAX_STAKE) * 100) / 100


def limit_price(best: float) -> float:
    from trading.exchange import tick_up
    return min(float(best), tick_up(float(best) * WALK))


def live_trader_backed(raw: bytes | None) -> set:
    """(market_id, selection_id) the live trader has matched a back on today, from its ledger."""
    if not raw:
        return set()
    try:
        led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}, low_memory=False)
    except Exception:
        return set()
    if not {"event", "market_id", "selection_id", "matched"} <= set(led.columns):
        return set()
    b = led[(led["event"] == "back") & (pd.to_numeric(led["matched"], errors="coerce") > 0)]
    return {(str(m), int(s)) for m, s in zip(b["market_id"], b["selection_id"]) if pd.notna(s)}


class MorningSession:
    """One morning: the day's selections, one order each at 10:00 UK, the ledger. ``x`` is an exchange
    (trading.exchange.LiveExchange or PaperExchange, or a stand-in); ``live_ledger`` returns the live trader's ledger
    bytes; ``stopped`` says whether the STOP object is there; ``on_order`` writes the ledger after every order."""

    def __init__(self, x, day: date, selections: pd.DataFrame, markets: list, mode: str,
                 live_ledger=lambda: None, stopped=lambda: False, on_order=lambda rows: None):
        self.x, self.day, self.mode = x, day, mode
        self.markets = {m.market_id: m for m in markets}
        self.sel = selections
        self.live_ledger, self.stopped, self.on_order = live_ledger, stopped, on_order
        self.entry, self.retry_end = uk_at(day, ENTRY_UK), uk_at(day, RETRY_UNTIL_UK)
        self.ledger: list[dict] = []
        self.done: set = set()
        self.staked = {m: 0.0 for m in MODELS}

    def _row(self, t, r, event, **kw) -> dict:
        m = self.markets.get(str(r.market_id))
        off = m.start.astimezone(UK).strftime("%H:%M") if m else ""
        return {"ts": t.isoformat(), "mode": self.mode, "model": r.model, "event": event,
                "market_id": str(r.market_id), "venue": m.venue if m else "", "off": off,
                "selection_id": int(r.selection_id), "runner": getattr(r, "horse_name", ""),
                "pred": getattr(r, "pred", None), **kw}

    def finished(self, t: datetime) -> bool:
        return t > self.retry_end or len(self.done) >= len(self.sel)

    def step(self, t: datetime) -> None:
        """One minute. Nothing at all before 10:00 UK or after 10:15 UK."""
        if t < self.entry or t > self.retry_end:
            return
        todo = [r for r in self.sel.itertuples(index=False) if (r.model, str(r.market_id), int(r.selection_id))
                not in self.done]
        if not todo:
            return
        if self.stopped():
            for r in todo:
                self.done.add((r.model, str(r.market_id), int(r.selection_id)))
                self.ledger.append(self._row(t, r, "stopped"))
            self.on_order(self.ledger)
            return
        live = live_trader_backed(self.live_ledger())
        books = self.x.books(sorted({str(r.market_id) for r in todo}))
        todo.sort(key=lambda r: ORDER.index(r.model))
        for r in todo:
            mid, sid, key = str(r.market_id), int(r.selection_id), (r.model, str(r.market_id), int(r.selection_id))
            m = self.markets.get(mid)
            if m is None or m.start - t <= MIN_AHEAD:
                self.done.add(key)
                self.ledger.append(self._row(t, r, "too_late"))
                continue
            if (mid, sid) in live:
                self.done.add(key)
                self.ledger.append(self._row(t, r, "skipped_live"))
                continue
            b = books.get(mid)
            q = b.runners.get(sid) if b is not None else None
            if b is None or q is None or q.status != "ACTIVE" or not q.back or b.status != "OPEN" or b.inplay:
                if t >= self.retry_end:
                    self.done.add(key)
                    self.ledger.append(self._row(t, r, "no_book"))
                continue                                     # tried again next minute until 10:15
            self.done.add(key)
            best = float(q.back[0][0])
            stake = to_win_stake(best)
            lim = limit_price(best)
            room = DAY_LIMIT - self.staked[r.model]
            if stake < MIN_STAKE:
                self.ledger.append(self._row(t, r, "below_min", best_back=best, asked=stake))
                continue
            if stake > room:
                self.ledger.append(self._row(t, r, "day_limit", best_back=best, asked=stake))
                continue
            f = self.x.back(mid, sid, lim, stake, ref(r.model, self.day, sid), min_fill=MIN_STAKE)
            ok = f.status == "SUCCESS" and f.matched > 0
            self.staked[r.model] += float(f.matched) if ok else 0.0
            self.ledger.append(self._row(t, r, "back" if ok else "not_filled", best_back=best, limit=lim,
                                         asked=stake, matched=float(f.matched) if ok else 0.0,
                                         avg_price=float(f.avg_price) if ok else None, bet_id=f.bet_id,
                                         status=f.status, error=f.error))
            self.on_order(self.ledger)


def run(session: MorningSession, now=lambda: datetime.now(timezone.utc), sleep=_time.sleep) -> list[dict]:
    """Wait for 10:00 UK, then a round each minute until every selection is done or 10:15 UK has passed."""
    while not session.finished(now()):
        t = now()
        if t < session.entry:
            sleep(min(60.0, max(1.0, (session.entry - t).total_seconds())))
            continue
        session.step(t)
        if not session.finished(now()):
            sleep(60.0)
    return session.ledger


# ------------------------------------------------------------------------------------------------- selection job

def select_day(day: date) -> pd.DataFrame:
    """The day's selections from the 06:00 card and the HRB database (horse_racing.db in the working directory)."""
    import lightgbm as lgb
    from morning import select as S
    from research.paper import morning_features as MF
    raw, meta = s3_get("paper/morning_edges/model/booster.txt"), s3_get("paper/morning_edges/model/meta.json")
    if raw is None or meta is None:
        raise SystemExit("the frozen steamer model is not in S3 (paper/morning_edges/model/)")
    booster, thr = lgb.Booster(model_str=raw.decode()), float(json.loads(meta)["threshold"])

    class Getter:
        def __call__(self, k):
            return s3_get(k)

        def ls(self, p):
            return [o["Key"] for o in _s3().list_objects_v2(Bucket=bucket(), Prefix=p).get("Contents", [])]
    card = S.read_card(Getter(), day.isoformat())
    if card is None:
        raise SystemExit(f"no 06:00 card for {day} in S3 (racecards/{day}_*.csv)")
    sel = S.select(S.history_with_card(MF.L.load(), card, day.isoformat()), day.isoformat(), booster, thr)
    log.info("%s: %d selections (%s), threshold %.4f", day, len(sel), sel.groupby("model").size().to_dict(), thr)
    return sel


# ------------------------------------------------------------------------------------------------- trade, settle

def _ledger_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=LEDGER_COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()


def _key(mode: str, day: date, name: str) -> str:
    return f"{PREFIX}/{mode}/{day:%Y-%m-%d}/{name}"


def trade(day: date, live: bool, wait_for_selections: bool = True) -> list[dict]:
    from trading.exchange import BetfairData, LiveExchange, PaperExchange
    from trading.matching import attach_ids
    mode = "live" if live else "paper"
    now = datetime.now(timezone.utc)
    if now > uk_at(day, RETRY_UNTIL_UK):
        log.warning("Started after 10:15 UK: nothing to do today")
        return []
    if now < uk_at(day, EARLIEST_START_UK):
        log.warning("Started before 07:30 UK: not this firing")
        return []
    raw = s3_get(f"{PREFIX}/selections/{day:%Y-%m-%d}.csv")
    while raw is None and wait_for_selections and datetime.now(timezone.utc) < uk_at(day, dtime(10, 10)):
        log.warning("The day's selections are not in S3 yet; waiting")
        _time.sleep(120)
        raw = s3_get(f"{PREFIX}/selections/{day:%Y-%m-%d}.csv")
    if raw is None:
        log.error("No selections for %s: nothing traded", day)
        return []
    sel = pd.read_csv(io.BytesIO(raw), dtype={"race_time": str})
    data = BetfairData()
    x = LiveExchange(data) if live else PaperExchange(data, bank=2 * DAY_LIMIT)
    x.login()
    start = datetime.combine(day, dtime(0, 0), tzinfo=UK).astimezone(timezone.utc)
    markets = x.markets(start, start + timedelta(days=1))
    sel = attach_ids(sel, markets)
    unmatched = sel[sel["market_id"].isna()]
    if len(unmatched):
        log.warning("%d selections not found on Betfair: %s", len(unmatched),
                    unmatched[["model", "track", "race_time", "horse_name"]].to_dict("records")[:10])
    sel = sel[sel["market_id"].notna()].copy()
    sel["selection_id"] = sel["selection_id"].astype("int64")
    log.info("%s %s: %d selections on Betfair (%s); to win GBP%.0f, at most GBP%.0f a bet, GBP%.0f a day a model",
             mode.capitalize(), day, len(sel), sel.groupby("model").size().to_dict(), TO_WIN, MAX_STAKE, DAY_LIMIT)
    local = Path("out/morning") / f"ledger_{mode}_{day:%Y-%m-%d}.csv"
    local.parent.mkdir(parents=True, exist_ok=True)

    def write(rows):
        body = _ledger_bytes(rows)
        local.write_bytes(body)
        try:
            s3_put(_key(mode, day, "ledger.csv"), body)
        except Exception as exc:                          # the local copy stands; the next order tries again
            log.warning("ledger copy to S3 failed: %s", exc)

    stop = {"at": None, "on": False}

    def stopped() -> bool:
        t = datetime.now(timezone.utc)
        if stop["at"] is None or (t - stop["at"]).total_seconds() >= 60:
            stop["at"] = t
            stop["on"] = any(s3_get(k) is not None for k in (f"{PREFIX}/STOP", "trading/STOP"))
        return stop["on"]
    s = MorningSession(x, day, sel, markets, mode, live_ledger=lambda: s3_get(f"trading/live/{day:%Y-%m-%d}/ledger.csv"),
                       stopped=stopped, on_order=write)
    rows = run(s)
    write(rows)
    ev = pd.DataFrame(rows)
    if len(ev):
        log.info("Done: %s", ev.groupby(["model", "event"]).size().to_dict())
        b = ev[ev["event"] == "back"]
        log.info("Matched GBP%.2f on %d bets (%s)", b["matched"].sum(), len(b),
                 b.groupby("model")["matched"].sum().round(2).to_dict())
    return rows


def settle(day: date, live: bool) -> dict:
    """Each back settled from Betfair's record of the settled bets (live), or from the closed books (paper):
    P&L before commission per bet, then per model the day's net and commission at 2% and 5% on net winnings."""
    from trading.exchange import BetfairData, LiveExchange, PaperExchange
    mode = "live" if live else "paper"
    raw = s3_get(_key(mode, day, "ledger.csv"))
    if raw is None:
        log.warning("No %s ledger for %s", mode, day)
        return {}
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    bets = led[led["event"] == "back"].copy()
    data = BetfairData()
    x = LiveExchange(data) if live else PaperExchange(data)
    x.login()
    mids = sorted(bets["market_id"].astype(str).unique())
    pnl, result = {}, {}
    if live and mids:
        for o in x.cleared(mids):
            r = str(o.get("customerOrderRef") or "")
            if not any(r.startswith(f"{p}{day:%m%d}") for p in MODELS.values()):
                continue                                  # the live trader's bets, or the owner's own
            k = (str(o["marketId"]), int(o["selectionId"]), r[:3])
            pnl[k] = pnl.get(k, 0.0) + float(o.get("profit") or 0.0)
            result[k] = "WINNER" if float(o.get("profit") or 0.0) > 0 else "LOSER"
    elif mids:
        books = x.books(mids, with_sp=True)
        for b in bets.itertuples(index=False):
            q = books.get(str(b.market_id))
            q = q.runners.get(int(b.selection_id)) if q is not None else None
            st = q.status if q is not None else ""
            k = (str(b.market_id), int(b.selection_id), MODELS[b.model])
            result[k] = st
            pnl[k] = (float(b.matched) * (float(b.avg_price) - 1) if st == "WINNER" else
                      -float(b.matched) if st == "LOSER" else 0.0)
    sp = {}                                               # the BSP of each runner backed, for the CLV
    if mids:
        try:
            for mid, q in x.books(mids, with_sp=True).items():
                for sid, r in q.runners.items():
                    if getattr(r, "bsp", None):
                        sp[(str(mid), int(sid))] = float(r.bsp)
        except Exception as exc:  # noqa: BLE001  the CLV is a reading: the settle stands without it
            log.warning("No BSP for the CLV: %s", exc)
    led["result"] = led["result"].astype(object) if "result" in led.columns else None   # read back blank: floats
    out = []
    for i, b in bets.iterrows():
        k = (str(b.market_id), int(b.selection_id), MODELS[b.model])
        led.loc[i, "result"] = result.get(k, "UNSETTLED")
        led.loc[i, "pnl"] = round(pnl[k], 2) if k in pnl else None
        bsp = sp.get(k[:2])
        led.loc[i, "bsp"] = bsp
        avg = pd.to_numeric(b.get("avg_price"), errors="coerce")
        led.loc[i, "clv"] = round(float(avg) / bsp - 1, 4) if bsp and bsp > 1 and avg == avg and avg > 1 else None
    s3_put(_key(mode, day, "ledger.csv"), _ledger_bytes(led.to_dict("records")))
    summary = {"day": str(day), "mode": mode}
    for model in MODELS:
        b = led[(led["event"] == "back") & (led["model"] == model)]
        p = pd.to_numeric(b["pnl"], errors="coerce")
        e = (1 / pd.to_numeric(b["avg_price"], errors="coerce")).sum()
        wins = int((b["result"] == "WINNER").sum())
        net = float(p.sum())
        summary[model] = {"selections": int((led["model"] == model).sum()), "bets": int(len(b)),
                          "staked": round(float(b["matched"].sum()), 2), "wins": wins, "expected": round(float(e), 3),
                          "unsettled": int(p.isna().sum()), "net_before_commission": round(net, 2),
                          **{f"pnl_{int(c * 100)}pc": round(net - c * max(net, 0.0), 2) for c in COMMISSION},
                          "skipped_live": int(((led["model"] == model) & (led["event"] == "skipped_live")).sum())}
        out.append(summary[model])
    s3_put(_key(mode, day, "summary.json"), json.dumps(summary, indent=2).encode())
    log.info("Settled %s: %s", day, json.dumps(summary))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true", help="the day's selections from the 06:00 card, to S3")
    g.add_argument("--trade", action="store_true", help="the 10:00 UK orders")
    g.add_argument("--settle", action="store_true", help="settle the day's ledger")
    ap.add_argument("--live", action="store_true", help="real orders (only with MORNING_LIVE=yes)")
    ap.add_argument("--date", default=None, help="UK racing day, YYYY-MM-DD (default today)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    day = date.fromisoformat(a.date) if a.date else uk_today()
    live = bool(a.live)
    if live and not live_switch_on():
        log.warning("--live without MORNING_LIVE=yes: paper")
        live = False
    if a.select:
        sel = select_day(day)
        buf = io.StringIO()
        sel.to_csv(buf, index=False)
        s3_put(f"{PREFIX}/selections/{day:%Y-%m-%d}.csv", buf.getvalue().encode())
        print(sel.to_string())
    elif a.trade:
        trade(day, live)
    else:
        settle(day, live)
    return 0


if __name__ == "__main__":
    sys.exit(main())
