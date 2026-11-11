"""The favourite back, on paper (the owner, 10 Nov 2026: "I want to back. Either back and trade out at BFSP, or back
just to win"; "Yes build it").

The back study's one rule that survived its false discovery rate (reports/greyhound_model.md section 39; fast-loop run
42906178212), pre-registered for its forward test from 11 Nov (reports/preregistration_greyhound_fav_back.md), run here
live on the live-key books so its decisions and fills are seen as they happen:

- at each race's **first traded book** (every runner has traded and our prices cover the field: the early lay's rule,
  greyhound/early_lay.py), the market's **favourite** (the shortest mid);
- a **back at the forecast SP x 1.20** on Betfair's ladder (rounded up), the forecast the early lay's own (the
  conditional logit on the book and the live model's price, ``greyhound.laytest.live_fit``), stake to win GBP100
  (``greyhound.tradeout.target_stake``, as the study stakes);
- **filled** on paper: ``taken`` if the best back price already reaches the limit; else ``strict`` (the registered
  rule: a later traded price beyond the limit, so the queue ahead certainly cleared) or ``through`` (the best back
  price reaches it, an upper bound); each logged once, with its time;
- **closed at the SP** (the back's winnings laid at the Betfair SP, settled by the result, as the study and the horse
  trader close) and, beside it, **held to win**.

**Paper only, by construction:** this module has no call that sends an order; it reads books and writes its ledger.
The registered gate is the study's procedure on the recorded books (the 2 Dec check-in); this is its execution shadow.

    python -m greyhound.fav_back --trade --until 21:30          # paper-models.yml, the live-key books
    python -m greyhound.fav_back --settle --days 3              # the days' backs against the SP and the result
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

from greyhound import costs

log = logging.getLogger(__name__)

PREFIX = "greyhound/fav_back"
MODE = "paper"
CONFIG = {
    "margin": 0.20,                 # the registered rule: the forecast SP x 1.20
    "watch_minutes": 60,            # races watched from this long before their off
    "min_rest_seconds": 60,         # no back with less than this left to rest
    "close_seconds": 5,             # the SP close (on paper) this long before the off
    "close_margin_seconds": 120,    # no new race off later than this before the trader stops
    "poll_seconds": 5,
    "near_poll_seconds": 1,
    "overrun_minutes": 15,
}
LEDGER_FIELDS = ["time_utc", "mode", "market_id", "selection_id", "track", "race_time", "trap", "dog", "action",
                 "limit", "back1", "lay1", "mid", "f_bsp", "p_model", "rank", "stake", "fill_price", "seconds_to_off",
                 "note"]
FILL_ACTIONS = ("fill_taken", "fill_strict", "fill_through")


def decide(d: pd.DataFrame, fit: dict, cfg: dict) -> pd.DataFrame:
    """The race at its first traded book (``early_lay.first_rows``): every runner's forecast SP and the market's
    favourite with its back limit (the forecast x (1 + margin), on the ladder rounded up), its stake and whether the
    limit is a price the study would place (``1 < limit <= MAX_PRICE``)."""
    from greyhound import tradeout as to
    from greyhound.early_lay import vol_in_fit
    from research.greyhound import limit_study as ls, tradeout_study as ts
    if not vol_in_fit(fit):
        d = d.assign(runner_total_matched=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = to.add_features(d)
        q, book = ts.predict(x, fit)
    x = x.assign(f_bsp=1.0 / (book * np.asarray(q, float)))
    x["fav"] = x["rank"].to_numpy(float) == 1                 # limit_study.gate_flags' 'fav': the shortest mid
    x["limit"] = ls.ladder(x.f_bsp.to_numpy(float) * (1 + cfg["margin"]), up=True)
    ok = np.isfinite(x["limit"]) & (x["limit"] > 1.0) & (x["limit"] <= to.MAX_PRICE)
    x["stake"] = np.round(to.target_stake(x["limit"].to_numpy(float)), 2)
    x["placed"] = x.fav & ok
    return x


def fill_now(limit: float, back1: float) -> bool:
    """Taken at once: the best back price on offer already reaches the limit (the study's ``fills``)."""
    return bool(np.isfinite(back1) and back1 >= limit)


def fill_later(limit: float, back1: float, last_traded: float) -> dict:
    """Which rules a later book fills: ``strict`` a traded price beyond the limit, ``through`` the best back price at
    or beyond it (limit_study.fills, for a back)."""
    return {"strict": bool(np.isfinite(last_traded) and last_traded > limit + 1e-9),
            "through": bool(np.isfinite(back1) and back1 >= limit)}


def sp_close_pnl(stake: float, price: float, bsp: float, won: bool) -> float:
    """A back of ``stake`` at ``price`` closed at the SP as the study closes it (limit_study.settle_orders' 'sp'): its
    winnings laid at the SP, so nothing if it wins and ``-stake + stake (price - 1) / (bsp - 1)`` if it loses;
    before commission."""
    return 0.0 if won else -stake + stake * (price - 1.0) / (bsp - 1.0)


def held_pnl(stake: float, price: float, won: bool) -> float:
    return stake * (price - 1.0) if won else -stake


class FavBack:
    """One paper book: the favourites at their first traded book, their backs resting until the off, the paper fills
    and the SP close, in a ledger copied to S3 (``greyhound/fav_back/paper/<day>/``)."""

    def __init__(self, store, x, cfg: dict, day: date, ledger_path: Path, fit: dict, kill=lambda: False,
                 now=lambda: datetime.now(timezone.utc)):
        self.store, self.x, self.cfg, self.day = store, x, cfg, day
        self.ledger_path, self.fit, self.kill, self.now = ledger_path, fit, kill, now
        self.first: dict = {}           # market -> selection -> its first traded book
        self.hist: dict = {}            # (market, selection) -> [(time, back1, lay1)]
        self.priced: set = set()        # markets decided
        self.closed: set = set()        # markets closed (SP close or lapsed)
        self.open: dict = {}            # market -> the back resting: {sid, limit, stake, fills: set}
        self.preds: pd.DataFrame | None = None
        self.until: datetime | None = None
        self._loaded_at = None
        self._beat_at = None
        self.last_error = ""
        self._resume()

    # the record ----------------------------------------------------------------------------------------------------
    def _resume(self) -> None:
        if not self.ledger_path.exists():
            return
        with self.ledger_path.open() as f:
            for r in csv.DictReader(f):
                m = r["market_id"]
                if r["action"] in ("back", "no_back"):
                    self.priced.add(m)
                if r["action"] == "back":
                    self.open[m] = {"sid": int(float(r["selection_id"])), "limit": float(r["limit"]),
                                    "stake": float(r["stake"]), "fills": set(), "row": r}
                if r["action"] in FILL_ACTIONS and m in self.open:
                    self.open[m]["fills"].add(r["action"])
                if r["action"] in ("sp_close", "lapsed"):
                    self.closed.add(m)
                    self.open.pop(m, None)

    def _write(self, rows: list[dict]) -> None:
        new = not self.ledger_path.exists()
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
        try:
            self.store.put(f"{PREFIX}/{MODE}/{self.day:%Y-%m-%d}/ledger.csv", self.ledger_path.read_bytes())
        except Exception as exc:                                # the local ledger is the record; S3 its copy
            log.warning("ledger not copied to S3 (%s)", exc)

    def load_predictions(self) -> None:
        t = self.now()
        if self._loaded_at and (t - self._loaded_at).total_seconds() < 600:
            return
        raw = self.store.get(f"greyhound/live/predictions/{self.day:%Y-%m-%d}.csv")
        self._loaded_at = t
        if raw:
            p = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
            p["selection_id"] = pd.to_numeric(p.selection_id, errors="coerce")
            p = p.dropna(subset=["selection_id"]).drop_duplicates(["market_id", "selection_id"], keep="last")
            p["off"] = pd.to_datetime(p.race_date + " " + p.race_time.astype(str).str[:5]).dt.tz_localize(
                "Europe/London").dt.tz_convert("UTC")
            self.preds = p

    def heartbeat(self, force: bool = False, **extra) -> None:
        t = self.now()
        if not force and self._beat_at and (t - self._beat_at).total_seconds() < 60:
            return
        self._beat_at = t
        st = {"time_utc": pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": MODE,
              "markets_watched": len(self.first), "markets_priced": len(self.priced),
              "backs_resting": len(self.open), "markets_closed": len(self.closed), "stopped": bool(self.kill()),
              "prices_loaded": self.preds is not None, "last_error": self.last_error[:300], **extra}
        try:
            self.store.put(f"{PREFIX}/{MODE}/{self.day:%Y-%m-%d}/status.json", json.dumps(st).encode())
        except Exception as exc:
            log.warning("status not written (%s)", exc)

    # one poll ------------------------------------------------------------------------------------------------------
    def step(self, near_only: bool = False) -> None:
        self.load_predictions()
        if self.preds is None:
            return
        t = pd.Timestamp(self.now())
        to_off = (self.preds.off - t).dt.total_seconds()
        horizon = self.cfg["close_seconds"] + 2 * self.cfg["near_poll_seconds"] + 5 if near_only \
            else self.cfg["watch_minutes"] * 60
        late = self.preds.market_id.isin(set(self.open)) & (to_off > -180)
        watch = self.preds[((to_off > 0) | late) & (to_off <= horizon) & ~self.preds.market_id.isin(self.closed)]
        mids = sorted(watch.market_id.dropna().astype(str).unique())
        if not mids:
            return
        offs = watch.groupby(watch.market_id.astype(str)).off.min()
        mids = sorted(mids, key=lambda m: offs.get(m, pd.Timestamp.max.tz_localize("UTC")))
        books = self.x.books(mids)
        stopped = self.kill()
        for mid in mids:
            b = books.get(mid)
            if b is None:
                continue
            g = watch[watch.market_id.astype(str) == mid]
            secs = float((pd.Timestamp(g.off.iloc[0]) - pd.Timestamp(self.now())).total_seconds())
            try:
                if mid in self.open:
                    self.track(mid, b, secs)
                if secs <= self.cfg["close_seconds"] or b.inplay or b.status != "OPEN":
                    if mid in self.open:
                        self.close(mid, secs)
                elif not near_only:
                    self.watch(mid, b, g, secs, stopped)
            except Exception as exc:                         # one market's failure never stops the others
                log.warning("%s: %s", mid, exc)
                self.last_error = f"{mid}: {exc}"

    def watch(self, mid: str, b, g: pd.DataFrame, secs: float, stopped: bool) -> None:
        from greyhound.early_lay import first_rows
        if mid not in self.priced and self.until is not None and pd.Timestamp(g.off.iloc[0]) > \
                pd.Timestamp(self.until) - pd.Timedelta(seconds=self.cfg["close_margin_seconds"]):
            return
        t = self.now()
        active = {sid: q for sid, q in b.runners.items() if q.status == "ACTIVE"}
        first = self.first.setdefault(mid, {})
        for sid, q in active.items():
            bb, bl = q.best_back, q.best_lay
            self.hist.setdefault((mid, sid), []).append((t, bb[0] if bb else np.nan, bl[0] if bl else np.nan))
            if sid not in first and q.last_traded and q.last_traded > 1 and bb:
                first[sid] = {"market_id": mid, "polled": t, "back1": bb[0], "back1_size": bb[1],
                              "lay1": bl[0] if bl else np.nan, "lay1_size": bl[1] if bl else np.nan,
                              "last_traded": float(q.last_traded), "traded": float(q.traded or 0.0),
                              "market_traded": float(b.total_matched or 0.0)}
        if mid in self.priced or secs < self.cfg["close_seconds"] + self.cfg["min_rest_seconds"]:
            return
        if not active or any(sid not in first for sid in active):
            return
        if not set(active) <= set(g.selection_id.dropna().astype(int)):
            return
        self.priced.add(mid)
        info = g.set_index(g.selection_id.astype(int))
        base = {"time_utc": pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": MODE, "market_id": mid,
                "seconds_to_off": round(secs)}
        if stopped:
            self._write([{**base, "action": "no_back", "note": "stopped (S3 STOP)"}])
            return
        prev = {}
        for sid in active:
            h = [e for e in self.hist.get((mid, sid), []) if (first[sid]["polled"] - e[0]).total_seconds() >= 90]
            prev[sid] = (h[-1][1], h[-1][2]) if h else None
        d = decide(first_rows({s: first[s] for s in active}, prev, g.assign(selection_id=g.selection_id.astype(int))),
                   self.fit, self.cfg)
        fav = d[d.fav]
        if not len(fav):
            self._write([{**base, "action": "no_back", "note": "no favourite"}])
            return
        r = fav.iloc[0]
        sid = int(r.selection_id)
        meta = info.loc[sid]
        row = {**base, "selection_id": sid, "track": meta.get("track", ""), "race_time": meta.get("race_time", ""),
               "trap": meta.get("trap", ""), "dog": meta.get("dog_name", ""), "limit": round(float(r["limit"]), 2)
               if np.isfinite(r["limit"]) else "", "back1": r.back1, "lay1": r.lay1, "mid": round(float(r["mid"]), 3),
               "f_bsp": round(float(r.f_bsp), 3), "p_model": round(float(r.p_model), 4), "rank": 1,
               "stake": float(r.stake) if np.isfinite(r.stake) else ""}
        if not bool(r.placed):
            self._write([{**row, "action": "no_back", "note": "the limit is off the study's prices (1 to 20)"}])
            return
        self._write([{**row, "action": "back"}])
        self.open[mid] = {"sid": sid, "limit": float(r["limit"]), "stake": float(r.stake), "fills": set(), "row": row}
        if fill_now(float(r["limit"]), float(r.back1)):
            self.open[mid]["fills"].add("fill_taken")
            self._write([{**row, "action": "fill_taken", "fill_price": r.back1, "note": "the back price reached it"}])

    def track(self, mid: str, b, secs: float) -> None:
        """The resting back's paper fills in this book (each rule logged once)."""
        o = self.open[mid]
        if "fill_taken" in o["fills"]:
            return
        q = b.runners.get(o["sid"])
        if q is None:
            return
        bb = q.best_back
        hit = fill_later(o["limit"], bb[0] if bb else np.nan, float(q.last_traded or np.nan))
        rows = []
        for rule, ok in hit.items():
            act = f"fill_{rule}"
            if ok and act not in o["fills"]:
                o["fills"].add(act)
                rows.append({**o["row"], "time_utc": pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "action": act, "fill_price": o["limit"], "seconds_to_off": round(secs)})
        if rows:
            self._write(rows)

    def close(self, mid: str, secs: float) -> None:
        """At the off: a filled back is closed at the SP (on paper: settled from the BSP after the race); else it
        lapses."""
        o = self.open.pop(mid)
        self.closed.add(mid)
        filled = o["fills"] & {"fill_taken", "fill_strict"}
        act = "sp_close" if filled else "lapsed"
        note = "paper: its winnings laid at the SP" if filled else \
            ("through only (never traded beyond)" if "fill_through" in o["fills"] else "unmatched")
        self._write([{**o["row"], "time_utc": pd.Timestamp(self.now()).strftime("%Y-%m-%dT%H:%M:%SZ"), "action": act,
                      "seconds_to_off": round(secs), "note": note}])

    def run(self, until: datetime, sleep=time.sleep) -> dict:
        self.until = until
        last_full = None
        hard_stop = until + timedelta(minutes=self.cfg["overrun_minutes"])
        while self.now() < until or (self.open and self.now() < hard_stop):
            try:
                if last_full is not None and (self.now() - last_full).total_seconds() < self.cfg["poll_seconds"]:
                    self.step(near_only=True)
                else:
                    last_full = self.now()
                    self.step()
            except Exception as exc:
                log.warning("poll failed (%s)", exc)
                self.last_error = f"poll: {exc}"
            self.heartbeat()
            sleep(self.cfg["near_poll_seconds"])
        self.heartbeat(force=True, finished=True)
        return {"day": f"{self.day:%Y-%m-%d}", "markets_priced": len(self.priced), "markets_closed": len(self.closed),
                "still_open": len(self.open)}


# --------------------------------------------------------------------------------------------------------------------
# Settlement and the running table
# --------------------------------------------------------------------------------------------------------------------

def closing(day: date, s3=None) -> pd.DataFrame:
    """Each runner's Betfair SP and whether it won, from the settled books: the recorder's, else the live-key
    record's."""
    from betfair_recorder import _read_day_csv
    from greyhound import tradeout as to
    out = []
    for books_name, markets_name in (("books_greyhound.csv", "markets_greyhound.csv"),
                                     ("books_greyhound_live.csv", "markets_greyhound_live.csv")):
        books, markets = _read_day_csv(day, books_name, s3=s3), _read_day_csv(day, markets_name, s3=s3)
        if books is None or markets is None or not len(books):
            continue
        sp = to.closing_prices(books, markets, day)
        fin = books[books.source == "final"] if "source" in books else books.iloc[0:0]
        win = fin.assign(market_id=fin.market_id.astype(str), selection_id=pd.to_numeric(fin.selection_id,
                                                                                         errors="coerce"))
        win = win[win.runner_status.astype(str) == "WINNER"][["market_id", "selection_id"]].drop_duplicates()
        sp = sp.assign(market_id=sp.market_id.astype(str)).merge(win.assign(won=1), on=["market_id", "selection_id"],
                                                                 how="left")
        sp["won"] = sp.won.fillna(0).astype(int)
        settled = set(win.market_id)
        out.append(sp[sp.market_id.isin(settled)])
    if not out:
        return pd.DataFrame(columns=["market_id", "selection_id", "bsp", "won"])
    return pd.concat(out, ignore_index=True).drop_duplicates(["market_id", "selection_id"])


def settle_frame(led: pd.DataFrame, close: pd.DataFrame, comm: float = costs.COMMISSION) -> pd.DataFrame:
    """Each back placed, its fills by rule and, where filled, its P&L closed at the SP and held to win, each net of
    the commission on its market's positive net (one back a market, so the market's net is the back's)."""
    if led is None or not len(led):
        return pd.DataFrame()
    led = led.assign(market_id=led.market_id.astype(str), selection_id=pd.to_numeric(led.selection_id,
                                                                                      errors="coerce"))
    if "fill_price" not in led:
        led["fill_price"] = np.nan
    backs = led[led.action == "back"].drop_duplicates("market_id", keep="last")
    acts = led[led.action.isin(FILL_ACTIONS)].groupby("market_id").action.agg(set)
    out = backs[["market_id", "selection_id", "track", "race_time", "trap", "dog", "limit", "mid", "f_bsp",
                 "stake"]].copy()
    out["limit"], out["stake"] = pd.to_numeric(out["limit"]), pd.to_numeric(out.stake)
    fills = out.market_id.map(acts).apply(lambda s: s if isinstance(s, set) else set())
    out["taken"] = fills.apply(lambda s: "fill_taken" in s)
    out["strict"] = out.taken | fills.apply(lambda s: "fill_strict" in s)
    out["through"] = out.strict | fills.apply(lambda s: "fill_through" in s)
    taken_price = led[led.action == "fill_taken"].drop_duplicates("market_id").set_index("market_id").fill_price
    out["price"] = pd.to_numeric(out.market_id.map(taken_price), errors="coerce").fillna(out["limit"])
    out = out.merge(close, on=["market_id", "selection_id"], how="left")
    net = lambda x: x - comm * max(x, 0.0)  # noqa: E731
    ok = out.bsp.notna() & (out.bsp > 1)
    out["settled"] = ok
    out["pnl_sp"] = [net(sp_close_pnl(s, p, b, w == 1)) if (f and o) else 0.0 for s, p, b, w, f, o in
                     zip(out.stake, out.price, out.bsp, out.won, out.strict, ok)]
    out["pnl_held"] = [net(held_pnl(s, p, w == 1)) if (f and o) else 0.0 for s, p, w, f, o in
                       zip(out.stake, out.price, out.won, out.strict, ok)]
    out["clv"] = np.where(out.strict & ok, out.price / out.bsp - 1.0, np.nan)
    out["pnl_sp_through"] = [net(sp_close_pnl(s, p, b, w == 1)) if (f and o) else 0.0 for s, p, b, w, f, o in
                             zip(out.stake, out.price, out.bsp, out.won, out.through, ok)]
    return out


def summary(s: pd.DataFrame) -> dict:
    if s is None or not len(s):
        return {"backs": 0}
    f = s[s.strict & s.settled]
    risk = float(f.stake.sum())
    return {"backs": int(len(s)), "settled": int(s.settled.sum()), "filled_strict": int(s.strict.sum()),
            "filled_through": int(s.through.sum()), "fill_rate_strict": round(float(s.strict.mean()), 3),
            "pnl_sp": round(float(s.pnl_sp.sum()), 2), "pnl_held": round(float(s.pnl_held.sum()), 2),
            "staked": round(risk, 2), "roi_sp%": round(100 * float(s.pnl_sp.sum()) / risk, 2) if risk else None,
            "roi_held%": round(100 * float(s.pnl_held.sum()) / risk, 2) if risk else None,
            "clv_mean%": round(100 * float(f.clv.mean()), 2) if len(f) else None,
            "pnl_sp_through": round(float(s.pnl_sp_through.sum()), 2)}


def settle(store, day: date, s3=None) -> dict:
    raw = store.get(f"{PREFIX}/{MODE}/{day:%Y-%m-%d}/ledger.csv")
    if not raw:
        return {"day": f"{day:%Y-%m-%d}", "note": "no ledger"}
    s = settle_frame(pd.read_csv(io.BytesIO(raw), dtype={"market_id": str}), closing(day, s3=s3))
    buf = io.StringIO()
    s.to_csv(buf, index=False)
    store.put(f"{PREFIX}/{MODE}/{day:%Y-%m-%d}/settled.csv", buf.getvalue().encode())
    out = {"day": f"{day:%Y-%m-%d}", **summary(s)}
    store.put(f"{PREFIX}/{MODE}/{day:%Y-%m-%d}/summary.json", json.dumps(out, indent=1).encode())
    return out


def table(store, day: date) -> str:
    """The running table's section: the day's backs, their fills, and (once the races settle) the SP close and the
    back to win."""
    head = "### The favourite back (paper: back at the forecast SP x 1.20, closed at the SP)\n\n"
    raw = store.get(f"{PREFIX}/{MODE}/{day:%Y-%m-%d}/ledger.csv")
    st = store.get(f"{PREFIX}/{MODE}/{day:%Y-%m-%d}/status.json")
    status = json.loads(st) if st else {}
    line = (f"Last seen {status.get('time_utc', '?')} UTC; {status.get('markets_priced', 0)} races priced, "
            f"{status.get('backs_resting', 0)} backs resting." if status else "No status yet.")
    if not raw:
        return head + line + " No backs yet.\n"
    led = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str})
    try:
        s = settle_frame(led, closing(day))
    except Exception as exc:                                   # the books not in yet: the ledger alone
        log.warning("not settled (%s)", exc)
        s = settle_frame(led, pd.DataFrame(columns=["market_id", "selection_id", "bsp", "won"]))
    sm = summary(s)
    lines = [head + line, "",
             f"Backs {sm.get('backs', 0)}; filled (strict) {sm.get('filled_strict', 0)}, through "
             f"{sm.get('filled_through', 0)}; settled {sm.get('settled', 0)}. Closed at the SP GBP{sm.get('pnl_sp', 0)}"
             f" ({sm.get('roi_sp%')}% of the stakes), held to win GBP{sm.get('pnl_held', 0)} ({sm.get('roi_held%')}%); "
             f"CLV {sm.get('clv_mean%')}%.", "",
             "| Race | Trap | Dog | Mid | Forecast | Limit | Stake | Filled | SP | Won | SP close | Held |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.sort_values("race_time").iterrows():
        filled = "taken" if r.taken else "strict" if r.strict else "through" if r.through else ""
        bsp = f"{r.bsp:.2f}" if pd.notna(r.get("bsp")) else ""
        won = "" if pd.isna(r.get("won")) else ("yes" if r.won == 1 else "no")
        lines.append(f"| {r.race_time} {r.track} | {r.trap} | {r.dog} | {r.mid} | {r.f_bsp} | {r.limit} | {r.stake} | "
                     f"{filled} | {bsp} | {won} | {r.pnl_sp:.2f} | {r.pnl_held:.2f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from betfair_recorder import uk_time_on, uk_today
    from greyhound import laytest
    from greyhound.live import kill_switch
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--trade", action="store_true", help="the paper book (no order is ever sent)")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--days", type=int, default=1, help="settle: the last N days")
    ap.add_argument("--until", default="21:30")
    ap.add_argument("--date", default=None)
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    day = date.fromisoformat(a.date) if a.date else uk_today()
    out = {}
    if a.settle:
        days = [day] if a.date else [day - timedelta(days=i) for i in range(a.days, 0, -1)]
        out["settle"] = [settle(store, d) for d in days]
    if a.trade:
        from trading.exchange import BetfairData, LiveExchange
        status_key = f"{PREFIX}/{MODE}/{day:%Y-%m-%d}/status.json"
        store.put(status_key, json.dumps({"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                          "mode": MODE, "state": "starting: fitting the forecast"}).encode())
        fit = laytest.live_fit(store, day)
        if fit is None:
            store.put(status_key, json.dumps({"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                              "mode": MODE, "state": "stopped: no forecast fit for today"}).encode())
            raise SystemExit("no forecast fit for today: nothing decided")
        x = LiveExchange(BetfairData(source="greyhound"))       # its books only: this module sends no order
        x.login()
        ledger = Path(os.getenv("GREYHOUND_LEDGER_DIR") or (Path.home() / ".ashcroft" / "greyhound")) / \
            "fav_back_paper" / f"{day:%Y-%m-%d}" / "ledger.csv"
        log.info("favourite back (paper): margin %.2f, to win GBP100, fit to %s", CONFIG["margin"], fit.get("fit_to"))
        out["trade"] = FavBack(store, x, dict(CONFIG), day, ledger, fit, kill=kill_switch(store)).run(
            uk_time_on(day, a.until))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
