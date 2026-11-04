"""The paper session: ball feed, engine, books and paper trader on one clock (build spec 7, Phase 1).

    python -m cricket.live.paper --feed-record feed.jsonl --books data/books/2026-11-04 --market 1.234 \\
        --root data/cricket/store --bank 1000 --out out/paper

Recorded mode (this command): a recorded feed (cricket/ingest/ball_feed.py ``FeedRecorder``, every event with the
time it reached us) and the recorded books of the same match (cricket/market/recorder.py) are merged in time order
and traded on paper exactly as live: a quote is priced on each delivery, and its signal is submitted at the
delivery's receipt time plus the time the quote took to compute, so a slow model pays for its slowness. Live mode is
the same session fed as events arrive; it needs the ball-feed vendor (build spec 5.3: not chosen yet).

Nothing here can place a bet: the trader is paper only and the books are read from the record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from cricket.ingest.ball_feed import FeedEvent
from cricket.live.engine import Engine
from cricket.market.book import Book, books_from_rows
from cricket.reports.phase_gate import feed_lag, gate, report_md
from cricket.trading.paper_trader import PaperTrader
from cricket.trading.signals import Rules, signals
from cricket.trading.sizing import Sizer


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def map_runners(teams: list[str], runners: dict[int, str]) -> dict[str, int]:
    """Cricsheet team name -> Betfair selection, by normalised name; a team left unmatched is left out."""
    by = {norm(n): int(s) for s, n in runners.items()}
    return {t: by[norm(t)] for t in teams if norm(t) in by}


def _t(text: str) -> dt.datetime:
    return dt.datetime.strptime(text.rstrip("Z")[:26], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=dt.timezone.utc)


class PaperSession:
    def __init__(self, engine: Engine, trader: PaperTrader, market_id: str, runners: dict[int, str],
                 clock=time.perf_counter):
        self.engine, self.trader, self.market_id = engine, trader, market_id
        self.runners = runners
        self.sel: dict[str, int] = {}
        self.book: Book | None = None
        self.last_event: dt.datetime | None = None
        self.quotes: list[dict] = []
        self.receipts: list[dt.datetime] = []
        self.match_id: str | None = None
        self.clock = clock
        self.notes: list[str] = []

    def on_book(self, book: Book) -> None:
        if book.market_id != self.market_id:
            return
        self.trader.on_book(book)
        self.book = book
        if book.status == "CLOSED":
            win = [s for s, r in book.runners.items() if r.status == "WINNER"]
            if win:
                self.trader.settle(self.market_id, win[0])

    def on_feed(self, ev: FeedEvent) -> None:
        now = _t(ev.receipt_utc)
        if ev.kind == "setup":
            self.match_id = ev.match_id
            self.sel = map_runners(ev.payload["teams"], self.runners)
            if len(self.sel) != 2:
                self.notes.append(f"runner mapping: no data found for {ev.payload['teams']} in {self.runners}")
        self.engine.on_event(ev)
        if ev.kind == "result":
            w = ev.payload.get("winner")
            if w in self.sel:
                self.trader.settle(self.market_id, self.sel[w])
            return
        if ev.kind != "delivery":
            return
        self.receipts.append(now)
        d = ev.payload
        if int(d["team_wickets"]) > 0 or int(d["runs_batter"]) in (4, 6):
            self.last_event = now
        t0 = self.clock()
        q = self.engine.price()
        took = self.clock() - t0
        at = now + dt.timedelta(seconds=took)
        self.quotes.append({"receipt": now.isoformat(), "submit_at": at.isoformat(), "innings": q["innings"],
                            "runs": q["runs"], "wickets": q["wickets"], "legal": q["legal"],
                            **{f"p_{k}": v for k, v in (q["p"] or {}).items()}, "quote_s": took})
        if q["p"] is None or self.book is None or len(self.sel) != 2:
            return
        p_sel = {self.sel[t]: p for t, p in q["p"].items() if t in self.sel}
        since = (at - self.last_event).total_seconds() if self.last_event else None
        for sig in signals(p_sel, self.book, self.trader.rules, since):
            self.trader.submit(sig, self.book, self.match_id, at)


def read_feed(path: Path) -> list[FeedEvent]:
    return [FeedEvent(**json.loads(x)) for x in path.read_text().splitlines() if x.strip()]


def read_books(path: Path, market_id: str) -> list[Book]:
    files = sorted(path.glob("books_*.csv.gz")) if path.is_dir() else [path]
    df = pd.concat([pd.read_csv(io.BytesIO(gzip.decompress(f.read_bytes())), dtype={"market_id": str})
                    for f in files], ignore_index=True)
    return books_from_rows(df[df.market_id == market_id])


def run_recorded(session: PaperSession, feed: list[FeedEvent], books: list[Book]) -> None:
    """Feed events and books merged by time: a book polled at the same instant as a receipt comes first."""
    stream = [(b.pt, 0, b) for b in books] + [(_t(e.receipt_utc), 1, e) for e in feed]
    for _, kind, x in sorted(stream, key=lambda r: (r[0], r[1])):
        if kind == 0:
            session.on_book(x)
        else:
            session.on_feed(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--version", default="latest")
    ap.add_argument("--feed-record", required=True)
    ap.add_argument("--books", required=True, help="a recorder day folder or one books_*.csv.gz")
    ap.add_argument("--market", required=True)
    ap.add_argument("--runners", required=True, help='JSON {"selection_id": "runner name"} (markets_*.csv.gz)')
    ap.add_argument("--bank", type=float, default=1000.0)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--commission", type=float, default=0.05)
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--out", default="out/paper")
    a = ap.parse_args(argv)
    from cricket.model.production import load_bundle
    from sources.common import Store
    store = Store(root=a.root)
    rules = Rules(threshold=a.threshold, commission=a.commission)
    trader = PaperTrader(rules, Sizer(bank=a.bank, commission=a.commission))
    s = PaperSession(Engine(load_bundle(store, a.version), paths=a.paths), trader, a.market,
                     {int(k): v for k, v in json.loads(a.runners).items()})
    books = read_books(Path(a.books), a.market)
    run_recorded(s, read_feed(Path(a.feed_record)), books)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    led = trader.ledger()
    led.to_csv(out / "paper_ledger.csv", index=False)
    pd.DataFrame(s.quotes).to_csv(out / "quotes.csv", index=False)
    res = gate(led, trader.tx.max_hour(), feed_lag(s.receipts, books), sorted({b.bet_delay for b in books}),
               trader.commission_paid)
    md = report_md(res, f"feed `{a.feed_record}`, books `{a.books}`, market {a.market}")
    (out / "phase1_gate.md").write_text(md)
    print(md)
    for n in s.notes:
        print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
