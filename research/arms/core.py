"""The replay's machinery for the strategy arms (read-only).

Ported from research/queries/done/stake_target_replay.py, the machinery every evening replay runs, so the arms import
it instead of patching it: a recorded day's books (the trader's own reads and the recorder's), its markets, the
model's prices, the real trader's fills, every runner's BSP and result from Betfair's price files; the day replayed a
minute at a time through the live Session, fills against the recorded sizes (an offer the replay takes stays taken,
on the back side and on the lay side), the account's money as Betfair holds it; the ledger scored at the price
matched and after the non-runners' reduction factors. Three kinds of day: the win markets (the live rule's), the
place markets (the recorder's books, Betfair's place files) and the evening before (books_evening: tomorrow's win
markets from 17:00 UK).
"""

from __future__ import annotations

import bisect
from contextlib import contextmanager
import gzip
import io
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import betfair_prices as bp  # noqa: E402
from trading import config as tc  # noqa: E402
from trading import strategy  # noqa: E402
from trading.exchange import Book, Fill, Market, PaperExchange, Quote  # noqa: E402
from trading.matching import attach_ids  # noqa: E402
from trading.session import HELD, Session  # noqa: E402

UK = ZoneInfo("Europe/London")
BALANCE = 5000.0                       # the account the replays hold (replay_1021 on)
STALE = timedelta(minutes=6)           # a book older than this is no book (the gaps between the sessions)
KINDS = {
    "win": {"books": ("books_trader", "books"), "cats": ("markets_trader", "markets"), "mtype": "WIN", "files": "win"},
    "place": {"books": ("books",), "cats": ("markets",), "mtype": "PLACE", "files": "place"},
    "evening": {"books": ("books_evening",), "cats": ("markets_evening", "markets_trader", "markets"),
                "mtype": "WIN", "files": "win"},
    # the evening before and the day itself as one session (the timing study, the owner 4 Nov): the evening's books,
    # then the overnight record once it is kept (books_overnight), then the trader's and the recorder's
    "evening_morning": {"books": ("books_evening", "books_overnight", "books_trader", "books"),
                        "cats": ("markets_evening", "markets_trader", "markets"), "mtype": "WIN", "files": "win"},
}
_S3 = None


def s3():
    global _S3
    if _S3 is None:
        import boto3
        _S3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
    return _S3


def bucket() -> str:
    return os.environ.get("PRED_BUCKET") or "ashcroft"


def get(key: str) -> bytes | None:
    try:
        return s3().get_object(Bucket=bucket(), Key=key)["Body"].read()
    except Exception as exc:                                # a file one writer never made
        print(f"  {key}: {type(exc).__name__}")
        return None


def csv(key: str, **kw) -> pd.DataFrame:
    raw = get(key)
    if raw is None:
        return pd.DataFrame()
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), dtype={"market_id": str, "selection_id": "Int64"}, low_memory=False, **kw)


def _levels(r, side):
    out = []
    for i in (1, 2, 3):
        p, s = r.get(f"{side}{i}"), r.get(f"{side}{i}_size")
        if p is not None and s is not None and pd.notna(p) and pd.notna(s) and p > 1 and s > 0:
            out.append((float(p), float(s)))
    return out


def price_files(day: date, market: str) -> pd.DataFrame:
    """Betfair's price file rows for the day's racing (files are named by the day after the racing, so both are read)."""
    rows = []
    for d in (day + timedelta(days=1), day):
        for f in (f"dwbfpricesuk{market}{d:%d%m%Y}.csv", f"dwbfpricesire{market}{d:%d%m%Y}.csv"):
            raw = get(f"{bp.S3_PREFIX}/{f}")
            if raw is None:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            t = bp.parse_file_text(text, f)
            rows.append(t[t["race_date"] == day.isoformat()])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def predictions(day: date) -> pd.DataFrame:
    raw = get(f"predictions/{day}.csv")
    if raw is None:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(raw), dtype={"race_time": str}, low_memory=False)


def load_day(day: date, kind: str = "win", preds: pd.DataFrame | None = None, preds_fn=None, intraday: bool = False):
    """The day's books, markets, prices, the real trader's fills, every runner's BSP and result; None while the
    price file is not published or nothing matched. ``preds`` replaces the model's prices; ``preds_fn(markets,
    places)`` makes them from the day's markets (the place arm: market_id, selection_id, predicted_bfsp).
    ``intraday`` (the owner, 4 Nov 2026: the paper agents trading as if live, results the same day): with no price
    file yet, the races run so far are settled from the recorder's own books (each runner's BSP as read after the off
    and, where read, its result): provisional, replaced when the price file is in."""
    k = KINDS[kind]
    print(f"\n== {day} ({kind}): books from s3://{bucket()}/betfair_live/{day}/")
    spf = price_files(day, k["files"])
    if spf.empty and not intraday:
        print(f"  Betfair's {k['files']} price file for {day}'s racing is not in the archive yet: the day waits")
        return None
    books = pd.concat([csv(f"betfair_live/{day}/{n}.csv.gz") for n in k["books"]], ignore_index=True)
    if "polled_utc" not in books.columns:
        books = pd.DataFrame(columns=["polled_utc", "market_id"])
    cat = pd.concat([csv(f"betfair_live/{day}/{n}.csv.gz") for n in k["cats"]], ignore_index=True)
    if books.empty or cat.empty:
        print("  no books or no catalogue recorded: the day is skipped")
        return None
    ledger = csv(f"trading/live/{day}/ledger.csv") if kind in ("win", "evening_morning") else pd.DataFrame()
    if preds is None:
        preds = predictions(day)
    print(f"  {len(books):,} book rows, {len(cat):,} catalogue rows, {len(ledger):,} ledger rows, {len(preds)} prices")
    books["t"] = pd.to_datetime(books["polled_utc"], utc=True)
    cat = cat[cat["market_type"].astype(str).str.upper().eq(k["mtype"]) & cat["country"].isin(["GB", "IE"])]
    cat = cat.drop_duplicates(["market_id", "selection_id"])
    markets = []
    for mid, g in cat.groupby("market_id"):
        start = pd.Timestamp(g["market_start_utc"].iloc[0]).tz_convert("UTC").to_pydatetime()
        markets.append(Market(str(mid), str(g["venue"].iloc[0]), str(g["country"].iloc[0]), start,
                              name=str(g["market_name"].iloc[0]),
                              runners={int(s): str(n) for s, n in zip(g["selection_id"], g["runner_name"])}))
    ids = {m.market_id for m in markets}
    books = books[books["market_id"].isin(ids)]
    places = {}
    if "number_of_winners" in books.columns:
        nw = pd.to_numeric(books["number_of_winners"], errors="coerce")
        places = {str(m): int(v) for m, v in books.assign(_nw=nw).dropna(subset=["_nw"])
                  .groupby("market_id")["_nw"].last().items()}
    if preds_fn is not None:
        preds = preds_fn(markets, places)
    if "market_id" in preds.columns and preds["market_id"].notna().all() and "selection_id" in preds.columns:
        preds = preds.copy()
    else:
        preds = attach_ids(preds, markets)
    preds = preds[preds["market_id"].notna()]
    print(f"  {len(markets)} GB/IE {k['mtype']} markets; {len(preds)} prices matched to them")

    polls: dict[str, list[tuple[datetime, Book]]] = defaultdict(list)
    cols = set(books.columns)
    for (mid, t), g in books.groupby(["market_id", "t"], sort=True):
        runners = {}
        for r in g.itertuples(index=False):
            rr = r._asdict()
            bsp = rr.get("sp_actual")
            lt = rr.get("last_traded")
            tm = rr.get("runner_total_matched")
            runners[int(rr["selection_id"])] = Quote(
                int(rr["selection_id"]), status=str(rr.get("runner_status") or "ACTIVE"),
                back=_levels(rr, "back"), lay=_levels(rr, "lay"),
                last_traded=float(lt) if lt is not None and pd.notna(lt) else None,
                traded=float(tm) if tm is not None and pd.notna(tm) else 0.0,
                bsp=float(bsp) if bsp is not None and pd.notna(bsp) and bsp > 1 else None)
        first = g.iloc[0]
        tm = first.get("market_total_matched") if "market_total_matched" in cols else None
        polls[str(mid)].append((t.to_pydatetime(), Book(str(mid), str(first["status"]), bool(first["inplay"]),
                                                        float(tm) if tm is not None and pd.notna(tm) else None,
                                                        runners, bsp_reconciled=bool(first.get("bsp_reconciled")))))

    if spf.empty:      # intraday: the recorder's BSPs (read after the off) and results, the races run so far
        spv = pd.to_numeric(books["sp_actual"], errors="coerce") if "sp_actual" in books.columns \
            else pd.Series(np.nan, index=books.index)
        last = books.assign(spv=spv)[spv > 1].sort_values("t").drop_duplicates(["market_id", "selection_id"],
                                                                              keep="last")
        settled = {}
        n_settled = 0
        for m in markets:
            g = last[last["market_id"] == m.market_id]
            for r in g.itertuples(index=False):
                st = "WINNER" if str(getattr(r, "runner_status", "")).upper() == "WINNER" else "LOSER"
                settled[(m.market_id, int(r.selection_id))] = (float(r.spv), st)
            n_settled += bool(len(g))
        print(f"  intraday: the recorder's BSPs for {n_settled} of {len(markets)} markets (provisional)")
    else:
        # the files matched to the exchange's markets by course, UK off time and horse, then by time and horse
        day_tracks = {bp.normalise_track(m.venue): m.venue for m in markets}
        spf = spf.copy()
        spf["venue"] = spf["menu_hint"].map(lambda h: bp.course_from_hint(h, day_tracks))
        spf["runner_name"] = spf["selection_name"]
        spf = spf.rename(columns={"event_id": "file_event_id", "selection_id": "file_selection_id"})
        spm = attach_ids(spf[["venue", "race_time", "runner_name", "bsp", "win_lose", "file_event_id",
                              "file_selection_id"]], markets)
        ex = pd.DataFrame([{"_t": bp.utc_iso_to_uk_hhmm(m.start.strftime("%Y-%m-%dT%H:%M:%S.000Z")),
                            "_h": bp.normalise_horse(n), "market_id": m.market_id, "selection_id": int(s)}
                           for m in markets for s, n in m.runners.items()]).drop_duplicates(["_t", "_h"], keep=False)
        miss = spm["market_id"].isna()
        spm["market_id"], spm["selection_id"] = spm["market_id"].astype(object), spm["selection_id"].astype(object)
        by_time = spm.loc[miss, ["race_time", "runner_name"]].assign(
            _t=lambda d: d["race_time"].astype(str).map(bp.race_time_to_24h),
            _h=lambda d: d["runner_name"].map(bp.normalise_horse)).merge(ex, on=["_t", "_h"], how="left")
        spm.loc[miss, "market_id"] = by_time["market_id"].to_numpy()
        spm.loc[miss, "selection_id"] = by_time["selection_id"].to_numpy()
        spm = spm[spm["market_id"].notna()].copy()
        print(f"  price files: {len(spf):,} runners, {len(spm):,} matched to the exchange's markets")
        if spm.empty:
            print("  no runner matched: the day is skipped")
            return None
        settled_at = datetime(day.year, day.month, day.day, 23, 0, tzinfo=timezone.utc)
        settled = {}
        n_settled = 0
        for m in markets:
            g = spm[spm["market_id"] == m.market_id]
            runners = {}
            for r in g.itertuples():
                if pd.notna(r.bsp) and r.bsp > 1:
                    st = "WINNER" if r.win_lose == 1 else "LOSER"
                    runners[int(r.selection_id)] = Quote(int(r.selection_id), status=st, bsp=float(r.bsp))
                    settled[(m.market_id, int(r.selection_id))] = (float(r.bsp), st)
            n_settled += bool(runners)
            polls[m.market_id] = [p for p in polls.get(m.market_id, []) if p[1].status != "CLOSED"]
            polls[m.market_id].append((settled_at, Book(m.market_id, "CLOSED", False, None, runners, True)))
        print(f"  settled books for {n_settled} of {len(markets)} markets")

    real_fills = defaultdict(list)                     # (mid, sid, price) -> [(t, matched)]
    if len(ledger):
        real = ledger[(ledger["event"] == "back") & (pd.to_numeric(ledger["matched"], errors="coerce") > 0)].copy()
        real["t"] = pd.to_datetime(real["ts"], utc=True)
        for r in real.itertuples(index=False):
            real_fills[(str(r.market_id), int(r.selection_id), round(float(r.best_back), 2))].append(
                (r.t, float(r.matched)))
    # the runners removed in the day, with Betfair's reduction factor and the time: the honest scoring
    rem = books[books["runner_status"].astype(str).eq("REMOVED")]
    removed = pd.DataFrame(columns=["market_id", "selection_id", "af", "rt"])
    if len(rem):
        rem = rem.assign(af=pd.to_numeric(rem.get("adjustment_factor"), errors="coerce").fillna(0.0),
                         rt=pd.to_datetime(rem.get("removal_utc"), utc=True, errors="coerce").fillna(rem["t"]))
        removed = rem.groupby(["market_id", "selection_id"]).agg(af=("af", "last"), rt=("rt", "min")).reset_index()
        removed["selection_id"] = removed["selection_id"].astype(int)
    return SimpleNamespace(day=day, kind=kind, markets=markets, polls=dict(polls),
                           poll_times={k_: [t for t, _ in v] for k_, v in polls.items()},
                           starts={m.market_id: m.start for m in markets}, real_fills=real_fills, preds=preds,
                           settled=settled, removed=removed, places=places, ledger=ledger,
                           last_poll=books["t"].max().to_pydatetime() if len(books) else None,
                           provisional=bool(spf.empty))


# ---------------------------------------------------------------------------------------------------------------
# Betfair as the recordings showed it, at the replay's clock
# ---------------------------------------------------------------------------------------------------------------

class ReplayData:
    """The day's books at the replay's clock; offers the replay took stay taken, backs and lays alike."""

    def __init__(self, D, stale: timedelta = STALE):
        self.D = D
        self.now = None
        self.stale = stale
        self.taken = defaultdict(float)
        self.taken_lay = defaultdict(float)

    def login(self):
        return None

    def markets(self, start, end, countries=("GB", "IE")):
        return [m for m in self.D.markets if start <= m.start < end and m.country in countries]

    def _at(self, mid):
        times = self.D.poll_times.get(mid)
        if not times:
            return None
        i = bisect.bisect_right(times, self.now) - 1
        if i < 0:
            return None
        t, b = self.D.polls[mid][i]
        if b.status == "CLOSED":
            return b
        return b if self.now - t <= self.stale else None

    def books(self, ids, with_sp=False):
        out = {}
        for mid in ids:
            b = self._at(mid)
            if b is None:
                continue
            runners = {}
            for sid, q in b.runners.items():
                back = []
                for p, s in q.back:
                    k = (mid, sid, round(p, 2))
                    before = sum(m for t, m in self.D.real_fills.get(k, []) if t <= self.now)
                    gone = max(0.0, self.taken[k] - before)
                    if s - gone > 0.01:
                        back.append((p, round(s - gone, 2)))
                lay = []
                for p, s in q.lay:
                    gone = self.taken_lay[(mid, sid, round(p, 2))]
                    if s - gone > 0.01:
                        lay.append((p, round(s - gone, 2)))
                runners[sid] = Quote(sid, q.status, back, lay, q.last_traded, q.traded, q.bsp)
            out[mid] = Book(b.market_id, b.status, b.inplay, b.total_matched, runners, b.bsp_reconciled)
        return out

    def cleared(self, ids):
        return []

    def funds(self):
        return None


class ReplayExchange(PaperExchange):
    """Paper fills against the recorded books, and the account's money as Betfair holds it: a back larger than the
    funds is refused, and a race's stakes come back ten minutes after its off. Lays fill against the lay levels."""

    def __init__(self, data: ReplayData, bank: float, balance=None):
        super().__init__(data, bank=bank)
        self.balance = balance
        self.fills = []                            # (market_id, money at risk, when matched)
        self.peak = 0.0

    def open_stakes(self) -> float:
        starts = self.data.D.starts
        return sum(s for mid, s, _ in self.fills if self.data.now < starts[mid] + timedelta(minutes=10))

    def funds(self):
        if self.balance is None:
            return None
        return {"available": round(self.balance - self.open_stakes(), 2), "exposure": -round(self.open_stakes(), 2)}

    def back(self, market_id, selection_id, price, size, ref, min_fill: float = 2.0):
        if self.balance is not None and size > self.balance - self.open_stakes() + 0.005:
            return Fill(market_id=market_id, selection_id=int(selection_id), side="BACK", order_type="LIMIT",
                        price=float(price), size=round(float(size), 2), status="FAILURE",
                        error="INSUFFICIENT_FUNDS (ERROR_IN_ORDER)")
        f = super().back(market_id, selection_id, price, size, ref, min_fill)
        if f.status == "SUCCESS":
            self.data.taken[(market_id, int(selection_id), round(float(price), 2))] += f.matched
            self.fills.append((market_id, f.matched, self.data.now))
            self.peak = max(self.peak, self.open_stakes())
        return f

    def lay(self, market_id, selection_id, price, size, ref, min_fill: float = 2.0) -> Fill:
        """A lay of ``size`` (the backer's stake) at ``price`` or shorter, filled level by level against the money
        offered to lay (the lowest price first); the liability is the money at risk."""
        self._n += 1
        fill = Fill(market_id=market_id, selection_id=int(selection_id), side="LAY", order_type="LIMIT",
                    price=float(price), size=round(float(size), 2), bet_id=f"paper-{self._n}")
        book = self._books.get(market_id)
        q = book.runners.get(int(selection_id)) if book else None
        if not book or book.status != "OPEN" or book.inplay or q is None or q.status != "ACTIVE":
            fill.status, fill.error = "FAILURE", "MARKET_NOT_OPEN_FOR_BETTING"
            return fill
        left, cost, got, takes = fill.size, 0.0, 0.0, []
        for p, s in q.lay:                                      # best (lowest) first
            if p - 1e-9 > price or left <= 0:
                break
            take = min(s, left)
            got += take
            cost += take * p
            left -= take
            takes.append((p, take))
        if got + 1e-9 < min(min_fill, fill.size):
            fill.status, fill.error = "FAILURE", "NOT_MATCHED"
            return fill
        fill.matched, fill.avg_price, fill.status = round(got, 2), round(cost / got, 4), "SUCCESS"
        liability = got * (fill.avg_price - 1.0)
        if self.balance is not None and liability > self.balance - self.open_stakes() + 0.005:
            fill.matched, fill.avg_price, fill.status, fill.error = 0.0, 0.0, "FAILURE", "INSUFFICIENT_FUNDS"
            return fill
        for p, take in takes:                                   # each level taken stays taken
            self.data.taken_lay[(market_id, int(selection_id), round(float(p), 2))] += take
        self.fills.append((market_id, liability, self.data.now))
        self.peak = max(self.peak, self.open_stakes())
        return fill


# ---------------------------------------------------------------------------------------------------------------
# one book for several agents (the owner, 4 Nov 2026: "We need to be careful that they dont conflict with each other
# e.g. taking the price for one, removes the price for others")
# ---------------------------------------------------------------------------------------------------------------

class SharedReplayData(ReplayData):
    """The recorded books as several sessions see them together. The live trader's own fills are already out of the
    books (the trader read them after its orders; a replay of the live rule nets them, as ReplayData does); every
    other agent's takes come out on top, for everyone after it: ``agent_taken`` on the backers' side (no netting:
    the real trader never took them), the lay side's ``taken_lay`` as before. ``resting``: the agents' resting
    quotes at each price, so a later quote at the same price queues behind an earlier one."""

    def __init__(self, D, stale: timedelta = STALE):
        super().__init__(D, stale=stale)
        self.agent_taken = defaultdict(float)
        self.resting = defaultdict(float)                  # (market, selection, side, price) -> GBP resting

    def books(self, ids, with_sp=False):
        out = super().books(ids, with_sp)
        if not self.agent_taken:
            return out
        for mid, b in out.items():
            for sid, q in b.runners.items():
                back = []
                for p, sz in q.back:
                    left = sz - self.agent_taken.get((mid, sid, round(p, 2)), 0.0)
                    if left > 0.01:
                        back.append((p, round(left, 2)))
                q.back = back
        return out


class AgentExchange(ReplayExchange):
    """An agent's own account on the shared book: what it takes from the backers' side comes out level by level
    for every agent after it (and never nets against the live trader's fills, which it did not make)."""

    def back(self, market_id, selection_id, price, size, ref, min_fill: float = 2.0):
        if self.balance is not None and size > self.balance - self.open_stakes() + 0.005:
            return Fill(market_id=market_id, selection_id=int(selection_id), side="BACK", order_type="LIMIT",
                        price=float(price), size=round(float(size), 2), status="FAILURE",
                        error="INSUFFICIENT_FUNDS (ERROR_IN_ORDER)")
        book = self._books.get(market_id)
        q = book.runners.get(int(selection_id)) if book else None
        levels = list(q.back) if q is not None else []
        f = PaperExchange.back(self, market_id, selection_id, price, size, ref, min_fill)
        if f.status == "SUCCESS":
            left = f.matched
            for p, s in levels:                                 # best (highest) first, as the fill took them
                if p + 1e-9 < price or left <= 1e-9:
                    break
                take = min(s, left)
                self.data.agent_taken[(market_id, int(selection_id), round(float(p), 2))] += take
                left -= take
            self.fills.append((market_id, f.matched, self.data.now))
            self.peak = max(self.peak, self.open_stakes())
        return f


def run_agents(D, specs: list[tuple], t0: datetime, t_end: datetime | None = None, stale: timedelta = STALE,
               step_minutes: int = 1) -> dict:
    """Several sessions on one book, a minute at a time: ``specs`` [(name, session class, cfg, policy, balance,
    mirror)]; the ``mirror`` session (the live rule's replay) steps first each minute, as the live trader's own fills
    came first, then the others in an order that turns each minute, so none always has the first pick. Returns
    {name: (ledger, session, exchange)}."""
    data = SharedReplayData(D, stale=stale)
    clock_t = [t0]
    runs = []
    for name, cls, cfg, policy, balance, mirror in specs:
        xc = ReplayExchange if mirror else AgentExchange
        x = xc(data, bank=cfg.limits.bank, balance=balance)
        s = cls(x, D.preds[["market_id", "selection_id", "predicted_bfsp"]], cfg, D.day,
                clock=lambda: clock_t[0], sleep=lambda _: None, **({"policy": policy} if policy is not None else {}))
        runs.append((name, s, x, mirror))
    out = {}
    with shared_plans():
        for _, s, _, _ in runs:
            s.load_markets()
        if not any(s.markets for _, s, _, _ in runs):
            return {name: (pd.DataFrame(), s, x) for name, s, x, _ in runs}
        last_off = max(m.start for _, s, _, _ in runs for m in s.markets.values())
        end = t_end if t_end is not None else last_off + timedelta(minutes=40)
        first = [r for r in runs if r[3]]
        rest = [r for r in runs if not r[3]]
        t, k = t0, 0
        while t <= end:
            data.now = t
            order = first + (rest[k % len(rest):] + rest[:k % len(rest)] if rest else [])
            for _, s, _, _ in order:
                s.step(t)
            t += timedelta(minutes=step_minutes)
            clock_t[0] = t
            k += 1
        data.now = datetime(D.day.year, D.day.month, D.day.day, 23, 30, tzinfo=timezone.utc)
        for name, s, x, _ in runs:
            s.settle_all(data.now)
            if hasattr(s, "settle_extra"):
                s.settle_extra(data.now)
            out[name] = (pd.DataFrame(s.ledger), s, x)
    return out


# ---------------------------------------------------------------------------------------------------------------
# the plan, cached: the rule's choices rest on the prices, not on the sizes or the target
# ---------------------------------------------------------------------------------------------------------------

_plan_cache: dict = {}
_orig_plan_race = strategy.plan_race
UNIT = tc.load_config(str(ROOT / "trading" / "config_live.json"), env={})
UNIT.limits.max_stake = 1e12


def set_plan_cfg(cfg, reset: bool = True) -> None:
    """The config the plans are made with (the bar, the guard, the closing model); the arms share the plans of one
    config, so a new one (the place arm's model) empties the cache."""
    global UNIT
    cfg.limits.max_stake = 1e12
    if reset or getattr(cfg, "closing_model_novol", None) != getattr(UNIT, "closing_model_novol", None):
        _plan_cache.clear()
    UNIT = cfg


def cached_plan(views, cfg, bank):
    key = tuple((v.selection_id, v.back, v.lay, v.last_traded, v.model_price, v.status) for v in views)
    if key not in _plan_cache:
        _plan_cache[key] = _orig_plan_race(views, UNIT, bank)
    size = {v.selection_id: v.back_size for v in views}
    out = []
    for d in _plan_cache[key]:
        d2 = replace(d)
        d2.target = min(d.target * cfg.target / UNIT.target, cfg.limits.max_stake)
        d2.back_size = size.get(d.selection_id, d.back_size)
        out.append(d2)
    return out


import trading.session as _ts  # noqa: E402


@contextmanager
def shared_plans():
    """The live Session planning through the arms' cache, for the span of one replay only: the trader's own
    plan_race is put back after, so importing this package (the tests collect it) leaves trading.session as it was."""
    before = _ts.plan_race
    _ts.plan_race = cached_plan
    try:
        yield
    finally:
        _ts.plan_race = before


def make_cfg(**over):
    cfg = tc.load_config(str(ROOT / "trading" / "config_live.json"), env={})
    for k, v in over.items():
        if k.startswith("limits."):
            setattr(cfg.limits, k[7:], v)
        else:
            setattr(cfg, k, v)
    return cfg


def hours_uk(now: datetime, since: str = "06:15") -> float:
    uk = now.astimezone(UK)
    h, m = since.split(":")
    return (uk.hour * 60 + uk.minute - int(h) * 60 - int(m)) / 60.0


# ---------------------------------------------------------------------------------------------------------------
# running a session over a day, and scoring its ledger
# ---------------------------------------------------------------------------------------------------------------

def run_session(D, session_cls, cfg, policy: dict | None, t0: datetime, t_end: datetime | None = None,
                balance: float = BALANCE, stale: timedelta = STALE, step_minutes: int = 1):
    """The day replayed a minute at a time from t0 through the live Session (or an arm's subclass), then settled
    from the closed books. Returns (the ledger, the session, the exchange)."""
    data = ReplayData(D, stale=stale)
    clock_t = [t0]
    x = ReplayExchange(data, bank=cfg.limits.bank, balance=balance)
    s = session_cls(x, D.preds[["market_id", "selection_id", "predicted_bfsp"]], cfg, D.day,
                    clock=lambda: clock_t[0], sleep=lambda _: None, **({"policy": policy} if policy is not None else {}))
    with shared_plans():
        s.load_markets()
        if not s.markets:
            return pd.DataFrame(), s, x
        last_off = max(m.start for m in s.markets.values())
        end = t_end if t_end is not None else last_off + timedelta(minutes=40)
        t = t0
        while t <= end:
            data.now = t
            s.step(t)
            t += timedelta(minutes=step_minutes)
            clock_t[0] = t
        data.now = datetime(D.day.year, D.day.month, D.day.day, 23, 30, tzinfo=timezone.utc)
        s.settle_all(data.now)
        if hasattr(s, "settle_extra"):
            s.settle_extra(data.now)
    return pd.DataFrame(s.ledger), s, x


def _clv(g: pd.DataFrame) -> float:
    return float((g.clv_fill * g.matched).sum() / g.matched.sum()) if len(g) and g.matched.sum() > 0 else np.nan


def score(led: pd.DataFrame, D, name: str, settled_pnl: float, day_limit: bool) -> dict:
    """A back ledger scored: CLV against the BSP at the price matched, the first fills and the top-ups apart, the
    result after commission, and the same after the non-runners' reduction factors (the honest figure)."""
    need = ["ts", "event", "result", "error", "market_id", "selection_id", "matched", "avg_price", "bsp", "clv", "pnl"]
    led = led.reindex(columns=list(dict.fromkeys([*led.columns, *need])))
    st = led[(led.event == "settle") & (led.result != "REMOVED")].copy()
    for c in ("matched", "clv", "pnl", "bsp"):
        st[c] = pd.to_numeric(st[c], errors="coerce")
    ok = st[np.isfinite(st.clv) & (st.matched > 0)]
    backs = led[(led.event == "back") & (pd.to_numeric(led.matched, errors="coerce") > 0)][
        ["ts", "market_id", "selection_id", "matched", "avg_price"]].copy()
    backs["matched"], backs["avg_price"] = (pd.to_numeric(backs[c], errors="coerce") for c in ("matched", "avg_price"))
    backs["k"] = backs.groupby(["market_id", "selection_id"]).cumcount()
    backs = backs.merge(ok[["market_id", "selection_id", "bsp", "result"]].drop_duplicates(["market_id", "selection_id"]),
                        on=["market_id", "selection_id"])
    backs["clv_fill"] = backs.avg_price / backs.bsp - 1
    first, tops = backs[backs.k == 0], backs[backs.k > 0]
    held = led[(led.event == "skip") & (led.error == HELD)][["market_id", "selection_id"]].drop_duplicates()
    red = reduced(backs, D)
    out = {"day": str(D.day), "arm": name, "horses": int(len(ok)), "staked": round(float(ok.matched.sum()), 2),
           "clv": round(float((ok.clv * ok.matched).sum() / ok.matched.sum()), 4) if len(ok) else None,
           "clv_x_stake": round(float((ok.clv * ok.matched).sum()), 2),
           "first_staked": round(float(first.matched.sum()), 2), "first_clv": round(_clv(first), 4),
           "topup_staked": round(float(tops.matched.sum()), 2), "topup_clv": round(_clv(tops), 4),
           "result": round(float(settled_pnl), 2), "horses_held": int(len(held)), "day_limit": bool(day_limit),
           **red}
    out["result_reduced"] = round(out["result"] - 0.98 * red["winners_cut"], 2)
    return out


def reduced(backs: pd.DataFrame, D) -> dict:
    """Each fill's price cut by the reduction factors of the runners removed after it was matched (Betfair's own
    rule), the CLV x stake at the cut prices and what the winners' cuts cost."""
    rem = D.removed
    if backs.empty:
        return {"clv_x_stake_reduced": 0.0, "winners_cut": 0.0}
    t = pd.to_datetime(backs.ts, utc=True, errors="coerce")
    fac = []
    for r, tt in zip(backs.itertuples(), t):
        x = rem[(rem.market_id == r.market_id) & (rem.selection_id != r.selection_id) & (rem.rt > tt)] if len(rem) else rem
        fac.append(float(np.prod(1 - x.af.to_numpy() / 100.0)) if len(x) else 1.0)
    p_red = np.maximum(1.01, backs.avg_price.to_numpy() * np.array(fac))
    clv_red = float((backs.matched * (p_red / backs.bsp - 1)).sum())
    won = backs.result.eq("WINNER").to_numpy()
    cut = float((backs.matched.to_numpy() * (backs.avg_price.to_numpy() - p_red))[won].sum())
    return {"clv_x_stake_reduced": round(clv_red, 2), "winners_cut": round(cut, 2)}


def score_side(rows: pd.DataFrame, D, name: str, side: str = "lay") -> dict:
    """A list of positions (market_id, selection_id, ts, matched, avg_price, bsp, result, liability) closed at the
    BSP, scored: the expected profit of the hedge at the price matched and after the reduction factors, and the
    result after 2% on each market's net."""
    if rows.empty:
        return {"day": str(D.day), "arm": name, "horses": 0, "staked": 0.0, "liability": 0.0, "clv": None,
                "clv_x_stake": 0.0, "clv_x_stake_reduced": 0.0, "result": 0.0, "result_reduced": 0.0,
                "clv_on_liability": None}
    rows = rows[rows.result.ne("REMOVED") & (rows.bsp > 1) & (rows.matched > 0)].copy()
    sign = -1.0 if side == "lay" else 1.0            # a lay closed at the BSP returns stake x (1 - price / BSP)
    rows["hedge"] = sign * rows.matched * (rows.avg_price / rows.bsp - 1.0)
    rem = D.removed
    t = pd.to_datetime(rows.ts, utc=True, errors="coerce")
    fac = []
    for r, tt in zip(rows.itertuples(), t):
        x = rem[(rem.market_id == r.market_id) & (rem.selection_id != r.selection_id) & (rem.rt > tt)] if len(rem) else rem
        fac.append(float(np.prod(1 - x.af.to_numpy() / 100.0)) if len(x) else 1.0)
    rows["p_red"] = np.maximum(1.01, rows.avg_price.to_numpy() * np.array(fac))
    rows["hedge_reduced"] = sign * rows.matched * (rows.p_red / rows.bsp - 1.0)
    by_mkt = rows.groupby("market_id")[["hedge", "hedge_reduced"]].sum()
    res = float((by_mkt.hedge - 0.02 * by_mkt.hedge.clip(lower=0)).sum())
    res_red = float((by_mkt.hedge_reduced - 0.02 * by_mkt.hedge_reduced.clip(lower=0)).sum())
    staked = float(rows.matched.sum())
    liab = float(rows.liability.sum()) if "liability" in rows else staked
    return {"day": str(D.day), "arm": name, "horses": int(rows.groupby(["market_id", "selection_id"]).ngroups),
            "staked": round(staked, 2), "liability": round(liab, 2),
            "clv": round(float(rows.hedge.sum() / staked), 4) if staked > 0 else None,
            "clv_x_stake": round(float(rows.hedge.sum()), 2),
            "clv_x_stake_reduced": round(float(rows.hedge_reduced.sum()), 2),
            "clv_on_liability": round(float(rows.hedge.sum() / liab), 4) if liab > 0 else None,
            "result": round(res, 2), "result_reduced": round(res_red, 2)}


def session_start(D, kind: str) -> datetime:
    """The replay's first minute: 05:00 UTC on the day (the live replays'), or 17:00 UK the evening before."""
    d = D.day
    if kind == "evening":
        return (datetime(d.year, d.month, d.day, 17, 0, tzinfo=UK) - timedelta(days=1)).astimezone(timezone.utc)
    return datetime(d.year, d.month, d.day, 5, 0, tzinfo=timezone.utc)


def fmt_money(x) -> str:
    return f"GBP{x:,.0f}" if x is not None and np.isfinite(x) else "-"


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_sizes(D, kind: str) -> dict:
    """How much the books carried: polls, markets with a two-sided quote, the spread, by UK hour (the evening arm's
    first question: is there a book at all)."""
    rows = []
    for mid, polls in D.polls.items():
        for t, b in polls:
            if b.status != "OPEN":
                continue
            two = [q for q in b.runners.values() if q.status == "ACTIVE" and q.back and q.lay]
            act = [q for q in b.runners.values() if q.status == "ACTIVE"]
            sp = [q.lay[0][0] / q.back[0][0] - 1 for q in two]
            bs = [q.back[0][1] for q in two]
            rows.append({"hour": t.astimezone(UK).strftime("%H"), "market_id": mid, "runners": len(act),
                         "two_sided": len(two), "spread": float(np.median(sp)) if sp else np.nan,
                         "back_size": float(np.median(bs)) if bs else np.nan,
                         "matched": b.total_matched if b.total_matched is not None else np.nan})
    if not rows:
        return {}
    f = pd.DataFrame(rows)
    g = f.groupby("hour").agg(polls=("market_id", "size"), markets=("market_id", "nunique"),
                              runners=("runners", "mean"), two_sided=("two_sided", "mean"),
                              spread=("spread", "median"), back_size=("back_size", "median"),
                              matched=("matched", "median")).round(3)
    return g.to_dict("index")
