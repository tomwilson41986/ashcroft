"""The owner's question (7 Oct): after what time should the trader stop? Time, not money, as the decision point.

The owner chose GBP400 to win with the GBP4,000 day limit, then asked for a stop time in place of the money. Each
recorded day with Betfair's price file (1-6 Oct) is replayed minute by minute through the books the trader and the
recorder kept, with the live rule (expected CLV >= 3% under the closing model), GBP400 to win, GBP300 a bet, and the
account's money held as Betfair holds it (as research/queries/done/stake_target_replay.py), settled from the price
files. The arms:

- a stop time and no money limit: no new back after 10:00, 11:00, ... 18:00 UK, or to the last race (21:30; always
  15 minutes before each off), with GBP5,000 in the account;
- the owner's choice: GBP4,000 a day, to the last race, GBP5,000;
- the marginal value of each hour: no stop, no limit and money without end, every back scored by the UK hour it was
  entered (the hours compete for nothing, so each hour's CLV x stake is its own).

Also the real fills of 1-6 Oct (the trader's ledgers, GBP250 to win) by the UK hour and the minutes to the off at
entry: their CLV at the BSP the day settled at. Each is scored on CLV x stake (the expected profit of the hedged
backs, before commission) and the result after 2%. Read CLV x stake, not the result: six days of results are mostly
noise. Read-only.
"""

from __future__ import annotations

import bisect
import gzip
import io
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import betfair_prices as bp  # noqa: E402
from trading import config as tc  # noqa: E402
from trading import strategy  # noqa: E402
from trading.exchange import Book, Fill, Market, PaperExchange, Quote  # noqa: E402
from trading.matching import attach_ids  # noqa: E402
from trading.session import HELD, Session  # noqa: E402

DAYS = {date(2026, 10, 1): 1480.11, date(2026, 10, 2): 1668.30, date(2026, 10, 3): 1712.60,
        date(2026, 10, 4): 2019.76, date(2026, 10, 5): 1884.64, date(2026, 10, 6): 1956.62}  # opening balances
NOW_BALANCE = 2577.13                            # the account at the close of 6 Oct (after a deposit of about GBP500)
STOPS = ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "21:30"]
GRID = ([(400.0, 5000.0, 300.0, 1e6, s) for s in STOPS]           # a stop time, no money limit
        + [(400.0, 5000.0, 300.0, 4000.0, "21:30")]                 # the owner's choice
        + [(400.0, None, 300.0, 1e6, "21:30")])                     # each hour's own value
STALE = timedelta(minutes=6)                     # a book older than this is no book (the gaps between the sessions)
pd.set_option("display.width", 250)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
OUT = Path("out/time_cutoff_replay")
OUT.mkdir(parents=True, exist_ok=True)


def _get(key: str) -> bytes | None:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:                                # a file one writer never made
        print(f"  {key}: {type(exc).__name__}")
        return None


def _csv(key: str) -> pd.DataFrame:
    raw = _get(key)
    if raw is None:
        return pd.DataFrame()
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), dtype={"market_id": str, "selection_id": "Int64"}, low_memory=False)


def _levels(r, side):
    out = []
    for i in (1, 2, 3):
        p, s = r[f"{side}{i}"], r[f"{side}{i}_size"]
        if pd.notna(p) and pd.notna(s) and p > 1 and s > 0:
            out.append((float(p), float(s)))
    return out


def load(day: date):
    """The day's books, markets, prices, the real trader's fills, and every runner's BSP and result from Betfair's
    price files (named by the day after the racing); None while the file is not published."""
    print(f"\n== {day}: books from s3://{bucket}/betfair_live/{day}/")
    sp_rows = []
    for d in (day + timedelta(days=1), day):
        for f in (f"dwbfpricesukwin{d:%d%m%Y}.csv", f"dwbfpricesirewin{d:%d%m%Y}.csv"):
            raw = _get(f"{bp.S3_PREFIX}/{f}")
            if raw is None:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            t = bp.parse_file_text(text, f)
            sp_rows.append(t[t["race_date"] == day.isoformat()])
    spf = pd.concat(sp_rows, ignore_index=True) if sp_rows else pd.DataFrame()
    if spf.empty:
        print(f"  Betfair's price file for {day}'s racing is not in the archive yet: the day waits for the next run")
        return None
    books = pd.concat([_csv(f"betfair_live/{day}/{n}.csv.gz") for n in ("books_trader", "books")], ignore_index=True)
    cat = pd.concat([_csv(f"betfair_live/{day}/{n}.csv.gz") for n in ("markets_trader", "markets")], ignore_index=True)
    ledger = _csv(f"trading/live/{day}/ledger.csv")
    preds = pd.read_csv(io.BytesIO(_get(f"predictions/{day}.csv")), dtype={"race_time": str})
    print(f"  {len(books):,} book rows, {len(cat):,} catalogue rows, {len(ledger):,} ledger rows, {len(preds)} prices")
    books["t"] = pd.to_datetime(books["polled_utc"], utc=True)
    cov = books["t"].drop_duplicates().dt.tz_convert("Europe/London").dt.hour.value_counts().sort_index()
    print("  minutes with books by UK hour (a gap is a stretch the replay cannot trade):", cov.to_dict())
    cat = cat[cat["market_type"].astype(str).str.upper().eq("WIN") & cat["country"].isin(["GB", "IE"])]
    cat = cat.drop_duplicates(["market_id", "selection_id"])
    markets = []
    for mid, g in cat.groupby("market_id"):
        start = pd.Timestamp(g["market_start_utc"].iloc[0]).tz_convert("UTC").to_pydatetime()
        markets.append(Market(str(mid), str(g["venue"].iloc[0]), str(g["country"].iloc[0]), start,
                              name=str(g["market_name"].iloc[0]),
                              runners={int(s): str(n) for s, n in zip(g["selection_id"], g["runner_name"])}))
    win_ids = {m.market_id for m in markets}
    books = books[books["market_id"].isin(win_ids)]
    preds = attach_ids(preds, markets)
    preds = preds[preds["market_id"].notna()]
    print(f"  {len(markets)} GB/IE win markets; {len(preds)} prices matched to them")

    polls: dict[str, list[tuple[datetime, Book]]] = defaultdict(list)
    for (mid, t), g in books.groupby(["market_id", "t"], sort=True):
        runners = {}
        for r in g.itertuples(index=False):
            rr = r._asdict()
            bsp = rr.get("sp_actual")
            runners[int(rr["selection_id"])] = Quote(
                int(rr["selection_id"]), status=str(rr.get("runner_status") or "ACTIVE"),
                back=_levels(rr, "back"), lay=_levels(rr, "lay"),
                last_traded=rr.get("last_traded") if pd.notna(rr.get("last_traded")) else None,
                traded=float(rr.get("runner_total_matched") or 0.0) if pd.notna(rr.get("runner_total_matched")) else 0.0,
                bsp=float(bsp) if pd.notna(bsp) and bsp > 1 else None)
        first = g.iloc[0]
        tm = first.get("market_total_matched")
        polls[str(mid)].append((t.to_pydatetime(), Book(str(mid), str(first["status"]), bool(first["inplay"]),
                                                        float(tm) if pd.notna(tm) else None, runners,
                                                        bsp_reconciled=bool(first.get("bsp_reconciled")))))

    # the files matched to the exchange's markets by course, UK off time and horse, then by time and horse
    day_tracks = {bp.normalise_track(m.venue): m.venue for m in markets}
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
    n_settled = 0
    for m in markets:
        g = spm[spm["market_id"] == m.market_id]
        runners = {int(r.selection_id): Quote(int(r.selection_id), status="WINNER" if r.win_lose == 1 else "LOSER",
                                              bsp=float(r.bsp)) for r in g.itertuples() if pd.notna(r.bsp) and r.bsp > 1}
        n_settled += bool(runners)
        polls[m.market_id] = [p for p in polls.get(m.market_id, []) if p[1].status != "CLOSED"]
        polls[m.market_id].append((settled_at, Book(m.market_id, "CLOSED", False, None, runners, True)))
    print(f"  settled books for {n_settled} of {len(markets)} markets")

    real = ledger[(ledger["event"] == "back") & (pd.to_numeric(ledger["matched"], errors="coerce") > 0)].copy()
    real["t"] = pd.to_datetime(real["ts"], utc=True)
    real_fills = defaultdict(list)                     # (mid, sid, price) -> [(t, matched)]
    for r in real.itertuples(index=False):
        real_fills[(str(r.market_id), int(r.selection_id), round(float(r.best_back), 2))].append((r.t, float(r.matched)))
    return SimpleNamespace(day=day, markets=markets, polls=polls, poll_times={k: [t for t, _ in v] for k, v in polls.items()},
                           starts={m.market_id: m.start for m in markets}, real_fills=real_fills, preds=preds)


D = None                                               # the day being replayed


class ReplayData:
    """Betfair as the day's recordings showed it, at the replay's clock; offers the replay took stay taken."""

    def __init__(self):
        self.now = None
        self.taken = defaultdict(float)

    def login(self):
        return None

    def markets(self, start, end, countries=("GB", "IE")):
        return [m for m in D.markets if start <= m.start < end and m.country in countries]

    def _at(self, mid):
        times = D.poll_times.get(mid)
        if not times:
            return None
        i = bisect.bisect_right(times, self.now) - 1
        if i < 0:
            return None
        t, b = D.polls[mid][i]
        if b.status == "CLOSED":
            return b
        return b if self.now - t <= STALE else None

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
                    before = sum(m for t, m in D.real_fills.get(k, []) if t <= self.now)
                    gone = max(0.0, self.taken[k] - before)
                    if s - gone > 0.01:
                        back.append((p, round(s - gone, 2)))
                runners[sid] = Quote(sid, q.status, back, list(q.lay), q.last_traded, q.traded, q.bsp)
            out[mid] = Book(b.market_id, b.status, b.inplay, b.total_matched, runners, b.bsp_reconciled)
        return out

    def cleared(self, ids):
        return []

    def funds(self):
        return None


class ReplayExchange(PaperExchange):
    """Paper fills against the recorded books, and the account's money as Betfair holds it: a back larger than the
    funds is refused, and a race's stakes come back ten minutes after its off."""

    def __init__(self, data, bank, balance=None):
        super().__init__(data, bank=bank)
        self.balance = balance
        self.fills = []                            # (market_id, stake, when matched)
        self.peak = 0.0                            # the most money open at once

    def open_stakes(self) -> float:
        return sum(s for mid, s, _ in self.fills if self.data.now < D.starts[mid] + timedelta(minutes=10))

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


class PolicySession(Session):
    """The live session with a policy on top-ups: none, a higher bar, or new horses first with money kept for them."""

    def __init__(self, *a, policy=None, **k):
        super().__init__(*a, **k)
        self.policy = policy or {}
        self._defer = None

    def execute(self, m, book, d, now):
        sp = self.policy.get("short_price")
        if sp is not None and d.price < sp and not (np.isfinite(d.edge) and d.edge >= self.policy["short_bar"]):
            return                                              # a short price below its own, higher bar
        pos = self.positions.get((m.market_id, d.selection_id))
        if pos is not None and pos.matched > 0:                 # a top-up
            pol = self.policy
            if pol.get("no_topups"):
                return
            bar = pol.get("topup_bar")
            if bar is not None and not (np.isfinite(d.edge) and d.edge >= bar):
                return
            if self._defer is not None:                         # new horses first: after this minute's new backs
                self._defer.append((m, book, d, now))
                return
            if pol.get("reserve") is not None and d.price > 1.0:
                left = self.cfg.target - pos.matched * (pos.avg_price - 1.0)
                want = max(0.0, min(left / (d.price - 1.0), self.cfg.limits.max_stake - pos.matched))
                f = self.x.funds()
                if f is not None and f["available"] - want < pol["reserve"]:
                    return
        super().execute(m, book, d, now)

    def step(self, now=None):
        if not self.policy.get("new_first"):
            return super().step(now)
        self._defer = []
        try:
            super().step(now)
        finally:
            deferred, self._defer = self._defer, None
        for m, book, d, t in deferred:
            self.execute(m, book, d, t)


# the rule's choices for each race at each minute rest on the prices, not on the sizes or the target: each is planned
# once (uncapped) and every replay takes the sizes it sees
_plan_cache: dict = {}
_orig_plan_race = strategy.plan_race
_UNIT = tc.load_config("trading/config_live.json", env={})
_UNIT.limits.max_stake = 1e12


def cached_plan(views, cfg, bank):
    key = tuple((v.selection_id, v.back, v.lay, v.last_traded, v.model_price, v.status) for v in views)
    if key not in _plan_cache:
        _plan_cache[key] = _orig_plan_race(views, _UNIT, bank)
    size = {v.selection_id: v.back_size for v in views}
    out = []
    for d in _plan_cache[key]:
        d2 = type(d)(**{**d.__dict__})
        d2.target = min(d.target * cfg.target / _UNIT.target, cfg.limits.max_stake)
        d2.back_size = size.get(d.selection_id, d.back_size)
        out.append(d2)
    return out


import trading.session as ts_mod  # noqa: E402
ts_mod.plan_race = cached_plan


def _clv(g: pd.DataFrame) -> float:
    return float((g.clv_fill * g.matched).sum() / g.matched.sum()) if len(g) and g.matched.sum() > 0 else np.nan


def replay(day: date, target: float, balance, max_stake: float, max_day: float, until: str) -> dict:
    cfg = tc.load_config("trading/config_live.json", env={})
    cfg.target, cfg.limits.max_stake, cfg.limits.max_daily_turnover = target, max_stake, max_day
    cfg.trade_from, cfg.trade_until, cfg.stop_before_off = "08:00", until, 15
    data = ReplayData()
    t0 = datetime(day.year, day.month, day.day, 6, 30, tzinfo=timezone.utc)
    clock_t = [t0]
    x = ReplayExchange(data, bank=cfg.limits.bank, balance=balance)
    s = PolicySession(x, D.preds[["market_id", "selection_id", "predicted_bfsp"]], cfg, day,
                      clock=lambda: clock_t[0], sleep=lambda _: None, policy={})
    s.load_markets()
    last_off = max(m.start for m in s.markets.values())
    t = t0
    while t <= last_off + timedelta(minutes=40):
        data.now = t
        s.step(t)
        t += timedelta(minutes=1)
        clock_t[0] = t
    data.now = t + timedelta(hours=12)                 # the settled books, recorded after the last race
    s.settle_all(t)
    led = pd.DataFrame(s.ledger)
    name = f"{int(target)}_{int(max_stake)}_{int(min(max_day, 1e6))}_{until.replace(':', '')}"
    out = score(led, day, name, balance, s.holds, s.state.turnover >= max_day - 2, s.state.settled_pnl)
    out.update({"target": target, "max_stake": max_stake, "max_day": max_day, "until": until,
                "peak_open": round(x.peak, 2)})
    HOURS.append(by_hour(led, day, until, balance, max_day))
    b = led[(led.get("event") == "back") & (pd.to_numeric(led.get("matched"), errors="coerce") > 0)] if len(led) else led
    out["last_back_uk"] = (pd.to_datetime(b["ts"], utc=True).max().tz_convert("Europe/London").strftime("%H:%M")
                           if len(b) else None)
    led.to_csv(OUT / f"ledger_{day}_{name}_{balance}.csv", index=False)
    return out


def score(led: pd.DataFrame, day, name, balance, holds, day_limit, settled_pnl) -> dict:
    """A replay's ledger scored: CLV against the BSP, the first fills and the top-ups apart, the horses held."""
    need = ["event", "result", "error", "market_id", "selection_id", "matched", "avg_price", "bsp", "clv", "pnl"]
    led = led.reindex(columns=list(dict.fromkeys([*led.columns, *need])))   # a policy that backed nothing logs nothing
    st = led[(led.event == "settle") & (led.result != "REMOVED")].copy()
    st["matched"], st["clv"], st["pnl"] = (pd.to_numeric(st[c], errors="coerce") for c in ("matched", "clv", "pnl"))
    ok = st[np.isfinite(st.clv) & (st.matched > 0)]
    backs = led[(led.event == "back") & (pd.to_numeric(led.matched, errors="coerce") > 0)][
        ["market_id", "selection_id", "matched", "avg_price"]].copy()       # in the ledger's order, which is time's
    backs["matched"], backs["avg_price"] = (pd.to_numeric(backs[c], errors="coerce") for c in ("matched", "avg_price"))
    backs["k"] = backs.groupby(["market_id", "selection_id"]).cumcount()
    backs = backs.merge(ok[["market_id", "selection_id", "bsp"]], on=["market_id", "selection_id"])
    backs["clv_fill"] = backs.avg_price / pd.to_numeric(backs.bsp, errors="coerce") - 1
    first, tops = backs[backs.k == 0], backs[backs.k > 0]
    held = led[(led.event == "skip") & (led.error == HELD)][["market_id", "selection_id"]].drop_duplicates()
    backed = set(map(tuple, backs[["market_id", "selection_id"]].astype(str).to_numpy()))
    out = {"day": str(day), "policy": name, "balance": balance if balance is not None else "none",
           "horses": int(len(ok)), "staked": round(float(ok.matched.sum()), 2),
           "clv": round(float((ok.clv * ok.matched).sum() / ok.matched.sum()), 4) if len(ok) else None,
           "clv_x_stake": round(float((ok.clv * ok.matched).sum()), 2),
           "first_staked": round(float(first.matched.sum()), 2), "first_clv": round(_clv(first), 4),
           "topup_staked": round(float(tops.matched.sum()), 2), "topup_clv": round(_clv(tops), 4),
           "result_before_commission": round(float(st.pnl.sum()), 2),
           "result_after_commission": round(float(settled_pnl), 2), "holds": holds,
           "horses_held": int(len(held)),
           "held_never_backed": int(sum(1 for r in held.astype(str).itertuples(index=False) if tuple(r) not in backed)),
           "day_limit_reached": bool(day_limit)}
    return out


HOURS: list = []


def by_hour(led: pd.DataFrame, day, until, balance, max_day) -> pd.DataFrame:
    """Each matched back by the UK hour it was entered and its minutes to the off: CLV at the BSP the race settled at."""
    if led.empty or "event" not in led:
        return pd.DataFrame()
    st = led[(led.event == "settle") & (led.result != "REMOVED")][["market_id", "selection_id", "bsp"]]
    st = st.drop_duplicates(["market_id", "selection_id"], keep="last")          # a day settled twice counts once
    b = led[(led.event == "back") & (pd.to_numeric(led.matched, errors="coerce") > 0)].copy()
    b = b.merge(st, on=["market_id", "selection_id"], suffixes=("", "_settled"))
    b["m"] = pd.to_numeric(b["matched"], errors="coerce")
    b["clv_fill"] = pd.to_numeric(b["avg_price"], errors="coerce") / pd.to_numeric(b["bsp_settled"], errors="coerce") - 1
    b = b[np.isfinite(b["clv_fill"])]
    b["hour"] = pd.to_datetime(b["ts"], utc=True).dt.tz_convert("Europe/London").dt.hour
    b["to_off"] = pd.to_numeric(b["minutes_to_off"], errors="coerce")
    b["day"], b["until"], b["balance"], b["max_day"] = str(day), until, str(balance), max_day
    return b[["day", "until", "balance", "max_day", "market_id", "selection_id", "hour", "to_off", "m", "avg_price",
              "clv_fill"]]


def table(b: pd.DataFrame, key) -> pd.DataFrame:
    g = b.assign(cx=b.m * b.clv_fill).groupby(key, observed=True)
    t = g.agg(backs=("m", "size"), staked=("m", "sum"), clv_x_stake=("cx", "sum"))
    t["clv"] = (t.clv_x_stake / t.staked).round(4)
    rng = np.random.default_rng(7)                                  # a 90% interval on the CLV, races resampled
    lo, hi = [], []
    for k, part in g:
        races = part.groupby("market_id").agg(cx=("cx", "sum"), m=("m", "sum"))
        if len(races) < 5:
            lo.append(np.nan); hi.append(np.nan); continue
        idx = rng.integers(0, len(races), (400, len(races)))
        r = races.cx.to_numpy()[idx].sum(1) / races.m.to_numpy()[idx].sum(1)
        lo.append(np.quantile(r, 0.05)); hi.append(np.quantile(r, 0.95))
    t["clv_lo90"], t["clv_hi90"] = np.round(lo, 4), np.round(hi, 4)
    t["share_of_clv_x_stake"] = (t.clv_x_stake / t.clv_x_stake.sum()).round(3)
    return t.round(2)


# -- the real fills, 1-6 Oct (GBP250 to win, the day's own money, the limits of the day)
real = []
for day in DAYS:
    led = _csv(f"trading/live/{day}/ledger.csv")
    if led.empty:
        continue
    led["market_id"] = led["market_id"].astype(str)
    real.append(by_hour(led, day, "live", "own", np.nan))
real = pd.concat(real, ignore_index=True) if real else pd.DataFrame()
if len(real):
    print(f"== the real fills, 1-6 Oct: {len(real):,} backs, GBP{real.m.sum():,.2f}, CLV "
          f"{(real.m * real.clv_fill).sum() / real.m.sum():+.4f}")
    print("\n-- by the UK hour of entry")
    print(table(real, "hour").to_string())
    bands = pd.cut(real.to_off, [15, 30, 60, 120, 240, 480, 1e4], right=False,
                   labels=["15-30 min", "30-60", "1-2 h", "2-4 h", "4-8 h", "8 h+"])
    print("\n-- by the minutes to the off at entry")
    print(table(real.assign(band=bands), "band").to_string())
    print("\n-- by day and hour: CLV x stake (GBP)")
    print(real.assign(cx=real.m * real.clv_fill).pivot_table(index="day", columns="hour", values="cx",
                                                             aggfunc="sum").round(0).to_string())

rows = []
for day in DAYS:
    D = load(day)
    if D is None:
        continue
    for target, bal, max_stake, max_day, until in GRID:
        r = replay(day, target, bal, max_stake, max_day, until)
        r["account"] = "without end" if bal is None else f"GBP{bal:,.0f}"
        rows.append(r)
        print({k: r[k] for k in ("day", "account", "max_day", "until", "horses", "staked", "clv", "clv_x_stake",
                                 "result_after_commission", "horses_held", "day_limit_reached", "last_back_uk",
                                 "peak_open")}, flush=True)
if not rows:
    raise SystemExit("no day could be settled")
res = pd.DataFrame(rows)
res.to_csv(OUT / "stops.csv", index=False)
key = ["account", "max_day", "until"]
pooled = res.groupby(key, sort=False).agg(days=("day", "nunique"), horses=("horses", "sum"), staked=("staked", "sum"),
                                          clv_x_stake=("clv_x_stake", "sum"), result=("result_after_commission", "sum"),
                                          held=("horses_held", "sum"), limit_days=("day_limit_reached", "sum"),
                                          peak_open=("peak_open", "max"))
pooled["clv"] = (pooled.clv_x_stake / pooled.staked).round(4)
print("\n== the days together, GBP400 to win (CLV x stake is the expected profit of the hedged backs, before commission)")
print(pooled.round(2).to_string())
print("\n== CLV x stake by day")
print(res.pivot_table(index=key, columns="day", values="clv_x_stake", sort=False).round(0).to_string())
print("\n== the last back (UK) by day")
print(res.pivot_table(index=key, columns="day", values="last_back_uk", aggfunc="first", sort=False).to_string())

hours = pd.concat([h for h in HOURS if len(h)], ignore_index=True)
hours.to_csv(OUT / "backs_by_hour.csv", index=False)
free = hours[(hours.balance == "None") & (hours.until == "21:30")]
print("\n== each hour's own value: GBP400 to win, no stop, no limit, money without end (backs by the UK hour entered)")
print(table(free, "hour").to_string())
bands = pd.cut(free.to_off, [15, 30, 60, 120, 240, 480, 1e4], right=False,
               labels=["15-30 min", "30-60", "1-2 h", "2-4 h", "4-8 h", "8 h+"])
print("\n-- by the minutes to the off at entry")
print(table(free.assign(band=bands), "band").to_string())
print("\n-- by day and hour: CLV x stake (GBP)")
print(free.assign(cx=free.m * free.clv_fill).pivot_table(index="day", columns="hour", values="cx",
                                                         aggfunc="sum").round(0).to_string())
five = hours[(hours.balance == "5000.0") & (hours.max_day >= 1e6) & (hours.until == "21:30")]
print("\n== GBP5,000 in the account, no stop, no limit: by the UK hour entered (the money binds)")
print(table(five, "hour").to_string())
