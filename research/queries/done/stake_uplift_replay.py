"""Should the to-win target go from GBP250 to GBP300, 350, 400 or 500? 1 Oct replayed through its own books.

The owner's question (2 Oct): is there the liquidity to win more on each horse? The trader backs at the best price
shown, filled at once or not at all, and tops a horse up each minute while its expected CLV stays at +3% or more; in
the morning only GBP10-25 is shown at the best price (1 Oct: median GBP11.6 on the horses backed), so a horse fills
over many minutes, and a bigger target needs more of them. This replays the day minute by minute through the books
Betfair showed (s3://$PRED_BUCKET/betfair_live/2026-10-01/: the trader's own reads, each minute, and the recorder's)
with the same rule, closing model, staking and limits, at each target:

- each minute's back fills against the book as recorded, at the price shown or better, up to the size shown;
- what the replay takes beyond what the real trader took at a price is kept off the book afterwards (an offer taken
  once is gone: the recorded books were left by the real orders, not the replay's);
- every race settled at its BSP from the settled book, each back laid at the SP for its winnings, 2% commission;
- the account's funds are not modelled here (the replay never runs short): the money each target needs at the
  busiest moment, the stakes of races not yet run, is reported beside it.

The rule's choices at each minute do not depend on the target, only the stakes do, so each race-minute is planned
once and the targets reuse it. The replay fills every order shown (1 Oct's real trader had 47 of 411 backs killed
as the price went): read the targets against each other and against the replay at GBP250, not against the account.
Read-only.
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

import boto3
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from trading import config as tc  # noqa: E402
from trading import strategy  # noqa: E402
from trading.exchange import Book, Market, PaperExchange, Quote  # noqa: E402
from trading.matching import attach_ids  # noqa: E402
from trading.session import Session  # noqa: E402

DAY = date(2026, 10, 1)
TARGETS = (250.0, 300.0, 350.0, 400.0, 500.0)
STALE = timedelta(minutes=6)                     # a book older than this is no book (the gaps between the sessions)
bucket = os.environ.get("PRED_BUCKET") or "ashcroft"
s3 = boto3.client("s3", region_name=os.environ.get("PRED_REGION") or "eu-west-2")
OUT = Path("out/stake_uplift")
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


print(f"1 Oct's books from s3://{bucket}/betfair_live/{DAY}/")
books = pd.concat([_csv(f"betfair_live/{DAY}/{n}.csv.gz") for n in ("books_trader", "books")], ignore_index=True)
cat = pd.concat([_csv(f"betfair_live/{DAY}/{n}.csv.gz") for n in ("markets_trader", "markets")], ignore_index=True)
ledger = _csv(f"trading/live/{DAY}/ledger.csv")
preds = pd.read_csv(io.BytesIO(_get(f"predictions/{DAY}.csv")), dtype={"race_time": str})
print(f"  {len(books):,} book rows ({books['source'].value_counts().to_dict()}), {len(cat):,} catalogue rows, "
      f"{len(ledger):,} ledger rows, {len(preds)} prices")

books["t"] = pd.to_datetime(books["polled_utc"], utc=True)
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


def _levels(r, side):
    out = []
    for i in (1, 2, 3):
        p, s = r[f"{side}{i}"], r[f"{side}{i}_size"]
        if pd.notna(p) and pd.notna(s) and p > 1 and s > 0:
            out.append((float(p), float(s)))
    return out


# every recorded poll of every market as a Book, in time order
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

poll_times = {mid: [t for t, _ in v] for mid, v in polls.items()}

# the real trader's fills at each limit price, to know which offers the recorded books had already lost
real = ledger[(ledger["event"] == "back") & (pd.to_numeric(ledger["matched"], errors="coerce") > 0)].copy()
real["t"] = pd.to_datetime(real["ts"], utc=True)
real_fills = defaultdict(list)                     # (mid, sid, price) -> [(t, matched)]
for r in real.itertuples(index=False):
    real_fills[(str(r.market_id), int(r.selection_id), round(float(r.best_back), 2))].append((r.t, float(r.matched)))


class ReplayData:
    """Betfair as the day's recordings showed it, at the replay's clock; offers the replay took stay taken."""

    def __init__(self):
        self.now = None
        self.taken = defaultdict(float)            # (mid, sid, price) -> what the replay matched there

    def login(self):
        return None

    def markets(self, start, end, countries=("GB", "IE")):
        return [m for m in markets if start <= m.start < end and m.country in countries]

    def _at(self, mid):
        times = poll_times.get(mid)
        if not times:
            return None
        i = bisect.bisect_right(times, self.now) - 1
        if i < 0:
            return None
        t, b = polls[mid][i]
        if b.status == "CLOSED":                   # the settled book stands for good
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
                    before = sum(m for t, m in real_fills.get(k, []) if t <= self.now)
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
    def back(self, market_id, selection_id, price, size, ref, min_fill: float = 2.0):
        f = super().back(market_id, selection_id, price, size, ref, min_fill)
        if f.status == "SUCCESS":
            self.data.taken[(market_id, int(selection_id), round(float(price), 2))] += f.matched
        return f


# the rule's choices for each race at each minute: they rest on the prices, not on the sizes or the target, so each is
# planned once (uncapped, at GBP250 to win) and every target rescales the stakes and takes the sizes it sees
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


def replay(target: float, max_stake: float = 300.0, max_day: float = 4000.0) -> dict:
    cfg = tc.load_config("trading/config_live.json", env={})
    cfg.target, cfg.limits.max_stake, cfg.limits.max_daily_turnover = target, max_stake, max_day
    data = ReplayData()
    clock_t = [datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc)]
    x = ReplayExchange(data, bank=cfg.limits.bank)
    s = Session(x, preds[["market_id", "selection_id", "predicted_bfsp"]], cfg, DAY, clock=lambda: clock_t[0],
                sleep=lambda _: None)
    s.load_markets()
    last_off = max(m.start for m in s.markets.values())
    t = clock_t[0]
    while t <= last_off + timedelta(minutes=40):
        data.now = t
        s.step(t)
        t += timedelta(minutes=1)
        clock_t[0] = t
    data.now = t + timedelta(hours=12)                 # the settled books, recorded after the last race
    s.settle_all(t)
    pos = [p for p in s.positions.values() if p.matched > 0]
    # the money the target needs at the busiest moment: stakes on races not yet run (Betfair returns a race's
    # stakes some 10 minutes after its off); each stake counted from the first minute it was matched
    first_fill = {}
    for r in s.ledger:
        if r["event"] == "back" and float(r.get("matched") or 0) > 0:
            first_fill.setdefault((r["market_id"], int(r["selection_id"])), []).append(
                (pd.Timestamp(r["ts"]).to_pydatetime(), float(r["matched"])))
    starts = {mid: m.start for mid, m in s.markets.items()}
    peak = 0.0
    for minute in pd.date_range(datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc), last_off, freq="1min"):
        open_stake = sum(m for (mid, _), fills in first_fill.items() for ft, m in fills
                         if ft <= minute < starts[mid] + timedelta(minutes=10))
        peak = max(peak, open_stake)
    exposure_peak = peak
    win = np.array([p.matched * (p.avg_price - 1) for p in pos])
    can_win = np.array([min(target, max_stake * (p.avg_price - 1)) for p in pos])
    settled = [r for r in s.ledger if r["event"] == "settle"]
    st = np.array([float(r["matched"]) for r in settled])
    cv = np.array([float(r["clv"]) if r.get("clv") is not None and np.isfinite(float(r["clv"])) else np.nan
                   for r in settled])
    ok = np.isfinite(cv)
    out = {"target": target, "max_stake": max_stake, "max_day": max_day, "horses": len(pos),
           "staked": round(float(sum(p.matched for p in pos)), 2),
           "reached_95pc": round(float(np.mean(win >= 0.95 * can_win)), 3),
           "median_fill": round(float(np.median(win / can_win)), 3),
           "clv": round(float(np.sum(st[ok] * cv[ok]) / np.sum(st[ok])), 4) if ok.any() else None,
           "pnl_after_commission": round(float(s.state.settled_pnl), 2), "races_settled": len({r["market_id"] for r in settled}),
           "peak_open_stakes": round(exposure_peak, 2), "day_limit_reached": s.state.turnover >= max_day - 2}
    pd.DataFrame(s.ledger).to_csv(OUT / f"ledger_{int(target)}_{int(max_stake)}_{int(max_day)}.csv", index=False)
    return out


rows = []
for T in TARGETS:
    rows.append(replay(T))
    print(rows[-1], flush=True)
for T in (400.0, 500.0):                            # the owner's limits lifted, to see what they hold back
    rows.append(replay(T, max_stake=600.0, max_day=8000.0))
    print(rows[-1], flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "targets.csv", index=False)
print()
print(res.to_string(index=False))
real_pos = real.groupby(["market_id", "selection_id"]).agg(staked=("matched", "sum"))
print(f"\nthe real day: {len(real_pos)} horses, GBP{real_pos.staked.sum():,.2f} staked (funds held it back from "
      f"10:19 UK to 16:35 UK; the replay never runs short)")
