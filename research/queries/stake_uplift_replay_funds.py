"""The to-win target at GBP250 to 500, replayed through 1 Oct's own books, with the account's money (fourth run).

The first run (research-query run 36965830970, research/queries/done/stake_uplift_replay.py) replayed 1 Oct
minute by minute through the books the trader and the recorder kept and found the day's GBP4,000 limit reached
at every target, but it could not settle: the delayed key's settled books carry no SP, and it never ran short of
money, where the real day was held for funds from 10:19 UK to 16:35 UK. This run:

- settles every race from Betfair's own price files for 1 Oct (s3://$PRED_BUCKET/betfair_prices_raw/: each
  runner's BSP and result), each back laid at the SP for its winnings, 2% commission (the account's rate);
- holds the account's money as Betfair does: a back larger than the funds is refused (INSUFFICIENT_FUNDS) and the
  trader holds and reads the funds again, as live; a race's stakes come back ten minutes after its off;
- runs each target with the balance of 1 Oct (GBP1,480.11, all of it free at 08:00 UK) and with more money in the
  account (GBP3,000 and GBP5,000), under the owner's limits (GBP300 a bet, GBP4,000 a day) and with them lifted;
- scores each: the horses backed, the stake, how many reached their target, the CLV against the BSP, the result
  after commission, the money at the busiest moment, and what the extra stake over GBP250 earned (the marginal
  CLV).

Each minute's back fills against the book as recorded at the price shown or better; an offer the replay takes
beyond the real trader's stays taken. The replay fills what is shown (1 Oct's real trader had 47 of 411 backs
killed as the price went), so read the targets against each other and against the GBP250 replay at the day's
balance, which stands for the real day. Read-only.
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
OUT = Path("out/stake_uplift_funds")
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

# the settled books: every runner's BSP and result from Betfair's price files for the day (the delayed key's own
# settled books carry no SP). The files are matched to the exchange's markets as the trader matches its prices: by
# course, UK off time and horse (their EVENT_ID and SELECTION_ID are not the exchange's ids)
import betfair_prices as bp  # noqa: E402
# Betfair names a day's file by the day after its racing (dwbfpricesukwin01102026.csv holds 30 Sep's races: the fourth
# run), so the files of 2 Oct and 1 Oct are read and only 1 Oct's races kept
sp_rows = []
for d in (DAY + timedelta(days=1), DAY):
    for f in (f"dwbfpricesukwin{d:%d%m%Y}.csv", f"dwbfpricesirewin{d:%d%m%Y}.csv"):
        try:
            when = s3.head_object(Bucket=bucket, Key=f"betfair_prices_raw/{f}")["LastModified"]
        except Exception:
            print(f"  {f}: not in the archive yet")
            continue
        raw = _get(f"betfair_prices_raw/{f}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        t = bp.parse_file_text(text, f)
        print(f"  {f} (archived {when:%Y-%m-%d %H:%M} UTC): {len(t):,} runners, race days "
              f"{sorted(t['race_date'].dropna().unique())}")
        sp_rows.append(t[t["race_date"] == DAY.isoformat()])
spf = pd.concat(sp_rows, ignore_index=True) if sp_rows else pd.DataFrame()
SETTLED_FROM = "Betfair's price files: every runner's BSP and result"
if spf.empty:
    # until the file is published: the real day's own settled bets (listClearedOrders), the BSP and result of each
    # horse the real trader backed; a horse it never backed has no BSP here and is void in the replay
    st_real = ledger[ledger["event"] == "settle"].copy()
    st_real["bsp"] = pd.to_numeric(st_real["bsp"], errors="coerce")
    st_real = st_real[st_real["bsp"] > 1]
    spf = pd.DataFrame({"market_id": st_real["market_id"].astype(str).to_numpy(),
                        "selection_id": st_real["selection_id"].astype(int).to_numpy(),
                        "bsp": st_real["bsp"].to_numpy(),
                        "win_lose": (st_real["result"] == "WINNER").astype(int).to_numpy()})
    SETTLED_FROM = (f"the real day's settled bets: the BSP and result of the {len(spf)} horses the real trader backed; "
                    f"any other horse a replay backs is void, so each replay is scored on those horses")
print(f"  settled from {SETTLED_FROM}")
# the course from the file's menu hint, read against the day's own venues (the third run passed none, and a hint
# naming the course in full, or an abbreviation the map lacks, matched nothing)
day_tracks = {bp.normalise_track(m.venue): m.venue for m in markets}
if "menu_hint" not in spf:                           # the real day's bets carry the exchange's own ids
    spm = spf.assign(file_event_id=pd.NA, file_selection_id=spf["selection_id"])
else:
    spf["venue"] = spf["menu_hint"].map(lambda h: bp.course_from_hint(h, day_tracks))
    print(f"  price-file meetings: {sorted(spf['menu_hint'].dropna().unique())} -> "
          f"{sorted(spf['venue'].dropna().unique())}")
    spf["runner_name"] = spf["selection_name"]
    spf = spf.rename(columns={"event_id": "file_event_id", "selection_id": "file_selection_id"})
    spm = attach_ids(spf[["venue", "race_time", "runner_name", "bsp", "win_lose", "file_event_id",
                          "file_selection_id"]], markets)
    # a runner the course left unmatched is found by its UK off time and name alone, where that pair is unique
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
    print(f"  price files: {len(spf):,} runners; {int((~miss).sum()):,} matched by course, time and horse, "
          f"{int(by_time['market_id'].notna().sum()):,} more by time and horse")
spm = spm[spm["market_id"].notna()].copy()
if spm.empty:
    raise SystemExit("no price-file runner matched the exchange's markets: the replay cannot settle")
same_sid = (spm["file_selection_id"].astype("Int64") == spm["selection_id"].astype("Int64")).mean()
print(f"  the file's SELECTION_ID equals the exchange's on {same_sid:.0%}; e.g. EVENT_ID "
      f"{spm['file_event_id'].iloc[0]} for market {spm['market_id'].iloc[0]}")
settled_at = datetime(2026, 10, 1, 23, 0, tzinfo=timezone.utc)
for m in markets:
    g = spm[spm["market_id"] == m.market_id]
    runners = {int(r.selection_id): Quote(int(r.selection_id), status="WINNER" if r.win_lose == 1 else "LOSER",
                                          bsp=float(r.bsp)) for r in g.itertuples() if pd.notna(r.bsp) and r.bsp > 1}
    polls[m.market_id] = [p for p in polls.get(m.market_id, []) if p[1].status != "CLOSED"]
    polls[m.market_id].append((settled_at, Book(m.market_id, "CLOSED", False, None, runners, True)))
print(f"  settled books for {sum(1 for m in markets if not spm[spm.market_id == m.market_id].empty)} of {len(markets)} markets")
poll_times = {mid: [t for t, _ in v] for mid, v in polls.items()}
starts = {m.market_id: m.start for m in markets}

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
    """Paper fills against the recorded books, and the account's money as Betfair holds it: a back larger than the
    funds is refused, and a race's stakes come back ten minutes after its off (None: never short)."""

    def __init__(self, data, bank, balance=None):
        super().__init__(data, bank=bank)
        self.balance = balance
        self.fills = []                            # (market_id, stake, when matched)

    def open_stakes(self) -> float:
        return sum(s for mid, s, _ in self.fills if self.data.now < starts[mid] + timedelta(minutes=10))

    def funds(self):
        if self.balance is None:
            return None
        return {"available": round(self.balance - self.open_stakes(), 2), "exposure": -round(self.open_stakes(), 2)}

    def back(self, market_id, selection_id, price, size, ref, min_fill: float = 2.0):
        if self.balance is not None and size > self.balance - self.open_stakes() + 0.005:
            from trading.exchange import Fill
            return Fill(market_id=market_id, selection_id=int(selection_id), side="BACK", order_type="LIMIT",
                        price=float(price), size=round(float(size), 2), status="FAILURE",
                        error="INSUFFICIENT_FUNDS (ERROR_IN_ORDER)")
        f = super().back(market_id, selection_id, price, size, ref, min_fill)
        if f.status == "SUCCESS":
            self.data.taken[(market_id, int(selection_id), round(float(price), 2))] += f.matched
            self.fills.append((market_id, f.matched, self.data.now))
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


def replay(target: float, max_stake: float = 300.0, max_day: float = 4000.0, balance=None) -> dict:
    cfg = tc.load_config("trading/config_live.json", env={})
    cfg.target, cfg.limits.max_stake, cfg.limits.max_daily_turnover = target, max_stake, max_day
    data = ReplayData()
    clock_t = [datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc)]
    x = ReplayExchange(data, bank=cfg.limits.bank, balance=balance)
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
    # the money the target needs at the busiest moment: stakes already matched on races not yet ten minutes past
    # their off (the second and third runs counted every fill of the day as open at once)
    peak = 0.0
    for minute in pd.date_range(datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc), last_off, freq="1min"):
        peak = max(peak, sum(st for mid, st, tf in x.fills if tf <= minute < starts[mid] + timedelta(minutes=10)))
    exposure_peak = peak
    win = np.array([p.matched * (p.avg_price - 1) for p in pos])
    can_win = np.array([min(target, max_stake * (p.avg_price - 1)) for p in pos])
    settled = [r for r in s.ledger if r["event"] == "settle"]
    st = np.array([float(r["matched"]) for r in settled])
    cv = np.array([float(r["clv"]) if r.get("clv") is not None and np.isfinite(float(r["clv"])) else np.nan
                   for r in settled])
    ok = np.isfinite(cv)
    pnl_before = float(sum(float(r["pnl"]) for r in settled))
    out = {"target": target, "max_stake": max_stake, "max_day": max_day,
           "balance": balance if balance is not None else "none", "horses": len(pos),
           "staked": round(float(sum(p.matched for p in pos)), 2),
           "staked_settled": round(float(sum(float(r["matched"]) for r in settled if r["result"] != "REMOVED")), 2),
           "horses_void": sum(1 for r in settled if r["result"] == "REMOVED"),
           "reached_95pc": round(float(np.mean(win >= 0.95 * can_win)), 3),
           "clv": round(float(np.sum(st[ok] * cv[ok]) / np.sum(st[ok])), 4) if ok.any() else None,
           "clv_x_stake": round(float(np.sum(st[ok] * cv[ok])), 2),
           "result_before_commission": round(pnl_before, 2),
           "result_after_commission": round(float(s.state.settled_pnl), 2),
           "races_settled": len({r["market_id"] for r in settled}), "holds": s.holds,
           "peak_open_stakes": round(exposure_peak, 2), "day_limit_reached": s.state.turnover >= max_day - 2}
    pd.DataFrame(s.ledger).to_csv(OUT / f"ledger_{int(target)}_{int(max_stake)}_{int(max_day)}_{balance}.csv", index=False)
    return out


rows = []
for balance in (1480.11, 3000.0, 5000.0, None):         # 1 Oct's balance, more money, and none held back
    for T in TARGETS:
        rows.append(replay(T, balance=balance))
        print(rows[-1], flush=True)
for balance in (5000.0, None):                       # the owner's limits lifted, to see what they hold back
    for T in (400.0, 500.0):
        rows.append(replay(T, max_stake=600.0, max_day=8000.0, balance=balance))
        print(rows[-1], flush=True)
res = pd.DataFrame(rows)
base = res[(res.target == 250.0) & (res.max_day == 4000.0)].set_index("balance")
res["marginal_clv_over_250"] = [
    round((r.clv_x_stake - base.at[r.balance, "clv_x_stake"]) / (r.staked_settled - base.at[r.balance, "staked_settled"]),
          4)
    if r.balance in base.index and r.staked_settled - base.at[r.balance, "staked_settled"] > 1 else None
    for r in res.itertuples()]
res.to_csv(OUT / "targets.csv", index=False)
print()
print(res.drop(columns=["clv_x_stake"]).to_string(index=False))
real_pos = real.groupby(["market_id", "selection_id"]).agg(staked=("matched", "sum"))
print(f"\nsettled from {SETTLED_FROM}")
print(f"the real day: {len(real_pos)} horses, GBP{real_pos.staked.sum():,.2f} staked, CLV +9.2%, +GBP194.57 before "
      f"commission (the replay at GBP250 with the day's balance stands for it)")
