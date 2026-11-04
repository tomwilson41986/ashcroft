"""Every trader's figures for the dashboard, from its own ledger in S3 (read-only).

Each trader's ledger is reduced to one table a day, a row per market: what was staked (matched), the expected value
at the fill where the ledger holds one, and the P&L once the market is settled, read the way that trader's own settle
step reads it:

- Horse racing, live and paper (``trading/<mode>/<day>/ledger.csv``, the PRED bucket): the settled P&L as the owner's
  day report takes it (``research/queries/done/day_report.py``): the ``settle`` rows' P&L less the ``commission`` rows,
  else Betfair's 2% of a winning market's net. Expected value: ``edge x matched`` on each back.
- Greyhounds, the paper trader (``sources/greyhound/live/paper/<day>/``): its ``settled.csv`` once the settle step has
  run, else today's backs against the recorder's settled greyhound books (``betfair_live/<day>/books_greyhound.csv.gz``)
  by the trader's own rule (``greyhound.live.settle_rows``: held to the result, the lay at SP). Expected value:
  ``edge x matched``.
- Greyhounds, the early lay (``sources/greyhound/early_lay/<mode>/<day>/ledger.csv``): the P&L the green-ups locked,
  and the lays closed at the SP or held, settled by the same books' result and SP. No expected value is logged.
- Football, live and paper (``sources/football/live/<mode>/<day>/``): its ``settled.csv`` (the backs against the gold
  results, written by the nightly settle). Expected value: ``ev x matched``.

The forecast is the owner's standard (the day report, 22 Oct 2026): never the raw expected value. The record is every
settled day before today; the ratio of the settled result to the expected value over it calibrates what is still to
run. A trader whose ledger holds no expected value is forecast by its record's mean day. The month is the month to
date, today's forecast, and the record's mean day for each day left. No record (fewer than ``MIN_RECORD`` days): no
forecast, and the page says so.
"""
from __future__ import annotations

import io
import math
import os
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RATE = 0.02                       # Betfair's commission where a ledger holds none (as the day report)
MIN_RECORD = 3                    # settled days before a forecast is made
MIN_RANGE = 5                     # settled days before a range is given
K = ["market_id", "selection_id"]
MARKET_COLS = ["market_id", "label", "off", "staked", "ev", "pnl", "settled", "open_liability"]

SPORTS = [("horse_racing", "Horse Racing"), ("greyhounds", "Greyhounds"), ("football", "Football"),
          ("cricket", "Cricket")]


# --------------------------------------------------------------------------------------------------------------------
# Storage: S3 (two buckets) or a local folder laid out as <root>/<bucket kind>/<key>
# --------------------------------------------------------------------------------------------------------------------

class S3Source:
    """Read-only: the PRED bucket (the horse traders) and the capture bucket (``sources/``, ``betfair_live/``)."""

    def __init__(self, s3=None):
        self._s3 = s3
        self.buckets = {"pred": os.getenv("PRED_BUCKET") or os.getenv("ULTRA_BETTING_S3_BUCKET") or "ashcroft",
                        "capture": os.getenv("CAPTURE_BUCKET") or os.getenv("ULTRA_BETTING_S3_BUCKET") or "ashcroft"}

    @property
    def s3(self):
        if self._s3 is None:
            import boto3
            from botocore.config import Config
            self._s3 = boto3.client("s3", region_name=os.getenv("PRED_REGION") or os.getenv("AWS_DEFAULT_REGION")
                                    or "eu-west-2", config=Config(connect_timeout=5, read_timeout=60,
                                                                  retries={"max_attempts": 3}))
        return self._s3

    def where(self, kind: str, key: str) -> str:
        return f"s3://{self.buckets[kind]}/{key}"

    def get(self, kind: str, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.buckets[kind], Key=key)["Body"].read()
        except Exception as exc:  # noqa: BLE001
            if "NoSuchKey" in str(exc) or "404" in str(exc) or "Not Found" in str(exc):
                return None
            raise

    def put(self, kind: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=self.buckets[kind], Key=key, Body=data, ContentType=content_type,
                           CacheControl="no-cache")

    def days(self, kind: str, prefix: str) -> list[str]:
        """The day folders (YYYY-MM-DD) directly under ``prefix``."""
        out, token = [], None
        while True:
            kw = {"Bucket": self.buckets[kind], "Prefix": prefix, "Delimiter": "/"}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            out += [p["Prefix"][len(prefix):].strip("/") for p in r.get("CommonPrefixes", [])]
            if not r.get("IsTruncated"):
                break
            token = r.get("NextContinuationToken")
        return sorted(d for d in out if _is_day(d))


class LocalSource:
    """The same reads from a folder (tests, or a local copy): ``<root>/pred/trading/...``, ``<root>/capture/...``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def where(self, kind: str, key: str) -> str:
        return str(self.root / kind / key)

    def get(self, kind: str, key: str) -> bytes | None:
        p = self.root / kind / key
        return p.read_bytes() if p.exists() else None

    def put(self, kind: str, key: str, data: bytes, content_type: str = "") -> None:
        p = self.root / kind / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def days(self, kind: str, prefix: str) -> list[str]:
        base = self.root / kind / prefix
        return sorted(p.name for p in base.iterdir() if p.is_dir() and _is_day(p.name)) if base.exists() else []


def _is_day(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return len(s) == 10
    except ValueError:
        return False


def _csv(raw: bytes | None, gz: bool = False) -> pd.DataFrame | None:
    if not raw or not raw.strip():
        return None
    try:
        return pd.read_csv(io.BytesIO(raw), compression="gzip" if gz else None, dtype={"market_id": str},
                           low_memory=False, on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return None


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=MARKET_COLS)


def _last_ts(led: pd.DataFrame, col: str) -> str | None:
    if led is None or col not in led.columns:
        return None
    t = pd.to_datetime(led[col], utc=True, errors="coerce").max()
    return None if pd.isna(t) else t.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------------------------------------
# Each trader's ledger to a table of markets
# --------------------------------------------------------------------------------------------------------------------

def horse_markets(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
    led = _csv(src.get(kind, f"{base}/{day}/ledger.csv"))
    if led is None or "event" not in led.columns:
        return None, None
    for c in ("matched", "edge", "pnl", "commission", "selection_id"):
        led[c] = _num(led[c]) if c in led.columns else np.nan
    b = led[(led.event == "back") & (led.matched > 0)].copy()
    b["ev"] = b.edge * b.matched
    st = led[led.event == "settle"].drop_duplicates(K, keep="last")
    pnl = st.groupby("market_id").pnl.sum()
    com = led[led.event == "commission"].groupby("market_id").commission.sum()
    m = b.groupby("market_id").agg(label=("venue", "first"), off=("off", "first"), staked=("matched", "sum"),
                                   ev=("ev", lambda s: s.sum(min_count=1)))
    m["settled"] = m.index.isin(pnl.index)
    gross = pnl.reindex(m.index)
    c = com.reindex(m.index).fillna(0.0)
    c = c.where(c > 0, RATE * gross.clip(lower=0).fillna(0.0))
    m["pnl"] = np.where(m.settled, gross - c, np.nan)
    m["open_liability"] = 0.0
    return m.reset_index()[MARKET_COLS], _last_ts(led, "ts")


def _greyhound_final(src, day: str) -> pd.DataFrame | None:
    """The recorder's settled greyhound books for the day (the paper trader's own settle source)."""
    parts = []
    for name in ("books_greyhound.csv.gz", "books_greyhound_live.csv.gz"):
        f = _csv(src.get("capture", f"betfair_live/{day}/{name}"), gz=True)
        if f is not None and "source" in f.columns:
            parts.append(f[f.source == "final"])
    if not parts:
        return None
    f = pd.concat(parts, ignore_index=True)
    f["selection_id"] = _num(f.selection_id)
    return f.sort_values("polled_utc").drop_duplicates(K, keep="last")


def greyhound_paper_markets(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
    led = _csv(src.get(kind, f"{base}/{day}/ledger.csv"))
    if led is None or "action" not in led.columns:
        return None, None
    s = _csv(src.get(kind, f"{base}/{day}/settled.csv"))
    if s is None:
        b = led[(led.action == "back") & (led.status == "SUCCESS")].copy()
        if len(b):
            final = _greyhound_final(src, day)
            if final is not None:
                from greyhound.live import settle_rows            # the trader's own rule, read-only
                s = settle_rows(b, final)
            else:
                s = b.assign(pnl=np.nan)
        else:
            s = b.assign(pnl=np.nan)
    if s is None or not len(s):
        return _empty(), _last_ts(led, "time_utc")
    for c in ("matched", "edge", "pnl"):
        s[c] = _num(s[c]) if c in s.columns else np.nan
    s["ev"] = s.edge * s.matched
    m = s.groupby("market_id").agg(label=("track", "first"), off=("race_time", "first"), staked=("matched", "sum"),
                                   ev=("ev", lambda x: x.sum(min_count=1)),
                                   pnl=("pnl", lambda x: x.sum(min_count=1)),
                                   settled=("pnl", lambda x: bool(x.notna().any())))
    m["pnl"] = m.pnl.where(m.settled)
    m["open_liability"] = 0.0
    return m.reset_index()[MARKET_COLS], _last_ts(led, "time_utc")


def early_lay_markets(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
    led = _csv(src.get(kind, f"{base}/{day}/ledger.csv"))
    if led is None or "action" not in led.columns:
        return None, None
    for c in ("matched", "lay_matched", "locked_pnl", "liability", "avg_price", "size", "selection_id"):
        led[c] = _num(led[c]) if c in led.columns else np.nan
    lays = led[led.action == "lay"]
    closes = led[led.action.isin(["green", "green_part", "sp_close", "hold"])].copy()
    if not len(lays):
        return _empty(), _last_ts(led, "time_utc")
    # the lays closed at the SP or held settle with the result: read it from the settled books when they are in
    opened = closes[closes.action.isin(["sp_close", "hold"])].copy()
    opened["pnl_res"] = np.nan
    if len(opened):
        final = _greyhound_final(src, day)
        if final is not None:
            opened = opened.merge(final[K + ["runner_status", "sp_actual"]], on=K, how="left")
            won = opened.runner_status == "WINNER"
            known = opened.runner_status.isin(["WINNER", "LOSER"])
            S, m, sp = opened["size"], opened.avg_price, _num(opened.sp_actual)
            lay_leg = np.where(won, -S * (m - 1), S * (1 - RATE))
            back_leg = np.where(won, S * (sp - 1) * (1 - RATE), -S)
            res = np.where(opened.action == "sp_close", lay_leg + back_leg, lay_leg)
            opened["pnl_res"] = np.where(known & ((opened.action == "hold") | (sp > 1)), res, np.nan)
    rows = []
    for mid, lg in lays.groupby("market_id"):
        cg = closes[closes.market_id == mid]
        og = opened[opened.market_id == mid]
        staked = float(cg.lay_matched.fillna(0).sum()) if len(cg) else float(lg.matched.fillna(0).sum())
        locked = float(cg[cg.action.isin(["green", "green_part"])].locked_pnl.fillna(0).sum())
        res_known = bool(og.pnl_res.notna().all()) if len(og) else True
        settled = bool(len(cg)) and res_known
        pnl = locked + float(og.pnl_res.fillna(0).sum()) if settled else np.nan
        open_liab = float(og[og.pnl_res.isna()].liability.fillna(0).sum()) if len(og) else 0.0
        if staked <= 0 and not len(cg):
            settled, pnl = True, 0.0                               # nothing matched: nothing to settle
        rows.append({"market_id": mid, "label": lg.track.iloc[0], "off": lg.race_time.iloc[0], "staked": staked,
                     "ev": np.nan, "pnl": pnl, "settled": settled, "open_liability": open_liab})
    return pd.DataFrame(rows, columns=MARKET_COLS), _last_ts(led, "time_utc")


def football_markets(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
    led = _csv(src.get(kind, f"{base}/{day}/ledger.csv"))
    if led is None or "action" not in led.columns:
        return None, None
    s = _csv(src.get(kind, f"{base}/{day}/settled.csv"))
    if s is None or not len(s):
        s = led[(led.action == "back") & (led.status == "SUCCESS")].copy()
        s["settled"], s["pnl"] = False, np.nan
    if not len(s):
        return _empty(), _last_ts(led, "time_utc")
    for c in ("matched", "ev", "pnl"):
        s[c] = _num(s[c]) if c in s.columns else np.nan
    s["settled"] = s.settled.astype(str).str.lower().isin(["true", "1", "1.0"])
    s["evx"] = s.ev * s.matched
    m = s.groupby("market_id").agg(label=("event", "first"), off=("kickoff_utc", "first"), staked=("matched", "sum"),
                                   ev=("evx", lambda x: x.sum(min_count=1)), settled=("settled", "all"),
                                   pnl=("pnl", lambda x: x.sum(min_count=1)))
    m["pnl"] = m.pnl.where(m.settled)
    m["open_liability"] = 0.0
    return m.reset_index()[MARKET_COLS], _last_ts(led, "time_utc")


# --------------------------------------------------------------------------------------------------------------------
# The traders
# --------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Trader:
    id: str
    sport: str
    name: str
    mode: str                   # live (real money) or paper
    what: str                   # one line: what it does
    kind: str                   # the bucket: pred or capture
    base: str                   # the key prefix holding <day>/ledger.csv
    reader: object
    glyph: str                  # the avatar's sport glyph
    initials: str
    record_from: str | None = None      # the forecast's record starts here (a rule change)
    refresh_days: int = 3       # days re-read every run; older settled days come from the cache
    notes: tuple = field(default_factory=tuple)


TRADERS = [
    Trader("horse-live", "horse_racing", "Win backs, live", "live",
           "UK and Irish win markets on the owner's account: backs at the price now, each laid at the SP",
           "pred", "trading/live", horse_markets, "horse", "HL", record_from="2026-10-10",
           notes=("The forecast's record starts 10 Oct 2026, the owner's rule of 9 Oct, as the day report.",)),
    Trader("horse-paper", "horse_racing", "Win backs, paper", "paper",
           "The paper trader: the same markets simulated against the book, nothing placed",
           "pred", "trading/paper", horse_markets, "horse", "HP"),
    Trader("greyhound-early-lay", "greyhounds", "Early lay, live", "live",
           "Lays the forecast drifters at 80% of the forecast SP, greened up before the off (from 3 Nov 2026)",
           "capture", "sources/greyhound/early_lay/live", early_lay_markets, "greyhound", "EL",
           notes=("No expected value is logged: the forecast is the record's mean day.",
                  "A lay closed at the SP or held settles when the recorder's settled books are in.")),
    Trader("greyhound-early-lay-paper", "greyhounds", "Early lay, paper", "paper",
           "The early lay's decisions logged, nothing sent", "capture", "sources/greyhound/early_lay/paper",
           early_lay_markets, "greyhound", "EP"),
    Trader("greyhound-paper", "greyhounds", "Win backs, paper", "paper",
           "The greyhound model's backs on the live key's books, laid at the SP, no bets placed",
           "capture", "sources/greyhound/live/paper", greyhound_paper_markets, "greyhound", "GP"),
    Trader("football-live", "football", "Match odds, live", "live",
           "Match odds and over/under 2.5, real stakes with FOOTBALL_LIVE=yes (GBP2 a bet, GBP20 a day)",
           "capture", "sources/football/live/live", football_markets, "football", "FL", refresh_days=10,
           notes=("Settled overnight against the results (football-model.yml): today's P&L stays open until then.",)),
    Trader("football-paper", "football", "Match odds, paper", "paper",
           "The football model's decisions on paper", "capture", "sources/football/live/paper", football_markets,
           "football", "FP", refresh_days=10,
           notes=("Settled overnight against the results (football-model.yml): today's P&L stays open until then.",)),
]


# --------------------------------------------------------------------------------------------------------------------
# Reading the days, with a cache of the settled ones
# --------------------------------------------------------------------------------------------------------------------

CACHE = "dashboard/cache"


def trader_days(src, t: Trader, today: date, cache: bool = True) -> tuple[dict[str, pd.DataFrame], str | None]:
    """Every day's market table for the trader, and the time of its last ledger row today."""
    out, last = {}, None
    for day in src.days(t.kind, t.base + "/"):
        if day > today.isoformat():
            continue
        old = date.fromisoformat(day) < today - timedelta(days=t.refresh_days)
        key = f"{CACHE}/{t.id}/{day}.csv"
        if cache and old:
            c = _csv(src.get("pred", key))
            if c is not None:
                c["settled"] = c.settled.astype(str).str.lower().isin(["true", "1", "1.0"])
                if c.settled.all() or date.fromisoformat(day) < today - timedelta(days=21):
                    out[day] = c
                    continue
        m, ts = t.reader(src, t.kind, t.base, day)
        if m is None:
            continue
        out[day] = m
        if day == today.isoformat():
            last = ts
        if cache and old:
            try:
                src.put("pred", key, m.to_csv(index=False).encode(), "text/csv")
            except Exception:  # noqa: BLE001  the cache is a convenience: the figures stand without it
                pass
    return out, last


# --------------------------------------------------------------------------------------------------------------------
# The figures
# --------------------------------------------------------------------------------------------------------------------

def _f(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, 2)


def summarise(t: Trader, days: dict[str, pd.DataFrame], today: date, last_ts: str | None = None) -> dict:
    ds = today.isoformat()
    rows = []
    for d, m in sorted(days.items()):
        m = m if len(m) else _empty()
        st = m[m.settled.astype(bool)]
        rows.append({"day": d, "staked": float(_num(m.staked).fillna(0).sum()),
                     "result": float(_num(st.pnl).fillna(0).sum()),
                     "ev_settled": float(_num(st.ev).sum(min_count=1)) if len(st) else np.nan,
                     "complete": bool(len(m) == 0 or m.settled.astype(bool).all()), "markets": int(len(m)),
                     "settled": int(len(st))})
    hist = pd.DataFrame(rows, columns=["day", "staked", "result", "ev_settled", "complete", "markets", "settled"])
    rec = hist[(hist.day < ds) & (hist.staked > 0)]
    if t.record_from:
        rec = rec[rec.day >= t.record_from]
    rec_full = rec[rec.complete]                              # the forecast's record: days fully settled

    tm = days.get(ds)
    today_out = {"day": ds, "traded": tm is not None, "pnl": None, "volume": None, "markets": 0, "settled": 0,
                 "open_liability": None, "forecast": None, "forecast_lo": None, "forecast_hi": None,
                 "forecast_7": None, "method": None, "last_order_utc": last_ts}
    share, f_day, f_lo, f_hi = 1.0, None, None, None
    has_ev = len(rec_full) and rec_full.ev_settled.notna().any() and rec_full.ev_settled.fillna(0).sum() > 0
    mean_day = float(rec_full.result.mean()) if len(rec_full) else None
    if tm is not None:
        st = tm[tm.settled.astype(bool)]
        left = tm[~tm.settled.astype(bool)]
        done = float(_num(st.pnl).fillna(0).sum())
        today_out.update({"pnl": _f(done), "volume": _f(_num(tm.staked).fillna(0).sum()), "markets": int(len(tm)),
                          "settled": int(len(st)),
                          "open_liability": _f(_num(tm.open_liability).fillna(0).sum())})
        if len(rec_full) >= MIN_RECORD:
            if has_ev and _num(tm.ev).notna().any():
                k_all = rec_full.result.sum() / rec_full.ev_settled.sum()
                last7 = rec_full.tail(7)
                k_7 = last7.result.sum() / last7.ev_settled.sum() if last7.ev_settled.sum() > 0 else None
                ev_day = float(_num(tm.ev).fillna(0).sum())
                ev_left = float(_num(left.ev).fillna(0).sum())
                share = ev_left / ev_day if ev_day > 0 else 0.0
                f_day = done + ev_left * k_all
                today_out["forecast_7"] = _f(done + ev_left * k_7) if k_7 is not None else None
                today_out["method"] = (f"settled + the expected value still to run x {k_all:.2f} (the record's result"
                                       f" / expected value, {len(rec_full)} days)")
                if len(rec_full) >= MIN_RANGE:
                    miss = rec_full.result - rec_full.ev_settled * k_all
                    f_lo = f_day + miss.quantile(0.1) * math.sqrt(share)
                    f_hi = f_day + miss.quantile(0.9) * math.sqrt(share)
            else:
                # no expected value: the markets a day still to come (the record's mean count less today's settled)
                # at the record's mean a market
                per_mkt = rec_full.result.sum() / rec_full.settled.sum() if rec_full.settled.sum() else 0.0
                n_day = float(rec_full.settled.mean())
                to_come = max(n_day - len(st), float(len(left)))
                share = min(1.0, to_come / n_day) if n_day > 0 else 0.0
                f_day = done + to_come * per_mkt
                today_out["method"] = (f"settled + {to_come:.0f} markets still to come x the record's mean a market "
                                       f"({per_mkt:+.2f}; {len(rec_full)} days, no expected value in the ledger)")
                if len(rec_full) >= MIN_RANGE:
                    miss = rec_full.result - rec_full.settled * per_mkt
                    f_lo = f_day + miss.quantile(0.1) * math.sqrt(share)
                    f_hi = f_day + miss.quantile(0.9) * math.sqrt(share)
            today_out.update({"forecast": _f(f_day), "forecast_lo": _f(f_lo), "forecast_hi": _f(f_hi)})

    # the month
    month = ds[:7]
    mh = hist[hist.day.str.startswith(month)]
    m_pnl = float(mh.result.sum()) if len(mh) else 0.0
    _, n_days = monthrange(today.year, today.month)
    days_left = n_days - today.day
    month_out = {"month": month, "pnl": _f(m_pnl) if len(mh) else None, "days": int((mh.staked > 0).sum()),
                 "forecast": None, "forecast_lo": None, "forecast_hi": None, "days_left": days_left, "method": None}
    if len(rec_full) >= MIN_RECORD:
        today_rest = (f_day - today_out["pnl"]) if (f_day is not None and today_out["pnl"] is not None) else 0.0
        fm = m_pnl + today_rest + days_left * mean_day
        month_out["forecast"] = _f(fm)
        month_out["method"] = (f"the month to date + today's forecast still to run + {days_left} days x the record's "
                               f"mean day ({_f(mean_day)}), if it trades every day left")
        if len(rec_full) >= MIN_RANGE:
            sd = float(rec_full.result.std(ddof=1))
            spread = 1.2816 * sd * math.sqrt(days_left + share)
            month_out["forecast_lo"], month_out["forecast_hi"] = _f(fm - spread), _f(fm + spread)

    traded = hist[hist.staked > 0]
    unsettled = hist[(hist.day < ds) & ~hist.complete]
    all_out = {"pnl": _f(hist.result.sum()) if len(hist) else None, "days": int(len(traded)),
               "first_day": traded.day.min() if len(traded) else None,
               "volume": _f(hist.staked.sum()) if len(hist) else None,
               "days_up": int((traded.result > 0).sum()),
               "open_days": unsettled.day.tolist()}
    spark = hist[hist.day >= (today - timedelta(days=29)).isoformat()][["day", "result", "staked"]]
    return {"id": t.id, "sport": t.sport, "name": t.name, "mode": t.mode, "what": t.what, "glyph": t.glyph,
            "initials": t.initials, "has_data": bool(len(hist)), "today": today_out, "month": month_out,
            "all_time": all_out,
            "record": {"days": int(len(rec_full)), "mean_day": _f(mean_day), "from": t.record_from},
            "daily": [{"day": r.day, "pnl": _f(r.result), "volume": _f(r.staked)} for r in spark.itertuples()],
            "notes": list(t.notes)}


def build(src, today: date | None = None, cache: bool = True, traders=TRADERS, source_label=None) -> dict:
    from zoneinfo import ZoneInfo
    today = today or datetime.now(ZoneInfo("Europe/London")).date()
    out_t, errors = [], []
    for t in traders:
        try:
            days, last = trader_days(src, t, today, cache=cache)
            s = summarise(t, days, today, last)
        except Exception as exc:  # noqa: BLE001  one trader's fault must not blank the page
            errors.append({"trader": t.id, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            s = summarise(t, {}, today)
            s["error"] = errors[-1]["error"]
        s["source"] = src.where(t.kind, f"{t.base}/<day>/ledger.csv")
        out_t.append(s)
    sports = []
    for sid, label in SPORTS:
        ts = [x for x in out_t if x["sport"] == sid]
        live = [x for x in ts if x["mode"] == "live" and x["has_data"]]

        def tot(path):
            vals = [x for x in (_dig(v, path) for v in live) if x is not None]
            return _f(sum(vals)) if vals else None
        sports.append({"id": sid, "label": label, "traders": ts,
                       "live_totals": {"today_pnl": tot("today.pnl"), "today_volume": tot("today.volume"),
                                       "today_forecast": tot("today.forecast"), "month_pnl": tot("month.pnl"),
                                       "month_forecast": tot("month.forecast"), "all_time_pnl": tot("all_time.pnl")}})
    return {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "today": today.isoformat(),
            "currency": "GBP", "sports": sports, "errors": errors,
            "source": source_label or "each trader's ledger in S3 (read-only); see each trader's source line"}


def _dig(d: dict, path: str):
    for p in path.split("."):
        d = d.get(p) if isinstance(d, dict) else None
    return d
