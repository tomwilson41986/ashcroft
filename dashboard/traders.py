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
- Greyhounds, the take and close (``sources/greyhound/take_close/<mode>/<day>/ledger.csv``): each back matched, closed
  by laying its winnings at the SP; the trader's own ``settled`` row a market, else the same sum on the settled books.
- Greyhounds, the lay at the SP (``sources/greyhound/sp_lay/<mode>/<day>/ledger.csv``): each lay for its liability
  at the Betfair SP, win and place, settled on the recorder's settled books; its paper forward test
  (``sources/greyhound/sp_offset/served/<day>.csv`` and ``settled/<day>.csv``) at GBP1 a lay.
- Football, live and paper (``sources/football/live/<mode>/<day>/``): its ``settled.csv`` (the backs against the gold
  results, written by the nightly settle). Expected value: ``ev x matched``.

The forecast is the owner's day report (22 Oct and 4 Nov 2026: "never give me inflated expected value", "always
forecast on today's markets"), never the expected value at the fill and never a ratio of the record's result to it:

- The horse traders (live and paper; ``forecast_by="day_report"``): the day report's own method
  (``dashboard/day_forecast.py``, from ``research/queries/done/day_report.py``): settled, plus each race to run marked
  now at the mid after the non-runners' cuts, plus the drift a pound staked from that mark to the settled result on the
  trader's past races marked at the same time before the off; Cash Out (the lays) beside it; the range the
  10th-90th percentile of the past races' misses.
- The others, whose ledgers carry no marks the dashboard can price: settled plus the markets still to come at the
  record's mean a market (fewer than ``MIN_RECORD`` settled days: no forecast, and the page says so).

The month is the month to date, today's forecast, and the record's mean day for each day left.
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
CLV_COLS = ["clv_ws", "clv_s", "clv_n", "clv_sum", "clv_sumsq"]
MARKET_COLS = ["market_id", "label", "off", "staked", "ev", "pnl", "settled", "open_liability"] + CLV_COLS
MIN_FLAG_BETS = 30                # bets with a CLV in the last 7 days before the edge flag judges
RACE_DAYS = 3                     # the latest traded days given race by race (the owner, 4 Nov: score them race by race)

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

    def day_files(self, kind: str, prefix: str, ext: str) -> list[str]:
        """The days of the files ``<prefix><YYYY-MM-DD>.<ext>`` directly under ``prefix``."""
        out, token = [], None
        while True:
            kw = {"Bucket": self.buckets[kind], "Prefix": prefix, "Delimiter": "/"}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            out += [o["Key"][len(prefix):] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                break
            token = r.get("NextContinuationToken")
        return sorted(n[:-len(ext) - 1] for n in out if n.endswith("." + ext) and _is_day(n[:-len(ext) - 1]))


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

    def day_files(self, kind: str, prefix: str, ext: str) -> list[str]:
        base = self.root / kind / prefix
        return sorted(p.stem for p in base.glob(f"*.{ext}") if _is_day(p.stem)) if base.exists() else []


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


def _clv_by_market(rows: pd.DataFrame, clv: str = "clv", stake: str = "matched") -> pd.DataFrame:
    """Each market's CLV sums over its positions with a CLV: stake-weighted (``clv_ws`` / ``clv_s``) and per bet
    (``clv_n``, ``clv_sum``, ``clv_sumsq``) for the spread."""
    if rows is None or not len(rows) or clv not in rows.columns:
        return pd.DataFrame(columns=CLV_COLS)
    r = rows.assign(_c=_num(rows[clv]), _w=_num(rows[stake]) if stake in rows.columns else np.nan)
    r = r[np.isfinite(r._c) & (r._w > 0)]
    if not len(r):
        return pd.DataFrame(columns=CLV_COLS)
    r = r.assign(_cw=r._c * r._w, _c2=r._c ** 2)
    return r.groupby("market_id").agg(clv_ws=("_cw", "sum"), clv_s=("_w", "sum"), clv_n=("_c", "size"),
                                      clv_sum=("_c", "sum"), clv_sumsq=("_c2", "sum"))


def _with_clv(m: pd.DataFrame, rows: pd.DataFrame, clv: str = "clv", stake: str = "matched") -> pd.DataFrame:
    c = _clv_by_market(rows, clv, stake)
    m = m.join(c, how="left") if len(m) else m.reindex(columns=list(m.columns) + CLV_COLS)
    for k in CLV_COLS:
        m[k] = _num(m[k]).fillna(0.0) if k in m.columns else 0.0
    return m


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
    m = _with_clv(m, st)                                   # the session's own CLV: average price / BSP - 1
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
    m = _with_clv(m, s)                                    # settle_rows' CLV: average price / BSP - 1
    return m.reset_index()[MARKET_COLS], _last_ts(led, "time_utc")


def early_lay_markets(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
    led = _csv(src.get(kind, f"{base}/{day}/ledger.csv"))
    if led is None or "action" not in led.columns:
        return None, None
    for c in ("matched", "lay_matched", "locked_pnl", "liability", "avg_price", "size", "selection_id", "lay_price"):
        led[c] = _num(led[c]) if c in led.columns else np.nan
    lays = led[led.action == "lay"]
    closes = led[led.action.isin(["green", "green_part", "sp_close", "hold"])].copy()
    if not len(lays):
        return _empty(), _last_ts(led, "time_utc")
    # the lays closed at the SP or held settle with the result: read it from the settled books when they are in
    opened = closes[closes.action.isin(["sp_close", "hold"])].copy()
    opened["pnl_res"] = np.nan
    final = _greyhound_final(src, day) if len(closes) else None
    lay_clv = pd.DataFrame(columns=["market_id", "clv", "lay_matched"])
    if final is not None:      # a lay's CLV: the BSP over the price laid, less 1 (laid under the SP is to the good)
        lc = closes.merge(final[K + ["sp_actual"]], on=K, how="left")
        lc["clv"] = _num(lc.sp_actual) / lc.lay_price - 1
        lay_clv = lc[lc.lay_price > 1]
    if len(opened):
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
    out = _with_clv(pd.DataFrame(rows).set_index("market_id"), lay_clv, "clv", "lay_matched")
    return out.reset_index()[MARKET_COLS], _last_ts(led, "time_utc")


def take_close_markets(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
    """The take and close (greyhound/take_close.py): each back matched (paper: what a fill-or-kill would take) is
    closed by laying its winnings at the SP. A market's P&L is the trader's own ``settled`` row (Betfair's result and
    SP, after commission on the market's net winnings); where it has none yet, the same sum on the recorder's settled
    books (``close_pnl``). The CLV of a back: its average price over the BSP, less 1."""
    led = _csv(src.get(kind, f"{base}/{day}/ledger.csv"))
    if led is None or "action" not in led.columns:
        return None, None
    for c in ("matched", "price", "pnl", "selection_id"):
        led[c] = _num(led[c]) if c in led.columns else np.nan
    backs = led[led.action.isin(["back_matched", "back_paper"]) & (led.matched > 0)].copy()
    if not len(backs):
        return _empty(), _last_ts(led, "time_utc")
    own = led[led.action == "settled"].drop_duplicates("market_id", keep="last").set_index("market_id").pnl
    backs["value"] = backs.matched * backs.price
    pos = backs.groupby(K, as_index=False).agg(matched=("matched", "sum"), value=("value", "sum"))
    pos["price"] = pos.value / pos.matched
    pos["won"], pos["bsp"] = None, np.nan
    final = _greyhound_final(src, day)
    if final is not None:
        f = final[K + ["runner_status", "sp_actual"]].assign(selection_id=_num(final.selection_id))
        pos = pos.drop(columns=["won", "bsp"]).merge(f, on=K, how="left")
        pos["won"] = pos.runner_status.map({"WINNER": True, "LOSER": False})
        pos["bsp"] = _num(pos.sp_actual)
    pos["clv"] = pos.price / pos.bsp - 1

    def books_pnl(g: pd.DataFrame) -> float:
        """The trader's close_pnl on the recorder's result and SP, less commission on the net: NaN until all are in."""
        gross = 0.0
        for r in g.itertuples():
            if r.won is True:
                continue
            if r.won is not False or not (r.bsp > 1):
                return np.nan
            gross += -r.matched + r.matched * (r.price - 1.0) / (r.bsp - 1.0)
        return gross - RATE * max(gross, 0.0)

    rows = []
    for mid, bg in backs.groupby("market_id"):
        pnl = float(own[mid]) if mid in own.index and np.isfinite(own[mid]) else books_pnl(pos[pos.market_id == mid])
        rows.append({"market_id": mid, "label": bg.track.iloc[0] if "track" in bg.columns else "",
                     "off": bg.race_time.iloc[0] if "race_time" in bg.columns else "",
                     "staked": float(bg.matched.sum()), "ev": np.nan, "pnl": pnl, "settled": bool(np.isfinite(pnl)),
                     "open_liability": 0.0 if np.isfinite(pnl) else float(bg.matched.sum())})
    out = _with_clv(pd.DataFrame(rows).set_index("market_id"), pos, "clv", "matched")
    return out.reset_index()[MARKET_COLS], _last_ts(led, "time_utc")


def sp_lay_markets(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
    """The lay at the SP (greyhound/sp_lay.py): each pick laid for a fixed liability at the Betfair SP, to win and,
    from 21 Nov 2026, to be placed (each its own market). Settled on the recorder's settled books, WIN and PLACE
    (``_greyhound_final``): a lay of liability L matched at SP b stakes L / (b - 1), so a loser (unplaced) pays that
    stake and a winner (placed) costs L; a removed runner's lay is void. 2% off each market's net winnings, as
    Betfair's. A lay the trader cancelled (a STOP before the off) is left out. No CLV: a lay at the SP takes the
    closing price itself."""
    led = _csv(src.get(kind, f"{base}/{day}/ledger.csv"))
    if led is None or "action" not in led.columns:
        return None, None
    for c in ("liability", "selection_id"):
        led[c] = _num(led[c]) if c in led.columns else np.nan
    led["kind"] = led["kind"].fillna("win").replace("", "win") if "kind" in led.columns else "win"
    st = led.get("status", pd.Series("", index=led.index)).astype(str)
    lays = led[((led.action == "lay") & st.isin(["PENDING", "UNKNOWN"])) | (led.action == "paper_lay")].copy()
    bet = lambda x: x.astype(str).str.replace(r"\.0$", "", regex=True)          # noqa: E731
    if "bet_id" in led.columns and len(lays):
        gone = led[(led.action == "cancel") & (st == "SUCCESS")]
        lays = lays[~bet(lays.bet_id).isin(set(bet(gone.bet_id)))]
    if not len(lays):
        return _empty(), _last_ts(led, "time_utc")
    final = _greyhound_final(src, day)
    if final is not None:
        f = final[K + ["runner_status", "sp_actual"]].assign(selection_id=_num(final.selection_id))
        lays = lays.merge(f, on=K, how="left")
    else:
        lays["runner_status"], lays["sp_actual"] = None, np.nan
    L, sp = lays.liability, _num(lays.sp_actual)
    lays["gross"] = np.select([lays.runner_status == "REMOVED", lays.runner_status == "WINNER",
                               (lays.runner_status == "LOSER") & (sp > 1)],
                              [0.0, -L, L / (sp - 1)], np.nan)
    rows = []
    for mid, g in lays.groupby("market_id"):
        gross = float(g.gross.sum()) if g.gross.notna().all() else np.nan
        pnl = gross - RATE * max(gross, 0.0) if np.isfinite(gross) else np.nan
        label = str(g.track.iloc[0]) if "track" in g.columns else ""
        rows.append({"market_id": mid, "label": label + (" (place)" if g.kind.iloc[0] == "place" else ""),
                     "off": g.race_time.iloc[0] if "race_time" in g.columns else "",
                     "staked": float(g.liability.sum()), "ev": np.nan, "pnl": pnl, "settled": bool(np.isfinite(pnl)),
                     "open_liability": 0.0 if np.isfinite(pnl) else float(g.liability.sum())})
    out = _with_clv(pd.DataFrame(rows).set_index("market_id"), None)
    return out.reset_index()[MARKET_COLS], _last_ts(led, "time_utc")


def sp_lay_forward_markets(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
    """The lay at the SP's paper forward test (research/greyhound/sp_offset_forward.py; ledger
    greyhound-sp-offset-forward-1120): the frozen residual's picks served before each race (``<base>/<day>.csv``),
    the ``lay_bottom10`` rule (every runner at or under the frozen cut in ``meta.json``) laid to win risking 1 at the
    Betfair SP, netted by race, commission off a gain. Read in the test's own unit, GBP1 a lay. A race is settled from
    the test's own ``settled/<day>.csv``; once that is written, a race it leaves out (not every served runner got an
    SP, as in the study) drops out. No orders, no CLV (the SP is the close)."""
    import json
    root = base.rsplit("/", 1)[0]
    served = _csv(src.get(kind, f"{base}/{day}.csv"))
    meta = src.get(kind, f"{root}/meta.json")
    if served is None or not meta or "s" not in served.columns:
        return None, None
    cut = float(json.loads(meta)["cuts"][SP_LAY_RULE])
    served["race_id"] = served.race_id.astype(str)
    p = served[(_num(served.s) <= cut) & served.market_id.notna()]
    if not len(p):
        return _empty(), None
    m = p.groupby("race_id").agg(label=("track", "first"), off=("race_time", "first"), staked=("s", "size"))
    m["staked"] = m.staked.astype(float)
    m["ev"], m["pnl"], m["settled"] = np.nan, np.nan, False
    done = _csv(src.get(kind, f"{root}/settled/{day}.csv"))
    if done is not None and "rule" in done.columns:
        d = done[done.rule == SP_LAY_RULE].assign(race_id=lambda x: x.race_id.astype(str))
        d = d.groupby("race_id").agg(risk=("risk", "sum"), net=("net", "sum"))
        m = m[m.index.isin(d.index)]                            # the races the test leaves out drop out
        m["staked"] = _num(d.risk.reindex(m.index))
        m["pnl"] = _num(d.net.reindex(m.index))
        m["settled"] = True
    m["open_liability"] = np.where(m.settled, 0.0, m.staked)
    m.index.name = "market_id"
    m = _with_clv(m, None)
    return m.reset_index()[MARKET_COLS], None


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
    m = _with_clv(m, s)                                    # the settle step's CLV: price x the close's chance - 1
    return m.reset_index()[MARKET_COLS], _last_ts(led, "time_utc")


# --------------------------------------------------------------------------------------------------------------------
# The traders
# --------------------------------------------------------------------------------------------------------------------

def paper_agent_markets(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
    """A horse paper agent's day (research/paper/horses.py): the table it wrote, a row a market, every market settled
    (the desk scores a day once Betfair's price file is in), the P&L after the non-runners' cuts and 2%."""
    m = _csv(src.get(kind, f"{base}/{day}/markets.csv"))
    if m is None:
        return None, None
    for c in MARKET_COLS:
        if c not in m.columns:
            m[c] = np.nan if c not in ("settled",) else True
    m["settled"] = m.settled.astype(str).str.lower().isin(["true", "1", "1.0"])
    for c in ["staked", "ev", "pnl", "open_liability"] + CLV_COLS:
        m[c] = _num(m[c]).fillna(0.0) if c != "pnl" else _num(m[c])
    return m[MARKET_COLS], None


def _morning_table(b: pd.DataFrame, price: str) -> pd.DataFrame:
    """The morning models' backs to a table of markets: P&L after 2% on each market's net winnings, settled once every
    back in the market has a result, and the CLV at the BSP (average price / BSP - 1)."""
    for c in ("matched", "pnl", "bsp", price):
        b[c] = _num(b[c]) if c in b.columns else np.nan
    b["clv"] = _num(b["clv"]) if "clv" in b.columns else np.nan
    b["clv"] = b.clv.where(b.clv.notna(), b[price] / b.bsp.where(b.bsp > 1) - 1)
    b["_set"] = b["result"].astype(str).isin(["WINNER", "LOSER", "REMOVED"]) if "result" in b.columns else False
    lab = "venue" if "venue" in b.columns else "label"
    m = b.groupby("market_id").agg(label=(lab, "first"), off=("off", "first"), staked=("matched", "sum"),
                                   settled=("_set", "all"), gross=("pnl", lambda x: x.sum(min_count=1)))
    m["ev"] = np.nan                                       # no expected value is logged: the forecast is the record's
    m["pnl"] = np.where(m.settled, m.gross - RATE * m.gross.clip(lower=0).fillna(0.0), np.nan)
    m["open_liability"] = 0.0
    m = _with_clv(m, b)
    return m.reset_index()[MARKET_COLS]


def morning_markets(model: str):
    """A morning model's day on the live account (morning/live.py): its backs at 10:00 UK, settled at 22:50 UTC from
    Betfair's record of the settled bets. One ledger holds both models; this reads one."""
    def read(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
        led = _csv(src.get(kind, f"{base}/{day}/ledger.csv"))
        if led is None or "event" not in led.columns or "model" not in led.columns:
            return None, None
        b = led[(led.event == "back") & (led.model == model)].copy()
        b = b[_num(b.get("matched", pd.Series(dtype=float))) > 0]
        return (_morning_table(b, "avg_price") if len(b) else _empty()), _last_ts(led, "ts")
    read.__name__ = f"morning_markets_{model}"
    return read


def morning_paper_markets(strategy: str):
    """The morning models' paper forward test (research/paper/morning_edges.py): the WIN backs of one strategy on the
    recorded books at 10:00 UK, held to the result after the non-runners' cuts; a file a day, written once settled."""
    def read(src, kind: str, base: str, day: str) -> tuple[pd.DataFrame | None, str | None]:
        bets = _csv(src.get(kind, f"{base}/{day}.csv"))
        if bets is None or "strategy" not in bets.columns:
            return None, None
        b = bets[(bets.strategy == strategy) & (bets.get("market", "win") == "win")
                 & (bets.get("event", "back") == "back")].copy()
        b = b[_num(b.get("matched", pd.Series(dtype=float))) > 0]
        return (_morning_table(b, "price") if len(b) else _empty()), None
    read.__name__ = f"morning_paper_markets_{strategy}"
    return read


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
    start_from: str | None = None       # the first day counted at all: earlier days in its ledger (a replay) are left out
    refresh_days: int = 3       # days re-read every run; older settled days come from the cache
    day_file: str | None = None         # a file a day (<base>/<day>.<ext>) rather than a folder a day
    notes: tuple = field(default_factory=tuple)
    switch: str | None = None   # the repository variable that turns its real orders on (shown on the tile)
    forecast_by: str = "record"  # "day_report": today's marks and the day report's drift (dashboard/day_forecast.py)
    stop_keys: tuple = ("greyhound/STOP", "trading/STOP")   # the S3 objects (bucket root) that stop its new orders
    orders: str = "lays"        # what a STOP halts, in the switch's words
    loss_limit: object = None   # a daily loss limit fixed in the trader's code (a number, or a function returning it),
                                # where no <switch>_DAILY_LOSS variable sets it


EARLY_LAY_FROM = "2026-11-11"  # the owner, 11 Nov 2026: the live early lay tracked from fresh
TAKE_CLOSE_FROM = "2026-11-18"  # the owner, 18 Nov 2026: the take and close live from that day
SP_LAY_FROM = "2026-11-21"      # the owner, 21 Nov 2026: the lay at the SP live from that day; its forward test's
                                # first day (research/greyhound/sp_offset_forward.py FIRST_DAY)
SP_LAY_RULE = "lay_bottom10"    # the rule the live trader lays (greyhound/sp_lay.py RULE)


def _sp_lay_limit() -> float:
    """The lay at the SP's daily loss limit, from its own code (greyhound/sp_lay.py CONFIG)."""
    from greyhound.sp_lay import CONFIG
    return float(CONFIG["daily_loss_limit"])

AGENTS_FROM = "2026-11-04"    # the owner, 4 Nov 2026: the paper agents "only start from now"; their replayed days are not P&L
AGENT_NOTES = ("A paper agent (research/paper/horses.py): no orders. It trades the recorded books with the other agents "
               "on one book: the live trader's fills are already out, and what one agent takes is gone for the others.",
               "Scored through the day as if live: hourly on the recorded books, each race run settled on the "
               "recorder's BSP (provisional), then for good once Betfair's price file is in; after the non-runners' "
               "cuts and 2%, at the replay's scale (live has run at 0.80 of it).")

TRADERS = [
    Trader("horse-live", "horse_racing", "Win backs, live", "live",
           "UK and Irish win markets on the owner's account: backs at the price now, each laid at the SP",
           "pred", "trading/live", horse_markets, "horse", "HL", record_from="2026-10-10", forecast_by="day_report",
           notes=("The forecast is the day report's: today's marks after the non-runners' cuts, plus the drift to the "
                  "result on the past races marked the same way, from 10 Oct 2026 (the owner's rule of 9 Oct).",)),
    Trader("horse-paper", "horse_racing", "Win backs, paper", "paper",
           "The paper trader: the same markets simulated against the book, nothing placed",
           "pred", "trading/paper", horse_markets, "horse", "HP", forecast_by="day_report"),
    # the morning models (the owner, 4 Nov 2026: "Create two live models; Steamers, Quick Returners", half stakes;
    # 11 Nov: "separate trading strategies that i can track in the dashboard"): each its own line, live and paper
    Trader("horse-morning-steamer", "horse_racing", "Steamers, live", "live",
           "The CLEAN steamer model's top 5%, WIN backs at 10:00 UK, GBP50 to win (at most GBP100 a bet)",
           "pred", "morning/live", morning_markets("steamer"), "horse", "ST",
           notes=("Real orders only with MORNING_LIVE=yes on the morning runner (morning-models.yml).",
                  "No expected value is logged: the forecast is the record's mean day. CLV at the BSP once settled.")),
    Trader("horse-morning-quick-return", "horse_racing", "Quick returners, live", "live",
           "Horses back within 1-4 days, WIN backs at 10:00 UK, GBP50 to win (at most GBP100 a bet)",
           "pred", "morning/live", morning_markets("quick_return"), "horse", "QR",
           notes=("Real orders only with MORNING_LIVE=yes on the morning runner (morning-models.yml).",
                  "No expected value is logged: the forecast is the record's mean day. CLV at the BSP once settled.")),
    Trader("horse-morning-steamer-paper", "horse_racing", "Steamers, paper", "paper",
           "The steamer paper forward test on the recorded books (GBP100 to win), from 5 Nov 2026",
           "pred", "paper/morning_edges/bets", morning_paper_markets("steam_am"), "horse", "SP",
           start_from="2026-11-05", day_file="csv",
           notes=("Paper, at twice the live stake: halve its P&L to read it at the live GBP50 to win.",
                  "A day is written once the price file is in (the next morning).")),
    Trader("horse-morning-quick-return-paper", "horse_racing", "Quick returners, paper", "paper",
           "The quick-return paper forward test on the recorded books (GBP100 to win), from 5 Nov 2026",
           "pred", "paper/morning_edges/bets", morning_paper_markets("qr_am"), "horse", "QP",
           start_from="2026-11-05", day_file="csv",
           notes=("Paper, at twice the live stake: halve its P&L to read it at the live GBP50 to win.",
                  "A day is written once the price file is in (the next morning).")),
    # the six horse paper agents (horse-agent-*) are retired (the owner, 4 Nov 2026: "Keep the current live rule. Retire
    # the others"); paper_agent_markets still reads their replayed days
    # the owner, 11 Nov 2026: "Show it. Start tracking from fresh": the switch on its tile, and a record from 11 Nov
    # (the 3-4 Nov live days before the audit put it on paper are left out of its figures and the live totals)
    Trader("greyhound-early-lay", "greyhounds", "Early lay, live", "live",
           "Lays the forecast drifters at 80% of the forecast SP, greened up before the off",
           "capture", "sources/greyhound/early_lay/live", early_lay_markets, "greyhound", "EL",
           start_from=EARLY_LAY_FROM, switch="GREYHOUND_EARLY_LAY",
           notes=("Tracked afresh from 11 Nov 2026 (the owner): its live days of 3-4 Nov are not counted.",
                  "No expected value is logged: the forecast is the record's mean day.",
                  "A lay closed at the SP or held settles when the recorder's settled books are in.")),
    Trader("greyhound-early-lay-paper", "greyhounds", "Early lay, paper", "paper",
           "The early lay's decisions logged, nothing sent", "capture", "sources/greyhound/early_lay/paper",
           early_lay_markets, "greyhound", "EP"),
    # the owner, 18 Nov 2026: "Run the take-and-close LIVE", "GBP100 a bet", its own STOP; 19 Nov: "add a tile"
    Trader("greyhound-take-close", "greyhounds", "Take and close, live", "live",
           "Backs a dog at its first traded book when the price is 8%+ over the forecast SP, laid at the SP at once",
           "capture", "sources/greyhound/take_close/live", take_close_markets, "greyhound", "TC",
           start_from=TAKE_CLOSE_FROM, switch="GREYHOUND_TAKE_CLOSE",
           stop_keys=("greyhound/take_close/STOP", "trading/STOP"), orders="backs",
           notes=("Live from 18 Nov 2026 (the owner): GBP100 a bet, its own STOP (greyhound/take_close/STOP); "
                  "greyhound/STOP does not stop it.",
                  "A race's P&L is the trader's own settle from Betfair's result and SP; the recorder's settled "
                  "books stand in until it is written.",
                  "No expected value is logged: the forecast is the record's mean day.")),
    # the owner, 21 Nov 2026: "Put any possible profitable strategies live" (the residual's bottom-10% lay at the SP,
    # to win and to be placed, GBP50 a lay, GBP1,000 a day); "Add the new greyhound traders"
    Trader("greyhound-sp-lay", "greyhounds", "Lay at the SP, live", "live",
           "Lays the residual's bottom 10% of each race at the Betfair SP, to win and to be placed, GBP50 of liability",
           "capture", "sources/greyhound/sp_lay/live", sp_lay_markets, "greyhound", "SL",
           start_from=SP_LAY_FROM, switch="GREYHOUND_SP_LAY", loss_limit=_sp_lay_limit,
           stop_keys=("greyhound/sp_lay/STOP", "trading/STOP"), orders="lays",
           notes=("Live from 21 Nov 2026 (the owner): GBP50 of liability a lay to win and GBP50 to be placed, one "
                  "GBP1,000 daily loss limit fixed in its code, its own STOP (greyhound/sp_lay/STOP); greyhound/STOP "
                  "does not stop it.",
                  "Staked is the liability laid. A race settles on the recorder's settled books (WIN and PLACE) at "
                  "the Betfair SP, 2% off each market's net winnings.",
                  "No CLV: a lay at the SP is matched at the closing price itself. No expected value is logged: the "
                  "forecast is the record's mean day.")),
    Trader("greyhound-sp-lay-paper", "greyhounds", "Lay at the SP, paper", "paper",
           "The same rule's registered forward test: the frozen residual's picks laid to win at the SP, GBP1 a lay",
           "capture", "sources/greyhound/sp_offset/served", sp_lay_forward_markets, "greyhound", "SF",
           start_from=SP_LAY_FROM, day_file="csv",
           notes=("The paper forward test (ledger greyhound-sp-offset-forward-1120): win only, read at GBP1 a lay "
                  "(x50 for the live stake); its gate is read once, at 120 settled days.",
                  "A day settles once the test's settle step has run on the settled books; races where a served "
                  "runner got no SP drop out, as in the study.")),
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

CACHE = "dashboard/cache/v2"           # v2: the tables carry CLV (4 Nov 2026)


def trader_days(src, t: Trader, today: date, cache: bool = True) -> tuple[dict[str, pd.DataFrame], str | None]:
    """Every day's market table for the trader, and the time of its last ledger row today."""
    out, last = {}, None
    listed = src.day_files(t.kind, t.base + "/", t.day_file) if t.day_file else src.days(t.kind, t.base + "/")
    for day in listed:
        if day > today.isoformat() or (t.start_from and day < t.start_from):
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


def summarise(t: Trader, days: dict[str, pd.DataFrame], today: date, last_ts: str | None = None,
              day_fc: dict | None = None) -> dict:
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
                 "method": None, "last_order_utc": last_ts}
    share, f_day, f_lo, f_hi = 1.0, None, None, None
    mean_day = float(rec_full.result.mean()) if len(rec_full) else None
    if tm is not None:
        st = tm[tm.settled.astype(bool)]
        left = tm[~tm.settled.astype(bool)]
        done = float(_num(st.pnl).fillna(0).sum())
        today_out.update({"pnl": _f(done), "volume": _f(_num(tm.staked).fillna(0).sum()), "markets": int(len(tm)),
                          "settled": int(len(st)),
                          "open_liability": _f(_num(tm.open_liability).fillna(0).sum())})
        if t.forecast_by == "day_report":
            f_day, f_lo, f_hi, share = _day_report_forecast(today_out, done, tm, day_fc)
        elif len(rec_full) >= MIN_RECORD:
            # the markets a day still to come (the record's mean count less today's settled) at the record's mean a
            # market: never the expected value at the fill, never scaled by the record's result over it
            per_mkt = rec_full.result.sum() / rec_full.settled.sum() if rec_full.settled.sum() else 0.0
            n_day = float(rec_full.settled.mean())
            to_come = max(n_day - len(st), float(len(left)))
            share = min(1.0, to_come / n_day) if n_day > 0 else 0.0
            f_day = done + to_come * per_mkt
            today_out["method"] = (f"settled + {to_come:.0f} markets still to come x the record's mean a market "
                                   f"({per_mkt:+.2f}; {len(rec_full)} days)")
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
    edge_out = edge(t, days, today, tm, rec_full)
    races_out = races(days)
    return {"id": t.id, "sport": t.sport, "name": t.name, "mode": t.mode, "what": t.what, "glyph": t.glyph,
            "initials": t.initials, "has_data": bool(len(hist)), "today": today_out, "month": month_out,
            "all_time": all_out,
            "record": {"days": int(len(rec_full)), "mean_day": _f(mean_day), "from": t.record_from},
            "daily": [{"day": r.day, "pnl": _f(r.result), "volume": _f(r.staked)} for r in spark.itertuples()],
            "clv_daily": clv_daily(days, hist),
            "edge": edge_out,
            "races": races_out,
            "forecast_races": (day_fc or {}).get("races", []) if t.forecast_by == "day_report" else [],
            "notes": list(t.notes)}


def _day_report_forecast(today_out: dict, done: float, tm: pd.DataFrame, fc: dict | None):
    """Today's forecast from the day report's figures (dashboard/day_forecast.py): settled + the races to run's
    forecast from the mid, the range from its draws, Cash Out beside it. No figures: no forecast, and why."""
    if not fc:
        today_out["method"] = "the day report's forecast was not run"
        return None, None, None, 1.0
    if not fc.get("ok"):
        today_out["method"] = f"no forecast: {fc.get('why', 'not read')}"
        return None, None, None, 1.0
    n_left = int(fc.get("races_left", 0))
    share = (n_left / len(tm)) if len(tm) else 0.0
    f_day = done + fc["forecast_left"]
    today_out.update({"mark_mid": _f(fc.get("mark_mid")), "cash_out": _f(fc.get("cash_out")),
                      "forecast_cash_out": _f(done + fc["forecast_left_cash_out"]), "races_left": n_left,
                      "unpriced": int(fc.get("unpriced", 0)), "books_to": fc.get("books_to"),
                      "forecast_past_days": fc.get("past_days"), "forecast_past_races": fc.get("past_races")})
    if not n_left:
        today_out["method"] = "every race settled: the day is its settled result"
        return done, done, done, 0.0
    unp = f"; {fc['unpriced']} horses with no price yet marked at nothing" if fc.get("unpriced") else ""
    today_out["method"] = (f"the day report: settled + {n_left} races to run marked now at the mid after the "
                           f"non-runners' cuts ({fc['mark_mid']:+,.2f}; Cash Out {fc['cash_out']:+,.2f}) + the drift "
                           f"to the result on {fc['past_races']} past races over {fc['past_days']} days marked the same "
                           f"way at the same time before the off; range 10th-90th{unp}")
    return f_day, done + fc["lo_left"], done + fc["hi_left"], share


def clv_daily(days: dict[str, pd.DataFrame], hist: pd.DataFrame) -> list[dict]:
    """Every traded day since the first (the owner, 5 Nov 2026: "show the CLV per day since we started"): the day's
    stake-weighted CLV against the closing price over its bets with one, the bets it counts, the stake, the P&L
    settled so far."""
    out = []
    for r in hist[hist.staked > 0].itertuples():
        c = _clv_of([days[r.day]]) if len(days.get(r.day, ())) else {"clv": None, "n": 0}
        out.append({"day": r.day, "clv": c["clv"], "bets": c["n"], "staked": _f(r.staked), "pnl": _f(r.result),
                    "complete": bool(r.complete)})
    return out


def races(days: dict[str, pd.DataFrame], n: int = RACE_DAYS) -> list[dict]:
    """The latest ``n`` traded days race by race: each market the trader staked on, its P&L once settled (else None)
    and its stake-weighted CLV."""
    out = []
    traded = [d for d in sorted(days) if len(days[d]) and _num(days[d].staked).fillna(0).sum() > 0]
    for d in traded[-n:]:
        m = days[d]
        for r in m.itertuples(index=False):
            staked = float(_num(pd.Series([r.staked])).fillna(0).iloc[0])
            settled = bool(r.settled) if not isinstance(r.settled, str) else r.settled.lower() in ("true", "1", "1.0")
            pnl = float(_num(pd.Series([r.pnl])).iloc[0]) if settled else np.nan
            if staked <= 0 and not math.isfinite(pnl):
                continue
            cs = float(_num(pd.Series([getattr(r, "clv_s", 0)])).fillna(0).iloc[0])
            cw = float(_num(pd.Series([getattr(r, "clv_ws", 0)])).fillna(0).iloc[0])
            out.append({"day": d, "market_id": str(r.market_id), "course": None if pd.isna(r.label) else str(r.label),
                        "off": None if pd.isna(r.off) else str(r.off), "staked": _f(staked),
                        "pnl": _f(pnl) if math.isfinite(pnl) else None,
                        "clv": round(cw / cs, 4) if cs > 0 else None})
    return out


def _clv_of(frames: list[pd.DataFrame]) -> dict:
    """Stake-weighted CLV over the frames' markets, with the per-bet spread for a 95% interval."""
    if not frames:
        return {"clv": None, "lo": None, "hi": None, "n": 0}
    m = pd.concat(frames, ignore_index=True)
    tot = {k: float(_num(m[k]).fillna(0).sum()) if k in m.columns else 0.0 for k in CLV_COLS}
    n = int(tot["clv_n"])
    if n == 0 or tot["clv_s"] <= 0:
        return {"clv": None, "lo": None, "hi": None, "n": n}
    w = tot["clv_ws"] / tot["clv_s"]
    mean = tot["clv_sum"] / n
    var = max(tot["clv_sumsq"] / n - mean ** 2, 0.0)
    se = math.sqrt(var / n) if n > 1 else None
    r = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {"clv": r(w), "lo": r(w - 1.96 * se) if se is not None else None,
            "hi": r(w + 1.96 * se) if se is not None else None, "n": n}


FLAG_LABEL = {"green": "Beating the market", "amber": "Edge may be fading", "red": "Not beating the market",
              "grey": "Too few bets to judge"}


def edge(t: Trader, days: dict[str, pd.DataFrame], today: date, tm: pd.DataFrame | None,
         rec_full: pd.DataFrame) -> dict:
    """CLV (today, the last 7 days with its 95% interval, the record), the expected value at the fill (today, and
    the record's settled result over it), and the edge flag. The flag reads the last 7 days' CLV against the
    record's: under MIN_FLAG_BETS bets, grey; at or under zero, red; above zero but its interval reaching zero, or
    under half the record's, amber; else green."""
    ds = today.isoformat()
    week0 = (today - timedelta(days=6)).isoformat()
    rec_from = t.record_from or "0000-00-00"
    c_today = _clv_of([tm]) if tm is not None else {"clv": None, "lo": None, "hi": None, "n": 0}
    c_7 = _clv_of([m for d, m in days.items() if week0 <= d <= ds and len(m)])
    c_rec = _clv_of([m for d, m in days.items() if rec_from <= d <= ds and len(m)])
    ev_today = float(_num(tm.ev).sum(min_count=1)) if tm is not None and len(tm) and _num(tm.ev).notna().any() else None
    vol = float(_num(tm.staked).fillna(0).sum()) if tm is not None else 0.0
    has_ev = len(rec_full) and rec_full.ev_settled.notna().any() and rec_full.ev_settled.fillna(0).sum() > 0
    k_all = float(rec_full.result.sum() / rec_full.ev_settled.sum()) if has_ev else None
    l7 = rec_full.tail(7)
    k_7 = float(l7.result.sum() / l7.ev_settled.sum()) if has_ev and l7.ev_settled.fillna(0).sum() > 0 else None
    if c_7["n"] < MIN_FLAG_BETS or c_7["clv"] is None:
        flag, why = "grey", f"{c_7['n']} bets with a closing price in the last 7 days; it needs {MIN_FLAG_BETS}"
    elif c_7["clv"] <= 0:
        flag, why = "red", f"the last 7 days' CLV is {c_7['clv']:+.1%}: its prices are no better than the close"
    elif (c_7["lo"] is not None and c_7["lo"] <= 0) or (c_rec["clv"] and c_rec["clv"] > 0 and c_7["clv"] < 0.5 * c_rec["clv"]):
        flag, why = "amber", (f"the last 7 days' CLV is {c_7['clv']:+.1%} against {c_rec['clv']:+.1%} on the record"
                              + (f"; its interval reaches {c_7['lo']:+.1%}" if c_7["lo"] is not None and c_7["lo"] <= 0 else ""))
    else:
        flag, why = "green", f"the last 7 days' CLV is {c_7['clv']:+.1%}, {c_7['lo']:+.1%} to {c_7['hi']:+.1%}"
    return {"clv_today": c_today["clv"], "clv_today_n": c_today["n"], "clv_7d": c_7["clv"], "clv_7d_lo": c_7["lo"],
            "clv_7d_hi": c_7["hi"], "clv_7d_n": c_7["n"], "clv_record": c_rec["clv"], "clv_record_n": c_rec["n"],
            "ev_today": _f(ev_today), "ev_today_pct": round(ev_today / vol, 4) if ev_today is not None and vol > 0 else None,
            "realised_vs_ev": round(k_all, 3) if k_all is not None else None,
            "realised_vs_ev_7d": round(k_7, 3) if k_7 is not None else None,
            "flag": flag, "flag_label": FLAG_LABEL[flag], "flag_reason": why}


STOP_KEYS = ("greyhound/STOP", "trading/STOP")     # the greyhound traders' kill switches (greyhound/live.py); a
# trader with its own (the take and close: greyhound/take_close/STOP) names them in Trader.stop_keys


def switch_state(src, t: Trader, env=None) -> dict | None:
    """Whether the trader can send real orders now, as the trader itself decides it (greyhound/early_lay.py
    switch_on, daily_loss_limit; greyhound/live.py kill_switch). Read-only: the repository variables reach this job
    as its environment (trader-dashboard.yml), the STOP objects are read from the capture bucket. ``state`` is
    'live' or 'paper' (None when the variable is not in this process's environment: a run by hand); the paper-models
    job's dispatch input can override the variable for one run, which no variable shows."""
    if not t.switch:
        return None
    env = os.environ if env is None else env
    name, lname = t.switch, f"{t.switch}_DAILY_LOSS"
    raw = env.get(name)
    state = None if raw is None else ("live" if str(raw).strip().lower() == "live" else "paper")
    if t.loss_limit is not None:                  # fixed in the trader's code: no variable sets it
        limit = float(t.loss_limit() if callable(t.loss_limit) else t.loss_limit)
    else:
        try:
            limit = float(str(env.get(lname, "")).strip())
            limit = limit if np.isfinite(limit) and limit > 0 else None
        except ValueError:
            limit = None
    stop, stop_err = False, None
    for k in t.stop_keys:
        try:
            if src.get("capture", k) is not None:
                stop = True
        except Exception as exc:  # noqa: BLE001  unreadable: the trader treats it as STOP, the page says unread
            stop_err = f"{k} unread ({type(exc).__name__})"
    if stop_err and not stop:
        stop = None
    armed = state == "live" and limit is not None and stop is False
    if state is None:
        label = f"{name} not read in this run"
    elif stop:
        label = f"Stopped: an S3 STOP object is set; no new {t.orders}"
    elif state != "live":
        label = f"Paper: {name} is '{raw}'; no real orders"
    elif limit is None:
        label = f"Switch live, but no daily loss limit ({lname}): no real orders"
    elif stop is None:
        label = f"Switch live, loss limit GBP{limit:,.0f} a day; STOP not read ({stop_err})"
    else:
        label = f"Live: real orders, daily loss limit GBP{limit:,.0f}"
    return {"name": name, "state": state, "loss_limit": limit, "stop": stop, "armed": armed, "label": label,
            "from": t.start_from}


def build(src, today: date | None = None, cache: bool = True, traders=TRADERS, source_label=None) -> dict:
    from zoneinfo import ZoneInfo
    today = today or datetime.now(ZoneInfo("Europe/London")).date()
    out_t, errors = [], []
    for t in traders:
        try:
            days, last = trader_days(src, t, today, cache=cache)
            fc = None
            if t.forecast_by == "day_report" and today.isoformat() in days:
                try:
                    from dashboard.day_forecast import record_days, today_forecast
                    fc = today_forecast(src, t.id, t.kind, t.base, today,
                                        record_days(sorted(days), today, t.record_from, t.start_from), cache=cache)
                except Exception as exc:  # noqa: BLE001  no forecast rather than a wrong one
                    fc = {"ok": False, "why": f"{type(exc).__name__}: {str(exc)[:160]}"}
            s = summarise(t, days, today, last, day_fc=fc)
        except Exception as exc:  # noqa: BLE001  one trader's fault must not blank the page
            errors.append({"trader": t.id, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            s = summarise(t, {}, today)
            s["error"] = errors[-1]["error"]
        s["source"] = src.where(t.kind, f"{t.base}/<day>/ledger.csv")
        if t.switch:
            try:
                s["switch"] = switch_state(src, t)
            except Exception as exc:  # noqa: BLE001
                s["switch"] = {"name": t.switch, "state": None, "armed": False, "label": f"not read: {type(exc).__name__}"}
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
