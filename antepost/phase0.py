"""Phase 0 of the ante-post plan (docs/antepost_plan.md s.3): the facts and the data, before any model.

Three parts, each read-only, each reported on its own so one failing leaves the others standing:

  rules     Betfair's own description of every GB/IE ANTEPOST_WIN market open now: one listMarketCatalogue call
            with MARKET_DESCRIPTION (the rules text, the BSP flag, the suspend time, turn in play). No order call.
  historic  The Historic Data bundle held in S3 (BASIC, Horse Racing, every country and market type, from Jan 2024;
            kept there by au/historic.py) streamed for its ANTEPOST_WIN markets: each market's clock (first and last
            message, its scheduled time and how often that moved, when it suspended for good, whether it went in
            play), each runner's status and removal time, its last traded price standing at each mark before the
            scheduled off; joined by selection id and race date (UK) to the race-day WIN market in betfair_prices
            (BSP, result) to say of each runner: ran in this race, ran in another race that day, or did not run.
  recorded  The daily snapshots betfair_recorder.py --antepost has kept since 7 Oct 2026
            (s3://$CAPTURE_BUCKET/betfair_live/<day>/books_antepost.csv.gz and markets_antepost.csv.gz): markets,
            runners, days, spread and size at the best prices, matched money; the races already run joined as above.

The "market as a whole" figures (every listed runner backed at its last traded price at a mark, laid at the BSP if it
ran, the stake lost if not) describe the data; they are not a rule and not a backtest: a last traded price is
evidence of a price, not of a fill, and nothing is selected.

    python -m antepost.phase0 --parts rules,historic,recorded --db horse_racing.db --out out/antepost_phase0
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import logging
import os
import re
import sqlite3
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
UK = ZoneInfo("Europe/London")
AP_RE = re.compile(rb'"marketType"\s*:\s*"ANTEPOST_WIN"')
#: marks before the scheduled off: days, then hours
MARKS = {f"d{d}": timedelta(days=d) for d in (180, 120, 90, 60, 45, 30, 21, 14, 7, 3, 2, 1)}
MARKS.update({"h6": timedelta(hours=6), "h1": timedelta(hours=1)})
BANDS = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 1001.0]
BUNDLE_PREFIX = "sources/betfair_historic/bundles/"
LIVE_PREFIX = "betfair_live/"
DEF_KEYS = ("bspMarket", "turnInPlayEnabled", "bettingType", "marketBaseRate", "discountAllowed", "eachWayDivisor",
            "numberOfWinners", "persistenceEnabled", "crossMatching", "raceType", "timezone", "regulators")


def _ts(ms) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def _iso_utc(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _iso(t: datetime | None) -> str | None:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ") if t else None


# ----------------------------------------------------------------------------------------------------------------
# Historic Data: one ante-post market's stream
# ----------------------------------------------------------------------------------------------------------------

class APMarket:
    """One ANTEPOST_WIN market read message by message: its clock, its runners and every last traded price."""

    def __init__(self, market_id: str):
        self.market_id = market_id
        self.defn: dict = {}
        self.first_pt = self.last_pt = None
        self.status_changes: list[tuple[datetime, str]] = []
        self.market_times: list[tuple[datetime, str]] = []
        self.inplay_at = None
        self.def_values: dict[str, set] = {k: set() for k in DEF_KEYS}
        self.first_seen: dict[int, datetime] = {}
        self.removed_at: dict[int, datetime] = {}
        self.adj: dict[int, float] = {}
        self.names: dict[int, str] = {}
        self.ltp: dict[int, list[tuple[datetime, float]]] = {}

    def read(self, pt: datetime, mc: dict) -> None:
        self.first_pt = self.first_pt or pt
        self.last_pt = pt
        d = mc.get("marketDefinition")
        if d:
            self.defn = d
            st = d.get("status")
            if st and (not self.status_changes or self.status_changes[-1][1] != st):
                self.status_changes.append((pt, st))
            mt = d.get("marketTime")
            if mt and (not self.market_times or self.market_times[-1][1] != mt):
                self.market_times.append((pt, mt))
            if d.get("inPlay") and self.inplay_at is None:
                self.inplay_at = pt
            for k in DEF_KEYS:
                v = d.get(k)
                if v is not None:
                    self.def_values[k].add(json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v)
            for r in d.get("runners") or []:
                sid = int(r["id"])
                self.first_seen.setdefault(sid, pt)
                if r.get("name"):
                    self.names[sid] = r["name"]
                if r.get("adjustmentFactor") is not None:
                    self.adj[sid] = float(r["adjustmentFactor"])
                if r.get("status") == "REMOVED" and sid not in self.removed_at:
                    self.removed_at[sid] = _iso_utc(r.get("removalDate")) or pt
        for rc in mc.get("rc") or []:
            if "ltp" in rc:
                self.ltp.setdefault(int(rc["id"]), []).append((pt, float(rc["ltp"])))

    # -- the market's clock -----------------------------------------------------------------------------------
    def off(self) -> datetime | None:
        return _iso_utc(self.defn.get("marketTime"))

    def last_open(self) -> datetime | None:
        """The last moment the market was open: the time of the status change after its last OPEN, else the last
        message while it was still open."""
        t = None
        for i, (pt, st) in enumerate(self.status_changes):
            if st == "OPEN":
                t = self.status_changes[i + 1][0] if i + 1 < len(self.status_changes) else self.last_pt
        return t

    def market_row(self) -> dict:
        d, off, close = self.defn, self.off(), self.last_open()
        suspends = sum(1 for i, (_, st) in enumerate(self.status_changes[:-1])
                       if st == "SUSPENDED" and self.status_changes[i + 1][1] == "OPEN")
        return {
            "market_id": self.market_id, "event_id": d.get("eventId"), "event_name": d.get("eventName"),
            "market_name": d.get("name"), "country": d.get("countryCode"), "venue": d.get("venue"),
            "market_time": _iso(off), "market_time_first": self.market_times[0][1] if self.market_times else None,
            "market_time_changes": max(0, len(self.market_times) - 1), "open_date": d.get("openDate"),
            "first_msg": _iso(self.first_pt), "last_msg": _iso(self.last_pt), "last_open": _iso(close),
            "final_status": d.get("status"), "went_in_play": _iso(self.inplay_at), "settled": d.get("settledTime"),
            "suspend_time": d.get("suspendTime"), "reopened_after_suspension": suspends,
            "status_path": " > ".join(st for _, st in self.status_changes)[:300],
            "runners": len(d.get("runners") or []),
            "removed": sum(1 for r in d.get("runners") or [] if r.get("status") == "REMOVED"),
            "days_listed": (off - self.first_pt).total_seconds() / 86400 if off and self.first_pt else None,
            "hours_closed_before_off": (off - close).total_seconds() / 3600 if off and close else None,
            **{f"def_{k}": "|".join(sorted(map(str, v))) for k, v in self.def_values.items()},
        }

    def runner_rows(self) -> list[dict]:
        off, close, out = self.off(), self.last_open(), []
        for r in self.defn.get("runners") or []:
            sid = int(r["id"])
            series = self.ltp.get(sid, [])
            removed = self.removed_at.get(sid)
            row = {
                "market_id": self.market_id, "selection_id": sid, "runner_name": self.names.get(sid),
                "final_status": r.get("status"), "bsp_in_market": r.get("bsp"),
                "first_seen": _iso(self.first_seen.get(sid)), "removed_at": _iso(removed),
                "removed_days_before_off": (off - removed).total_seconds() / 86400 if off and removed else None,
                "adjustment_factor": self.adj.get(sid), "n_trades": len(series),
                "ltp_first": series[0][1] if series else None, "ltp_last": series[-1][1] if series else None,
                "ltp_last_at": _iso(series[-1][0]) if series else None,
            }
            for k, delta in MARKS.items():
                row[f"listed_{k}"] = row[f"ltp_{k}"] = None
                if off is None:
                    continue
                t = off - delta
                seen = self.first_seen.get(sid)
                open_then = close is None or t <= close
                listed = seen is not None and seen <= t and (removed is None or removed > t) and open_then
                row[f"listed_{k}"] = bool(listed)
                if listed:
                    prior = [p for (pt, p) in series if pt <= t]
                    row[f"ltp_{k}"] = prior[-1] if prior else None
            out.append(row)
        return out


def parse_stream(text: bytes) -> list[APMarket]:
    """A stream file's ANTEPOST_WIN markets (an event's file may hold others: they are left out)."""
    if text[:3] == b"BZh":
        text = bz2.decompress(text)
    markets: dict[str, APMarket] = {}
    kind: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)
        pt = _ts(msg.get("pt", 0))
        for mc in msg.get("mc") or []:
            mid = mc.get("id")
            if mid is None:
                continue
            d = mc.get("marketDefinition")
            if d and d.get("marketType"):
                kind[mid] = d["marketType"]
            if kind.get(mid) == "ANTEPOST_WIN":
                markets.setdefault(mid, APMarket(mid)).read(pt, mc)
    return [m for m in markets.values() if m.defn]


def _parse_item(item: tuple[str, bytes]):
    name, raw = item
    try:
        text = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
        if not AP_RE.search(text):
            return name, [], []
        ms = parse_stream(text)
        return name, [m.market_row() for m in ms], [dict(r, source_file=name) for m in ms for r in m.runner_rows()]
    except Exception as exc:                                            # an unreadable file is counted, not fatal
        return name, None, str(exc)


def stream_bundle(fileobj, chunk: int = 4000) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Every member of a Historic Data tar (a file object read once, front to back) -> ante-post markets, runners."""
    from concurrent.futures import ProcessPoolExecutor
    rep = {"files": 0, "unreadable": 0, "files_with_antepost": 0}
    markets, runners, batch = [], [], []

    def flush(pool):
        for name, mrows, rrows in pool.map(_parse_item, batch, chunksize=64):
            if mrows is None:
                rep["unreadable"] += 1
                continue
            if mrows:
                rep["files_with_antepost"] += 1
                markets.extend(mrows)
                runners.extend(rrows)

    with ProcessPoolExecutor() as pool:
        try:
            with tarfile.open(fileobj=fileobj, mode="r|*") as t:
                for m in t:
                    if not (m.isfile() and m.name.endswith(".bz2") and "PaxHeaders" not in m.name):
                        continue
                    batch.append((m.name, t.extractfile(m).read()))
                    rep["files"] += 1
                    if len(batch) >= chunk:
                        flush(pool)
                        batch = []
                        if rep["files"] % 100000 < chunk:
                            log.info("%d files read, %d ante-post markets", rep["files"], len(markets))
        except (tarfile.ReadError, EOFError) as exc:                    # a bundle cut short is read up to the cut
            rep["truncated"] = f"{exc} after {rep['files']} files"
        if batch:
            flush(pool)
    mk = pd.DataFrame(markets).drop_duplicates("market_id", keep="last") if markets else pd.DataFrame()
    rn = pd.DataFrame(runners).drop_duplicates(["market_id", "selection_id"], keep="last") if runners else pd.DataFrame()
    return mk, rn, rep


# ----------------------------------------------------------------------------------------------------------------
# The race-day join: did the horse run in this race?
# ----------------------------------------------------------------------------------------------------------------

def uk_date(t) -> str | None:
    t = _iso_utc(t) if isinstance(t, str) else t
    return t.astimezone(UK).date().isoformat() if t else None


def race_day_join(markets: pd.DataFrame, runners: pd.DataFrame, db: str | None) -> tuple[pd.DataFrame, dict]:
    """Each ante-post runner against the WIN markets of its race's day (betfair_prices: UK/IE, the BSP and result).
    The race is the WIN event most of the market's runners ran in that day; a runner found in another event that day
    ran elsewhere; one found in none did not run. A market none of whose runners is found is left unjoined."""
    rep: dict = {}
    if runners.empty:
        return runners, {"skipped": "no runners"}
    if not db or not Path(db).exists():
        return runners.assign(label=None), {"skipped": f"no database at {db}"}
    mk = markets.set_index("market_id")
    rn = runners.copy()
    rn["race_date"] = rn["market_id"].map(mk["market_time"]).map(uk_date)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dates = sorted(d for d in rn["race_date"].dropna().unique())
    wins = []
    for i in range(0, len(dates), 200):
        part = dates[i:i + 200]
        q = (f"SELECT race_date, event_id, event_name, track_bf, event_dt, selection_id, bsp, win_lose "
             f"FROM betfair_prices WHERE lower(market_type)='win' AND race_date IN ({','.join('?' * len(part))})")
        wins.append(pd.read_sql_query(q, con, params=part))
    cover = pd.read_sql_query("SELECT min(race_date) lo, max(race_date) hi FROM betfair_prices "
                              "WHERE lower(market_type)='win'", con).iloc[0].to_dict()
    con.close()
    rep["betfair_prices_win_dates"] = cover
    w = pd.concat(wins, ignore_index=True) if wins else pd.DataFrame()
    if w.empty:
        return rn.assign(label=None), dict(rep, skipped="no WIN rows on the race dates")
    w = w.drop_duplicates(["race_date", "selection_id", "event_id"])
    j = rn.merge(w, on=["race_date", "selection_id"], how="left")
    race = (j.dropna(subset=["event_id"]).groupby("market_id")["event_id"]
            .agg(lambda s: s.value_counts().index[0]).rename("race_event_id"))
    j = j.merge(race, left_on="market_id", right_index=True, how="left")
    j["in_race"] = j["event_id"].notna() & (j["event_id"] == j["race_event_id"])
    # one row a runner: its row in the race if it ran there, else any row that day
    j = j.sort_values("in_race", ascending=False).drop_duplicates(["market_id", "selection_id"])
    j["label"] = np.where(j["race_event_id"].isna(), "race_not_found",
                          np.where(j["in_race"], "ran", np.where(j["event_id"].notna(), "ran_elsewhere", "no_run")))
    for c in ("bsp", "win_lose"):
        j.loc[~j["in_race"], c] = np.nan
    return j, rep


# ----------------------------------------------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------------------------------------------

def _q(s: pd.Series, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return {"n": int(len(s)), **({f"q{int(q * 100)}": round(float(s.quantile(q)), 3) for q in qs} if len(s) else {})}


def summarise_historic(mk: pd.DataFrame, rn: pd.DataFrame, jrep: dict) -> dict:
    out: dict = {"join": jrep}
    if mk.empty:
        return dict(out, markets=0)
    t = pd.to_datetime(mk["market_time"], utc=True, errors="coerce")
    out["markets"] = int(len(mk))
    out["by_country_year"] = {f"{c}|{y}": int(n) for (c, y), n in
                              mk.groupby([mk["country"].fillna("?"), t.dt.year.fillna(0).astype(int)]).size().items()}
    out["market_time_range"] = [str(t.min()), str(t.max())]
    out["definition_values"] = {k: mk[f"def_{k}"].value_counts().head(8).to_dict() for k in DEF_KEYS}
    out["final_status"] = mk["final_status"].value_counts().to_dict()
    out["went_in_play_share"] = round(float(mk["went_in_play"].notna().mean()), 4)
    out["market_time_changed_share"] = round(float((mk["market_time_changes"] > 0).mean()), 4)
    out["reopened_after_suspension_share"] = round(float((mk["reopened_after_suspension"] > 0).mean()), 4)
    out["days_listed"] = _q(mk["days_listed"])
    out["hours_closed_before_off"] = _q(mk["hours_closed_before_off"], (0.05, 0.25, 0.5, 0.75, 0.95))
    out["runners_per_market"] = _q(mk["runners"])
    out["markets_sample"] = mk.sort_values("market_time").tail(25)[
        ["market_id", "market_time", "country", "venue", "event_name", "market_name", "runners", "removed",
         "last_open", "hours_closed_before_off", "went_in_play", "status_path"]].to_dict("records")
    if rn.empty:
        return out
    out["runners"] = int(len(rn))
    out["runner_final_status"] = rn["final_status"].value_counts().to_dict()
    rem = rn[rn["final_status"] == "REMOVED"]
    out["removed_days_before_off"] = _q(rem["removed_days_before_off"], (0.05, 0.25, 0.5, 0.75, 0.95))
    out["removed_with_adjustment_factor"] = _q(rem["adjustment_factor"])
    out["bsp_in_market_nonnull"] = int(pd.to_numeric(rn["bsp_in_market"], errors="coerce").notna().sum())
    if "label" in rn and rn["label"].notna().any():
        gbie = rn[rn["market_id"].isin(mk.loc[mk["country"].isin(["GB", "IE"]), "market_id"])]
        out["gbie_runners"] = int(len(gbie))
        out["label"] = gbie["label"].value_counts().to_dict()
        out["status_by_label"] = {f"{a}|{b}": int(n) for (a, b), n in
                                  gbie.groupby(["final_status", "label"]).size().items()}
        joined = gbie[gbie["label"] != "race_not_found"]
        out["races_joined"] = int(joined["market_id"].nunique())
        out["by_mark"] = mark_table(joined)
    return out


def mark_table(j: pd.DataFrame) -> dict:
    """At each mark: runners listed, the share that ran, by price band; for those that ran, their price against the
    BSP; and the market as a whole: every listed runner with a price, backed at it, laid at the BSP if it ran."""
    out = {}
    for k in MARKS:
        lst = j[j[f"listed_{k}"] == True]                                  # noqa: E712 (None is not listed)
        if lst.empty:
            continue
        ran = lst["label"] == "ran"
        row = {"listed": int(len(lst)), "races": int(lst["market_id"].nunique()), "ran_share": round(float(ran.mean()), 4)}
        p = pd.to_numeric(lst[f"ltp_{k}"], errors="coerce")
        band = pd.cut(p, BANDS)
        row["ran_share_by_band"] = {str(b): {"n": int(len(g)), "ran": round(float((g["label"] == "ran").mean()), 4)}
                                    for b, g in lst.groupby(band, observed=True)}
        bsp = pd.to_numeric(lst["bsp"], errors="coerce")
        r = lst[ran & p.notna() & bsp.notna() & (bsp > 1)]
        if len(r):
            ratio = p[r.index] / bsp[r.index]
            row["ran_price_vs_bsp"] = {"n": int(len(r)), "median_log_ratio": round(float(np.log(ratio).median()), 4),
                                       "mean_ratio_minus_1": round(float((ratio - 1).mean()), 4),
                                       "share_price_above_bsp": round(float((ratio > 1).mean()), 4)}
        w = lst[p.notna()]
        if len(w):
            pay = np.where(w["label"] == "ran", p[w.index] / bsp.reindex(w.index), 0.0)
            pay = pd.Series(pay, index=w.index).dropna()
            row["market_whole_back_to_bsp"] = {"n": int(len(pay)), "mean_return": round(float(pay.mean() - 1), 4),
                                               "note": "descriptive: last traded price, before commission, no fill"}
        out[k] = row
    return out


# ----------------------------------------------------------------------------------------------------------------
# The parts
# ----------------------------------------------------------------------------------------------------------------

def _strip_html(s: str) -> str:
    s = re.sub(r"(?i)<br\s*/?>|</p>|</li>", "\n", s or "")
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&nbsp;", " ", s)
    s = re.sub(r"&amp;", "&", s)
    return re.sub(r"\n\s*\n+", "\n", s).strip()


def part_rules(out: Path) -> dict:
    from betfair_client import BetfairClient
    c = BetfairClient()
    c.login()
    try:
        now = datetime.now(timezone.utc)
        cat = c._api_call("listMarketCatalogue", {
            "filter": {"eventTypeIds": ["7"], "marketTypeCodes": ["ANTEPOST_WIN"], "marketCountries": ["GB", "IE"],
                       "marketStartTime": {"from": _iso(now), "to": _iso(now + timedelta(days=400))}},
            "marketProjection": ["EVENT", "MARKET_START_TIME", "MARKET_DESCRIPTION", "RUNNER_DESCRIPTION"],
            "maxResults": 200, "sort": "FIRST_TO_START"}) or []
    finally:
        try:
            c.logout()
        except Exception:
            pass
    rows, texts = [], {}
    for m in cat:
        d, ev = m.get("description") or {}, m.get("event") or {}
        text = _strip_html(d.get("rules") or "")
        texts.setdefault(text, []).append(m.get("marketId"))
        rows.append({"market_id": m.get("marketId"), "market_name": m.get("marketName"), "event": ev.get("name"),
                     "venue": ev.get("venue"), "country": ev.get("countryCode"),
                     "market_start": m.get("marketStartTime"), "runners": len(m.get("runners") or []),
                     "total_matched": m.get("totalMatched"),
                     **{k: d.get(k) for k in ("bspMarket", "turnInPlayEnabled", "persistenceEnabled",
                                               "marketBaseRate", "discountAllowed", "bettingType", "suspendTime",
                                               "marketTime", "settleTime", "rulesHasDate", "clarifications",
                                               "raceType", "eachWayDivisor")}})
    (out / "rules_markets.json").write_text(json.dumps(rows, indent=1, default=str))
    (out / "rules_texts.json").write_text(json.dumps([{"markets": v, "text": k} for k, v in texts.items()], indent=1))
    return {"markets": len(rows), "markets_list": rows,
            "distinct_rules_texts": [{"markets": len(v), "example_market": v[0], "text": k[:20000]}
                                     for k, v in sorted(texts.items(), key=lambda kv: -len(kv[1]))]}


def _s3():
    from betfair_recorder import s3_client, bucket
    return s3_client(), bucket()


def part_historic(out: Path, db: str | None) -> dict:
    s3, b = _s3()
    keys = [o for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=BUNDLE_PREFIX)
            for o in page.get("Contents", [])]
    rep: dict = {"bundles": [{"key": o["Key"], "bytes": o["Size"]} for o in keys]}
    horse = [o["Key"] for o in keys if "horse" in o["Key"].lower() and o["Key"].endswith(".tar")]
    if not horse:
        return dict(rep, skipped="no horse racing bundle under " + BUNDLE_PREFIX)
    mks, rns = [], []
    for key in horse:
        from botocore.config import Config
        import boto3
        big = boto3.client("s3", region_name=os.getenv("CAPTURE_REGION") or "eu-west-2",
                           config=Config(connect_timeout=10, read_timeout=300, retries={"max_attempts": 5}))
        body = big.get_object(Bucket=b, Key=key)["Body"]
        mk, rn, srep = stream_bundle(body)
        rep.setdefault("stream", {})[key] = srep
        mks.append(mk.assign(bundle=key))
        rns.append(rn)
    mk = pd.concat(mks, ignore_index=True).drop_duplicates("market_id", keep="last")
    rn = pd.concat(rns, ignore_index=True).drop_duplicates(["market_id", "selection_id"], keep="last")
    rn, jrep = race_day_join(mk, rn, db)
    mk.to_parquet(out / "historic_markets.parquet", index=False)
    rn.to_parquet(out / "historic_runners.parquet", index=False)
    rep["summary"] = summarise_historic(mk, rn, jrep)
    return rep


def _read_live(s3, b, suffix: str) -> pd.DataFrame:
    frames = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=LIVE_PREFIX):
        for o in page.get("Contents", []):
            if o["Key"].endswith(suffix):
                raw = s3.get_object(Bucket=b, Key=o["Key"])["Body"].read()
                df = pd.read_csv(io.BytesIO(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw), dtype=str)
                frames.append(df.assign(day=o["Key"].split("/")[1]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def part_recorded(out: Path, db: str | None) -> dict:
    s3, b = _s3()
    books = _read_live(s3, b, "books_antepost.csv.gz")
    cat = _read_live(s3, b, "markets_antepost.csv.gz")
    rep: dict = {"book_rows": int(len(books)), "catalogue_rows": int(len(cat))}
    if books.empty:
        return dict(rep, skipped="no books_antepost files found")
    for c in ("back1", "back1_size", "lay1", "lay1_size", "last_traded", "market_total_matched"):
        books[c] = pd.to_numeric(books[c], errors="coerce")
    rep["days"] = sorted(books["day"].unique().tolist())
    rep["markets"] = int(books["market_id"].nunique())
    cm = cat.drop_duplicates(["market_id", "selection_id"], keep="last") if not cat.empty else cat
    if not cm.empty:
        m1 = cm.drop_duplicates("market_id", keep="last")
        rep["markets_list"] = m1[["market_id", "market_name", "event_name", "venue", "country", "market_start_utc"]]\
            .sort_values("market_start_utc").to_dict("records")
    last = books.sort_values("polled_utc").drop_duplicates(["market_id", "selection_id"], keep="last")
    rep["runners"] = int(len(last))
    rep["runner_status_last"] = last["runner_status"].value_counts().to_dict()
    mm = books.sort_values("polled_utc").drop_duplicates(["market_id", "day"], keep="last")
    rep["matched_per_market_day"] = _q(mm["market_total_matched"])
    act = books[(books["runner_status"] == "ACTIVE") & books["back1"].notna() & books["lay1"].notna()]
    act = act.assign(rel_spread=act["lay1"] / act["back1"] - 1, band=pd.cut(act["back1"], BANDS))
    rep["spread_by_band"] = {str(k): {"rel_spread": _q(g["rel_spread"]), "back1_size": _q(g["back1_size"]),
                                      "lay1_size": _q(g["lay1_size"])} for k, g in act.groupby("band", observed=True)}
    rep["two_sided_share"] = round(float(len(act) / max(1, (books["runner_status"] == "ACTIVE").sum())), 4)
    if not cm.empty:
        start = pd.to_datetime(cm.drop_duplicates("market_id").set_index("market_id")["market_start_utc"], utc=True,
                               errors="coerce")
        past = start[start < pd.Timestamp.now(tz="UTC")]
        rep["markets_past_off"] = int(len(past))
        if len(past):
            mk = pd.DataFrame({"market_id": past.index, "market_time": past.map(_iso).values})
            rn = last[last["market_id"].isin(past.index)][["market_id", "selection_id", "runner_status"]].copy()
            rn["selection_id"] = pd.to_numeric(rn["selection_id"], errors="coerce").astype("Int64")
            rn, jrep = race_day_join(mk, rn, db)
            rep["past_join"] = jrep
            if "label" in rn:
                rep["past_label"] = rn["label"].value_counts().to_dict()
                rep["past_status_by_label"] = {f"{a}|{b}": int(n) for (a, b), n in
                                               rn.groupby(["runner_status", "label"]).size().items()}
    books.to_parquet(out / "recorded_books.parquet", index=False)
    if not cat.empty:
        cat.to_parquet(out / "recorded_catalogue.parquet", index=False)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", default="rules,historic,recorded")
    ap.add_argument("--db", default="horse_racing.db")
    ap.add_argument("--out", default="out/antepost_phase0")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"run_at": _iso(datetime.now(timezone.utc))}
    parts = {"rules": lambda: part_rules(out), "historic": lambda: part_historic(out, a.db),
             "recorded": lambda: part_recorded(out, a.db)}
    for p in a.parts.split(","):
        try:
            rep[p] = parts[p]()
        except Exception as exc:
            log.exception("part %s failed", p)
            rep[p] = {"error": f"{type(exc).__name__}: {exc}"}
        (out / "antepost_phase0.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
