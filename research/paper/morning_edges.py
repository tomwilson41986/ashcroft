"""The morning edges paper trader (the owner, 4 Nov 2026: "set this up as a paper trader model. Ensure it doesn't bet
before 10:00 so it doesnt take prices from the live trader"). Spec: docs/paper_trader_morning_edges.md.

Paper only. Nothing here can place an order: it reads S3 (the recorded books, Betfair's price files, the frozen
model, the Timeform cache) and the HRB database, and replays each racing day once its price files are archived.

Three strategies, frozen 4 Nov 2026 (reports/strategy_research_2026-11-04.md):
  qr_am      back within 1-4 days (flat/AW/hurdle/chase, 5+ runners): WIN and PLACE backed at the 10:00 UK book
  steam_am   the CLEAN steamer model's top 5% (research/paper/morning_features.py): WIN and PLACE at the 10:00 book
  tf_pb_bsp  Flat/AW, last run a career-best Timeform rating by 5lb+: WIN and PLACE at BSP

The live trader comes first, three ways:
  * NO ORDER BEFORE 10:00 UK (ENTRY_UK). The session does nothing at an earlier minute (tests/test_morning_edges.py).
  * A horse the live trader has backed that day (trading/live/<day>/ledger.csv) is left alone, whenever it backed it:
    the paper trader never competes with it for a price, now or for a later top-up. Counted as `skipped_live`.
  * Fills come from the recorded book at the mark, which already lacks what the live trader had matched.

Staking: to win GBP100 in each market (stake = 100 / (price - 1)), GBP200 a bet at most, GBP4,000 staked a day per
strategy per market, GBP5,000 paper bank. A back takes the best price and walks down the ladder to 95% of it; what
is left unfilled stays unfilled and is recorded. Non-runners are void; a winner's price is cut by Betfair's reduction
factor for runners removed after the bet. P&L at 2% and 5% commission on each market's net winnings.

Ledger: s3://$PRED_BUCKET/paper/morning_edges/: model/ (frozen), selections/<day>.csv, bets/<day>.csv, days/<day>.json.
The gate (fixed 4 Nov, read when a strategy has 1,000 settled win bets and 60 racing days; promotion only on the
owner's word): win A/E vs filled price > 1 with chi2 p < 0.05; place A/E > 1; win+place P&L at 5% with a day-block
bootstrap 90% interval above zero; 80%+ of the stake filled and median fill no worse than 3% under the morning WAP;
no month with win A/E under 0.85 on 100+ bets. Stop early: 300+ bets and win A/E under 0.90, or a GBP3,000 drawdown.

    python -m research.paper.morning_edges --train --cutoff 2026-10-31      # once: fit and freeze the model
    python -m research.paper.morning_edges --days pending [--backfill-from 2026-09-30]
    python -m research.paper.morning_edges --report-only
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as C2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "strat"))

import betfair_prices as bp  # noqa: E402
from research.arms import core  # noqa: E402
from research.paper import morning_features as MF  # noqa: E402
from trading.matching import attach_ids  # noqa: E402

ENTRY_UK = dtime(10, 0)          # no order before this, UK clock (the live trader's morning is its own)
RETRY_UNTIL_UK = dtime(10, 15)   # a selection with no book at 10:00 is tried each minute to here, then dropped
TO_WIN = 100.0
MAX_STAKE = 200.0
DAY_LIMIT = 4000.0
BANK = 5000.0
WALK = 0.95                      # walk the ladder down to 95% of the best price
MIN_AHEAD = timedelta(minutes=2)  # a race off within 2 minutes of the bet is not bet
FORWARD_FROM = date(2026, 11, 5)
PREFIX = "paper/morning_edges"
STRATS = {"qr_am": "qr", "steam_am": "steam"}
TF_PREFIX = f"{PREFIX}/timeform"


def entry_utc(day: date) -> datetime:
    return datetime.combine(day, ENTRY_UK, tzinfo=core.UK).astimezone(timezone.utc)


def retry_utc(day: date) -> datetime:
    return datetime.combine(day, RETRY_UNTIL_UK, tzinfo=core.UK).astimezone(timezone.utc)


def _put(key: str, body: bytes) -> None:
    core.s3().put_object(Bucket=core.bucket(), Key=key, Body=body)


def _keys(prefix: str) -> list[str]:
    out, token = [], None
    while True:
        kw = {"Bucket": core.bucket(), "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = core.s3().list_objects_v2(**kw)
        out += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        token = r.get("NextContinuationToken")


# ---------------------------------------------------------------------------------------------------------------
# the session: backs the day's selections at the 10:00 book, never earlier
# ---------------------------------------------------------------------------------------------------------------

class MorningEdgeSession:
    """A session run_agents can step (research/arms/core.py): ``policy`` carries the day's selections (market_id,
    selection_id, label) and the market type. At every minute before 10:00 UK it does nothing."""

    def __init__(self, x, preds, cfg, day, clock=None, sleep=None, policy=None):
        self.x, self.day = x, day
        self.policy = policy or {}
        self.sel = self.policy.get("selections", pd.DataFrame(columns=["market_id", "selection_id"]))
        self.live_backed = set(self.policy.get("live_backed", set()))
        self.entry, self.retry_end = entry_utc(day), retry_utc(day)
        self.markets = {}
        self.ledger = []
        self.done = set()
        self.staked = 0.0

    def load_markets(self):
        ids = set(self.sel["market_id"].astype(str))
        self.markets = {m.market_id: m for m in self.x.data.D.markets if m.market_id in ids}

    def step(self, t: datetime) -> None:
        if t < self.entry or t > self.retry_end:
            return                                           # the live trader's morning: no order before 10:00 UK
        todo = [r for r in self.sel.itertuples(index=False)
                if (str(r.market_id), int(r.selection_id)) not in self.done
                and str(r.market_id) in self.markets and self.markets[str(r.market_id)].start - t > MIN_AHEAD]
        if not todo:
            return
        books = self.x.books(sorted({str(r.market_id) for r in todo}))
        for r in todo:
            mid, sid = str(r.market_id), int(r.selection_id)
            row = {"ts": t.isoformat(), "market_id": mid, "selection_id": sid, "label": getattr(r, "label", ""),
                   "event": "", "best_back": np.nan, "size": 0.0, "price": np.nan, "matched": 0.0}
            if (mid, sid) in self.live_backed:
                row["event"] = "skipped_live"
                self.done.add((mid, sid))
                self.ledger.append(row)
                continue
            b = books.get(mid)
            q = b.runners.get(sid) if b is not None else None
            if b is None or q is None or q.status != "ACTIVE" or not q.back or b.status != "OPEN" or b.inplay:
                if t >= self.retry_end:
                    row["event"] = "no_book"
                    self.done.add((mid, sid))
                    self.ledger.append(row)
                continue                                     # retried next minute until 10:15
            best = float(q.back[0][0])
            size = min(TO_WIN / (best - 1.0), MAX_STAKE, DAY_LIMIT - self.staked)
            row["best_back"], row["size"] = best, round(size, 2)
            self.done.add((mid, sid))
            if size < 2.0:
                row["event"] = "day_limit"
                self.ledger.append(row)
                continue
            f = self.x.back(mid, sid, max(1.01, best * WALK), round(size, 2), f"me-{mid}-{sid}", min_fill=2.0)
            row["event"] = "back" if f.status == "SUCCESS" else f"failed:{f.error}"
            if f.status == "SUCCESS":
                row["price"], row["matched"] = float(f.avg_price), float(f.matched)
                self.staked += float(f.matched)
            self.ledger.append(row)

    def settle_all(self, now):
        return None


# ---------------------------------------------------------------------------------------------------------------
# settle: held to the result, Betfair's reduction factors, commission per market
# ---------------------------------------------------------------------------------------------------------------

def settle(bets: pd.DataFrame, D) -> pd.DataFrame:
    """Each matched back held to the result: void if the runner did not run; a winner's price cut by the reduction
    factors of runners removed after the bet. Adds bsp, result, pnl (before commission)."""
    b = bets[bets["matched"] > 0].copy()
    if b.empty:
        return b
    b["bsp"] = [D.settled.get((m, int(s)), (np.nan, "REMOVED"))[0] for m, s in zip(b.market_id, b.selection_id)]
    b["result"] = [D.settled.get((m, int(s)), (np.nan, "REMOVED"))[1] for m, s in zip(b.market_id, b.selection_id)]
    b = b[pd.to_numeric(b["bsp"], errors="coerce") > 1].copy()
    if b.empty:
        return b
    rem, t = D.removed, pd.to_datetime(b["ts"], utc=True)
    fac = []
    for r, tt in zip(b.itertuples(), t):
        x = rem[(rem.market_id == r.market_id) & (rem.selection_id != r.selection_id) & (rem.rt > tt)] if len(rem) else rem
        fac.append(float(np.prod(1 - x.af.to_numpy() / 100.0)) if len(x) else 1.0)
    P = b["price"].to_numpy(float)
    Pr = np.maximum(1.01, 1 + (P - 1) * np.array(fac))
    won = b["result"].eq("WINNER").to_numpy()
    b["price_reduced"] = Pr
    b["pnl"] = np.where(won, b["matched"] * (Pr - 1), -b["matched"])
    return b


def commission(pnl_by_market: pd.Series, rate: float) -> float:
    return float((pnl_by_market - rate * pnl_by_market.clip(lower=0)).sum())


# ---------------------------------------------------------------------------------------------------------------
# the day
# ---------------------------------------------------------------------------------------------------------------

_DF = None
_FEAT = None


def history() -> pd.DataFrame:
    """The HRB database, loaded once (strat_lib.load)."""
    global _DF
    if _DF is None:
        _DF = MF.L.load()
    return _DF


def features_all() -> pd.DataFrame:
    """The features for every runner, built once. Each day's features read only earlier days and the card; a check
    against card_view runs on the first day of every job (``check_card_view``)."""
    global _FEAT
    if _FEAT is None:
        _FEAT = MF.build(history())
    return _FEAT


def check_card_view(day: str, booster) -> None:
    """The day's predictions from the full build equal those from the card view (results and prices blanked)."""
    full = features_all()
    a = full[full["race_date"] == day]
    b = MF.build(MF.card_view(history(), day))
    b = b[b["race_date"] == day]
    if a.empty:
        return
    pa = pd.Series(booster.predict(a[MF.CLEAN]), index=a["horse_name"].to_numpy())
    pb = pd.Series(booster.predict(b[MF.CLEAN]), index=b["horse_name"].to_numpy())
    common = pa.index.intersection(pb.index)
    gap = float(np.max(np.abs(pa.loc[common].to_numpy() - pb.loc[common].to_numpy()))) if len(common) else 0.0
    print(f"  card-view check {day}: {len(common)} runners, largest prediction difference {gap:.2e}")
    if gap > 1e-9:
        raise RuntimeError(f"a selection on {day} would read the day's result or price: {gap}")


def load_model():
    import lightgbm as lgb
    raw = core.get(f"{PREFIX}/model/booster.txt")
    meta = core.get(f"{PREFIX}/model/meta.json")
    if raw is None or meta is None:
        return None, None, {}
    m = json.loads(meta)
    return lgb.Booster(model_str=raw.decode()), float(m["threshold"]), m


def selections(day: str, booster, thr) -> pd.DataFrame:
    x = features_all()
    x = x[x["race_date"] == day].copy()
    if x.empty:
        return pd.DataFrame()
    out = [x[MF.rule_qr(x)].assign(strategy="qr_am", pred=np.nan)]
    if booster is not None:
        x["pred"] = booster.predict(x[MF.CLEAN])
        out.append(x[(x["pred"] >= thr) & (x["field"] >= 5)].assign(strategy="steam_am"))
    keep = ["strategy", "race_date", "race_time", "track", "horse_name", "code", "field", "dslr", "pred"]
    return pd.concat(out, ignore_index=True)[keep]


def live_backed(D) -> set:
    """Every (market, selection) the live trader backed that day, at any time."""
    led = D.ledger
    if led is None or len(led) == 0 or "event" not in led:
        return set()
    b = led[(led["event"] == "back") & (pd.to_numeric(led["matched"], errors="coerce") > 0)]
    return {(str(m), int(s)) for m, s in zip(b["market_id"], b["selection_id"])}


def wap_lookup(day: date, mtype: str) -> dict:
    """The price file's morning WAP per (UK off time, horse): what the research priced at, for the fill check."""
    spf = core.price_files(day, mtype)
    if spf.empty or "morningwap" not in spf:
        return {}
    return {(bp.race_time_to_24h(t), bp.normalise_horse(h)): (w, v)
            for t, h, w, v in zip(spf["race_time"], spf["selection_name"], spf["morningwap"], spf["morning_vol"])}


def _off(m) -> str:
    return bp.utc_iso_to_uk_hhmm(m.start.strftime("%Y-%m-%dT%H:%M:%S.000Z"))


def live_keys(D) -> set:
    """The live trader's backed horses as (UK off time, horse): the same horse in the place market is skipped too."""
    ids = live_backed(D)
    starts = {m.market_id: m for m in D.markets}
    return {(_off(starts[m]), bp.normalise_horse(starts[m].runners.get(s, ""))) for m, s in ids if m in starts}


NO_PREDS = pd.DataFrame(columns=["market_id", "selection_id", "predicted_bfsp"])


def run_market(day: date, kind: str, sel: pd.DataFrame, live: set | None) -> tuple[pd.DataFrame, object, set]:
    """Replay one market type (win or place) for the morning strategies on one shared book. ``live``: the live
    trader's horses as (off, horse); None on the win markets, where they are read from its ledger."""
    D = core.load_day(day, kind, preds=NO_PREDS.copy())
    if D is None:
        return pd.DataFrame(), None, live or set()
    if live is None:
        live = live_keys(D)
    m = attach_ids(sel.rename(columns={"track": "venue", "horse_name": "runner_name"}), D.markets)
    m = m[m["market_id"].notna()].copy()
    m["selection_id"] = m["selection_id"].astype(int)
    m["label"] = m["runner_name"]
    starts = {mm.market_id: mm for mm in D.markets}
    skip = {(str(r.market_id), int(r.selection_id)) for r in m.itertuples()
            if (_off(starts[str(r.market_id)]), bp.normalise_horse(r.runner_name)) in live}
    specs = []
    for name in STRATS:
        s_ = m[m["strategy"] == name].sort_values("race_time")
        specs.append((name, MorningEdgeSession, core.make_cfg(), {"selections": s_, "live_backed": skip}, BANK, False))
    out = core.run_agents(D, specs, entry_utc(day) - timedelta(minutes=5), t_end=retry_utc(day) + timedelta(minutes=1))
    rows = [led.assign(strategy=name, market=kind) for name, (led, _, _) in out.items()
            if led is not None and not led.empty]
    bets = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"  {kind}: {len(m)} selections matched to markets (of {len(sel)}), {len(skip)} the live trader's, "
          f"{len(bets)} ledger rows")
    return bets, D, live


def refresh_timeform(months_back: int = 1) -> None:
    """Keep the Railway Timeform cache in S3 current: any month missing (the first run pulls 2014 on, about 310
    paced requests), and the current and previous month again (late results). research/strat/fetch_timeform.py."""
    import time as _t
    from research.strat import fetch_timeform as FT
    have = {k.rsplit("/", 1)[-1] for k in _keys(TF_PREFIX + "/")}
    today = pd.Timestamp.today()
    recent = {str(p) for p in pd.period_range(today - pd.DateOffset(months=months_back), today, freq="M")}
    for m in pd.period_range("2014-01", today.strftime("%Y-%m"), freq="M"):
        for c in ("GBR", "IRE"):
            name = f"{c}_{m}.parquet"
            if name in have and str(m) not in recent:
                continue
            try:
                df = FT.month_frame(c, m.to_timestamp())
                buf = io.BytesIO()
                df.astype(str).replace({"None": None, "nan": None}).to_parquet(buf, index=False)
                _put(f"{TF_PREFIX}/{name}", buf.getvalue())
                print(f"  timeform {name}: {len(df)} rows")
            except Exception as exc:
                print(f"  timeform {name}: {type(exc).__name__} {exc}")
            _t.sleep(1.0)


_TF = None


def timeform_all() -> pd.DataFrame:
    """The Timeform cache from S3, read once per run."""
    global _TF
    if _TF is None:
        frames = []
        for k in _keys(TF_PREFIX + "/"):
            raw = core.get(k)
            if raw:
                frames.append(pd.read_parquet(io.BytesIO(raw), columns=["meetingDate", "horseCode", "horseName",
                                                                       "performanceRating", "scheduledTimeOfRaceLocal"]))
        _TF = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _TF


def tf_selections(day: str) -> pd.DataFrame:
    """tf_pb_bsp: Flat/AW runners whose last Timeform run (before the day) was a career-best by 5lb+."""
    tf = timeform_all()
    if tf.empty:
        print("  no Timeform cache in S3: tf_pb_bsp skipped")
        return pd.DataFrame()
    tf = tf.copy()
    tf["d"] = tf["meetingDate"].astype(str).str[:10]
    tf = tf[tf["d"] < day].copy()
    tf["r"] = pd.to_numeric(tf["performanceRating"], errors="coerce").where(lambda s: s > 0)
    tf = tf.sort_values(["horseCode", "d", "scheduledTimeOfRaceLocal"])
    tf["best_before"] = tf.groupby("horseCode")["r"].shift(1).groupby(tf["horseCode"]).cummax()
    last = tf.groupby("horseCode").tail(1)
    last = last[(last["r"] - last["best_before"]) >= 5]
    last = last.assign(hn=MF_norm(last["horseName"])).drop_duplicates("hn", keep="last")
    x = features_all()
    x = x[(x["race_date"] == day) & x["code"].isin(["flat", "aw"])].copy()
    x["hn"] = MF_norm(x["horse_name"])
    return x[x["hn"].isin(set(last["hn"]))].assign(strategy="tf_pb_bsp")


def MF_norm(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.upper().str.replace(r"\s*\([A-Z]{2,4}\)\s*$", "", regex=True)
            .str.replace(r"[^A-Z0-9]", "", regex=True))


def settle_tf(day: str, sel: pd.DataFrame) -> pd.DataFrame:
    """tf_pb_bsp backed at BSP, win and place, held to the result (the HRB database's results and SPs). Sized to win
    GBP100 at the morning WAP (a price known before the off), GBP200 a bet at most; no morning price, no bet."""
    if sel.empty:
        return pd.DataFrame()
    h = history()
    r = h[(h["race_date"] == day)][["horse_name", "track", "race_time", "bsp", "pbsp", "won", "placed", "morningwap",
                                    "morning_vol", "p_morningwap", "p_morning_vol", "fin"]]
    m = sel[["horse_name", "track", "race_time"]].merge(r, on=["horse_name", "track", "race_time"], how="left")
    m = m[(m["bsp"] > 1) & (m["morningwap"] > 1) & (m["morning_vol"] >= MF.L.MIN_AM_VOL)].copy()
    rows = []
    for mk, price, res in (("win", "bsp", "won"), ("place", "pbsp", "placed")):
        x = m[m[price] > 1].copy()
        if x.empty:
            continue
        if mk == "win":
            sizing = x["morningwap"]
        else:                                  # the morning place price; none matched in the morning, no place bet
            x = x[(x["p_morningwap"] > 1) & (x["p_morning_vol"] >= MF.L.MIN_AM_VOL / 5)].copy()
            sizing = x["p_morningwap"]
        if x.empty:
            continue
        x["matched"] = np.minimum(TO_WIN / (sizing - 1), MAX_STAKE)
        x["price"] = x[price]
        x["pnl"] = np.where(x[res] == 1, x["matched"] * (x["price"] - 1), -x["matched"])
        x["result"] = np.where(x[res] == 1, "WINNER", "LOSER")
        x["market_id"] = x["track"] + "|" + x["race_time"].astype(str) + "|" + mk
        rows.append(x.assign(strategy="tf_pb_bsp", market=mk, event="back_sp", best_back=np.nan,
                             label=x["horse_name"], ts=entry_utc(date.fromisoformat(day)).isoformat()))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def day_job(day_iso: str, booster, thr, write: bool, check: bool = False) -> list[dict]:
    day = date.fromisoformat(day_iso)
    if check and booster is not None:
        check_card_view(day_iso, booster)
    sel = selections(day_iso, booster, thr)
    tfs = tf_selections(day_iso)
    print(f"  selections: {sel.groupby('strategy').size().to_dict() if len(sel) else {}}, tf_pb_bsp {len(tfs)}")
    if write and len(sel):
        _put(f"{PREFIX}/selections/{day_iso}.csv", pd.concat([sel, tfs.reindex(columns=sel.columns) if len(tfs) else None])
             .to_csv(index=False).encode())
    allbets, rows = [], []
    live = None
    waps = {}
    for kind in ("win", "place"):
        bets, D, live = run_market(day, kind, sel, live)
        if D is None:
            continue
        waps[kind] = wap_lookup(day, kind)
        st = settle(bets[bets["event"] == "back"], D) if len(bets) else pd.DataFrame()
        if len(st):
            starts = {m.market_id: m for m in D.markets}
            st["off"] = st["market_id"].map(lambda m: bp.utc_iso_to_uk_hhmm(
                starts[m].start.strftime("%Y-%m-%dT%H:%M:%S.000Z")) if m in starts else None)
            st["wap"] = [waps[kind].get((o, bp.normalise_horse(lb)), (np.nan, np.nan))[0]
                         for o, lb in zip(st["off"], st["label"])]
        other = bets[bets["event"] != "back"] if len(bets) else pd.DataFrame()
        allbets += [st, other]
    allbets.append(settle_tf(day_iso, tfs))
    bets = pd.concat([b for b in allbets if b is not None and len(b)], ignore_index=True) if allbets else pd.DataFrame()
    if bets.empty:
        print("  no bets")
        return [{"day": day_iso, "note": "no bets (no books, no price file, or no selections)"}]
    for strat in ("qr_am", "steam_am", "tf_pb_bsp"):
        for mk in ("win", "place"):
            b = bets[(bets["strategy"] == strat) & (bets["market"] == mk)]
            matched = b[b.get("matched", 0) > 0] if len(b) else b
            if len(b) == 0:
                continue
            pm = matched.groupby("market_id")["pnl"].sum() if len(matched) else pd.Series(dtype=float)
            p = 1 / matched["price"] if len(matched) else pd.Series(dtype=float)
            won = matched["result"].eq("WINNER") if len(matched) else pd.Series(dtype=bool)
            want = b["size"].sum() if "size" in b and strat != "tf_pb_bsp" else matched["matched"].sum()
            rows.append({
                "day": day_iso, "forward": day >= FORWARD_FROM, "strategy": strat, "market": mk,
                "selections": int(len(b)), "bets": int(len(matched)),
                "skipped_live": int((b["event"] == "skipped_live").sum()),
                "no_book": int((b["event"] == "no_book").sum()),
                "staked": round(float(matched["matched"].sum()), 2), "wanted": round(float(want), 2),
                "wins": int(won.sum()), "expected": round(float(p.sum()), 4),
                "pnl_2pc": round(commission(pm, 0.02), 2), "pnl_5pc": round(commission(pm, 0.05), 2),
                "wap_gap_median": (round(float((matched["price"] / matched["wap"] - 1).median()), 4)
                                   if "wap" in matched and matched["wap"].notna().any() else None)})
    if write:
        _put(f"{PREFIX}/bets/{day_iso}.csv", bets.to_csv(index=False).encode())
        _put(f"{PREFIX}/days/{day_iso}.json", json.dumps(rows, default=str, indent=1).encode())
    for r in rows:
        print("  ", {k: v for k, v in r.items() if k not in ("day", "forward")})
    return rows


# ---------------------------------------------------------------------------------------------------------------
# the ledger and the gate
# ---------------------------------------------------------------------------------------------------------------

def ledger() -> pd.DataFrame:
    rows = []
    for k in _keys(f"{PREFIX}/days/"):
        raw = core.get(k)
        if raw:
            rows += [r for r in json.loads(raw) if "strategy" in r]
    return pd.DataFrame(rows)


def _boot(x: np.ndarray, n: int = 4000, seed: int = 7) -> tuple[float, float]:
    if len(x) < 2:
        return (np.nan, np.nan)
    m = np.random.default_rng(seed).choice(x, size=(n, len(x)), replace=True).sum(axis=1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def gate(f: pd.DataFrame) -> pd.DataFrame:
    """Each strategy against the gate fixed on 4 Nov, on the forward days."""
    fw = f[f["forward"].astype(bool)]
    out = []
    for s in ("qr_am", "steam_am", "tf_pb_bsp"):
        w, p = fw[(fw.strategy == s) & (fw.market == "win")], fw[(fw.strategy == s) & (fw.market == "place")]
        if w.empty:
            continue
        n, W, E = int(w.bets.sum()), float(w.wins.sum()), float(w.expected.sum())
        x2 = (W - E) ** 2 / E + (W - E) ** 2 / (n - E) if 0 < E < n else np.nan
        pv = float(C2.sf(x2, 1)) if np.isfinite(x2) else np.nan
        Pw, Pe = float(p.wins.sum()), float(p.expected.sum())
        daily = pd.concat([w, p]).groupby("day")["pnl_5pc"].sum().to_numpy()
        lo, hi = _boot(daily)
        fill = float(w.staked.sum() / w.wanted.sum()) if w.wanted.sum() else np.nan
        gs = pd.to_numeric(w.wap_gap_median, errors="coerce")
        gap = float(gs.median()) if gs.notna().any() else np.nan
        mon = w.assign(m=w.day.str[:7]).groupby("m").agg(n=("bets", "sum"), W=("wins", "sum"), E=("expected", "sum"))
        bad_month = bool(((mon.n >= 100) & (mon.W / mon.E < 0.85)).any())
        cum = np.cumsum(daily)
        dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0.0
        ae = W / E if E else np.nan
        ready = n >= 1000 and w.day.nunique() >= 60
        ok = (ae > 1 and pv < 0.05 and Pe > 0 and Pw / Pe > 1 and lo > 0 and fill >= 0.80
              and (not np.isfinite(gap) or gap >= -0.03) and not bad_month)
        stop = (n >= 300 and ae < 0.90) or dd < -3000
        out.append({"strategy": s, "days": int(w.day.nunique()), "win_bets": n, "win_A/E": round(ae, 3),
                    "chi2_p": pv, "place_A/E": round(Pw / Pe, 3) if Pe else None,
                    "pnl_5pc": round(float(daily.sum()), 2), "ci90_total": f"{lo:+.0f} to {hi:+.0f}",
                    "fill_ratio": round(fill, 3), "wap_gap_median": gap, "skipped_live": int(w.skipped_live.sum()),
                    "drawdown": round(dd, 2),
                    "gate": "STOP" if stop else ("PASSED" if ready and ok else ("failed" if ready else "running"))})
    return pd.DataFrame(out)


def report(f: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    if f.empty:
        print("the morning-edges ledger is empty")
        return
    f = f.copy()
    f["forward"] = f["forward"].astype(bool)
    for label, sub in (("FORWARD (from 5 Nov 2026, the gate's days)", f[f.forward]),
                       ("BACKFILL (before 5 Nov: the fill check, shown apart)", f[~f.forward])):
        if sub.empty:
            continue
        g = sub.groupby(["strategy", "market"]).agg(days=("day", "nunique"), bets=("bets", "sum"),
                                                     skipped_live=("skipped_live", "sum"), staked=("staked", "sum"),
                                                     wanted=("wanted", "sum"), wins=("wins", "sum"),
                                                     expected=("expected", "sum"), pnl_2pc=("pnl_2pc", "sum"),
                                                     pnl_5pc=("pnl_5pc", "sum"),
                                                     wap_gap=("wap_gap_median",
                                                              lambda v: pd.to_numeric(v, errors="coerce").median()))
        g["A/E"] = g.wins / g.expected
        g["fill_ratio"] = g.staked / g.wanted
        g["roi_5pc%"] = 100 * g.pnl_5pc / g.staked
        print(f"\n== the morning edges, {label}")
        print(g.round(3).to_string())
    gt = gate(f)
    if len(gt):
        print("\n== the gate (fixed 4 Nov 2026; read at 1,000 win bets and 60 days; the owner decides)")
        print(gt.to_string(index=False))


# ---------------------------------------------------------------------------------------------------------------

def train(cutoff: str, force: bool) -> None:
    if core.get(f"{PREFIX}/model/meta.json") is not None and not force:
        print("a frozen model is already in S3; --force to replace it (only before the forward start)")
        return
    booster, thr, n = MF.train_steamer(history(), cutoff)
    meta = {"threshold": thr, "rows": n, "cutoff": cutoff, "start": "2018-01-01", "features": MF.CLEAN,
            "trained_utc": datetime.now(timezone.utc).isoformat(), "spec": "docs/paper_trader_morning_edges.md"}
    _put(f"{PREFIX}/model/booster.txt", booster.model_to_string().encode())
    _put(f"{PREFIX}/model/meta.json", json.dumps(meta, indent=1).encode())
    print(f"frozen: {n:,} training rows to {cutoff}, threshold {thr:.4f}")


def pending_days(today: date, backfill_from: date | None) -> list[str]:
    done = {k.rsplit("/", 1)[-1][:10] for k in _keys(f"{PREFIX}/days/")}
    start = backfill_from or FORWARD_FROM
    last = today - timedelta(days=1)
    return [(start + timedelta(n)).isoformat() for n in range((last - start).days + 1)
            if (start + timedelta(n)).isoformat() not in done]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="fit the CLEAN steamer model and freeze it in S3")
    ap.add_argument("--cutoff", default="2026-10-31")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--days", default="pending", help="'pending', a..b, or a comma list")
    ap.add_argument("--backfill-from", default=None)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args(argv)
    if a.train:
        train(a.cutoff, a.force)
        return 0
    if not a.report_only:
        booster, thr, meta = load_model()
        refresh_timeform()
        if booster is None:
            print("no frozen model in S3: run --train first (steam_am waits; qr_am and tf_pb_bsp run)")
        if a.days == "pending":
            bf = date.fromisoformat(a.backfill_from) if a.backfill_from else None
            days = pending_days(datetime.now(timezone.utc).date(), bf)
        elif ".." in a.days:
            x, y = (date.fromisoformat(s) for s in a.days.split(".."))
            days = [(x + timedelta(n)).isoformat() for n in range((y - x).days + 1)]
        else:
            days = [s for s in a.days.split(",") if s]
        print(f"days to replay: {days}")
        for i, d in enumerate(days):
            try:
                day_job(d, booster, thr, write=not a.no_write, check=(i == 0))
            except Exception:
                print(f"  {d}: failed\n{traceback.format_exc()}")
    report(ledger())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
