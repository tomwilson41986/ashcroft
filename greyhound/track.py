"""The greyhound model's forward test (the owner's ask, 5 Oct 2026: "start tracking this"): paper bets only.

The model is frozen once, fitted on every GBGB race before the first tracked day, and the rules are fixed with it
(``RULES``, written into the frozen model's meta before any tracked day is seen). Each day after racing:

1. The day's GB races from GBGB (the raw day file, fetched if the store lacks it), priced by the frozen model from the
   dogs' earlier days only (the engine is lag-safe: the day's own results never reach its features).
2. The day's recorded Betfair greyhound books (``betfair_live/<day>/books_greyhound.csv``, live-record.yml, the
   delayed key's feed) read at fixed marks: the first book recorded, and 60, 10 and 1 minutes before the off.
3. At each mark, a paper back of ``STAKE`` on every dog whose edge at the best back price clears the threshold, filled
   only when the best back offered at least the stake; settled at that price and, for comparison, at the BSP, with
   CLV against the BSP.

Nothing here can place an order; nothing reads a price of the race into the model. The ledger is kept a day a file
in the sources store (``greyhound/track/days/<day>.parquet``) with the day's priced runners
(``greyhound/track/pred/<day>.parquet``), and the running summary in ``greyhound/track/summary.json``.

    python -m greyhound.track --freeze --cutoff 2026-10-05        # once: fit and freeze the model and rules
    python -m greyhound.track --days 3                             # the nightly run: the last 3 days not yet final
    python -m greyhound.track --summary
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from greyhound.model import COM, PARAMS, logloss, modelling_rows, race_norm

log = logging.getLogger(__name__)

PREFIX = "greyhound/track"
MODEL_KEY = f"{PREFIX}/model.txt"
META_KEY = f"{PREFIX}/meta.json"
STAKE = 2.0                       # paper stake (Betfair's minimum is GBP1): the fill test is that the book offered it
CAP = 20.0                        # no bet above this price
MARKS = ("first", "off_60", "off_10", "off_1")
EDGES = (0.1, 0.2, 0.3)
#: fixed before the first tracked day, with the model; the summary judges the primary rule only
RULES = {
    "primary": {"mark": "first", "edge": 0.2},
    "reported": [{"mark": m, "edge": e} for m in MARKS for e in EDGES],
    "stake": STAKE, "cap": CAP, "commission": COM,
    "gate": "judged after 2,000 primary bets or 30 tracked days, whichever is later: small real stakes only if the "
            "return at the taken price has a 90% lower bound above 0 and the mean CLV against the BSP is above 0",
}


# --------------------------------------------------------------------------------------------------------------------
# The frozen model
# --------------------------------------------------------------------------------------------------------------------

def freeze(df: pd.DataFrame, features: list[str], cutoff: str, date_from: str = "2019-01-01") -> tuple[str, dict]:
    """Fit on every modelling race in [date_from, cutoff), early-stopped on the last tenth; the booster as text and
    its meta (the features, the cutoff, the rules)."""
    import lightgbm as lgb
    d = modelling_rows(df)
    d = d[(d.t >= date_from) & (d.t < cutoff)]
    cut = d.t.quantile(0.9)
    tr, es = d[d.t < cut], d[d.t >= cut]
    m = lgb.train(PARAMS, lgb.Dataset(tr[features], tr.won), 3000, valid_sets=[lgb.Dataset(es[features], es.won)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    meta = {"cutoff": cutoff, "date_from": date_from, "fit_rows": len(d), "rounds": m.best_iteration,
            "features": features, "rules": RULES, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return m.model_to_string(num_iteration=m.best_iteration), meta


def load_frozen(store):
    import lightgbm as lgb
    text, meta = store.get(MODEL_KEY), store.get(META_KEY)
    if not text or not meta:
        return None, None
    return lgb.Booster(model_str=text.decode()), json.loads(meta)


def price_day(df: pd.DataFrame, booster, features: list[str], day: str) -> pd.DataFrame:
    """The day's modelling races, priced by the frozen model and normalised within each race."""
    d = modelling_rows(df)
    d = d[d.race_date == day]
    out = d[["race_id", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "field", "won",
             "sp_decimal"]].copy()
    out["p_model"] = race_norm(pd.Series(booster.predict(d[features]), index=d.index), d.race_id) if len(d) else []
    return out


# --------------------------------------------------------------------------------------------------------------------
# The books and the paper bets
# --------------------------------------------------------------------------------------------------------------------

def book_marks(books: pd.DataFrame, markets: pd.DataFrame, day: date) -> pd.DataFrame:
    """The recorder's marks (betfair_recorder.marks) and the first book recorded before the off."""
    from betfair_recorder import marks
    mk = marks(books, markets, day)
    b = books.copy()
    b["polled"] = pd.to_datetime(b.polled_utc, utc=True, errors="coerce")
    offs = pd.to_datetime(markets.drop_duplicates("market_id").set_index("market_id").market_start_utc, utc=True,
                          errors="coerce")
    b["off"] = b.market_id.map(offs)
    b["minutes_to_off"] = (b.off - b.polled).dt.total_seconds() / 60.0
    pre = b[(b.source != "final") & (b.status == "OPEN") & b.inplay.astype(str).isin(["0", "False", "false"])
            & (b.minutes_to_off > 0)]
    first = pre.sort_values("polled").drop_duplicates(["market_id", "selection_id"]).assign(mark="first")
    first["bsp"] = np.nan
    first["race_date"] = f"{day:%Y-%m-%d}"
    cols = ["market_id", "selection_id", "mark", "polled_utc", "minutes_to_off", "back1", "back1_size", "bsp"]
    return pd.concat([mk[cols], first[cols]], ignore_index=True)


def join_markets(pred: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """The day's priced runners on Betfair's GB win markets, by UK date, off time, track and trap; a race is kept only
    when every one of its runners is found."""
    w = markets[(markets.market_type == "WIN") & (markets.country.fillna("GB") == "GB")].drop_duplicates(
        ["market_id", "selection_id"]).copy()
    uk = pd.to_datetime(w.market_start_utc, utc=True, errors="coerce").dt.tz_convert("Europe/London")
    w["race_date"], w["race_time"] = uk.dt.strftime("%Y-%m-%d"), uk.dt.strftime("%H:%M")
    w["track"] = w.venue.astype(str).str.strip()
    w["trap"] = pd.to_numeric(w.runner_name.astype(str).str.extract(r"^(\d+)\.")[0], errors="coerce")
    w["market_id"] = w.market_id.astype(str)
    m = pred.merge(w[["race_date", "race_time", "track", "trap", "market_id", "selection_id"]],
                   on=["race_date", "race_time", "track", "trap"], how="inner")
    whole = m.groupby("race_id").trap.transform("size") == m.field
    return m[whole & (m.groupby("market_id").race_id.transform("nunique") == 1)]


def paper_bets(priced: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    """Every reported rule's paper backs on the day (one row a rule a dog), with the fill test and the settlement."""
    mk = mk.copy()
    mk["market_id"] = mk.market_id.astype(str)
    mk["selection_id"] = pd.to_numeric(mk.selection_id, errors="coerce")
    bsp = mk[mk.mark == "final"][["market_id", "selection_id", "bsp"]]
    d = priced.merge(bsp, on=["market_id", "selection_id"], how="left")
    rows = []
    for mark in MARKS:
        at = mk[mk.mark == mark][["market_id", "selection_id", "polled_utc", "minutes_to_off", "back1", "back1_size"]]
        e = d.merge(at, on=["market_id", "selection_id"], how="inner")
        e = e[e.back1 > 1]
        e["edge"] = e.p_model * (1 + (e.back1 - 1) * (1 - COM)) - 1
        for th in EDGES:
            sel = e[(e.edge > th) & (e.back1 <= CAP)].copy()
            sel["mark"], sel["threshold"] = mark, th
            rows.append(sel)
    if not rows or not sum(len(r) for r in rows):
        return pd.DataFrame()
    b = pd.concat(rows, ignore_index=True)
    b["filled"] = b.back1_size.fillna(0) >= STAKE
    b["pnl"] = np.where(b.won == 1, (b.back1 - 1) * (1 - COM), -1.0) * STAKE
    b["pnl_bsp"] = np.where(b.bsp > 1, np.where(b.won == 1, (b.bsp - 1) * (1 - COM), -1.0) * STAKE, np.nan)
    b["clv"] = b.back1 / b.bsp - 1
    return b


def track_day(store, day: date, df: pd.DataFrame, booster, meta: dict, s3=None) -> dict:
    """One day: price it, read its books, write its ledger. A day whose settled books are missing is written with
    its BSPs empty and done again the next night."""
    from betfair_recorder import _read_day_csv
    ds = f"{day:%Y-%m-%d}"
    pred = price_day(df, booster, meta["features"], ds)
    books = _read_day_csv(day, "books_greyhound.csv", s3=s3)
    markets = _read_day_csv(day, "markets_greyhound.csv", s3=s3)
    out = {"day": ds, "races_priced": int(pred.race_id.nunique())}
    if books is None or markets is None or not len(pred):
        out["note"] = "no recorded greyhound books" if books is None or markets is None else "no GBGB races"
        return out
    priced = join_markets(pred, markets)
    mk = book_marks(books, markets, day)
    bets = paper_bets(priced, mk)
    final = mk[mk.mark == "final"]
    keep = priced.merge(mk.assign(market_id=mk.market_id.astype(str),
                                  selection_id=pd.to_numeric(mk.selection_id, errors="coerce"))
                        .pivot_table(index=["market_id", "selection_id"], columns="mark", values="back1",
                                     aggfunc="first").add_prefix("back1_").reset_index(),
                        on=["market_id", "selection_id"], how="left")
    keep = keep.merge(final.assign(market_id=final.market_id.astype(str),
                                   selection_id=pd.to_numeric(final.selection_id, errors="coerce"))
                      [["market_id", "selection_id", "bsp"]], on=["market_id", "selection_id"], how="left")
    store.put_parquet(f"{PREFIX}/pred/{ds}.parquet", keep)
    store.put_parquet(f"{PREFIX}/days/{ds}.parquet", bets)
    out.update({"races_on_betfair": int(priced.race_id.nunique()), "bsp_settled": bool(keep.bsp.notna().any()),
                "unmatched_tracks": sorted(set(pred.track) - set(priced.track)),
                "primary_bets": int(((bets.mark == RULES["primary"]["mark"])
                                     & (bets.threshold == RULES["primary"]["edge"])).sum()) if len(bets) else 0})
    return out


# --------------------------------------------------------------------------------------------------------------------
# The summary
# --------------------------------------------------------------------------------------------------------------------

def _settle(b: pd.DataFrame, col: str) -> dict:
    b = b.dropna(subset=[col])
    if not len(b):
        return {"bets": 0}
    r = b[col] / STAKE
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan
    return {"bets": int(len(b)), "staked": round(float(STAKE * len(b)), 2), "pnl": round(float(b[col].sum()), 2),
            "roi%": round(100 * float(r.mean()), 2),
            "90%": [round(100 * float(r.mean() - 1.645 * se), 2), round(100 * float(r.mean() + 1.645 * se), 2)]}


def summary(store) -> dict:
    keys = sorted(k for k in store.listing(f"{PREFIX}/days/") if k.endswith(".parquet"))
    bets = [store.get_parquet(k) for k in keys]
    bets = pd.concat([b for b in bets if b is not None and len(b)], ignore_index=True) if bets else pd.DataFrame()
    pkeys = sorted(k for k in store.listing(f"{PREFIX}/pred/") if k.endswith(".parquet"))
    pred = [store.get_parquet(k) for k in pkeys]
    pred = pd.concat([p for p in pred if p is not None and len(p)], ignore_index=True) if pred else pd.DataFrame()
    out = {"days": len(keys), "first_day": keys[0][-18:-8] if keys else None, "last_day": keys[-1][-18:-8] if keys else None,
           "rules": RULES}
    if len(pred):
        ll = {"runners": len(pred), "races": int(pred.race_id.nunique()), "model": round(logloss(pred.p_model, pred.won), 5)}
        for col in ("bsp", "back1_first", "back1_off_1"):
            if col in pred.columns:
                q = pred[col].where(pred[col] > 1)
                whole = q.notna().groupby(pred.race_id).transform("all")
                if whole.any():
                    p = race_norm(1.0 / q.where(whole), pred.race_id)
                    ll[col] = round(logloss(p[whole], pred.won[whole]), 5)
                    ll[f"model_on_{col}_races"] = round(logloss(pred.p_model[whole], pred.won[whole]), 5)
        out["logloss"] = ll
    if len(bets):
        rules = {}
        for (mark, th), g in bets.groupby(["mark", "threshold"]):
            f = g[g.filled]
            rules[f"{mark}: edge>{th:.1f}"] = {"signals": int(len(g)), "filled": int(len(f)),
                                               "per_day": round(len(f) / max(1, g.race_date.nunique()), 1),
                                               "at_price": _settle(f, "pnl"), "at_bsp": _settle(f, "pnl_bsp"),
                                               "clv_vs_bsp%": round(100 * float(f.clv.mean()), 2) if f.clv.notna().any()
                                               else None}
        out["rules_by_mark"] = rules
        p = RULES["primary"]
        prim = bets[(bets.mark == p["mark"]) & (bets.threshold == p["edge"]) & bets.filled]
        out["primary"] = rules.get(f"{p['mark']}: edge>{p['edge']:.1f}")
        out["primary_by_day"] = {d: {"bets": int(len(g)), "pnl": round(float(g.pnl.sum()), 2),
                                     "pnl_bsp": round(float(g.pnl_bsp.sum()), 2)}
                                 for d, g in prim.groupby("race_date")}
    return out


def markdown(s: dict) -> str:
    lines = [f"## Greyhound forward test: {s.get('first_day')} to {s.get('last_day')} ({s.get('days', 0)} days)", ""]
    p = s.get("primary")
    if p:
        ap, ab = p["at_price"], p["at_bsp"]
        lines += [f"**Primary rule** (first recorded price, edge > 0.2, GBP{STAKE:g} paper): {p['filled']} bets "
                  f"({p['per_day']}/day), return at the price {ap.get('roi%')}% (90% {ap.get('90%')}), "
                  f"CLV vs BSP {p['clv_vs_bsp%']}%, the same bets at BSP {ab.get('roi%')}%", ""]
    if s.get("logloss"):
        lines += ["Log-loss: " + ", ".join(f"{k} {v}" for k, v in s["logloss"].items()), ""]
    if s.get("rules_by_mark"):
        lines += ["| Rule | Filled / signals | Per day | At the price (90%) | CLV | At BSP |", "|---|---|---|---|---|---|"]
        for k, r in s["rules_by_mark"].items():
            lines.append(f"| {k} | {r['filled']} / {r['signals']} | {r['per_day']} | {r['at_price'].get('roi%')}% "
                         f"{r['at_price'].get('90%')} | {r['clv_vs_bsp%']}% | {r['at_bsp'].get('roi%')}% |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# The day's GBGB results
# --------------------------------------------------------------------------------------------------------------------

def runs_through(store, day: date, first_year: int = 2018) -> pd.DataFrame:
    """Every GBGB run to ``day``: the year tables, and the raw day files after the tables' last day (each fetched from
    GBGB when the store lacks it)."""
    from greyhound.data import clean
    from sources import gbgb
    keys = sorted(k for k in store.listing("gbgb/") if k.endswith(".parquet") and "/runs_" in k
                  and first_year <= int(k[-12:-8]) <= day.year)
    frames = [store.get_parquet(k) for k in keys]
    df = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True)
    last = date.fromisoformat(str(df.race_date.max()))
    extra, s = [], None
    d = min(last, day - timedelta(days=gbgb.REFRESH_DAYS))   # the last days again: GBGB corrects them for a day or two
    while d <= day:
        raw = store.get(gbgb.raw_key(d))
        if raw is None or (day - d).days < 1:
            s = s or gbgb.session()
            got = gbgb.fetch_day(s, d)
            store.put(gbgb.raw_key(d), gbgb.gz(json.dumps(got, separators=(",", ":")).encode()))
            raw = gbgb.gz(json.dumps(got).encode())
        extra += gbgb.rows(json.loads(gbgb.gunzip(raw)))
        d += timedelta(days=1)
    if extra:
        new = pd.DataFrame(extra)
        df = pd.concat([df[~df.race_date.isin(set(new.race_date))], new], ignore_index=True)
    df = df.drop_duplicates(["race_id", "trap", "dog_id"], keep="last")
    return clean(df)


def main(argv=None) -> int:
    from greyhound.metrics import GreyhoundMetricsEngine
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="the sources store root (default S3)")
    ap.add_argument("--freeze", action="store_true", help="fit and freeze the model and rules (once; --refreeze again)")
    ap.add_argument("--refreeze", action="store_true")
    ap.add_argument("--cutoff", default=None, help="the frozen model fits on races before this day (the first tracked)")
    ap.add_argument("--days", type=int, default=0, help="track the last N days (to yesterday, UK)")
    ap.add_argument("--date", default=None, help="track this one day")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--out", default=None, help="write the summary JSON here (and markdown beside it)")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    from betfair_recorder import uk_today
    today = uk_today()
    report = {}
    booster, meta = load_frozen(store)
    days = []
    if a.date:
        days = [date.fromisoformat(a.date)]
    elif a.days:
        first = date.fromisoformat(meta["cutoff"]) if meta else None
        days = [today - timedelta(days=i) for i in range(a.days, 0, -1)]
        days = [d for d in days if first is None or d >= first]
    need_runs = (a.freeze and (booster is None or a.refreeze)) or days
    if need_runs:
        last = max(days) if days else date.fromisoformat(a.cutoff) - timedelta(days=1)
        eng = GreyhoundMetricsEngine()
        df = eng.calculate_all(runs_through(store, last))
        if a.freeze and (booster is None or a.refreeze):
            if not a.cutoff:
                ap.error("--freeze needs --cutoff (the first day to track)")
            text, meta = freeze(df, eng.features, a.cutoff)
            store.put(MODEL_KEY, text.encode())
            store.put(META_KEY, json.dumps(meta, indent=1).encode())
            import lightgbm as lgb
            booster = lgb.Booster(model_str=text)
            report["frozen"] = {k: v for k, v in meta.items() if k != "features"}
        elif a.freeze:
            report["frozen"] = "already frozen: " + meta["frozen_utc"]
        if days and booster is None:
            ap.error("no frozen model: run --freeze --cutoff first")
        days = [d for d in days if d >= date.fromisoformat(meta["cutoff"])]
        report["days"] = []
        for d in days:
            done = store.get_parquet(f"{PREFIX}/pred/{d:%Y-%m-%d}.parquet")
            if done is not None and len(done) and done.bsp.notna().any() and not a.date:
                report["days"].append({"day": f"{d:%Y-%m-%d}", "note": "already final"})
                continue
            report["days"].append(track_day(store, d, df, booster, meta))
    if a.summary or days:
        s = summary(store)
        store.put(f"{PREFIX}/summary.json", json.dumps(s, indent=1, default=str).encode())
        report["summary"] = s
        if a.out:
            Path(a.out).with_suffix(".md").write_text(markdown(s))
    txt = json.dumps(report, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
