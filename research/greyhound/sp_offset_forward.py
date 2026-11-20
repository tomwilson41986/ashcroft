"""The residual on the BSP, forward and on paper (the owner, 20 Nov 2026: the discovery round, build 6; ledger
greyhound-sp-offset-forward-1120). No Betfair key, no order: a signal written before each race, settled at the SP.

The residual study (research/greyhound/sp_offset.py) read six rules at the SP on years the model never saw. This is
the same model and the same cuts on days nobody has seen:

- ``--freeze``: the residual as the seed-averaged study read it (greyhound-sp-offset-seeds-1120): the mean of one
  fit a seed (``SEEDS``) on the feature cache, every wholly priced race to its last day less the races a twin row
  touches, without the features the cut check barred (``DROP_FEATURES``), each fit early-stopped on the last tenth
  of the days; the cuts read off those days. Stored once. It is never fitted again.
- ``--serve``: the day's card built as the live price run builds it (the recorder's catalogue, GBGB history to the
  day before, the same engine blocks as the cache), the frozen residual centred within each race, the six rules' picks
  written with the time. A race is written once, at its first serve at least ``LEAD_MIN`` minutes before its off, and
  never changed: the first serve wins.
- ``--settle``: a day's picks against the settled books: a back of 1 or a lay risking 1 at the Betfair SP, netted by
  race, commission off a gain (research/greyhound/sp_harness.py). Only races where every served runner got an SP
  count, as in the study (wholly priced races).
- ``--read``: every settled forward day, each rule alone and pooled with the study's test years
  (``stage2_daily.csv``), the deflated Sharpe at the one greyhound trial count.

    python -m research.greyhound.sp_offset_forward --freeze
    python -m research.greyhound.sp_offset_forward --serve [--date 2026-11-21]
    python -m research.greyhound.sp_offset_forward --settle --days 3 --read --out out/study
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

log = logging.getLogger(__name__)

TAG = "greyhound-sp-offset-forward-1120"
PREFIX = "greyhound/sp_offset"
MODEL_KEY, META_KEY, DEV_DAILY_KEY = f"{PREFIX}/models.json", f"{PREFIX}/meta.json", f"{PREFIX}/stage2_daily.csv"
SEEDS = tuple(range(1, 21))                              # both halves of the seed-averaged study
DROP_FEATURES = ("gp_bsp_ae_car", "gp_bsp_mkt_car")      # the cut check (greyhound-cut-check-1120)
FIRST_DAY = "2026-11-21"          # the first forward day: after the freeze, after every day any study has read
LEAD_MIN = 10.0                   # a race is served only this long or more before its off
MIN_FIELD = 4                     # the races a model prices (greyhound/model.py modelling_rows)
SERVED_COLS = ["day", "race_id", "market_id", "selection_id", "trap", "dog_id", "dog_name", "track", "race_time",
               "off_utc", "field", "s", "served_utc", "code_hash"]


# --------------------------------------------------------------------------------------------------------------------
# Freeze
# --------------------------------------------------------------------------------------------------------------------

def freeze_fit(d: pd.DataFrame, feats: list[str], seeds: tuple[int, ...] | None = None):
    """The fits, one a seed (one fit with the study's own seed when none is given): every row of ``d`` (wholly priced
    races grouped by race in time order, with ``day``), the last tenth of its days for early stopping and for the
    cuts, which are read off the mean of the fits. Returns the boosters and their meta."""
    from research.greyhound import sp_offset as so
    day = d.day.to_numpy()
    days = np.unique(day)
    cut = days[int(len(days) * 0.9)]
    tr, es = np.flatnonzero(day < cut), np.flatnonzero(day >= cut)
    race = pd.factorize(d.race_id)[0]
    won = d.won.to_numpy(float)
    base = np.log(np.clip(d.bsp_p_norm.to_numpy(float), 1e-9, 1.0))
    X = d[feats].to_numpy(np.float32, na_value=np.nan)
    Xtr, Xes = X[tr], X[es]
    del X
    models = [so.fit(Xtr, won[tr], race[tr], base[tr], Xes, won[es], race[es], base[es],
                     params=None if sd is None else dict(so.PARAMS, seed=int(sd))) for sd in (seeds or (None,))]
    f_es = np.mean([m.predict(Xes, num_iteration=m.best_iteration, raw_score=True) for m in models], axis=0)
    s_es = f_es - pd.Series(f_es).groupby(race[es]).transform("mean").to_numpy()
    meta = {"tag": TAG, "features": list(feats), "rounds": [int(m.best_iteration or 0) for m in models],
            "seeds": [int(x) for x in seeds] if seeds else None,
            "cuts": {name: float(np.quantile(s_es, q)) for name, _, q in so.STAGE2},
            "fit_from": str(day.min()), "fit_to": str(day.max()), "early_stop_from": str(cut),
            "fit_rows": int(len(d)), "fit_races": int(d.race_id.nunique())}
    return models, meta


def freeze(store, force: bool = False) -> dict:
    from greyhound import fastloop as fl
    from research.greyhound.twin_check import flags
    if store.get(META_KEY) and not force:
        return {"note": "already frozen: not fitted again", **json.loads(store.get(META_KEY))}
    df, cmeta = fl.load_cache(store)
    feats = [f for f in fl.feature_set(cmeta, "card_safe") if f in df.columns and f not in fl.LABELS
             and f not in DROP_FEATURES]
    df = df.assign(day=pd.to_datetime(df.t).dt.strftime("%Y-%m-%d"))
    keep = df.bsp_p_norm.notna().to_numpy()
    touched = flags(df).touched.to_numpy()               # read on every row: a twin may be an unpriced copy
    left_out = int(df.race_id[keep & touched].nunique())
    d = df[keep & ~touched]
    del df
    if str(d.day.max()) >= FIRST_DAY:
        raise SystemExit(f"the cache reaches {d.day.max()}: a forward day ({FIRST_DAY} on) would be in the fit")
    models, meta = freeze_fit(d, feats, seeds=SEEDS)
    meta.update({"cache_hash": cmeta.get("hash"), "cache_last_day": cmeta.get("last_day"), "code_hash": fl.code_hash(),
                 "touched_races_left_out": left_out, "features_barred": list(DROP_FEATURES),
                 "first_forward_day": FIRST_DAY, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    store.put(MODEL_KEY, json.dumps([m.model_to_string(num_iteration=m.best_iteration) for m in models]).encode())
    store.put(META_KEY, json.dumps(meta, indent=1).encode())
    return meta


def load_frozen(store):
    """The frozen fits (a list of boosters) and their meta; (None, None) when nothing is frozen."""
    import lightgbm as lgb
    text, meta = store.get(MODEL_KEY), store.get(META_KEY)
    if not text or not meta:
        return None, None
    return [lgb.Booster(model_str=t) for t in json.loads(text)], json.loads(meta)


def score(models, x: np.ndarray) -> np.ndarray:
    """The residual: the mean raw score of the frozen fits."""
    return np.mean([m.predict(x, raw_score=True) for m in models], axis=0)


# --------------------------------------------------------------------------------------------------------------------
# Serve
# --------------------------------------------------------------------------------------------------------------------

def signal(card: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """The card's runners with the residual centred on their race (``s``). Races under ``MIN_FIELD`` runners on the
    card are left out: the model never saw one."""
    c = card.assign(_f=np.asarray(scores, float))
    c = c[c.groupby("race_id")._f.transform("size") >= MIN_FIELD]
    return c.assign(s=c._f - c.groupby("race_id")._f.transform("mean")).drop(columns="_f")


def first_serve_wins(old: pd.DataFrame | None, new: pd.DataFrame, now: datetime, lead_min: float = LEAD_MIN) -> pd.DataFrame:
    """The day's served rows after this serve: every race already written is kept as it was; a new race is added only
    if its off is at least ``lead_min`` minutes after ``now``."""
    off = pd.to_datetime(new.off_utc, utc=True, errors="coerce")
    ok = off >= pd.Timestamp(now) + pd.Timedelta(minutes=lead_min)
    seen = set() if old is None or not len(old) else set(old.race_id.astype(str))
    add = new[ok.to_numpy() & ~new.race_id.astype(str).isin(seen).to_numpy()]
    return add if old is None or not len(old) else pd.concat([old, add], ignore_index=True)


def card_frame(store, day: date, s3=None):
    """The day's card through the engine as the feature cache was built (greyhound/fastloop.py ``build_cache``: the
    same blocks, runs and price years), the card rows only, with each runner's Betfair market and selection."""
    from betfair_recorder import _read_day_csv
    from greyhound import fastloop as fl
    from greyhound.card import cards, with_cards
    from greyhound.data import clean
    from greyhound.live import drop_removed, removed_runners
    from greyhound.metrics import GreyhoundMetricsEngine
    from greyhound.model import load_prices
    from greyhound.track import runs_through
    markets = _read_day_csv(day, "markets_greyhound.csv", s3=s3)
    if markets is None or not len(markets):
        return None
    trial_rows: list = []
    hist = runs_through(store, day - timedelta(days=1), trials_out=trial_rows)
    card = cards(markets, hist, trials=trial_rows[0] if trial_rows else None)
    card = card[card.race_date == f"{day:%Y-%m-%d}"]
    card = drop_removed(card, removed_runners(_read_day_csv(day, "books_greyhound.csv", s3=s3)))
    if not len(card):
        return None
    eng = GreyhoundMetricsEngine(blocks=fl.BLOCKS)
    df = eng.calculate_all(clean(with_cards(hist, card)), prices=load_prices(store, fl.PRICE_YEARS))
    c = df[df.is_card.fillna(False).astype(bool)]
    key = card[["race_id", "trap", "market_id", "selection_id"]].drop_duplicates(["race_id", "trap"])
    return c.merge(key, on=["race_id", "trap"], how="left")


def serve(store, day: date, s3=None, now: datetime | None = None) -> dict:
    from greyhound import fastloop as fl
    ds = f"{day:%Y-%m-%d}"
    if ds < FIRST_DAY:
        return {"day": ds, "note": f"before the first forward day ({FIRST_DAY}): not served"}
    m, meta = load_frozen(store)
    if m is None:
        return {"day": ds, "note": "no frozen residual stored: not served (run --freeze once)"}
    c = card_frame(store, day, s3=s3)
    if c is None or not len(c):
        return {"day": ds, "note": "no data found: no greyhound catalogue or no GB win markets on the card"}
    missing = [f for f in meta["features"] if f not in c.columns]
    if missing:
        raise SystemExit(f"the frozen residual reads features this build no longer makes: {missing[:10]}")
    now = now or datetime.now(timezone.utc)
    x = c[meta["features"]].to_numpy(np.float32, na_value=np.nan)
    sig = signal(c, score(m, x))
    off = pd.to_datetime(sig.race_date.astype(str) + " " + sig.race_time.astype(str), errors="coerce") \
        .dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    new = sig.assign(day=ds, off_utc=off.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     served_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"), code_hash=fl.code_hash())[SERVED_COLS]
    raw = store.get(f"{PREFIX}/served/{ds}.csv")
    old = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str, "race_id": str}) if raw else None
    out = first_serve_wins(old, new, now)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    store.put(f"{PREFIX}/served/{ds}.csv", buf.getvalue().encode())
    return {"day": ds, "card_races": int(c.race_id.nunique()), "races_served_before": 0 if old is None else int(old.race_id.nunique()),
            "races_served": int(out.race_id.nunique()), "runners_served": int(len(out)),
            "code_hash": fl.code_hash(), "frozen_code_hash": meta.get("code_hash"),
            "code_changed_since_freeze": fl.code_hash() != meta.get("code_hash")}


# --------------------------------------------------------------------------------------------------------------------
# Settle and read
# --------------------------------------------------------------------------------------------------------------------

def settle_day(served: pd.DataFrame, close: pd.DataFrame, cuts: dict, comm: float) -> tuple[pd.DataFrame, dict]:
    """A day's served runners against the settled books: (rule, day, race_id, risk, gross, net) a race a rule, and
    the counts. A race counts only if every served runner has an SP above 1 (the study's wholly priced races)."""
    from research.greyhound.sp_harness import settle
    from research.greyhound.sp_offset import picks
    s = served.assign(market_id=served.market_id.astype(str),
                      selection_id=pd.to_numeric(served.selection_id, errors="coerce"))
    cl = close.assign(market_id=close.market_id.astype(str), selection_id=pd.to_numeric(close.selection_id, errors="coerce"))
    j = s.merge(cl[["market_id", "selection_id", "bsp", "won"]], on=["market_id", "selection_id"], how="left")
    j["bsp"] = pd.to_numeric(j.bsp, errors="coerce")
    priced = (j.bsp > 1).groupby(j.race_id).transform("all").to_numpy()
    whole = j[priced]
    counts = {"races_served": int(j.race_id.nunique()), "races_wholly_priced": int(whole.race_id.nunique()),
              "races_left_out": int(j.race_id.nunique() - whole.race_id.nunique())}
    p = picks(whole, cuts)
    out = [settle(b, comm).assign(rule=name) for name, b in p.groupby("rule")]
    cols = ["rule", "day", "race_id", "risk", "gross", "net"]
    return (pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)), counts


def settle(store, day: date, s3=None) -> dict:
    from greyhound.costs import commission
    from greyhound.fav_back import closing
    ds = f"{day:%Y-%m-%d}"
    raw = store.get(f"{PREFIX}/served/{ds}.csv")
    if not raw:
        return {"day": ds, "note": "not served"}
    _, meta = load_frozen(store)
    if meta is None:
        return {"day": ds, "note": "no frozen residual stored"}
    served = pd.read_csv(io.BytesIO(raw), dtype={"market_id": str, "race_id": str})
    close = closing(day, s3=s3)
    if not len(close):
        return {"day": ds, "note": "no data found: no settled books yet"}
    per_race, counts = settle_day(served, close, meta["cuts"], commission())
    buf = io.StringIO()
    per_race.to_csv(buf, index=False)
    store.put(f"{PREFIX}/settled/{ds}.csv", buf.getvalue().encode())
    by = per_race.groupby("rule")[["risk", "net"]].sum() if len(per_race) else pd.DataFrame(columns=["risk", "net"])
    return {"day": ds, **counts, "rules": {k: {"bets": int(v.risk), "net": round(float(v.net), 2)} for k, v in by.iterrows()}}


def read(forward: pd.DataFrame, dev_daily: pd.DataFrame | None, n_trials: int) -> dict:
    """Each rule on the forward days alone and pooled with the study's test years. ``forward``: the settled rows
    (rule, day, race_id, risk, net); ``dev_daily``: rule, day, net, risk."""
    from research.greyhound.power import required_t
    from research.greyhound.sp_harness import read as read_rule
    from research.greyhound.sp_offset import STAGE2
    out = {"tag": TAG, "n_trials": int(n_trials), "t_required": round(required_t(n_trials), 2),
           "forward_days": int(forward.day.nunique()) if len(forward) else 0, "rules": []}
    for name, _, _ in STAGE2:
        f = forward[forward.rule == name] if len(forward) else forward
        row = {"rule": name, "forward": read_rule(f, n_trials) if len(f) else {"bets": 0, "note": "no data found"}}
        if dev_daily is not None and len(dev_daily):
            d = dev_daily[dev_daily.rule == name].assign(race_id=lambda x: "dev_" + x.day.astype(str))
            both = pd.concat([d[["day", "race_id", "risk", "net"]], f[["day", "race_id", "risk", "net"]]], ignore_index=True)
            # the study's days are day sums: one row a day, so bets = the money at risk and the interval is by day
            row["pooled"] = read_rule(both, n_trials) if len(both) else {"bets": 0}
            row["study"] = read_rule(d[["day", "race_id", "risk", "net"]], n_trials) if len(d) else {"bets": 0}
        fw, po = row["forward"], row.get("pooled") or {}
        lo = (po.get("roi_90%") or [None])[0]
        row["pass"] = bool(po.get("dsr_passed") and lo is not None and lo > 0 and (fw.get("roi_%") or 0) > 0)
        out["rules"].append(row)
    out["passed"] = [r["rule"] for r in out["rules"] if r["pass"]]
    return out


def markdown(res: dict) -> str:
    md = [f"## The residual on the BSP, forward on paper ({TAG})", "",
          f"{res['forward_days']} settled forward days. The bar: a t of {res['t_required']} at {res['n_trials']:,} "
          "greyhound trials (the deflated Sharpe), on the study's test years and the forward days together, with the "
          "forward days' own return above zero. A back of 1 or a lay risking 1 at the SP, 2% off a race's gain.", "",
          "| Rule | Forward bets | Forward return % | 90% | t | Study return % | Pooled return % | 90% | t | Deflated Sharpe | Pass |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rules"]:
        f, s, p = r["forward"], r.get("study") or {}, r.get("pooled") or {}
        md.append(f"| {r['rule']} | {f.get('bets')} | {f.get('roi_%')} | {f.get('roi_90%')} | {f.get('t')} | "
                  f"{s.get('roi_%')} | {p.get('roi_%')} | {p.get('roi_90%')} | {p.get('t')} | {p.get('dsr')} | {r['pass']} |")
    md += ["", f"Passed: {res['passed'] or 'none'}."]
    return "\n".join(md)


def mean_daily(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The study's days from several runs of it (the seed halves): each rule's net and risk a day, averaged over the
    runs (a day a run did not bet counts as nothing)."""
    x = pd.concat(frames, ignore_index=True).groupby(["rule", "day"], as_index=False)[["net", "risk"]].sum()
    x[["net", "risk"]] = x[["net", "risk"]] / len(frames)
    return x


def load_settled(store, first: str = FIRST_DAY, last: str | None = None) -> pd.DataFrame:
    last = last or f"{datetime.now(timezone.utc).date():%Y-%m-%d}"
    out, d = [], date.fromisoformat(first)
    while f"{d:%Y-%m-%d}" <= last:
        raw = store.get(f"{PREFIX}/settled/{d:%Y-%m-%d}.csv")
        if raw:
            out.append(pd.read_csv(io.BytesIO(raw), dtype={"race_id": str}))
        d += timedelta(days=1)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["rule", "day", "race_id", "risk", "gross", "net"])


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--date", help="UK racing day, YYYY-MM-DD (default today)")
    ap.add_argument("--days", type=int, default=3, help="--settle: this many days back from --date")
    ap.add_argument("--dev-daily", nargs="+",
                    help="the study's stage2_daily csv(s) to store as the study's days (once); several (the seed "
                         "halves) are averaged day by day")
    ap.add_argument("--out", default="out/sp_offset_forward")
    a = ap.parse_args(argv)
    store = Store()
    from zoneinfo import ZoneInfo
    day = date.fromisoformat(a.date) if a.date else datetime.now(ZoneInfo("Europe/London")).date()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    if a.dev_daily:
        if store.get(DEV_DAILY_KEY):
            res["dev_daily"] = "already stored: not replaced"
        else:
            buf = io.StringIO()
            mean_daily([pd.read_csv(f) for f in a.dev_daily]).to_csv(buf, index=False)
            store.put(DEV_DAILY_KEY, buf.getvalue().encode())
            res["dev_daily"] = f"stored (the mean of {len(a.dev_daily)} file(s))"
    if a.freeze:
        res["freeze"] = {k: v for k, v in freeze(store).items() if k != "features"}
    if a.serve:
        res["serve"] = serve(store, day)
    if a.settle:
        res["settle"] = [settle(store, day - timedelta(days=i)) for i in range(a.days)]
    md = ["```", json.dumps(res, indent=1, default=str), "```"]
    if a.read:
        from research.greyhound.trial_count import greyhound_trials
        raw = store.get(DEV_DAILY_KEY)
        dev = pd.read_csv(io.BytesIO(raw)) if raw else None
        res["read"] = read(load_settled(store), dev, greyhound_trials())
        md = [markdown(res["read"]), ""] + md
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
