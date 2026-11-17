"""The Australian greyhound rule on paper, forward, against Betfair's books (the owner, 17 Nov 2026: "Record the
Australian books live for a paper test"; ledger greyhound-au-paper-1117).

The rule (frozen here, research/greyhound/au_model.py's model and blend; ledger greyhound-au-model-1117 read the 0.2
edge as a hypothesis): back a runner at the Betfair SP when the blend's expected value at the price just before the
off, after ``COMMISSION``, exceeds ``EDGE``, the price at most ``MAX_PRICE``; one unit a bet, settled at the BSP.
The study decided at the BSP itself, which is not known before the off; here the decision is made on the last book
read before the scheduled off (each runner's projected SP, ``sp_near``, else ``sp_far``, else the best back), so it
is a rule an order could follow (a back at the SP with a limit), and the bet is settled at the actual BSP.

- ``--freeze``: the model (LightGBM on GreyhoundAPI's Australian results of the ``TRAIN_YEARS`` before 2026, as the
  study) and the blend (a conditional logit on logit(model) and logit(market), fitted on every 2026 race of Betfair's
  Historic Data with a price for every runner) to ``greyhound/au_paper/bundle.json``. Run once; the rule is then fixed.
- ``--price --date D``: GreyhoundAPI's Australian cards for the UTC day D (fetched now), each runner's model chance
  from its form before D, to ``greyhound/au_paper/prices/D.parquet`` with the time it was made. Never rewritten.
- ``--settle --date D``: the day's recorded Australian books (greyhound/book_record.py, tags ``greyhound_au_n`` and
  ``greyhound_au_d``) and the prices: the decisions, the BSP and the result, to ``greyhound/au_paper/ledger/D.parquet``.
- ``--summary``: every settled day, the return at 5% and 2% commission with a 90% interval by day.

Nothing here can place an order: no Betfair call is made at all (the books are read from S3).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PREFIX = "greyhound/au_paper"
TAGS = ("greyhound_au_n", "greyhound_au_d")
EDGE = 0.20
COMMISSION = 0.05
MAX_PRICE = 30.0
TRAIN_YEARS = 5
TEST_YEAR = 2026
LGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1)


# ---- the model's features (as the study builds them) -------------------------------------------------------------------

def feature_frame(d: pd.DataFrame) -> pd.DataFrame:
    from research.greyhound import au_feasibility as af, au_model as am
    X0, _ = af.features(d)
    X1 = am.linked_features(d, am.linked(d))
    return pd.concat([X0, X1], axis=1)


def freeze(store) -> dict:
    import lightgbm as lgb
    from research.greyhound import au_feasibility as af, au_model as am
    api = af.load_api(store, 2016)
    api = api[api.position.notna()]
    d, jn = am.join_prices(api, am.load_historic(store, [TEST_YEAR]))
    d = af.figures(d)
    d = d[d.won.notna()].reset_index(drop=True)
    X = feature_frame(d)
    d["year"] = d.race_date.str[:4].astype(int)
    tr = (d.year < TEST_YEAR) & (d.year >= TEST_YEAR - TRAIN_YEARS)
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X[tr], d.won[tr])
    te = d.year == TEST_YEAR
    t = d[te].copy()
    t["p_model"] = af.race_norm(m.predict_proba(X[te])[:, 1], t.race_id.to_numpy())
    bsp = pd.to_numeric(t.bf_bsp, errors="coerce").where(lambda x: x > 1)
    t["price"] = bsp.fillna(pd.to_numeric(t.bf_close, errors="coerce").where(lambda x: x > 1))
    g = t.groupby("race_id")
    s = t[g.price.transform(lambda x: x.notna().all()) & (g.won.transform("sum") == 1)].copy()
    rid = s.race_id.to_numpy()
    s["p_mkt"] = af.race_norm(1.0 / s.price.to_numpy(float), rid)
    Xb = np.column_stack([af._logit(s.p_model.to_numpy()), af._logit(s.p_mkt.to_numpy())])
    b, se = af.clogit(Xb, s.won.to_numpy(float), rid)
    bundle = {"frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "features": list(X.columns), "model": m.booster_.model_to_string(),
              "blend": [float(b[0]), float(b[1])], "blend_se": [float(se[0]), float(se[1])],
              "rule": {"edge": EDGE, "commission": COMMISSION, "max_price": MAX_PRICE},
              "train_years": [TEST_YEAR - TRAIN_YEARS, TEST_YEAR - 1], "blend_rows": int(len(s)),
              "blend_races": int(s.race_id.nunique()), "join": jn}
    store.put(f"{PREFIX}/bundle.json", json.dumps(bundle).encode())
    return {k: v for k, v in bundle.items() if k != "model"}


def load_bundle(store) -> dict | None:
    raw = store.get(f"{PREFIX}/bundle.json")
    return json.loads(raw) if raw else None


# ---- the day's cards and prices --------------------------------------------------------------------------------------

def card_rows(races: list, day: date) -> pd.DataFrame:
    """One row a runner (not scratched or withdrawn) of each Australian race scheduled on the UTC day, in the API's
    results columns, the result columns empty."""
    rows = []
    for r in races:
        if r.get("region") != "AU" and (r.get("track") or {}).get("region") != "AU":
            continue
        st = (r.get("scheduled_start") or {})
        utc = str(st.get("utc") or "")
        if utc[:10] != day.isoformat() or r.get("status") not in (None, "scheduled", "open", "pending"):
            continue
        runners = [p for p in r.get("runners") or [] if str(p.get("runner_status", "runner")) == "runner"]
        for p in runners:
            rows.append({"race_id": r.get("race_id"), "dog_id": p.get("dog_id"), "dog_name": p.get("dog_name"),
                         "track": (r.get("track") or {}).get("name"), "start_utc": utc,
                         "start_local": st.get("local"), "trap": p.get("trap"), "position": np.nan,
                         "run_time_s": np.nan, "adjusted_time_s": np.nan, "sectional_s": np.nan,
                         "distance_m": r.get("distance_m"), "class_number": r.get("class_number"),
                         "grade": r.get("grade"), "field_size": len(runners), "weight_kg": np.nan,
                         "bf_back": np.nan, "bf_lay": np.nan, "status": r.get("status"), "result_status": None})
    return pd.DataFrame(rows)


def fetch_cards(client=None) -> list:
    from sources.greyhoundapi import Client
    c = client or Client()
    out = []
    for path in ("/racecards/today", "/racecards/upcoming"):
        for data, _ in c.pages(path):
            out += data
    seen, uniq = set(), []
    for r in out:
        if r.get("race_id") not in seen:
            seen.add(r.get("race_id"))
            uniq.append(r)
    return uniq


def price(store, day: date, races: list | None = None) -> dict:
    import lightgbm as lgb
    from research.greyhound import au_feasibility as af
    key = f"{PREFIX}/prices/{day.isoformat()}.parquet"
    if store.get_parquet(key) is not None:
        return {"note": f"prices for {day} already made: not rewritten"}
    bundle = load_bundle(store)
    if bundle is None:
        return {"note": "no data found: no frozen bundle (run --freeze)"}
    card = card_rows(races if races is not None else fetch_cards(), day)
    if not len(card):
        return {"note": f"no data found: no Australian card for {day}"}
    hist = af.load_api(store, 2016)
    hist = hist[hist.position.notna() & (hist.start_local.astype(str).str[:10] < day.isoformat())]
    allr = pd.concat([hist, card], ignore_index=True)
    allr["dog_id"] = pd.to_numeric(allr.dog_id, errors="coerce")
    d = af.figures(allr)
    X = feature_frame(d)
    is_card = d.position.isna().to_numpy()
    booster = lgb.Booster(model_str=bundle["model"])
    Xc = X.loc[is_card, bundle["features"]]
    t = d.loc[is_card, ["race_id", "dog_id", "dog_name", "track", "start_utc", "trap"]].copy()
    t["p_model"] = af.race_norm(booster.predict(Xc), t.race_id.to_numpy())
    t["priced_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    store.put_parquet(key, t.reset_index(drop=True))
    return {"day": day.isoformat(), "races": int(t.race_id.nunique()), "runners": int(len(t)),
            "first_off": str(t.start_utc.min()), "priced_utc": t.priced_utc.iloc[0]}


# ---- settlement ------------------------------------------------------------------------------------------------------

def decision_prices(books: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Each runner's price on the last live book read before its market's scheduled off: the projected SP (near,
    else far), else the best back."""
    b = books[books.source.astype(str) == "live"].copy()
    m = markets[["market_id", "market_start_utc"]].drop_duplicates("market_id")
    b = b.merge(m.astype({"market_id": str}), on="market_id", how="inner") if len(b) else b
    if not len(b):
        return pd.DataFrame(columns=["market_id", "selection_id", "dec_price", "dec_utc", "price_kind"])
    b["t"] = pd.to_datetime(b.polled_utc, utc=True, errors="coerce")
    b["off"] = pd.to_datetime(b.market_start_utc, utc=True, errors="coerce")
    b = b[(b.t <= b.off) & (b.runner_status.astype(str) == "ACTIVE")]
    last = b.groupby("market_id").t.transform("max")
    b = b[b.t == last]
    near, far, back = (pd.to_numeric(b[c], errors="coerce") for c in ("sp_near", "sp_far", "back1"))
    ok = lambda x: x.where((x > 1) & (x < 1000))  # noqa: E731
    b["dec_price"] = ok(near).fillna(ok(far)).fillna(ok(back))
    b["price_kind"] = np.where(ok(near).notna(), "sp_near", np.where(ok(far).notna(), "sp_far", "back1"))
    b["dec_utc"] = b.t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return b[["market_id", "selection_id", "dec_price", "dec_utc", "price_kind"]].assign(
        selection_id=lambda x: pd.to_numeric(x.selection_id, errors="coerce"))


def match_runners(prices: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """The card's runners on the Betfair WIN markets: venue, the scheduled off to the minute and the dog's name."""
    from research.greyhound.au_feasibility import norm
    p = prices.copy()
    p["k_min"] = pd.to_datetime(p.start_utc, utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    p["k_dog"], p["k_trk"] = norm(p.dog_name), norm(p.track)
    m = markets.copy()
    m["k_min"] = pd.to_datetime(m.market_start_utc, utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    m["k_dog"] = norm(m.runner_name.astype(str).str.replace(r"^\s*\d+\.\s*", "", regex=True))
    m["k_trk"] = norm(m.venue)
    m = m.drop_duplicates(["market_id", "selection_id"])
    return p.merge(m[["k_min", "k_dog", "k_trk", "market_id", "selection_id"]], on=["k_min", "k_dog", "k_trk"],
                   how="inner").assign(selection_id=lambda x: pd.to_numeric(x.selection_id, errors="coerce"),
                                       market_id=lambda x: x.market_id.astype(str))


def decide(frame: pd.DataFrame, blend, edge: float = EDGE, commission: float = COMMISSION,
           max_price: float = MAX_PRICE) -> pd.DataFrame:
    """The blend on the decision prices, race by race (only races where every active runner is matched and priced),
    and the bets."""
    from research.greyhound import au_feasibility as af
    f = frame.copy()
    rid = f.market_id.to_numpy()
    f["p_model"] = af.race_norm(f.p_model.to_numpy(float), rid)
    f["p_mkt"] = af.race_norm(1.0 / f.dec_price.to_numpy(float), rid)
    Xb = np.column_stack([af._logit(f.p_model.to_numpy()), af._logit(f.p_mkt.to_numpy())])
    f["p_blend"] = af.apply_clogit(Xb, np.asarray(blend, float), rid)
    f["ev"] = f.p_blend * (f.dec_price - 1.0) * (1 - commission) - (1 - f.p_blend)
    f["bet"] = (f.ev > edge) & (f.dec_price <= max_price)
    return f


def settle(store, day: date, s3=None) -> dict:
    from betfair_recorder import _read_day_csv
    from greyhound import tradeout as to
    prices = store.get_parquet(f"{PREFIX}/prices/{day.isoformat()}.parquet")
    bundle = load_bundle(store)
    if prices is None or bundle is None:
        return {"note": f"no data found: no prices or no bundle for {day}"}
    books, markets = [], []
    for tag in TAGS:
        b, m = _read_day_csv(day, f"books_{tag}.csv", s3=s3), _read_day_csv(day, f"markets_{tag}.csv", s3=s3)
        if b is not None and len(b) and m is not None and len(m):
            books.append(b.assign(market_id=b.market_id.astype(str)))
            markets.append(m.assign(market_id=m.market_id.astype(str)))
    if not books:
        return {"note": f"no data found: no recorded Australian books for {day}"}
    books, markets = pd.concat(books, ignore_index=True), pd.concat(markets, ignore_index=True)
    markets = markets.drop_duplicates(["market_id", "selection_id"])
    mt = matched = match_runners(prices, markets)
    dec = decision_prices(books, markets)
    f = matched.merge(dec, on=["market_id", "selection_id"], how="inner")
    priced_at = pd.to_datetime(prices.priced_utc.iloc[0], utc=True)
    f = f[pd.to_datetime(f.start_utc, utc=True) > priced_at]
    # every active runner of the market at the decision book matched and priced, or the race is left out
    active = dec.groupby("market_id").selection_id.nunique()
    have = f.groupby("market_id").selection_id.nunique()
    full = (have.reindex(active.index).fillna(0) == active)
    f = f[f.market_id.isin(full[full].index)]
    sp = to.closing_prices(books, markets, day)
    fin = books[books.source.astype(str) == "final"]
    win = fin[fin.runner_status.astype(str) == "WINNER"][["market_id", "selection_id"]].drop_duplicates()
    win = win.assign(selection_id=pd.to_numeric(win.selection_id, errors="coerce"), won=1)
    sp = sp.assign(market_id=sp.market_id.astype(str)).merge(win, on=["market_id", "selection_id"], how="left")
    settled = set(win.market_id)
    sp["won"] = sp.won.fillna(0).astype(int)
    out = decide(f, bundle["blend"]).merge(sp[["market_id", "selection_id", "bsp", "won"]],
                                          on=["market_id", "selection_id"], how="left")
    out = out[out.market_id.isin(settled)]
    bets = out[out.bet & (pd.to_numeric(out.bsp, errors="coerce") > 1)].copy()
    for c in (COMMISSION, 0.02):
        bets[f"ret_{int(round(c * 100))}"] = np.where(bets.won == 1, (bets.bsp - 1.0) * (1 - c), -1.0)
    bets["day"] = day.isoformat()
    store.put_parquet(f"{PREFIX}/ledger/{day.isoformat()}.parquet", bets.reset_index(drop=True))
    return {"day": day.isoformat(), "card_runners": int(len(prices)), "matched_to_betfair": int(len(mt)),
            "markets_recorded": int(markets.market_id.nunique()), "markets_settled": len(settled),
            "races_decided": int(out.market_id.nunique()), "bets": int(len(bets)),
            "pnl_5": round(float(bets.ret_5.sum()), 3) if len(bets) else 0.0,
            "pnl_2": round(float(bets.ret_2.sum()), 3) if len(bets) else 0.0,
            "price_kinds": out.price_kind.value_counts().to_dict() if len(out) else {}}


def summary(store, draws: int = 2000) -> dict:
    keys = sorted(k for k in store.listing(f"{PREFIX}/ledger/") if k.endswith(".parquet"))
    parts = [store.get_parquet(k) for k in keys]
    b = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True) if parts else pd.DataFrame()
    out = {"days": len(keys), "bets": int(len(b))}
    if not len(b):
        return out
    for c in ("ret_5", "ret_2"):
        g = b.groupby("day")[c].agg(["sum", "size"])
        idx = np.random.default_rng(0).integers(0, len(g), (draws, len(g)))
        roi = g["sum"].to_numpy()[idx].sum(1) / g["size"].to_numpy()[idx].sum(1)
        out[c] = {"roi%": round(100 * float(b[c].mean()), 2), "pnl_units": round(float(b[c].sum()), 2),
                  "roi_90": [round(100 * float(np.quantile(roi, 0.05)), 2), round(100 * float(np.quantile(roi, 0.95)), 2)]}
    out["win_rate"] = round(float(b.won.mean()), 4)
    out["mean_bsp"] = round(float(b.bsp.mean()), 2)
    out["mean_dec_price"] = round(float(b.dec_price.mean()), 2)
    return out


def main(argv=None) -> int:
    from sources.common import Store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--price", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--date", default=None, help="UTC day, YYYY-MM-DD")
    a = ap.parse_args(argv)
    store = Store()
    day = date.fromisoformat(a.date) if a.date else datetime.now(timezone.utc).date()
    rep: dict = {}
    if a.freeze:
        rep["freeze"] = freeze(store)
    if a.settle:
        rep["settle"] = settle(store, day)
    if a.price:
        rep["price"] = price(store, day)
    if a.summary:
        rep["summary"] = summary(store)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
