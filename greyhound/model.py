"""The greyhound win model: market-blind LightGBM on the metrics, fitted walk-forward, scored against the markets.

Market-blind, as the horse model is (it predicts from form, not from the price), so it can be priced the evening or
morning before racing and set against the early market, where the feasibility study found the lead
(reports/greyhound_feasibility.md).

- ``walk_forward``: for each test period, fit on every earlier period (with an early-stopping slice at the end of the
  fit window), predict the period; probabilities normalised within each race.
- ``score``: log-loss and Brier against the bookmakers' SP (GBGB), and a blend test: does the model add to the SP?
- ``against_betfair``: the predictions joined to Betfair's GB win markets (the Historic Data table): log-loss
  against the first traded price, the T-1 price and the BSP, and the value rules (back at the first price, or decide
  at T-1 and back at the SP, where the model's expected return clears an edge), with CLV against the BSP.

    python -m greyhound.model --store <sources root> --from 2025-10-01 --folds 2026-01,2026-04,2026-07
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COM = 0.02
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=300, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1)


def race_norm(p: pd.Series, race: pd.Series) -> pd.Series:
    return p / p.groupby(race).transform("sum")


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def modelling_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The runs a model prices: graded and open races with a market, four or more runners, not trials."""
    return df[~df.is_trial & (df.field >= 4)].copy()


def walk_forward(df: pd.DataFrame, features: list[str], folds: list[str], date_from: str | None = None,
                 rounds: int = 3000) -> pd.DataFrame:
    """Out-of-sample predictions: fold i fits on [date_from, folds[i]) and predicts [folds[i], folds[i+1])."""
    import gc

    import lightgbm as lgb
    mask = ~df.is_trial.to_numpy(bool) & (df.field.to_numpy() >= 4)
    if "is_card" in df:
        mask &= ~df.is_card.to_numpy(bool)
    if date_from:
        mask &= (df.t >= date_from).to_numpy()
    # the modelling rows' features once, as float32; each fold slices it (copies of the frame do not fit a runner)
    rows = np.flatnonzero(mask)
    meta = df.iloc[rows][["race_id", "t", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "won",
                          "sp_p_norm", "sp_decimal", "field"]].reset_index(drop=True)
    X = np.empty((len(rows), len(features)), dtype=np.float32)
    for j, f in enumerate(features):
        X[:, j] = df[f].to_numpy(np.float32, na_value=np.nan)[rows]
    y, t_all = meta.won.to_numpy(), meta.t.to_numpy().astype("datetime64[ns]")
    out = []
    edges = [np.datetime64(pd.Timestamp(f), "ns") for f in folds] + [
        np.datetime64(meta.t.max() + pd.Timedelta(days=1), "ns")]
    for i in range(len(folds)):
        fit_i = np.flatnonzero(t_all < edges[i])
        test_i = np.flatnonzero((t_all >= edges[i]) & (t_all < edges[i + 1]))
        if not len(fit_i) or not len(test_i):
            continue
        cut = np.quantile(t_all[fit_i].astype("int64"), 0.9).astype("datetime64[ns]")
        tr_i, es_i = fit_i[t_all[fit_i] < cut], fit_i[t_all[fit_i] >= cut]
        print(f"fold {folds[i]}: fit {len(tr_i):,} + {len(es_i):,} rows, test {len(test_i):,}, {len(features)} features",
              flush=True)
        dtr = lgb.Dataset(X[tr_i], y[tr_i], feature_name=list(features), free_raw_data=True)
        des = lgb.Dataset(X[es_i], y[es_i], reference=dtr, free_raw_data=True)
        m = lgb.train(PARAMS, dtr, rounds, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False)])
        del dtr, des
        gc.collect()
        t = meta.iloc[test_i].copy()
        t["p_raw"] = m.predict(X[test_i], num_iteration=m.best_iteration)
        t["p_model"] = race_norm(t.p_raw, t.race_id)
        t["fold"] = folds[i]
        t["rounds"] = m.best_iteration
        out.append(t)
        imp = pd.Series(m.feature_importance("gain"), index=features)
        last_importance = (imp / imp.sum()).sort_values(ascending=False)
    del X
    pred = pd.concat(out, ignore_index=True)
    # set after the concat: pandas compares frames' attrs when it concatenates, and a Series there cannot be compared
    pred.attrs["importance"] = last_importance if out else pd.Series(dtype=float)
    return pred


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_test(pred: pd.DataFrame, market: str) -> dict:
    """Does the model add to the market? A conditional-logit-style blend (logistic on both log-odds, normalised in
    the race) fitted on the first half of the predictions, scored on the second half against the market alone."""
    from sklearn.linear_model import LogisticRegression
    d = pred.dropna(subset=[market]).sort_values("t")
    half = d.t.quantile(0.5)
    a, b = d[d.t < half], d[d.t >= half]
    if a.won.nunique() < 2 or len(b) == 0:
        return {"skipped": "too few races to fit a blend"}
    X = lambda x: np.c_[_logit(x[market]), _logit(x.p_model)]  # noqa: E731
    lr = LogisticRegression(C=1.0).fit(X(a), a.won)
    pb = race_norm(pd.Series(lr.predict_proba(X(b))[:, 1], index=b.index), b.race_id)
    return {"weights": {"market": round(float(lr.coef_[0][0]), 3), "model": round(float(lr.coef_[0][1]), 3)},
            "logloss_second_half": {"market": round(logloss(b[market], b.won), 5),
                                    "model": round(logloss(b.p_model, b.won), 5),
                                    "blend": round(logloss(pb, b.won), 5)}}


def score(pred: pd.DataFrame) -> dict:
    d = pred.dropna(subset=["sp_p_norm"])
    out = {"runners": len(pred), "races": int(pred.race_id.nunique()),
           "logloss": {"model": round(logloss(d.p_model, d.won), 5), "sp": round(logloss(d.sp_p_norm, d.won), 5),
                       "uniform": round(logloss(1.0 / d.field, d.won), 5)},
           "by_fold": {}}
    for f, g in d.groupby("fold"):
        out["by_fold"][f] = {"runners": len(g), "model": round(logloss(g.p_model, g.won), 5),
                             "sp": round(logloss(g.sp_p_norm, g.won), 5), "rounds": int(g.rounds.iloc[0])}
    out["blend_with_sp"] = blend_test(d, "sp_p_norm")
    # calibration: predicted against actual by decile of the model's chance
    d = d.assign(dec=pd.qcut(d.p_model, 10, labels=False, duplicates="drop"))
    out["calibration"] = {int(k): [round(float(v.p_model.mean()), 4), round(float(v.won.mean()), 4)]
                          for k, v in d.groupby("dec")}
    return out


def _settle(sel: pd.DataFrame, price: str) -> dict:
    if not len(sel):
        return {"bets": 0}
    p = sel[price]
    pnl = np.where(sel.won == 1, (p - 1) * (1 - COM), -1.0)
    se = pnl.std(ddof=1) / np.sqrt(len(pnl)) if len(pnl) > 1 else np.nan
    return {"bets": int(len(sel)), "per_day": round(len(sel) / max(1, sel.race_date.nunique()), 1),
            "roi%": round(100 * pnl.mean(), 2), "90%": [round(100 * (pnl.mean() - 1.645 * se), 2),
                                                        round(100 * (pnl.mean() + 1.645 * se), 2)],
            "clv_vs_bsp%": round(100 * float((sel[price] / sel.bsp - 1).mean()), 2)}


def against_betfair(pred: pd.DataFrame, bf: pd.DataFrame, edges=(0.0, 0.1, 0.2, 0.3), cap: float = 20.0) -> dict:
    """The predictions on Betfair's GB win markets (one row a runner: market_time, venue, runner_name '1. Name',
    bsp, ltp_first, ltp_t_1, won)."""
    w = bf[(bf.market_type == "WIN") & (bf.country == "GB") & bf.runner_status.isin(["WINNER", "LOSER"])].copy()
    uk = pd.to_datetime(w.market_time, utc=True).dt.tz_convert("Europe/London")
    w["race_date"], w["race_time"] = uk.dt.strftime("%Y-%m-%d"), uk.dt.strftime("%H:%M")
    w["track"] = w.venue.astype(str)
    w["trap"] = pd.to_numeric(w.runner_name.str.extract(r"^(\d+)\.")[0], errors="coerce")
    m = pred.merge(w[["race_date", "race_time", "track", "trap", "market_id", "bsp", "ltp_first", "ltp_t_1"]],
                   on=["race_date", "race_time", "track", "trap"], how="inner")
    m = m[m.groupby("market_id").race_id.transform("size") == m.groupby("market_id").field.transform("first")]
    m = m.dropna(subset=["bsp"])
    out = {"runners": len(m), "markets": int(m.market_id.nunique())}
    ll = {"model": logloss(m.p_model, m.won)}
    for col in ("bsp", "ltp_t_1", "ltp_first"):
        q = m[col].where(m[col] > 1)
        p = race_norm(1.0 / q, m.market_id)
        ok = p.notna()
        ll[col] = logloss(p[ok], m.won[ok])
        m[f"p_{col}"] = p
    out["logloss"] = {k: round(v, 5) for k, v in ll.items()}
    out["blend_with_bsp"] = blend_test(m.assign(t=pd.to_datetime(m.race_date)), "p_bsp")
    rules = {}
    for price in ("ltp_first", "ltp_t_1"):
        e = m[m[price] > 1].copy()
        e["edge"] = e.p_model * (1 + (e[price] - 1) * (1 - COM)) - 1
        for th in edges:
            sel = e[(e.edge > th) & (e[price] <= cap)]
            rules[f"{price}: edge>{th:.1f}"] = {"at_that_price": _settle(sel, price), "at_bsp": _settle(sel, "bsp")}
    out["rules"] = rules
    return out


def against_price_files(pred: pd.DataFrame, prices: pd.DataFrame, edges=(0.0, 0.1, 0.2, 0.3), cap: float = 20.0,
                        min_vols=(0.0, 20.0, 100.0)) -> dict:
    """The predictions on Betfair's greyhound price files (every year archived): log-loss against the BSP, the
    pre-play and the morning weighted average prices, and the early rule at the MORNING price where the morning
    traded volume on the runner reaches ``min_vol`` (the fillability the basic plan cannot show).

    A market is kept only when its result in the file (WIN_LOSE) agrees with GBGB's on every runner: a join on
    the wrong race (a time an hour out) would otherwise score one race's prices on another's result. A price's
    log-loss is read only on the markets where it prices the whole field: a WAP on two runners of six,
    normalised, says nothing of the race."""
    w = prices[(prices.market == "win") & (prices.country == "GB")].copy()
    w = w.rename(columns={"dog": "dog_bf"})
    cols = ["race_date", "race_time", "track", "trap", "event_id", "bsp", "ppwap", "morningwap", "morning_vol",
            "pp_vol"] + (["win_lose"] if "win_lose" in w.columns else [])
    m = pred.merge(w[cols], on=["race_date", "race_time", "track", "trap"], how="inner")
    m = m[m.groupby("event_id").race_id.transform("size") == m.groupby("event_id").field.transform("first")]
    m = m.dropna(subset=["bsp"])
    joined = int(m.event_id.nunique())
    if "win_lose" in m.columns:
        agree = (m.win_lose.isna() | (m.win_lose == m.won)).groupby(m.event_id).transform("all")
        m = m[agree]
    out = {"runners": len(m), "markets": int(m.event_id.nunique()), "markets_joined": joined,
           "markets_result_disagrees": joined - int(m.event_id.nunique()),
           "morning_vol_per_runner": {q: round(float(m.morning_vol.quantile(q)), 1) for q in (0.25, 0.5, 0.75, 0.9)},
           "pp_vol_per_runner": {q: round(float(m.pp_vol.quantile(q)), 1) for q in (0.25, 0.5, 0.75)}}
    ll, cover = {"model": logloss(m.p_model, m.won)}, {}
    for col in ("bsp", "ppwap", "morningwap"):
        q = m[col].where(m[col] > 1)
        whole = q.notna().groupby(m.event_id).transform("all")
        p = race_norm(1.0 / q.where(whole), m.event_id)
        cover[col] = {"runners_priced": round(float(q.notna().mean()), 4),
                      "markets_whole_field": round(float(whole.groupby(m.event_id).first().mean()), 4)}
        ll[col] = logloss(p[whole], m.won[whole]) if whole.any() else None
        if whole.any():
            ll[f"model_on_{col}_markets"] = logloss(m.p_model[whole], m.won[whole])
        m[f"p_{col}"] = p
    out["price_coverage"] = cover
    out["logloss"] = {k: (round(v, 5) if v is not None else None) for k, v in ll.items()}
    out["blend_with_bsp"] = blend_test(m.assign(market_id=m.event_id), "p_bsp")
    rules = {}
    e = m[m.morningwap > 1].copy()
    e["edge"] = e.p_model * (1 + (e.morningwap - 1) * (1 - COM)) - 1
    for vol in min_vols:
        for th in edges:
            sel = e[(e.edge > th) & (e.morningwap <= cap) & (e.morning_vol.fillna(0) >= vol)]
            rules[f"morning: edge>{th:.1f}, vol>={vol:g}"] = {"at_morning": _settle(sel, "morningwap"),
                                                             "at_bsp": _settle(sel, "bsp")}
    out["rules"] = rules
    by_year = {}
    for y, g in m.groupby(m.race_date.str[:4]):
        sel = e[(e.race_date.str[:4] == y) & (e.edge > 0.2) & (e.morningwap <= cap)]
        by_year[y] = {"runners": len(g), "model": round(logloss(g.p_model, g.won), 5),
                      "bsp": round(logloss(g.p_bsp.fillna(1 / g.field), g.won), 5),
                      "morning_edge>0.2": {"at_morning": _settle(sel, "morningwap"), "at_bsp": _settle(sel, "bsp")}}
    out["by_year"] = by_year
    return out


#: the columns the fits and the scores read beside the features
KEEP_COLS = ["race_id", "raceid", "t", "race_date", "race_time", "track", "trap", "dog_id", "dog_name", "won",
             "sp_p_norm", "sp_decimal", "field", "is_trial", "is_card"]


def slim(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The frame the walk-forward needs, in place: the columns the scores read and the features as float32, nothing
    else (the full history at 267 features is about 5m rows; in float64 beside the raw columns it does not fit a
    runner)."""
    keep = set(KEEP_COLS) | set(features)
    df.drop(columns=[c for c in df.columns if c not in keep], inplace=True)
    for f in dict.fromkeys(features):
        if f in df.columns and f not in KEEP_COLS and df[f].dtype != np.float32:
            df[f] = pd.to_numeric(df[f], errors="coerce").astype(np.float32)
    return df


PRICE_COLS = ["market", "country", "event_id", "race_date", "race_time", "track", "trap", "bsp", "ppwap", "morningwap",
              "morning_vol", "pp_vol", "win_lose"]


def load_prices(store, years: str | None = None):
    """The greyhound price-file tables, GB win markets only, the columns the scores and the BSP history read, the years
    asked for: the whole archive (every market and country from 2014) does not fit beside the metrics."""
    import re
    keys = sorted(k for k in store.listing("greyhound_prices/") if re.search(r"prices_(\d{4})\.parquet$", k))
    if years:
        lo, hi = (int(x) for x in years.split("-"))
        keys = [k for k in keys if lo <= int(re.search(r"prices_(\d{4})", k).group(1)) <= hi]
    parts = []
    for k in keys:
        t = store.get_parquet(k)
        if t is None or not len(t):
            continue
        t = t[(t.market == "win") & (t.country == "GB")]
        parts.append(t[[c for c in PRICE_COLS if c in t.columns]].copy())
    return pd.concat(parts, ignore_index=True) if parts else None


def main(argv=None) -> int:
    from greyhound.data import load_runs
    from greyhound.metrics import GreyhoundMetricsEngine
    from sources.common import Store
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=None, help="the sources store root (default S3)")
    ap.add_argument("--years", default=None, help="GBGB years to read, e.g. 2018-2026")
    ap.add_argument("--from", dest="date_from", default=None, help="the first day a model fits on")
    ap.add_argument("--folds", required=True, help="test period starts, e.g. 2024-01,2025-01,2026-01")
    ap.add_argument("--features", default=None, help="a saved metrics frame (parquet) instead of building one")
    ap.add_argument("--save-features", default=None)
    ap.add_argument("--save-pred", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", action="store_true",
                    help="fit the base features and base + the parity block (greyhound/parity.py) on the same folds")
    a = ap.parse_args(argv)
    store = Store(root=a.store)
    prices = load_prices(store, a.years)
    eng = GreyhoundMetricsEngine(blocks=("parity", "hrb"))
    if a.features and Path(a.features).exists():
        df = pd.read_parquet(a.features)
        sets = json.loads(Path(a.features + ".features.json").read_text())
    else:
        years = None
        if a.years:
            lo, hi = (int(x) for x in a.years.split("-"))
            years = list(range(lo, hi + 1))
        df = eng.calculate_all(load_runs(store, years=years), prices=prices)
        sets = {"base": eng.block_features["base"], "all": eng.features}
        if "hrb" in eng.block_features:                    # the step before the last block, to score what it adds
            sets["base+parity"] = eng.block_features["base"] + eng.block_features.get("parity", [])
        print(f"features built: {len(df):,} runs, {len(eng.features)} features, {eng.timings}", flush=True)
        df = slim(df, eng.features)
        if a.save_features:
            df.to_parquet(a.save_features)
            Path(a.save_features + ".features.json").write_text(json.dumps(sets))
    if isinstance(sets, list):                              # a features file from before the blocks
        sets = {"all": sets}
    names = ([n for n in ("base", "base+parity", "all") if n in sets] if a.compare and "base" in sets else ["all"])
    bf = store.get_parquet("betfair_historic/markets_greyhound_racing_2026.parquet")
    report = {"timings": getattr(eng, "timings", None), "sets": {}}
    for name in names:
        feats = sets[name]
        pred = walk_forward(df, feats, a.folds.split(","), a.date_from)
        importance = pred.attrs.pop("importance")          # off the frame: parquet writes attrs as JSON
        r = {"features": len(feats), "score": score(pred)}
        if bf is not None:
            r["betfair_2026"] = against_betfair(pred, bf)
        if prices is not None:
            r["price_files"] = against_price_files(pred, prices)
        r["top_features"] = {k: round(float(v), 4) for k, v in importance.head(30).items()}
        report["sets"][name] = r
        if a.save_pred and name == names[-1]:
            pred.to_parquet(a.save_pred)
    txt = json.dumps(report, indent=1, default=str)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
