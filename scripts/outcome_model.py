#!/usr/bin/env python3
"""Walk-forward win-probability model on top of the market, scored by what a
rank-1 backer would have made.

Model, per fold
    The market-only conditional logit on ln π and (ln π)² is fitted on the
    training window; LightGBM with the race-softmax objective is then boosted
    from that fit as a fixed offset, so the trees can only add what the price
    does not already say (Benter's combined model, fitted in one pass). Early
    stopping uses the last --val-months of the training window. Refit every
    --fold-months; every prediction is for races the fit never saw.

    π is the BSP-implied probability. That is the price the bet settles at,
    and it stands in for the exchange price a minute before the off, which is
    when a live version of this would be run.

Strategies scored (flat 1-point stakes, at BSP, commission on winnings)
    rank1            back the model's top-rated runner in every race
    rank1_ev>τ       ... only when its expected value at BSP exceeds τ
    fav              back the market favourite (the baseline to beat)
Each with ROI, a bootstrap 90% interval, strike rate, per-year and per-half
figures, and the model's log-likelihood gain over the market.

Protocol
    Races on or after --lockbox-from are never read unless --final is given.
    Every run writes a ledger entry (ledger_entry_<tag>.json) that is committed
    to reports/research_ledger.jsonl, development runs included, so the number
    of configurations tried and of looks at the holdout is on the record.

Usage
    python scripts/outcome_model.py --db horse_racing.db --feature-cache .feature_cache \\
        --out-dir reports/outcome_model --tag iter1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import residual_screen as rs  # noqa: E402

log = logging.getLogger("outcome_model")

COMMISSION = 0.05
EV_GRID = (0.0, 0.02, 0.05, 0.10, 0.20)
#: Race attributes carried into the out-of-sample file, for segment tables.
SEGMENT_COLS = ("race_type", "race_code", "surface_type", "number_of_runners", "race_class", "track")
KELLY_FRACTION = 0.25


# ---------------------------------------------------------------------------
# Betting arithmetic
# ---------------------------------------------------------------------------

def bet_returns(won: np.ndarray, price: np.ndarray, commission: float = COMMISSION) -> np.ndarray:
    """Profit per 1-point back bet at `price`, commission on net winnings."""
    return np.where(won > 0, (price - 1.0) * (1.0 - commission), -1.0)


def expected_value(p: np.ndarray, price: np.ndarray, commission: float = COMMISSION) -> np.ndarray:
    return p * (price - 1.0) * (1.0 - commission) - (1.0 - p)


def roi_summary(r: np.ndarray, won: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    r = np.asarray(r, float)
    n = len(r)
    if n == 0:
        return {"bets": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(r, size=(n_boot, n), replace=True).mean(axis=1) if n > 1 else np.array([r.mean()])
    return {"bets": int(n), "roi": float(r.mean()), "lo90": float(np.percentile(boots, 5)),
            "hi90": float(np.percentile(boots, 95)), "profit": float(r.sum()), "strike": float(np.mean(won > 0))}


def pick_rank1(race: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Boolean mask of each race's top-scoring row (first on ties). Rows sorted by race."""
    starts, seg = rs.race_blocks(pd.factorize(pd.Series(race))[0])
    m = np.maximum.reduceat(score, starts)
    top = score >= m[seg]
    first = np.zeros(len(score), bool)
    idx = np.flatnonzero(top)
    first_idx = idx[np.r_[True, seg[idx][1:] != seg[idx][:-1]]]
    first[first_idx] = True
    return first


def score_strategies(oos: pd.DataFrame, commission: float = COMMISSION, ev_grid=EV_GRID) -> dict:
    """Every strategy's ROI table on a frame of out-of-sample predictions."""
    oos = oos.sort_values(["date", "race"], kind="stable").reset_index(drop=True)
    race = oos["race"].to_numpy()
    won = oos["won"].to_numpy(float)
    price = oos["bsp"].to_numpy(float)
    ret = bet_returns(won, price, commission)
    r1 = pick_rank1(race, oos["p_model"].to_numpy(float))
    fav = pick_rank1(race, oos["p_mkt"].to_numpy(float))
    ev = expected_value(oos["p_model"].to_numpy(float), price, commission)
    out = {"fav": roi_summary(ret[fav], won[fav]), "rank1": roi_summary(ret[r1], won[r1]),
           "rank1_is_fav_share": float(np.mean(fav[r1]))}
    for t in ev_grid:
        m = r1 & (ev > t + 1e-9)          # float dust at a fair price is not an edge
        out[f"rank1_ev>{t:g}"] = roi_summary(ret[m], won[m])
    years = pd.to_datetime(oos["date"]).dt.year.to_numpy()
    out["rank1_by_year"] = {int(y): roi_summary(ret[r1 & (years == y)], won[r1 & (years == y)], n_boot=500)
                            for y in np.unique(years)}
    half = np.zeros(len(oos), bool)
    codes = pd.factorize(pd.Series(race))[0]
    half[codes >= codes.max() / 2] = True
    for t in (None,) + tuple(ev_grid):
        m = r1 if t is None else r1 & (ev > t + 1e-9)
        name = "rank1" if t is None else f"rank1_ev>{t:g}"
        out[f"{name}_halves"] = [roi_summary(ret[m & ~half], won[m & ~half], n_boot=500)["roi"] if (m & ~half).any() else None,
                                 roi_summary(ret[m & half], won[m & half], n_boot=500)["roi"] if (m & half).any() else None]
    out["segments"] = segment_tables(oos, r1, ev, ret, won)
    out["kelly"] = {f"rank1_ev>{t:g}": kelly_path(oos["p_model"].to_numpy(float), price, won, r1 & (ev > t + 1e-9),
                                                 commission)
                    for t in ev_grid}
    bands = [(1, 2), (2, 3), (3, 5), (5, 8), (8, 15), (15, 1000)]
    out["rank1_by_price"] = {f"{a}-{b}": roi_summary(ret[r1 & (price >= a) & (price < b)],
                                                     won[r1 & (price >= a) & (price < b)], n_boot=500)
                             for a, b in bands}
    return out


def kelly_path(p: np.ndarray, price: np.ndarray, won: np.ndarray, mask: np.ndarray,
               commission: float = COMMISSION, fraction: float = KELLY_FRACTION) -> dict:
    """Fractional-Kelly bankroll over the masked bets in date order (bank starts at 1).

    Kelly on a back bet at net odds b = (price-1)(1-c): stake f* = (p·b − (1−p)) / b
    of the bank, scaled by `fraction`, never negative, capped at 5% a bet."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {"bets": 0}
    b = (price[idx] - 1.0) * (1.0 - commission)
    f = np.clip(fraction * (p[idx] * b - (1 - p[idx])) / np.maximum(b, 1e-9), 0.0, 0.05)
    growth = np.where(won[idx] > 0, 1.0 + f * b, 1.0 - f)
    bank = np.cumprod(growth)
    peak = np.maximum.accumulate(np.r_[1.0, bank])[1:]
    return {"bets": int(len(idx)), "final_bank": float(bank[-1]), "max_drawdown": float(np.max(1 - bank / peak)),
            "log_growth_per_bet": float(np.mean(np.log(growth))), "turnover": float(f.sum())}


def segment_tables(oos: pd.DataFrame, r1: np.ndarray, ev: np.ndarray, ret: np.ndarray, won: np.ndarray) -> dict:
    """Rank-1 ROI (all, and EV > 0) and the model's gain over the market, by segment.

    For reading, not selecting: a segment chosen because it looks good here is a
    hypothesis for the holdout, not a result."""
    out = {}
    starts, seg = rs.race_blocks(pd.factorize(oos["race"])[0])
    y = oos["won"].to_numpy(float)
    d = (rs.race_loglik(np.log(np.clip(oos["p_model"].to_numpy(float), 1e-12, 1)), y, starts, seg)
         - rs.race_loglik(np.log(np.clip(oos["p_mkt"].to_numpy(float), 1e-12, 1)), y, starts, seg))
    race_first = starts
    cols = {c: oos[c] for c in SEGMENT_COLS if c in oos.columns and c != "track"}
    if "number_of_runners" in cols:
        cols["field"] = pd.cut(pd.to_numeric(oos["number_of_runners"], errors="coerce"), [0, 5, 8, 12, 16, 60],
                               labels=["2-5", "6-8", "9-12", "13-16", "17+"]).astype(str)
        del cols["number_of_runners"]
    for name, col in cols.items():
        col = col.astype(str).fillna("?")
        table = {}
        for level in col.value_counts().index[:12]:
            m_rows = (col == level).to_numpy()
            m_race = m_rows[race_first]
            if m_race.sum() < 200:
                continue
            a = roi_summary(ret[r1 & m_rows], won[r1 & m_rows], n_boot=300)
            e = roi_summary(ret[r1 & m_rows & (ev > 1e-9)], won[r1 & m_rows & (ev > 1e-9)], n_boot=300)
            dd = d[m_race]
            table[str(level)] = {"races": int(m_race.sum()), "rank1_roi": a.get("roi"), "rank1_lo90": a.get("lo90"),
                                 "ev0_bets": e.get("bets", 0), "ev0_roi": e.get("roi"),
                                 "dll_mnats": float(1000 * dd.mean()),
                                 "dll_t": float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if len(dd) > 1 else None}
        out[name] = table
    return out


# ---------------------------------------------------------------------------
# The walk-forward
# ---------------------------------------------------------------------------

def fold_starts(first: str, last_exclusive: pd.Timestamp, months: int) -> list[pd.Timestamp]:
    out, t = [], pd.Timestamp(first)
    while t < last_exclusive:
        out.append(t)
        t = t + pd.DateOffset(months=months)
    return out


def fit_boosted(X, tr: rs.Part, va: rs.Part, market_beta, params=None, rounds=2000, early=100):
    import lightgbm as lgb

    def fobj(preds, ds):
        p = rs.softmax_blocks(preds, *tr.blk)
        return p - tr.y, np.maximum(p * (1 - p), 1e-6)

    def feval(preds, ds):
        return "race_nll", float(-rs.race_loglik(preds, va.y, *va.blk).mean()), False

    base = {"learning_rate": 0.05, "num_leaves": 31, "feature_fraction": 0.5,
            "min_data_in_leaf": int(min(2000, max(20, len(tr.idx) // 50))),
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 63,
            "verbosity": -1, "seed": 42, "objective": fobj}
    base.update(params or {})
    dtr = lgb.Dataset(X[tr.idx], label=tr.y, init_score=tr.M @ market_beta, free_raw_data=False)
    dva = lgb.Dataset(X[va.idx], label=va.y, init_score=va.M @ market_beta, reference=dtr)
    booster = lgb.train(base, dtr, num_boost_round=rounds, valid_sets=[dva], feval=feval,
                        callbacks=[lgb.early_stopping(early, verbose=False)])
    return booster, int(booster.best_iteration or rounds)


def walk_forward(sample: pd.DataFrame, X: np.ndarray, first_fold: str, end: pd.Timestamp, fold_months: int,
                 val_months: int, params=None, feature_names=None,
                 mode: str = "offset") -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """mode "offset": trees boosted from the market-only logit (one pass).
    mode "free": Benter's two stages -- a market-free fundamental model f on the
    winner, then Stage C, softmax(a ln pi + b (ln pi)^2 + c ln f), fitted on the
    validation months, whose f the fundamental model never trained on."""
    y = sample["y"].to_numpy(float)
    codes = pd.factorize(sample["race"])[0]
    pos = np.arange(len(sample))
    date = sample["date"].to_numpy()
    M = rs.market_columns(sample["ln_pi"].to_numpy(float))

    def part(mask):
        return rs.Part(pos[mask], codes[mask], y[mask], M[mask])

    preds, folds, gains = [], [], []
    for f0 in fold_starts(first_fold, end, fold_months):
        f1 = min(f0 + pd.DateOffset(months=fold_months), end)
        t0, t1 = np.datetime64(f0), np.datetime64(f1)
        v0 = np.datetime64(f0 - pd.DateOffset(months=val_months))
        tr_all, te = date < t0, (date >= t0) & (date < t1)
        fit_m, va_m = date < v0, (date >= v0) & (date < t0)
        if te.sum() == 0 or fit_m.sum() == 0 or va_m.sum() == 0:
            continue
        t_start = time.time()
        p_all = part(tr_all)
        bm = rs.fit_clogit(p_all.M, p_all.y, *p_all.blk)
        p_te = part(te)
        off = p_te.M @ bm
        if mode == "free":
            p_va = part(va_m)
            booster, best = fit_boosted(X, part(fit_m), p_va, np.zeros(rs.MARKET_TERMS), params=params)
            f_va = booster.predict(X[p_va.idx], raw_score=True, num_iteration=best)
            f_te = booster.predict(X[p_te.idx], raw_score=True, num_iteration=best)
            # the fundamental model's log-probability is its raw score less the race's log-sum-exp
            lf_va = np.log(rs.softmax_blocks(f_va, *p_va.blk))
            lf_te = np.log(rs.softmax_blocks(f_te, *p_te.blk))
            c = rs.fit_clogit(np.column_stack([p_va.M, lf_va]), p_va.y, *p_va.blk)
            eta = np.column_stack([p_te.M, lf_te]) @ c
        else:
            booster, best = fit_boosted(X, part(fit_m), part(va_m), bm, params=params)
            eta = off + booster.predict(X[p_te.idx], raw_score=True, num_iteration=best)
        p_model = rs.softmax_blocks(eta, *p_te.blk)
        p_mkt = rs.softmax_blocks(off, *p_te.blk)
        d = rs.race_loglik(eta, p_te.y, *p_te.blk) - rs.race_loglik(off, p_te.y, *p_te.blk)
        preds.append(pd.DataFrame({"row": p_te.idx, "p_model": p_model, "p_mkt": p_mkt, "fold": str(f0.date())}))
        g = rs.gain_summary(d, p_te.null)
        folds.append({"fold": str(f0.date()), "to": str(f1.date()), "races": int(p_te.n_races),
                      "best_iteration": best, "dll_mnats": g["dll_mnats"], "t": g["t"],
                      "seconds": round(time.time() - t_start)})
        if feature_names is not None:
            imp = booster.feature_importance("gain", iteration=best)
            gains.append(pd.Series(imp, index=feature_names, name=str(f0.date())))
        log.info("fold %s: %d races, %d rounds, ΔLL %+.2f mnats/race (t %+.2f), %ds", f0.date(), p_te.n_races,
                 best, g["dll_mnats"], g["t"], folds[-1]["seconds"])
    p = pd.concat(preds, ignore_index=True)
    keep = ["race", "date", "horse_name", "bfsp", "y"] + [c for c in SEGMENT_COLS if c in sample.columns]
    oos = sample.iloc[p["row"].to_numpy()][keep].reset_index(drop=True)
    oos = oos.rename(columns={"bfsp": "bsp", "y": "won"})
    oos[["p_model", "p_mkt", "fold"]] = p[["p_model", "p_mkt", "fold"]].to_numpy()
    oos["p_model"] = oos["p_model"].astype(float); oos["p_mkt"] = oos["p_mkt"].astype(float)
    importance = pd.concat(gains, axis=1) if gains else pd.DataFrame()
    return oos, folds, importance


def loglik_summary(oos: pd.DataFrame) -> dict:
    oos = oos.sort_values(["date", "race"], kind="stable")
    starts, seg = rs.race_blocks(pd.factorize(oos["race"])[0])
    y = oos["won"].to_numpy(float)
    lm = rs.race_loglik(np.log(np.clip(oos["p_model"].to_numpy(float), 1e-12, 1)), y, starts, seg)
    lk = rs.race_loglik(np.log(np.clip(oos["p_mkt"].to_numpy(float), 1e-12, 1)), y, starts, seg)
    null = rs.uniform_loglik(starts, len(y))
    g = rs.gain_summary(lm - lk, null)
    g["R2_model"] = float(1 - lm.sum() / null.sum())
    g["R2_market"] = float(1 - lk.sum() / null.sum())
    return g


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run(args) -> dict:
    t_start = time.time()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    from model.bfsp_features import ALL_FEATURE_COLS

    df = rs.load_frame(args.db, args.start_date, args.feature_cache)
    lock = pd.Timestamp(args.lockbox_from)
    if not args.final:
        n0 = len(df)
        df = df[pd.to_datetime(df["race_date"]) < lock]
        log.info("lockbox: %d rows on or after %s withheld", n0 - len(df), lock.date())
    sample = rs.screening_sample(df, None)
    prod = [c for c in dict.fromkeys(ALL_FEATURE_COLS) if c in df.columns]
    drops = tuple(p for p in (args.drop or "").split(",") if p)
    feats = [c for c in prod if not c.startswith(drops)] if drops else prod
    rows_in_sample = df.index.get_indexer(sample.index)
    cols = [np.array(pd.to_numeric(df[c], errors="coerce"), dtype=np.float32)[rows_in_sample] for c in feats]
    blocks = [b for b in (args.blocks or "").split(",") if b]
    if blocks:
        keep = df[[c for c in df.columns if c not in set(ALL_FEATURE_COLS)]].copy()
        del df
        keep, added = rs.attach_blocks(keep, blocks, args.db)
        key = pd.DataFrame({"race": rs.race_key(keep), "horse_name": keep["horse_name"].values})
        lookup = pd.DataFrame({"race": sample["race"].values, "horse_name": sample["horse_name"].values,
                               "pos": np.arange(len(sample))})
        m = key.reset_index().merge(lookup, on=["race", "horse_name"], how="inner").drop_duplicates("pos")
        for b, bcols in added.items():
            for c in bcols:
                arr = np.full(len(sample), np.nan, dtype=np.float32)
                arr[m["pos"].to_numpy()] = np.array(pd.to_numeric(keep[c], errors="coerce"),
                                                    dtype=np.float32)[m["index"].to_numpy()]
                cols.append(arr); feats.append(c)
        del keep
    else:
        del df
    X = np.column_stack(cols)
    del cols
    log.info("design: %d runners x %d features; %d races %s -> %s", X.shape[0], X.shape[1],
             sample["race"].nunique(), sample["date"].min().date(), sample["date"].max().date())

    end = sample["date"].max() + pd.Timedelta(days=1)
    first = args.lockbox_from if args.final else args.first_fold
    cfg = {"tag": args.tag, "mode": args.mode, "features": len(feats), "feature_hash": config_hash({"f": feats}),
           "drop": args.drop,
           "blocks": args.blocks, "fold_months": args.fold_months, "val_months": args.val_months,
           "params": args.params, "first_fold": first, "final": bool(args.final)}
    params = json.loads(args.params) if args.params else None
    oos, folds, importance = walk_forward(sample, X, first, end, args.fold_months, args.val_months, params, feats,
                                          mode=args.mode)
    if args.final:
        oos = oos[pd.to_datetime(oos["date"]) >= lock]
    oos.to_csv(out / f"oos_{args.tag}.csv.gz", index=False)
    if len(importance):
        importance.mean(axis=1).sort_values(ascending=False).rename("gain").to_csv(out / f"importance_{args.tag}.csv")
    res = {"config": cfg, "config_hash": config_hash(cfg), "folds": folds, "loglik": loglik_summary(oos),
           "strategies": score_strategies(oos), "seconds": round(time.time() - t_start)}
    (out / f"result_{args.tag}.json").write_text(json.dumps(res, indent=2, default=float))
    (out / f"summary_{args.tag}.md").write_text(render(res))
    # Every configuration tried goes on the record, development runs included:
    # the count is what a final number has to be deflated by.
    entry = {"when": pd.Timestamp.now(tz="UTC").isoformat(), "kind": "HOLDOUT LOOK" if args.final else "dev",
             "config_hash": res["config_hash"], "config": cfg, "loglik": res["loglik"],
             "rank1": res["strategies"]["rank1"],
             "best_ev_rule": max(((k, v) for k, v in res["strategies"].items()
                                  if k.startswith("rank1_ev>") and not k.endswith("_halves") and v.get("bets", 0) >= 300),
                                 key=lambda kv: kv[1]["roi"], default=(None, None))}
    (out / f"ledger_entry_{args.tag}.json").write_text(json.dumps(entry, indent=2, default=float))
    return res


def render(res: dict) -> str:
    c, ll, s = res["config"], res["loglik"], res["strategies"]
    L = [f"# Outcome model `{c['tag']}`" + ("  —  LOCKED HOLDOUT" if c["final"] else ""), ""]
    L.append(f"{c['features']} features, folds of {c['fold_months']} months from {c['first_fold']}, "
             f"config {res['config_hash']}.")
    L.append("")
    L.append(f"Log-likelihood over the market: {ll['dll_mnats']:+.3f} mnats/race (SE {ll['se_mnats']:.3f}, "
             f"t {ll['t']:+.2f}), ΔR² {ll['dR2']:+.5f}; R² model {ll['R2_model']:.4f} vs market {ll['R2_market']:.4f}, "
             f"{ll['races']:,} races.")
    L.append("")
    L.append("| strategy | bets | ROI | 90% CI | strike | 1st half | 2nd half |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ["fav", "rank1"] + [f"rank1_ev>{t:g}" for t in EV_GRID]:
        v = s.get(k, {})
        if not v or not v.get("bets"):
            continue
        h = s.get(f"{k}_halves", [None, None])
        fmt = lambda x: "" if x is None else f"{x:+.2%}"
        L.append(f"| {k} | {v['bets']:,} | {v['roi']:+.2%} | {v['lo90']:+.2%} to {v['hi90']:+.2%} | "
                 f"{v['strike']:.1%} | {fmt(h[0])} | {fmt(h[1])} |")
    L.append("")
    L.append(f"The model's rank-1 is the market favourite in {s['rank1_is_fav_share']:.1%} of races.")
    L.append("")
    L.append("Rank 1 by year: " + ", ".join(f"{y} {v['roi']:+.2%} ({v['bets']:,})" for y, v in s["rank1_by_year"].items() if v.get("bets")))
    L.append("")
    L.append("Rank 1 by BSP band: " + ", ".join(f"{b} {v['roi']:+.2%} ({v['bets']:,})" for b, v in s["rank1_by_price"].items() if v.get("bets")))
    L.append("")
    k = s.get("kelly", {})
    if k:
        L.append("Quarter-Kelly bankroll (bank 1.0, 5% cap a bet): " + "; ".join(
            f"{name} {v['bets']:,} bets -> {v['final_bank']:.3f} (max DD {v['max_drawdown']:.1%})"
            for name, v in k.items() if v.get("bets")))
        L.append("")
    for segname, table in s.get("segments", {}).items():
        L.append(f"| {segname} | races | rank-1 ROI | lo90 | EV>0 bets | EV>0 ROI | ΔLL mnats | t |")
        L.append("|---|---|---|---|---|---|---|---|")
        for level, v in table.items():
            fmt = lambda x, p=True: "" if x is None else (f"{x:+.2%}" if p else f"{x:+.2f}")
            L.append(f"| {level} | {v['races']:,} | {fmt(v['rank1_roi'])} | {fmt(v['rank1_lo90'])} | {v['ev0_bets']:,} | "
                     f"{fmt(v['ev0_roi'])} | {fmt(v['dll_mnats'], False)} | {fmt(v['dll_t'], False)} |")
        L.append("")
    L.append("| fold | races | rounds | ΔLL mnats | t |")
    L.append("|---|---|---|---|---|")
    for f in res["folds"]:
        L.append(f"| {f['fold']} | {f['races']:,} | {f['best_iteration']} | {f['dll_mnats']:+.2f} | {f['t']:+.2f} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(ROOT / "horse_racing.db"))
    ap.add_argument("--feature-cache", default=None)
    ap.add_argument("--start-date", default="2021-01-01")
    ap.add_argument("--first-fold", default="2023-01-01")
    ap.add_argument("--fold-months", type=int, default=3)
    ap.add_argument("--val-months", type=int, default=3)
    ap.add_argument("--lockbox-from", default="2026-04-01")
    ap.add_argument("--final", action="store_true", help="score the locked holdout (logged to the ledger)")
    ap.add_argument("--drop", default="", help="comma list of feature-name prefixes to withhold")
    ap.add_argument("--blocks", default="", help="opt-in blocks, as residual_screen.py")
    ap.add_argument("--params", default="", help="LightGBM params as JSON, merged over the defaults")
    ap.add_argument("--mode", default="offset", choices=["offset", "free"],
                    help="offset: trees boosted from the market; free: market-free Stage F, then Stage C")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "outcome_model"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run(args)
    print(render(res))


if __name__ == "__main__":
    main()
