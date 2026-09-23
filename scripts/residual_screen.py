#!/usr/bin/env python3
"""Which of our features know something the market does not?

Benter's test, one input at a time. For each candidate feature x, a conditional
logit on the winner of each race,

    P(i wins) ∝ exp(a1·ln π_i + a2·(ln π_i)² + b1·z_i + b2·z_i² [+ b3·missing_i])

is compared with the market alone (b = 0) on races the fit never saw. π is the
race-normalised market probability and z the feature standardised on the
training rows. The unit is the change in out-of-sample log-likelihood per race,
in millinats: above zero, the feature carries information about the result that
the price does not.

This asks what the deployed model's feature importance cannot. That model is
trained to reproduce the price, so its gain measures how much a feature tells it
about the *price*. A feature the market already prices perfectly scores high
there and zero here; a feature the market ignores is worthless to it by
construction, however well it predicts winners.

Screens (each feature, and then every feature at once)
  bsp       π from Betfair SP. Fitted before --split, scored from it.
            -> is there value at the price we settle at?
  morning   π from Betfair's morning WAP (the betfair_prices table). Fitted on
            alternate calendar months, scored on the others, both ways round.
            -> is there value at the price we can actually bet?
  move      OLS of ln(π_bsp / π_morning), the move from the morning to the off,
            on the feature, with the morning price as control and race fixed
            effects; same cross-fitting.
            -> does the feature predict the closing-line move?
  alone     the feature with no market at all (how predictive it is, full stop)

Joint tests (Benter's own fundamental-model form, and a non-linear one)
  linear    every production feature at once, linear, ridge-penalised, on top of
            the market (penalty chosen on the last months of the training window)
  boosted   LightGBM with the race-softmax objective, started from the fitted
            market-only logit as a fixed offset, so every tree can only add what
            the price does not already say; early-stopped on the same months

Opt-in feature blocks that production does not use (--blocks) are screened the
same way, feature by feature and jointly, including on top of the production
features' joint index.

Usage
    python scripts/residual_screen.py --db horse_racing.db --feature-cache .feature_cache \\
        --out-dir reports/residual_screen --blocks perf,kalman,blandford,pedigree,connections
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("residual_screen")

MARKET_TERMS = 2               # ln π and (ln π)²: the favourite-longshot bias, curvature included
MIN_RACES_TO_SCORE = 200
CHUNK_ROWS = 60_000            # Hessian accumulation block for wide designs


# ---------------------------------------------------------------------------
# Race-block arithmetic. Rows are sorted so each race is contiguous; `starts`
# indexes the first row of each race and `seg` maps every row to its race.
# ---------------------------------------------------------------------------

def race_blocks(codes) -> tuple[np.ndarray, np.ndarray]:
    codes = np.asarray(codes)
    if len(codes) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(codes)])
    return starts, np.repeat(np.arange(len(starts)), sizes)


def softmax_blocks(eta, starts, seg):
    m = np.maximum.reduceat(eta, starts)
    e = np.exp(eta - m[seg])
    return e / np.add.reduceat(e, starts)[seg]


def race_loglik(eta, y, starts, seg):
    """Log-probability each race's model gave its winner (exactly one y == 1 per race)."""
    m = np.maximum.reduceat(eta, starts)
    lse = m + np.log(np.add.reduceat(np.exp(eta - m[seg]), starts))
    return np.add.reduceat(eta * y, starts) - lse


def uniform_loglik(starts, n_rows) -> np.ndarray:
    """Per-race log-likelihood of picking the winner at random, the McFadden null."""
    return -np.log(np.diff(np.r_[starts, n_rows]))


def fit_clogit(X, y, starts, seg, ridge=None, beta0=None, max_iter=60, tol=1e-10):
    """Maximum-likelihood conditional (race-softmax) logit by damped Newton.

    `ridge` is a per-coefficient L2 penalty on the summed log-likelihood; give
    the market terms 0. The penalised objective is concave, so Newton with step
    halving gets there in a handful of iterations. Wide designs accumulate the
    Hessian in race-aligned chunks to bound memory."""
    X = np.asarray(X, float)
    n, k = X.shape
    pen = np.zeros(k) if ridge is None else np.broadcast_to(np.asarray(ridge, float), (k,)).astype(float)
    beta = np.zeros(k) if beta0 is None else np.asarray(beta0, float).copy()
    cuts = _chunk_starts(starts, n) if k > 40 else None

    def objective(b):
        return race_loglik(X @ b, y, starts, seg).sum() - 0.5 * float(np.dot(pen * b, b))

    f = objective(beta)
    for _ in range(max_iter):
        p = softmax_blocks(X @ beta, starts, seg)
        g = X.T @ (y - p) - pen * beta
        H = _neg_hessian(X, p, starts, cuts)
        H[np.diag_indices(k)] += pen + 1e-9
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t, f_new = 1.0, objective(beta + step)
        while f_new < f and t > 1e-8:
            t *= 0.5
            f_new = objective(beta + t * step)
        if f_new < f:                      # no ascent left at machine precision
            break
        beta = beta + t * step
        done = (f_new - f) < tol * (1.0 + abs(f))
        f = f_new
        if done:
            break
    return beta


def _chunk_starts(starts, n):
    """Row offsets that cut the frame into ~CHUNK_ROWS pieces on race boundaries."""
    cuts, last = [0], 0
    for s in starts:
        if s - last >= CHUNK_ROWS:
            cuts.append(int(s)); last = s
    return np.r_[cuts, n]


def _neg_hessian(X, p, starts, cuts):
    """Σ_races [Σ_i p_i x_i x_iᵀ − x̄ x̄ᵀ], x̄ = Σ_i p_i x_i: the information matrix."""
    k = X.shape[1]
    if cuts is None:
        px = X * p[:, None]
        xbar = np.add.reduceat(px, starts, axis=0)
        return X.T @ px - xbar.T @ xbar
    H = np.zeros((k, k))
    for a, b in zip(cuts[:-1], cuts[1:]):
        Xc, pc = X[a:b], p[a:b]
        local = starts[(starts >= a) & (starts < b)] - a
        px = Xc * pc[:, None]
        xbar = np.add.reduceat(px, local, axis=0)
        H += Xc.T @ px - xbar.T @ xbar
    return H


def demean_blocks(A, starts, seg):
    A = np.asarray(A, float)
    sizes = np.diff(np.r_[starts, len(A)]).astype(float)
    sums = np.add.reduceat(A, starts, axis=0)
    means = sums / (sizes[:, None] if A.ndim == 2 else sizes)
    return A - means[seg]


def within_race_varies(v: np.ndarray, starts) -> float:
    """Share of races in which the feature takes more than one value."""
    filled = np.where(np.isnan(v), -9.87654321e12, v)
    return float(np.mean(np.maximum.reduceat(filled, starts) != np.minimum.reduceat(filled, starts)))


# ---------------------------------------------------------------------------
# Turning one feature into model columns. Every constant is learned on the
# training rows only.
# ---------------------------------------------------------------------------

def feature_design(values, train_mask, *, categorical=False, quadratic=True, max_levels=6):
    """Return (columns n×k, kind). k == 0: nothing to screen.

    numeric   z and z² after winsorising at the training 0.5 / 99.5 percentiles
              (missing -> 0, the training mean), plus a missing flag when between
              0.5 % and 99.5 % of training rows are missing
    binary    z (+ missing flag)
    category  one-hot of the most common training levels, the commonest dropped"""
    v = pd.Series(values)
    tr = np.asarray(train_mask, bool)
    if categorical or v.dtype == object or str(v.dtype) in ("category", "bool", "string"):
        s = v.astype("string").fillna("<NA>")
        counts = s[tr].value_counts()
        levels = [lv for lv in counts.index[1:max_levels + 1] if counts[lv] >= 200]
        if not levels:
            return np.zeros((len(v), 0)), "constant"
        return np.column_stack([(s == lv).to_numpy(float) for lv in levels]), "category"
    x = np.array(pd.to_numeric(v, errors="coerce"), dtype=float)     # a writable copy
    x[~np.isfinite(x)] = np.nan
    miss = np.isnan(x)
    miss_rate = float(miss[tr].mean()) if tr.any() else 1.0
    xt = x[tr & ~miss]
    if len(xt) < 200:
        return np.zeros((len(v), 0)), "constant"
    lo, hi = np.percentile(xt, [0.5, 99.5])
    xc = np.clip(x, lo, hi)
    mu, sd = float(np.mean(np.clip(xt, lo, hi))), float(np.std(np.clip(xt, lo, hi)))
    if not sd > 0:
        return np.zeros((len(v), 0)), "constant"
    z = np.where(miss, 0.0, (xc - mu) / sd)
    cols, kind = [z], "numeric"
    if len(np.unique(xt[: 200_000])) <= 2:
        kind = "binary"
    elif quadratic:
        z2 = z ** 2
        cols.append(z2 - z2[tr].mean())
    if 0.005 < miss_rate < 0.995:
        cols.append(miss.astype(float))
    return np.column_stack(cols), kind


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def gain_summary(d: np.ndarray, null_ll: np.ndarray) -> dict:
    """Per-race log-likelihood gains -> millinats per race, SE, t and ΔR² (McFadden)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return {"dll_mnats": np.nan, "se_mnats": np.nan, "t": np.nan, "dR2": np.nan, "races": n}
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    return {"dll_mnats": 1000 * mean, "se_mnats": 1000 * se, "t": mean / se if se > 0 else np.nan,
            "dR2": float(d.sum() / abs(null_ll.sum())), "races": n}


class Part:
    """One scoring sample: rows of whole races, contiguous, with the market columns."""

    def __init__(self, idx, codes, y, market_X):
        self.idx = np.asarray(idx)
        self.y = np.asarray(y, float)
        self.M = np.asarray(market_X, float)
        self.starts, self.seg = race_blocks(np.asarray(codes))
        self.n_races = len(self.starts)
        self.null = uniform_loglik(self.starts, len(self.y))

    @property
    def blk(self):
        return self.starts, self.seg


class OutcomeScreen:
    """Market-only and market+feature conditional logits over one or more
    (train, test) splits; the per-race test gains are pooled across splits."""

    def __init__(self, splits: list[tuple[Part, Part]]):
        self.splits = splits
        self.market = []
        for tr, te in splits:
            b = fit_clogit(tr.M, tr.y, *tr.blk)
            self.market.append((b, race_loglik(te.M @ b, te.y, *te.blk)))
        self.null = np.concatenate([te.null for _, te in splits])

    def market_r2(self) -> float:
        ll = np.concatenate([m[1] for m in self.market])
        return float(1 - ll.sum() / self.null.sum())

    def gains(self, F_full: np.ndarray, ridge: float = 0.0, with_market: bool = True,
              base_extra: np.ndarray | None = None) -> np.ndarray:
        """Per-race out-of-sample log-likelihood gain from adding F (frame-indexed rows).

        The base model is the market (plus `base_extra`, one frame-indexed column
        such as another model's index, when given); the full model adds F with
        an L2 penalty `ridge` on F's coefficients only. with_market=False fits F
        alone and returns its gain over picking at random."""
        out = []
        for (tr, te), (bm, ll_m) in zip(self.splits, self.market):
            Ftr, Fte = F_full[tr.idx], F_full[te.idx]
            if not with_market:
                b = fit_clogit(Ftr, tr.y, *tr.blk, ridge=np.full(Ftr.shape[1], ridge))
                out.append(race_loglik(Fte @ b, te.y, *te.blk) - te.null)
                continue
            Btr, Bte = tr.M, te.M
            if base_extra is None:
                bb, base_ll = bm, ll_m
            else:
                Btr = np.column_stack([Btr, base_extra[tr.idx]])
                Bte = np.column_stack([Bte, base_extra[te.idx]])
                bb = fit_clogit(Btr, tr.y, *tr.blk)
                base_ll = race_loglik(Bte @ bb, te.y, *te.blk)
            pen = np.r_[np.zeros(Btr.shape[1]), np.full(Ftr.shape[1], ridge)]
            b = fit_clogit(np.column_stack([Btr, Ftr]), tr.y, *tr.blk, ridge=pen,
                           beta0=np.r_[bb, np.zeros(Ftr.shape[1])])
            out.append(race_loglik(np.column_stack([Bte, Fte]) @ b, te.y, *te.blk) - base_ll)
        return np.concatenate(out)


class MoveScreen:
    """OLS of the morning-to-off log-probability move on a feature, race fixed
    effects, morning price as control; per-race OOS reduction in squared error."""

    def __init__(self, splits: list[tuple[Part, Part]], move: np.ndarray):
        self.splits = splits
        self.move = move
        self.base = []
        for tr, te in splits:
            C, yv = demean_blocks(tr.M, *tr.blk), demean_blocks(move[tr.idx], *tr.blk)
            b = np.linalg.lstsq(C, yv, rcond=None)[0]
            Ce, ye = demean_blocks(te.M, *te.blk), demean_blocks(move[te.idx], *te.blk)
            r = ye - Ce @ b
            self.base.append((np.add.reduceat(r ** 2, te.starts), ye))
        self.sst = sum(float((ye ** 2).sum()) for _, ye in self.base)

    def gains(self, F_full: np.ndarray, ridge: float = 0.0) -> np.ndarray:
        out = []
        for (tr, te), (sse_base, ye) in zip(self.splits, self.base):
            A = demean_blocks(np.column_stack([tr.M, F_full[tr.idx]]), *tr.blk)
            yv = demean_blocks(self.move[tr.idx], *tr.blk)
            pen = np.r_[np.zeros(tr.M.shape[1]), np.full(A.shape[1] - tr.M.shape[1], ridge)]
            b = np.linalg.solve(A.T @ A + np.diag(pen + 1e-9), A.T @ yv)
            Ae = demean_blocks(np.column_stack([te.M, F_full[te.idx]]), *te.blk)
            r = ye - Ae @ b
            out.append(sse_base - np.add.reduceat(r ** 2, te.starts))
        return np.concatenate(out)


def boosted_gain(X, split_tr: Part, split_va: Part | None, split_te: Part, market_beta, params=None,
                 rounds=1500, early=100):
    """LightGBM race-softmax boosting from the market-only logit as a fixed offset.

    The offset means the trees start from the price and can only add what it
    does not already say. With `split_va` the round count is early-stopped on
    it; without, exactly `rounds` are grown (never stop on the test part: that
    picks the model with the answers). Returns (per-race test gain over the
    market, rounds used)."""
    import lightgbm as lgb

    def offset(part):
        return part.M @ market_beta

    dtr = lgb.Dataset(X[split_tr.idx], label=split_tr.y, init_score=offset(split_tr), free_raw_data=False)

    def fobj(preds, ds):
        p = softmax_blocks(preds, *split_tr.blk)
        return p - split_tr.y, np.maximum(p * (1 - p), 1e-6)

    base = {"learning_rate": 0.05, "num_leaves": 31, "feature_fraction": 0.5,
            "min_data_in_leaf": int(min(2000, max(20, len(split_tr.idx) // 50))),
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "max_bin": 63,
            "verbosity": -1, "seed": 42, "objective": fobj}
    base.update(params or {})
    if split_va is not None:
        dva = lgb.Dataset(X[split_va.idx], label=split_va.y, init_score=offset(split_va), reference=dtr)

        def feval(preds, ds):
            return "race_nll", float(-race_loglik(preds, split_va.y, *split_va.blk).mean()), False

        booster = lgb.train(base, dtr, num_boost_round=rounds, valid_sets=[dva], feval=feval,
                            callbacks=[lgb.early_stopping(early, verbose=False)])
        used = int(booster.best_iteration or rounds)
    else:
        booster = lgb.train(base, dtr, num_boost_round=rounds)
        used = int(rounds)
    eta = offset(split_te) + booster.predict(X[split_te.idx], raw_score=True, num_iteration=used)
    ll_m = race_loglik(offset(split_te), split_te.y, *split_te.blk)
    return race_loglik(eta, split_te.y, *split_te.blk) - ll_m, used


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_frame(db: str, start_date: str, cache_dir: str | None) -> pd.DataFrame:
    from evaluate_oos import build_feature_frame
    from model import feature_cache

    df, info = feature_cache.build_or_load(lambda: build_feature_frame(db, start_date), db,
                                           start_date=start_date, cache_dir=cache_dir)
    log.info("feature frame: %d rows x %d columns (%s)", len(df), len(df.columns),
             "cache hit" if info.get("cached") else "built")
    return df


def race_key(df: pd.DataFrame) -> pd.Series:
    return (pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d") + "|" + df["track"].astype(str)
            + "|" + df["race_time"].astype(str))


def morning_prices(db: str) -> pd.DataFrame:
    """Betfair morning WAP per runner, keyed like the feature frame."""
    q = """SELECT rr.race_date, rr.track, rr.race_time, rr.horse_name,
                  bp.morningwap, bp.morning_vol, bp.bsp AS bf_bsp
           FROM betfair_prices bp JOIN race_results rr ON rr.id = bp.race_results_id
           WHERE LOWER(bp.market_type) = 'win'"""
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            mp = pd.read_sql_query(q, conn)
    except Exception as exc:                        # no table, no morning screens
        log.warning("no morning prices (%s)", exc)
        return pd.DataFrame()
    mp["race_date"] = pd.to_datetime(mp["race_date"]).dt.strftime("%Y-%m-%d")
    return mp.drop_duplicates(["race_date", "track", "race_time", "horse_name"])


def screening_sample(df: pd.DataFrame, mp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows of complete, single-winner races, sorted so each race is contiguous.

    Complete means every row has a BSP, the book of 1/BSP is between 0.95 and
    1.15 (no big runner missing) and there are 3–40 runners."""
    d = pd.DataFrame(index=df.index)
    d["race"] = race_key(df)
    d["date"] = pd.to_datetime(df["race_date"]).dt.normalize()
    d["horse_name"] = df["horse_name"].values
    won = df["won"] if "won" in df.columns else (pd.to_numeric(df["placing_numerical"], errors="coerce") == 1)
    d["y"] = pd.to_numeric(won, errors="coerce").fillna(0).astype(float)
    d["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    for c in ("race_type", "race_code", "surface_type", "number_of_runners", "race_class", "track"):
        if c in df.columns:
            d[c] = df[c].values
    d = d[d["bfsp"] > 1.0]
    g = d.groupby("race")
    ok = (g["y"].transform("sum") == 1) & g["y"].transform("size").between(3, 40)
    book = (1.0 / d["bfsp"]).groupby(d["race"]).transform("sum")
    d = d[ok & book.between(0.95, 1.15)].copy()
    inv = 1.0 / d["bfsp"]
    d["ln_pi"] = np.log(inv) - np.log(inv.groupby(d["race"]).transform("sum"))
    if mp is not None and len(mp):
        k = d["race"].str.split("|", expand=True)
        keyed = pd.DataFrame({"race_date": k[0].values, "track": k[1].values, "race_time": k[2].values,
                              "horse_name": d["horse_name"].values}, index=d.index)
        m = keyed.merge(mp, on=["race_date", "track", "race_time", "horse_name"], how="left")
        d["morningwap"] = m["morningwap"].to_numpy()
        d["morning_vol"] = m["morning_vol"].to_numpy()
        have = d["morningwap"] > 1.0
        full = have.groupby(d["race"]).transform("all")
        inv = (1.0 / d["morningwap"]).where(have)
        d["ln_pi_m"] = np.where(full, np.log(inv) - np.log(inv.groupby(d["race"]).transform("sum")), np.nan)
    else:
        d["ln_pi_m"] = np.nan
    return d.sort_values(["date", "race"], kind="stable")


def market_columns(ln_pi: np.ndarray) -> np.ndarray:
    return np.column_stack([ln_pi, ln_pi ** 2])


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

def post_race_names() -> set[str]:
    from model.custom_metrics import POST_RACE_ONLY
    from model.draw_metrics import DRAW_POST_RACE_ONLY
    from model.pace_metrics import PACE_POST_RACE_ONLY
    from model.primitives import POST_RACE_PRIMITIVES
    return (set(POST_RACE_ONLY) | set(POST_RACE_PRIMITIVES) | set(PACE_POST_RACE_ONLY)
            | set(DRAW_POST_RACE_ONLY) | {"perf_lbs", "kf_post", "kf_innov"})


def attach_blocks(df: pd.DataFrame, blocks: list[str], db: str) -> tuple[pd.DataFrame, dict]:
    """Opt-in research blocks production does not train on. A block that fails
    is logged and skipped rather than sinking the run."""
    added: dict[str, list[str]] = {}
    if "raceid" not in df.columns:
        df["raceid"] = race_key(df)
    for b in blocks:
        n0, t0 = len(df), time.time()
        try:
            if b == "perf":
                from model.perf_figures import PERF_FIGURE_FEATURES, add_perf_figure_features
                df = add_perf_figure_features(df); cols = list(PERF_FIGURE_FEATURES)
            elif b == "kalman":
                from model.state_space import add_kalman_features
                if "perf_lbs" not in df.columns:
                    from model.perf_figures import add_perf_figure_features
                    df = add_perf_figure_features(df)
                df = add_kalman_features(df, "perf_lbs", prefix="kf")
                cols = ["kf_rating", "kf_sd", "kf_n", "kf_z", "kf_rank", "kf_vs_max"]
            elif b == "blandford":
                from model.blandford_features import BLANDFORD_FEATURES, add_blandford_features
                df = add_blandford_features(df, db_path=db); cols = list(BLANDFORD_FEATURES)
            elif b == "markets":
                from model.market_block import add_same_race_market_features
                df, cols = add_same_race_market_features(df)
            elif b == "comments":
                from model.comment_features import add_comment_features
                df, cols = add_comment_features(df)
            elif b in ("pedigree", "connections"):
                if "nmfp" not in df.columns:
                    from model.primitives import add_run_primitives
                    df = add_run_primitives(df)
                if b == "pedigree":
                    from model.pedigree import add_pedigree_features
                    df, cols = add_pedigree_features(df)
                else:
                    from model.connections import add_connection_features
                    df, cols = add_connection_features(df)
            else:
                log.warning("unknown block %s", b); continue
        except Exception as exc:                    # noqa: BLE001 - a research block may not fit this frame
            log.warning("block %s failed (%s: %s); skipped", b, type(exc).__name__, exc)
            continue
        if len(df) != n0:
            log.warning("block %s changed the row count %d -> %d; de-duplicating", b, n0, len(df))
            df = df.drop_duplicates(subset=["raceid", "horse_name"]).reset_index(drop=True)
        banned = post_race_names()
        added[b] = [c for c in dict.fromkeys(cols) if c in df.columns and c not in banned]
        log.info("block %s: %d features in %.0fs", b, len(added[b]), time.time() - t0)
    return df, added


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def screen_features(frame_vals: dict, names: list[str], sample: pd.DataFrame, bsp: OutcomeScreen | None,
                    morning: OutcomeScreen | None, move: MoveScreen | None, alone: bool, train_mask: np.ndarray,
                    starts_all) -> list[dict]:
    rows = []
    t0 = time.time()
    for i, name in enumerate(names):
        v = frame_vals[name]
        rec = {"feature": name}
        cat = name.endswith("_cat")
        vv = np.array(pd.to_numeric(pd.Series(v), errors="coerce"), dtype=float) if not cat else None
        rec["miss_rate"] = float(np.mean(pd.isna(v)))
        if vv is not None:
            rec["varies_in_race"] = within_race_varies(vv, starts_all)
        else:
            rec["varies_in_race"] = within_race_varies(pd.factorize(pd.Series(v))[0].astype(float), starts_all)
        if rec["varies_in_race"] < 0.01:
            rec["kind"] = "race-level"
            rows.append(rec); continue
        F, kind = feature_design(v, train_mask, categorical=cat)
        rec["kind"] = kind
        if F.shape[1] == 0:
            rows.append(rec); continue
        if bsp is not None:
            d = bsp.gains(F)
            s = gain_summary(d, bsp.null)
            rec.update({f"bsp_{k}": v_ for k, v_ in s.items() if k != "races"})
            half = _half_split(bsp)
            rec["bsp_dll_first_half"] = 1000 * d[:half].mean()
            rec["bsp_dll_second_half"] = 1000 * d[half:].mean()
            if alone:
                a = bsp.gains(F, with_market=False)
                rec["alone_R2"] = float(a.sum() / abs(bsp.null.sum()))
        if morning is not None:
            s = gain_summary(morning.gains(F), morning.null)
            rec.update({f"morning_{k}": v_ for k, v_ in s.items() if k != "races"})
        if move is not None:
            dm = move.gains(F)
            se = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else np.nan
            rec["move_dR2_x1000"] = 1000 * float(dm.sum() / move.sst)
            rec["move_t"] = float(dm.mean() / se) if se and se > 0 else np.nan
        rows.append(rec)
        if (i + 1) % 25 == 0:
            log.info("  %d/%d features screened (%.0fs)", i + 1, len(names), time.time() - t0)
    return rows


def _half_split(screen: OutcomeScreen) -> int:
    n = sum(te.n_races for _, te in screen.splits)
    return n // 2


def build_parts(sample: pd.DataFrame, split: str, val_months: int):
    """Row-index parts: BSP forward split (+ an inner validation tail of the
    training window) and the morning month-parity cross-fit."""
    y = sample["y"].to_numpy(float)
    codes = pd.factorize(sample["race"])[0]
    pos = np.arange(len(sample))
    date = sample["date"].to_numpy()
    split_ts = np.datetime64(pd.Timestamp(split))
    val_ts = np.datetime64(pd.Timestamp(split) - pd.DateOffset(months=val_months))
    M = market_columns(sample["ln_pi"].to_numpy(float))

    def part(mask, market=M):
        return Part(pos[mask], codes[mask], y[mask], market[mask])

    tr, te = date < split_ts, date >= split_ts
    inner_tr, inner_va = date < val_ts, (date >= val_ts) & (date < split_ts)
    parts = {"bsp": (part(tr), part(te)), "inner": (part(inner_tr), part(inner_va))}

    has_m = np.isfinite(sample["ln_pi_m"].to_numpy(float))
    if has_m.sum() > 0:
        Mm = market_columns(np.nan_to_num(sample["ln_pi_m"].to_numpy(float)))
        month = sample["date"].dt.year.to_numpy() * 12 + sample["date"].dt.month.to_numpy()
        a, b = has_m & (month % 2 == 0), has_m & (month % 2 == 1)
        parts["morning"] = [(part(a, Mm), part(b, Mm)), (part(b, Mm), part(a, Mm))]
    return parts


def joint_design(frame_vals: dict, names: list[str], train_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Every feature at once, linear only: z per numeric feature, one-hots per
    category; one missing flag per distinct missingness pattern."""
    cols, labels, seen = [], [], set()
    for name in names:
        v = frame_vals[name]
        F, kind = feature_design(v, train_mask, categorical=name.endswith("_cat"), quadratic=False)
        if F.shape[1] == 0:
            continue
        if kind in ("numeric", "binary") and F.shape[1] == 2:           # z + missing flag
            key = hash(F[:, 1].tobytes())
            cols.append(F[:, :1]); labels.append(name)
            if key not in seen:
                seen.add(key); cols.append(F[:, 1:]); labels.append(f"{name}__missing")
        else:
            cols.append(F); labels.extend([name] if F.shape[1] == 1 else [f"{name}#{j}" for j in range(F.shape[1])])
    return np.column_stack(cols) if cols else np.zeros((len(train_mask), 0)), labels


def run(args) -> dict:
    t_start = time.time()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    from model.bfsp_features import ALL_FEATURE_COLS
    from model.feature_registry import stage_of

    df = load_frame(args.db, args.start_date, args.feature_cache)
    if args.lockbox_from:
        # The locked holdout: no screen, fit or threshold ever sees these races.
        cut = pd.Timestamp(args.lockbox_from)
        n0 = len(df)
        df = df[pd.to_datetime(df["race_date"]) < cut]
        log.info("lockbox: %d rows on or after %s withheld", n0 - len(df), cut.date())
    mp = morning_prices(args.db) if not args.no_morning else pd.DataFrame()
    sample = screening_sample(df, mp)
    log.info("screening sample: %d runners in %d races (%s to %s); %d races with a full morning book",
             len(sample), sample["race"].nunique(), sample["date"].min().date(), sample["date"].max().date(),
             sample.loc[np.isfinite(sample["ln_pi_m"]), "race"].nunique())

    prod = [c for c in dict.fromkeys(ALL_FEATURE_COLS) if c in df.columns]
    if args.limit:
        prod = prod[: args.limit]
    rows_in_sample = df.index.get_indexer(sample.index)
    frame_vals = {c: df[c].to_numpy()[rows_in_sample] for c in prod}
    # The blocks need only the raw columns; dropping the production features
    # now halves the peak memory of everything that follows.
    df = df[[c for c in df.columns if c not in set(ALL_FEATURE_COLS)]]
    parts = build_parts(sample, args.split, args.val_months)
    tr_bsp, te_bsp = parts["bsp"]
    train_mask = np.zeros(len(sample), bool); train_mask[tr_bsp.idx] = True
    starts_all, _ = race_blocks(pd.factorize(sample["race"])[0])
    log.info("BSP split at %s: %d training races, %d test races", args.split, tr_bsp.n_races, te_bsp.n_races)

    bsp = OutcomeScreen([parts["bsp"]])
    morning = move = None
    if "morning" in parts and min(p.n_races for pr in parts["morning"] for p in pr) >= MIN_RACES_TO_SCORE:
        morning = OutcomeScreen(parts["morning"])
        mv = (sample["ln_pi"].to_numpy(float) - np.nan_to_num(sample["ln_pi_m"].to_numpy(float))).clip(-2.0, 2.0)
        move = MoveScreen(parts["morning"], mv)
    result = {"sample": {"runners": int(len(sample)), "races": int(sample["race"].nunique()),
                         "from": str(sample["date"].min().date()), "to": str(sample["date"].max().date()),
                         "split": args.split,
                         "bsp_train_races": tr_bsp.n_races, "bsp_test_races": te_bsp.n_races,
                         "morning_races": int(sum(te.n_races for _, te in parts["morning"])) if morning else 0},
              "market_R2": {"bsp_test": bsp.market_r2(), "morning": morning.market_r2() if morning else None}}
    log.info("market R² (out of sample): BSP %.4f%s", result["market_R2"]["bsp_test"],
             f", morning {result['market_R2']['morning']:.4f}" if morning else "")

    # 1. one feature at a time
    rows = screen_features(frame_vals, prod, sample, bsp, morning, move, not args.no_alone, train_mask, starts_all)
    per = pd.DataFrame(rows)
    per["block"] = "production"
    per["stage"] = per["feature"].map(stage_of)
    imp_path = ROOT / "data" / "models" / "bfsp_feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        imp["gain_share"] = imp["importance"] / imp["importance"].sum()
        imp["gain_rank"] = imp["importance"].rank(ascending=False, method="min")
        per = per.merge(imp[["feature", "gain_share", "gain_rank"]], on="feature", how="left")
    per.to_csv(out / "per_feature.csv", index=False)
    log.info("per-feature screen done (%.0fs)", time.time() - t_start)

    # 2. all production features at once, linear (Benter's form), ridge chosen on the inner split
    Xj, labels = joint_design(frame_vals, prod, train_mask)
    log.info("joint design: %d columns", Xj.shape[1])
    inner = OutcomeScreen([parts["inner"]])
    grid = {}
    for lam in args.ridge_grid:
        grid[lam] = gain_summary(inner.gains(Xj, ridge=lam), inner.null)["dll_mnats"]
        log.info("  ridge %g: inner validation %+.3f mnats/race", lam, grid[lam])
    lam = max(grid, key=grid.get)
    joint = {"columns": Xj.shape[1], "ridge_grid": grid, "ridge": lam}
    joint["bsp_all"] = gain_summary(bsp.gains(Xj, ridge=lam), bsp.null)
    f_cols = [j for j, l in enumerate(labels) if stage_of(l.split("#")[0].split("__")[0]) == "F"]
    joint["bsp_stage_f_only"] = gain_summary(bsp.gains(Xj[:, f_cols], ridge=lam), bsp.null)
    fund = bsp.gains(Xj[:, f_cols], ridge=lam, with_market=False)
    joint["fundamental_alone_R2"] = float(fund.sum() / abs(bsp.null.sum()))
    if morning is not None:
        joint["morning_all"] = gain_summary(morning.gains(Xj, ridge=lam), morning.null)
        dm = move.gains(Xj, ridge=lam)
        joint["move_all_dR2_x1000"] = 1000 * float(dm.sum() / move.sst)
        joint["move_all_t"] = float(dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))))
    log.info("joint linear: %s", json.dumps({k: v for k, v in joint.items() if k != "ridge_grid"}, default=float))
    result["joint_linear"] = joint

    # the production index, for scoring the opt-in blocks on top of it
    bm_all = fit_clogit(np.column_stack([tr_bsp.M, Xj[tr_bsp.idx]]), tr_bsp.y, *tr_bsp.blk,
                        ridge=np.r_[np.zeros(MARKET_TERMS), np.full(Xj.shape[1], lam)])
    prod_index = Xj @ bm_all[MARKET_TERMS:]

    # 3. non-linear: boosting from the market offset
    if not args.no_boost:
        itr, iva = parts["inner"]
        bm_inner = fit_clogit(itr.M, itr.y, *itr.blk)
        Xnum = np.column_stack([pd.to_numeric(pd.Series(frame_vals[c]), errors="coerce").to_numpy(np.float32)
                                for c in prod])
        try:
            d, best = boosted_gain(Xnum, itr, iva, te_bsp, bm_inner)
            result["boosted_bsp"] = {**gain_summary(d, te_bsp.null), "best_iteration": best}
            log.info("boosted over BSP: %s", result["boosted_bsp"])
        except Exception as exc:                      # noqa: BLE001 - report, don't sink the screen
            log.warning("boosted test failed: %s", exc)
            best = None
        if morning is not None and best:
            db_ = []
            for tr_m, te_m in parts["morning"]:
                bm_m = fit_clogit(tr_m.M, tr_m.y, *tr_m.blk)
                g_, _ = boosted_gain(Xnum, tr_m, None, te_m, bm_m, rounds=best or 200)
                db_.append(g_)
            result["boosted_morning"] = {**gain_summary(np.concatenate(db_), morning.null),
                                         "rounds": best, "note": "rounds fixed from the BSP run; no early stopping"}
            log.info("boosted over morning: %s", result["boosted_morning"])
        del Xnum

    # 4. opt-in blocks
    del Xj
    blocks = [b for b in (args.blocks or "").split(",") if b]
    if blocks:
        df = df.copy()
        df, added = attach_blocks(df, blocks, args.db)
        key_df = pd.DataFrame({"race": race_key(df), "horse_name": df["horse_name"].values})
        lookup = pd.DataFrame({"race": sample["race"].values, "horse_name": sample["horse_name"].values,
                               "pos": np.arange(len(sample))})
        m = key_df.reset_index().merge(lookup, on=["race", "horse_name"], how="inner").drop_duplicates("pos")
        block_rows, block_joint = [], {}
        for b, cols in added.items():
            vals = {}
            for c in cols:
                arr = np.full(len(sample), np.nan, dtype=object if c.endswith("_cat") else float)
                arr[m["pos"].to_numpy()] = df[c].to_numpy()[m["index"].to_numpy()]
                vals[c] = arr
            r = screen_features(vals, cols, sample, bsp, morning, move, not args.no_alone, train_mask, starts_all)
            for x in r:
                x["block"] = b
            block_rows += r
            Xb, _ = joint_design(vals, cols, train_mask)
            if Xb.shape[1]:
                block_joint[b] = {"columns": Xb.shape[1],
                                  "bsp_over_market": gain_summary(bsp.gains(Xb, ridge=lam), bsp.null),
                                  "bsp_over_market_and_production": gain_summary(
                                      bsp.gains(Xb, ridge=lam, base_extra=prod_index), bsp.null)}
                if morning is not None:
                    block_joint[b]["morning_over_market"] = gain_summary(morning.gains(Xb, ridge=lam), morning.null)
                log.info("block %s joint: %s", b, block_joint[b])
        if block_rows:
            bp_ = pd.DataFrame(block_rows)
            bp_["stage"] = bp_["feature"].map(stage_of)
            per = pd.concat([per, bp_], ignore_index=True)
            per.to_csv(out / "per_feature.csv", index=False)
        result["blocks"] = block_joint

    result["seconds"] = round(time.time() - t_start)
    (out / "screen.json").write_text(json.dumps(result, indent=2, default=float))
    (out / "summary.md").write_text(render_summary(per, result))
    return result


def render_summary(per: pd.DataFrame, result: dict) -> str:
    L = ["# Residual information screen", ""]
    s = result["sample"]
    L.append(f"{s['runners']:,} runners in {s['races']:,} complete races, {s['from']} to {s['to']}. "
             f"BSP screens fitted before {s['split']} ({s['bsp_train_races']:,} races) and scored after it "
             f"({s['bsp_test_races']:,} races); morning screens cross-fitted by calendar month over "
             f"{s['morning_races']:,} races.")
    L.append("")
    mr = result["market_R2"]
    L.append(f"Market R² out of sample: BSP {mr['bsp_test']:.4f}" +
             (f", morning WAP {mr['morning']:.4f}" if mr.get("morning") else "") + ".")
    L.append("")
    j = result.get("joint_linear", {})
    if j:
        L.append("## Every production feature at once")
        L.append("")
        L.append("| model | ΔLL mnats/race | SE | t | ΔR² |")
        L.append("|---|---|---|---|---|")
        for k in ("bsp_all", "bsp_stage_f_only", "morning_all"):
            if k in j:
                g = j[k]
                L.append(f"| linear, {k} | {g['dll_mnats']:+.3f} | {g['se_mnats']:.3f} | {g['t']:+.2f} | {g['dR2']:+.5f} |")
        for k in ("boosted_bsp", "boosted_morning"):
            if k in result:
                g = result[k]
                L.append(f"| {k} | {g['dll_mnats']:+.3f} | {g['se_mnats']:.3f} | {g['t']:+.2f} | {g['dR2']:+.5f} |")
        L.append("")
        L.append(f"Linear fundamental model alone (Stage F columns, no market): R² {j['fundamental_alone_R2']:.4f}. "
                 f"Ridge {j['ridge']:g} on {j['columns']} columns.")
        if "move_all_dR2_x1000" in j:
            L.append(f"Morning-to-off move explained by all features jointly: ΔR² ×1000 = "
                     f"{j['move_all_dR2_x1000']:+.2f} (t {j['move_all_t']:+.2f}).")
        L.append("")
    cols = [c for c in ("feature", "block", "stage", "kind", "gain_rank", "bsp_dll_mnats", "bsp_t",
                        "bsp_dll_first_half", "bsp_dll_second_half", "alone_R2", "morning_dll_mnats",
                        "morning_t", "move_dR2_x1000", "move_t") if c in per.columns]
    fmt = lambda df_: markdown_table(df_[cols]) if len(df_) else "(none)"
    if "bsp_t" in per.columns:
        L += ["## Most information beyond BSP (per feature, out of sample)", "", fmt(per.sort_values("bsp_t", ascending=False).head(40)), ""]
        n_scr = int(per["bsp_t"].notna().sum())
        L.append(f"{int((per['bsp_t'] > 2).sum())} of {n_scr} screened features have t > 2 beyond BSP; "
                 f"about {0.023 * n_scr:.0f} would by chance.")
        L.append("")
    if "morning_t" in per.columns:
        L += ["## Most information beyond the morning price", "", fmt(per.sort_values("morning_t", ascending=False).head(30)), ""]
    if "move_t" in per.columns:
        L += ["## Best predictors of the morning-to-off move", "", fmt(per.sort_values("move_t", ascending=False).head(30)), ""]
    if "gain_rank" in per.columns and "bsp_t" in per.columns:
        L += ["## The deployed model's top-30 features, on this test", "", fmt(per.sort_values("gain_rank").head(30)), ""]
    if result.get("blocks"):
        L += ["## Opt-in blocks, jointly", "", "```", json.dumps(result["blocks"], indent=1, default=float), "```", ""]
    return "\n".join(L)


def markdown_table(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(ROOT / "horse_racing.db"))
    ap.add_argument("--feature-cache", default=None, metavar="DIR")
    ap.add_argument("--start-date", default="2021-01-01")
    ap.add_argument("--split", default="2024-07-01", help="BSP screens: fit before, score from")
    ap.add_argument("--lockbox-from", default="2026-04-01",
                    help="withhold every race on or after this date (the locked holdout); '' to disable")
    ap.add_argument("--val-months", type=int, default=6, help="inner validation tail of the training window")
    ap.add_argument("--ridge-grid", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    ap.add_argument("--blocks", default="", help="comma list: perf,kalman,blandford,pedigree,connections,comments,markets")
    ap.add_argument("--limit", type=int, default=0, help="screen only the first N production features (smoke runs)")
    ap.add_argument("--no-alone", action="store_true")
    ap.add_argument("--no-boost", action="store_true")
    ap.add_argument("--no-morning", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "residual_screen"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = run(args)
    print(json.dumps({k: res[k] for k in ("sample", "market_R2", "joint_linear") if k in res}, indent=1, default=float))


if __name__ == "__main__":
    main()
