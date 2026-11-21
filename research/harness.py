"""The validation harness's scoring (the owner's model brief, 18 Nov 2026, Phase 1; horse racing only).

One place that scores a set of out-of-sample predictions on both of the owner's targets, with every interval
resampled by race day:

    bfsp_error        the mean absolute log error of the predicted price against the BFSP, overall and by segment
    win_scores        log loss and Brier of the race-normalised probability against the market's (the BSP,
                      normalised), and the calibration by decile (ECE)
    benter            a conditional logit on [log p_model, log p_market], fitted walk-forward by fold (each fold
                      scored by a fit on the folds before it): the weights and the log-likelihood over the market alone
    order_agreement   within each race, Spearman's correlation of the price order with an outcome (race-centred
                      pounds beaten first, the rule of 18 Nov), the BSP's order beside it

Inputs are plain frames with these columns (the research loop's oos_*.csv carry them): race_date, race_time, track,
horse_name, predicted_bfsp, bfsp, won, fold_idx, number_of_runners, race_code, surface_type; an outcome column for
order_agreement. Nothing here reads the database or the locked holdout; the caller decides the window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

B = 1000
SEED = 20261118
PRICE_BANDS = [1, 3, 6, 12, 25, 50, 1e9]
PRICE_LABELS = ["<3", "3-6", "6-12", "12-25", "25-50", "50+"]
FIELD_BANDS = [0, 7, 11, 15, 99]
FIELD_LABELS = ["2-7", "8-11", "12-15", "16+"]


def race_ids(df: pd.DataFrame) -> np.ndarray:
    """A race is its date, time and course (the results table has no race id)."""
    key = df["race_date"].astype(str) + "|" + df["race_time"].astype(str) + "|" + df["track"].astype(str)
    return pd.factorize(key)[0]


def day_ids(df: pd.DataFrame) -> np.ndarray:
    return pd.factorize(df["race_date"].astype(str))[0]


def day_bootstrap(day: np.ndarray, num: np.ndarray, den: np.ndarray | None = None, level: float = 0.95,
                  draws: int = B, seed: int = SEED) -> tuple[float, float, float]:
    """A ratio of sums (a mean when den is None) and its interval, race days drawn with replacement."""
    num = np.asarray(num, dtype=float)
    den = np.ones_like(num) if den is None else np.asarray(den, dtype=float)
    days, inv = np.unique(np.asarray(day), return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(days))
    sd = np.bincount(inv, weights=den, minlength=len(days))
    est = sn.sum() / sd.sum()
    idx = np.random.default_rng(seed).integers(0, len(days), size=(draws, len(days)))
    bs = sn[idx].sum(axis=1) / sd[idx].sum(axis=1)
    a = (1.0 - level) / 2.0
    return float(est), float(np.quantile(bs, a)), float(np.quantile(bs, 1.0 - a))


def normalise(price: np.ndarray, race: np.ndarray) -> np.ndarray:
    """The race-normalised implied probability of each runner's price (the book made to sum to 1)."""
    q = 1.0 / np.asarray(price, dtype=float)
    return q / pd.Series(q).groupby(race).transform("sum").to_numpy()


def check_race_sums(p: np.ndarray, race: np.ndarray, tol: float = 1e-9) -> int:
    """How many races' probabilities do not sum to 1 (0 is the contract)."""
    s = pd.Series(np.asarray(p, dtype=float)).groupby(race).sum()
    return int((np.abs(s - 1.0) > tol).sum())


def code_of(df: pd.DataFrame) -> np.ndarray:
    jumps = df["race_code"].astype(str).str.contains("Hurdle|Chase|NH|National Hunt", case=False)
    aw = df["surface_type"].astype(str).str.contains("Polytrack|Tapeta|Fibresand|All", case=False)
    return np.where(jumps, "jumps", np.where(aw, "aw", "flat turf"))


def bfsp_error(df: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Mean absolute log error against the BFSP, overall and by code, price band, field size, quarter, year, track and
    race type (the last two when the frame carries them)."""
    ok = (df["bfsp"] > 1) & (df["predicted_bfsp"] > 1)
    e = df.loc[ok].copy()
    e["err"] = np.abs(np.log(e["predicted_bfsp"]) - np.log(e["bfsp"]))
    e["day"] = day_ids(e)
    e["code"] = code_of(e)
    e["band"] = pd.cut(e["bfsp"], PRICE_BANDS, labels=PRICE_LABELS)
    e["field"] = pd.cut(e["number_of_runners"], FIELD_BANDS, labels=FIELD_LABELS)
    e["quarter"] = pd.to_datetime(e["race_date"]).dt.to_period("Q").astype(str)
    e["year"] = pd.to_datetime(e["race_date"]).dt.year.astype(str)
    segs = ["code", "band", "field", "quarter", "year"] + [c for c in ("track", "race_type") if c in e.columns]
    rows = [("all", "all", len(e), *day_bootstrap(e["day"].to_numpy(), e["err"].to_numpy(), level=level))]
    for seg in segs:
        for v, s in e.groupby(seg, observed=True):
            rows.append((seg, str(v), len(s), *day_bootstrap(s["day"].to_numpy(), s["err"].to_numpy(), level=level)))
    return pd.DataFrame(rows, columns=["segment", "value", "runners", "error", "lo", "hi"])


def _one_winner(df: pd.DataFrame) -> pd.DataFrame:
    ok = (df["bfsp"] > 1) & (df["predicted_bfsp"] > 1) & df["won"].notna()
    w = df.loc[ok].copy()
    w["race"] = race_ids(w)
    w = w[w.groupby("race")["won"].transform("sum") == 1].copy()
    w["race"] = race_ids(w)
    w["day"] = day_ids(w)
    w["p_mkt"] = normalise(w["bfsp"].to_numpy(), w["race"].to_numpy())
    w["p_mod"] = normalise(w["predicted_bfsp"].to_numpy(), w["race"].to_numpy())
    return w


def win_scores(df: pd.DataFrame, level: float = 0.95) -> dict:
    """Log loss and Brier per runner, the model's and the market's, their difference, and the ECE by decile."""
    w = _one_winner(df)
    y = w["won"].astype(float).to_numpy()
    day = w["day"].to_numpy()
    out = {"races": int(w["race"].nunique()), "runners": len(w)}
    ll = {}
    for name in ("p_mod", "p_mkt"):
        p = np.clip(w[name].to_numpy(), 1e-9, 1 - 1e-9)
        ll[name] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        out[f"{name}_logloss"] = day_bootstrap(day, ll[name], level=level)
        out[f"{name}_brier"] = day_bootstrap(day, (p - y) ** 2, level=level)
    out["logloss_model_minus_market"] = day_bootstrap(day, ll["p_mod"] - ll["p_mkt"], level=level)
    dec = pd.qcut(w["p_mod"], 10, labels=False, duplicates="drop")
    cal = w.groupby(dec).agg(n=("won", "size"), p=("p_mod", "mean"), obs=("won", "mean"), mkt=("p_mkt", "mean"))
    out["ece"] = float((cal["n"] * (cal["p"] - cal["obs"]).abs()).sum() / cal["n"].sum())
    out["calibration"] = cal
    return out


def _clogit_nll(beta, X, race, y):
    s = X @ beta
    mx = pd.Series(s).groupby(race).transform("max").to_numpy()
    den = np.bincount(race, weights=np.exp(s - mx))
    first = pd.Series(mx).groupby(race).first().to_numpy()
    return -(s[y == 1].sum() - (np.log(den) + first).sum())


def clogit_fit(X: np.ndarray, race: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights of a conditional (within-race) logit; race must be 0..R-1 and each race one winner."""
    return minimize(_clogit_nll, np.ones(X.shape[1]), args=(X, race, y), method="BFGS").x


def _race_loglik(s: np.ndarray, race: np.ndarray, y: np.ndarray) -> np.ndarray:
    mx = pd.Series(s).groupby(race).transform("max").to_numpy()
    lse = np.log(pd.Series(np.exp(s - mx)).groupby(race).transform("sum").to_numpy()) + mx
    ll = pd.Series(np.where(y == 1, s - lse, 0.0)).groupby(race).sum()
    return ll.to_numpy()


def benter(df: pd.DataFrame, level: float = 0.95) -> dict:
    """The walk-forward Benter test: each fold scored by weights fitted on the folds before it. The gain is the
    log-likelihood per race over the market alone (positive: the model adds to the market)."""
    w = _one_winner(df)
    w["lm"], w["lk"] = np.log(w["p_mod"]), np.log(w["p_mkt"])
    folds = sorted(w["fold_idx"].unique())
    weights, gains, days = [], [], []
    for f in folds[1:]:
        tr, te = w[w["fold_idx"] < f], w[w["fold_idx"] == f]
        rtr = pd.factorize(tr["race"])[0]
        b2 = clogit_fit(tr[["lm", "lk"]].to_numpy(), rtr, tr["won"].to_numpy().astype(int))
        b1 = clogit_fit(tr[["lk"]].to_numpy(), rtr, tr["won"].to_numpy().astype(int))
        weights.append({"fold": int(f), "beta_model": float(b2[0]), "beta_market": float(b2[1]),
                        "beta_market_alone": float(b1[0])})
        rte = pd.factorize(te["race"])[0]
        yte = te["won"].to_numpy().astype(int)
        g = _race_loglik(te[["lm", "lk"]].to_numpy() @ b2, rte, yte) - _race_loglik(te[["lk"]].to_numpy() @ b1, rte, yte)
        gains.append(g)
        days.append(pd.Series(te["day"].to_numpy()).groupby(rte).first().to_numpy())
    if not gains:
        return {"weights": weights, "races": 0, "gain_per_race": None}
    g, d = np.concatenate(gains), np.concatenate(days)
    return {"weights": weights, "races": len(g), "gain_per_race": day_bootstrap(d, g, level=level)}


def order_agreement(df: pd.DataFrame, outcome: str, min_runners: int = 4, level: float = 0.95) -> dict:
    """Within each race (at least min_runners with the outcome), Spearman's correlation of the model's price order
    and the BSP's with the outcome (higher is better for the horse), the mean over races and the difference."""
    ok = (df["bfsp"] > 1) & (df["predicted_bfsp"] > 1) & df[outcome].notna()
    d = df.loc[ok].copy()
    d["race"] = race_ids(d)
    d = d[d.groupby("race")[outcome].transform("size") >= min_runners].copy()
    d["day"] = day_ids(d)
    d["s_mod"], d["s_mkt"] = -np.log(d["predicted_bfsp"]), -np.log(d["bfsp"])
    rk = d.groupby("race")[[outcome, "s_mod", "s_mkt"]].rank()
    rk["race"], rk["day"] = d["race"], d["day"]
    rows = []
    for race, g in rk.groupby("race"):
        rows.append((g["day"].iloc[0], g[outcome].corr(g["s_mod"]), g[outcome].corr(g["s_mkt"])))
    r = pd.DataFrame(rows, columns=["day", "mod", "mkt"]).dropna()
    if r.empty:
        return {"outcome": outcome, "races": 0, "note": "no data found"}
    day = r["day"].to_numpy()
    return {"outcome": outcome, "races": len(r),
            "model": day_bootstrap(day, r["mod"].to_numpy(), level=level),
            "market": day_bootstrap(day, r["mkt"].to_numpy(), level=level),
            "model_minus_market": day_bootstrap(day, (r["mod"] - r["mkt"]).to_numpy(), level=level)}
