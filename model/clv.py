"""
Closing-line value (CLV): the objective is not to beat BSP but to *forecast*
it well enough to get on earlier at longer prices.

Why CLV is the right score. Back a horse early at odds o and it closes at
BSP b. If you green up by laying at b, the locked-in profit per unit staked
is exactly o/b − 1 (before commission) whether it wins or loses. If you hold
to settlement, the expected profit is the same quantity in expectation once
you accept that 1/b is the best available estimate of the true probability.
So the strategy's edge is the realised CLV of the bets it selects, and the
model's job is to predict b from what is known in the morning.

Model (Benter's blend, applied to prices):
    ln b ≈ β0 + β1 ln π_morning + β2 ln f̂ (+ β3 ln volume share, ...)
where f̂ is the fundamentals-only BSP forecast (train_bfsp.py, walk-forward)
and π_morning the morning market price, both as race-normalised
probabilities. Since the target is a log *price*, the coefficients on the
log probabilities are negative; |β2| clearly > 0 means fundamentals predict
the move beyond the morning price; the early-bet rule then backs runners
whose predicted BSP is shorter than the price on offer by a margin.

    price_move_model      walk-forward fit of the blend
    early_bet_rule        select + size (predicted CLV above a threshold)
    clv_report            realised CLV by predicted-CLV tier, hit rate,
                          hold-to-settlement EV, volume constraints,
                          race-bootstrap CI
    steam_predictability  does f̂ vs morning price predict the direction of the move?

Inputs: one row per runner with bsp (closing), morning price, model BSP
forecast, race id, date; optional morning traded volume and outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.bet_analysis import cluster_bootstrap_roi
from model.diagnostics import race_normalise


def prepare_clv_frame(df: pd.DataFrame, bsp_col: str = "bfsp", early_col: str = "morningwap", pred_col: str = "predicted_bfsp",
                      race_col: str = "raceid", vol_col: str | None = "morning_vol") -> pd.DataFrame:
    d = df.copy()
    for c in (bsp_col, early_col, pred_col):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d[bsp_col] > 1.0) & (d[early_col] > 1.0) & (d[pred_col] > 1.0)].copy()
    d["ln_bsp"] = np.log(d[bsp_col]); d["ln_early"] = np.log(d[early_col]); d["ln_pred"] = np.log(d[pred_col])
    # race-normalised implied probabilities (so the blend sees distributions, not raw prices)
    d["_pe"] = 1 / d[early_col]; d["p_early"] = race_normalise(d, "_pe", race_col)
    d["_pm"] = 1 / d[pred_col]; d["p_pred"] = race_normalise(d, "_pm", race_col)
    d["ln_n"] = np.log(d.groupby(race_col)[race_col].transform("size"))
    if vol_col and vol_col in d.columns:
        v = pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0)
        d["vol_share"] = v / v.groupby(d[race_col]).transform("sum").replace(0, np.nan)
        d["ln_vol"] = np.log1p(v)
    d["realised_clv"] = d[early_col] / d[bsp_col] - 1.0          # back early, green up at BSP
    d["move"] = d["ln_early"] - d["ln_bsp"]                       # > 0: price shortened (steamer)
    return d.drop(columns=["_pe", "_pm"])


def price_move_model(d: pd.DataFrame, date_col: str = "race_date", n_folds: int = 6, extra_cols=None, ridge: float = 1e-3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward ridge regression of ln BSP on [ln p_early, ln p_pred, ln n, extras].

    Returns the frame with ``ln_bsp_hat`` / ``bsp_hat`` / ``pred_clv`` (= early/bsp_hat − 1)
    and a per-fold coefficient table. The fold structure is chronological.
    """
    d = d.copy(); d[date_col] = pd.to_datetime(d[date_col])
    cols = ["ln_p_early", "ln_p_pred", "ln_n"] + list(extra_cols or [])
    d["ln_p_early"] = np.log(d["p_early"].clip(1e-6)); d["ln_p_pred"] = np.log(d["p_pred"].clip(1e-6))
    dates = np.sort(d[date_col].unique()); edges = np.linspace(0, len(dates), n_folds + 1).astype(int)
    d["ln_bsp_hat"] = np.nan; rows = []
    for k in range(1, n_folds):
        tr = d[date_col].isin(dates[: edges[k]]).values; te = d[date_col].isin(dates[edges[k]: edges[k + 1]]).values
        if tr.sum() < 500 or te.sum() == 0:
            continue
        X = d.loc[tr, cols].fillna(0.0).values; X1 = np.column_stack([np.ones(len(X)), X]); y = d.loc[tr, "ln_bsp"].values
        beta = np.linalg.solve(X1.T @ X1 + ridge * np.eye(X1.shape[1]), X1.T @ y)
        Xt = np.column_stack([np.ones(te.sum()), d.loc[te, cols].fillna(0.0).values])
        d.loc[te, "ln_bsp_hat"] = Xt @ beta
        resid = y - X1 @ beta
        rows.append({"fold": k, "n_train": int(tr.sum()), **dict(zip(["intercept"] + cols, beta)), "train_rmse": float(np.sqrt(np.mean(resid ** 2)))})
    d["bsp_hat"] = np.exp(d["ln_bsp_hat"]).clip(upper=1000.0)
    d["pred_clv"] = np.exp(d["ln_early"]) / d["bsp_hat"] - 1.0
    return d, pd.DataFrame(rows)


def early_bet_rule(d: pd.DataFrame, min_pred_clv: float = 0.10, commission: float = 0.05, max_early_odds: float = 50.0,
                   min_vol: float | None = None, vol_col: str = "morning_vol") -> pd.DataFrame:
    """Select runners whose forecast BSP is shorter than the morning price by ≥ min_pred_clv.
    Predicted net edge if greened up: pred_clv·(1−commission)."""
    b = d[d["pred_clv"].notna() & (d["pred_clv"] >= min_pred_clv) & (np.exp(d["ln_early"]) <= max_early_odds)].copy()
    if min_vol is not None and vol_col in b.columns:
        b = b[pd.to_numeric(b[vol_col], errors="coerce").fillna(0) >= min_vol]
    b["net_clv"] = b["realised_clv"] * (1 - commission)
    b["ret_back"] = b["net_clv"]  # greened-up return per unit (for the race bootstrap helper)
    return b


def clv_report(d: pd.DataFrame, bets: pd.DataFrame, race_col: str = "raceid", y_col: str | None = "won", commission: float = 0.05,
               tiers=(0.0, 0.05, 0.10, 0.20, 0.35, 0.60, np.inf)) -> dict:
    out = {"n_runners": int(len(d)), "n_bets": int(len(bets)), "bets_per_race": float(len(bets) / max(d[race_col].nunique(), 1))}
    if len(bets):
        lo, hi = cluster_bootstrap_roi(bets, "net_clv", race_col, n_boot=300)
        out.update({"mean_net_clv": float(bets["net_clv"].mean()), "median_net_clv": float(bets["net_clv"].median()),
                    "clv_ci90": (lo, hi), "hit_rate_shortened": float((bets["realised_clv"] > 0).mean()),
                    "mean_pred_clv": float(bets["pred_clv"].mean())})
        if y_col and y_col in bets.columns:
            y = pd.to_numeric(bets[y_col].replace({True: 1, False: 0}), errors="coerce").fillna(0).values > 0
            early = np.exp(bets["ln_early"].values)
            out["hold_roi_net"] = float(np.mean(np.where(y, (early - 1) * (1 - commission), -1.0)))
            out["bsp_implied_ev_net"] = float(np.mean((early - 1) * (1 - commission) / np.exp(bets["ln_bsp"].values) - (1 - 1 / np.exp(bets["ln_bsp"].values))))
    # by predicted-CLV tier over all runners
    t = d[d["pred_clv"].notna()].copy(); t["tier"] = pd.cut(t["pred_clv"], tiers, right=False)
    tab = t.groupby("tier", observed=True).agg(n=("realised_clv", "size"), pred_clv=("pred_clv", "mean"), realised_clv=("realised_clv", "mean"),
                                                hit_rate=("realised_clv", lambda x: (x > 0).mean()), avg_early=("ln_early", lambda x: np.exp(x).mean())).reset_index()
    tab["tier"] = tab["tier"].astype(str); out["by_pred_clv_tier"] = tab
    # baseline: back everything in the morning (is the morning market itself biased?)
    out["all_runners_mean_clv"] = float(d["realised_clv"].mean()); out["all_runners_hit_rate"] = float((d["realised_clv"] > 0).mean())
    return out


def steam_predictability(d: pd.DataFrame) -> dict:
    """Does the fundamentals forecast anticipate the morning→BSP move? Correlation of
    (ln p_pred − ln p_early) with the realised move, and the direction hit rate."""
    ok = d["ln_pred"].notna() & d["ln_early"].notna() & d["ln_bsp"].notna()
    signal = np.log(d.loc[ok, "p_pred"].clip(1e-6)) - np.log(d.loc[ok, "p_early"].clip(1e-6))
    move = d.loc[ok, "move"].values
    strong = np.abs(signal) > np.quantile(np.abs(signal), 0.75)
    return {"corr_signal_move": float(np.corrcoef(signal, move)[0, 1]), "direction_hit_rate": float(np.mean(np.sign(signal) == np.sign(move))),
            "direction_hit_rate_strong_signal": float(np.mean(np.sign(signal[strong]) == np.sign(move[strong]))), "n": int(ok.sum())}
