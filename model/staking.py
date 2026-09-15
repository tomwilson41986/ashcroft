"""
Staking analysis: what the model's probabilities are worth when they, and
nothing else, decide the bets.

The market is excluded from every *decision* here. It enters only where it
cannot be avoided -- as the price the bet is settled at (BSP, net of
commission). That asymmetry is the whole point, and it is worth stating
plainly: a Kelly stake is a function of two numbers, the probability you
believe and the price someone offers. Strip the market out of the price as
well and the model prices its own bets, its implied odds equal its own
probabilities, the edge is identically zero and Kelly stakes nothing. So
"Kelly without the market" means: model probabilities, market price, no
market-derived filter, no blending.

    add_kelly          per-runner Kelly fraction at the settlement price
    bankroll_path      compounding bank, races in running order
    growth_stats       terminal multiple, per-race log growth, drawdown, ruin
    kelly_ladder       full / half / quarter / ... Kelly side by side
    rank_staking       the model's n-th choice: flat, proportional, level-profit
    topk_staking       backing the model's top k in every race
    shrinkage_scan     how far probabilities must be flattened to stop losing
    price_forecast_by_rank   predicted BFSP vs actual BFSP, by model rank

Returns are per unit staked, commission taken on net winnings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.bet_analysis import cluster_bootstrap_roi, prepare_bets


# ---------------------------------------------------------------------------
# Kelly at the settlement price
# ---------------------------------------------------------------------------

def add_kelly(d: pd.DataFrame, p_col: str = "p_model", price_col: str = "bsp",
              commission: float = 0.05, cap: float = 1.0) -> pd.DataFrame:
    """Kelly fraction of bank per runner, plus the growth the model expects.

    ``b_net`` is the net-of-commission payout per unit staked, so the usual
    f* = (p·b - q)/b holds with b = b_net. ``g_model`` is the expected log
    growth *if the model's probability were true* -- the promise. The
    realised path is what tests it."""
    out = d.copy()
    b = (out[price_col].astype(float) - 1.0) * (1.0 - commission)
    p = out[p_col].astype(float).clip(1e-9, 1 - 1e-9)
    out["b_net"] = b
    # The model's own price: what it can filter on at bet time without looking
    # at the market. Realised BSP is not available when the bet is struck, so a
    # rule conditioned on it is look-ahead, not a strategy.
    out["model_price"] = (out["predicted_bfsp"].astype(float) if "predicted_bfsp" in out.columns
                          else 1.0 / p)
    out["kelly_full"] = np.clip((p * b - (1.0 - p)) / b, 0.0, cap)
    out["edge_model"] = p * b - (1.0 - p)            # expected profit per unit staked
    f = out["kelly_full"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["g_model"] = np.where(f > 0, p * np.log1p(f * b) + (1 - p) * np.log1p(-f), 0.0)
    return out


def _race_order(d: pd.DataFrame, race_col: str = "raceid") -> pd.Series:
    """Chronological key per race (date + off time when available)."""
    if "race_date" in d.columns:
        t = d["race_time"].astype(str) if "race_time" in d.columns else "00:00"
        key = pd.to_datetime(d["race_date"].astype(str) + " " + t, errors="coerce", format="mixed")
        if key.isna().all():
            key = pd.to_datetime(d["race_date"], errors="coerce")
        return key.fillna(pd.to_datetime(d["race_date"], errors="coerce"))
    return pd.Series(np.arange(len(d)), index=d.index)


def bankroll_path(d: pd.DataFrame, fraction: float = 0.25, stake_col: str = "kelly_full",
                  race_col: str = "raceid", commission: float = 0.05,
                  max_exposure: float = 1.0, bank0: float = 1.0) -> pd.DataFrame:
    """Compound the bank race by race in running order.

    Stakes inside one race are laid out simultaneously against the same bank,
    so the bank multiplies by ``1 + Σ fᵢ(bᵢ·1{win} - 1)``. Because every stake
    is a fraction of the current bank, the whole path is a cumulative product
    -- no simulation loop, and no approximation beyond scaling the stakes down
    proportionally if they would commit more than ``max_exposure`` of the bank.

    The path is carried in logs: a losing Kelly run drives the bank far below
    the smallest float, and a cumprod would silently flatten to zero long
    before the losing actually stops."""
    t = d.copy()
    t["_f"] = float(fraction) * t[stake_col].astype(float)
    t = t[t["_f"] > 0]
    if t.empty:
        return pd.DataFrame(columns=["raceid", "when", "bets", "staked", "pnl", "log_mult", "log_bank", "bank"])
    order = _race_order(d, race_col)
    # ret_back is already net profit per unit staked (+b_net on a winner, -1 on a
    # loser), so the race P&L is just the stake-weighted sum of it.
    g = (t.assign(_pnl=t["_f"] * t["ret_back"])
           .groupby(race_col).agg(stake=("_f", "sum"), raw_pnl=("_pnl", "sum"), bets=("_f", "size")))
    when = order.groupby(d[race_col]).min()
    g = g.join(when.rename("when")).sort_values("when")
    scale = np.where(g["stake"] > max_exposure, max_exposure / g["stake"].replace(0, np.nan), 1.0)
    g["scale"] = np.nan_to_num(scale, nan=1.0)
    g["staked"] = g["stake"] * g["scale"]
    g["pnl"] = g["raw_pnl"] * g["scale"]
    mult = 1.0 + g["pnl"].values
    g["ruin"] = mult <= 0.0
    g["log_mult"] = np.log(np.maximum(mult, 1e-300))
    if g["ruin"].any():                       # bank gone: nothing compounds after it
        first = int(np.argmax(g["ruin"].values))
        g = g.iloc[: first + 1]
    g["log_bank"] = np.log(bank0) + g["log_mult"].cumsum()
    g["bank"] = np.exp(np.clip(g["log_bank"], -700, 700))
    return g.reset_index()


def growth_stats(path: pd.DataFrame, n_races_total: int | None = None) -> dict:
    """Terminal multiple, per-race log growth, worst drawdown, time under water."""
    if path.empty:
        return {"races_bet": 0, "bets": 0, "turnover_bank_multiples": 0.0, "log10_terminal_bank": 0.0,
                "log_growth_per_race": 0.0, "roi_on_turnover_pct": np.nan, "max_drawdown_pct": 0.0,
                "pct_races_under_water": 0.0, "ruined": False, "races_offered": int(n_races_total or 0),
                "mean_race_pnl_pct": 0.0, "races_to_halve_bank": np.nan, "races_to_1pct": np.nan}
    log_bank = path["log_bank"].values
    peak = np.maximum.accumulate(log_bank)
    dd = 1.0 - np.exp(np.clip(log_bank - peak, -700, 0))
    under = peak > log_bank + 1e-12
    ruined = bool(path["ruin"].any()) if "ruin" in path.columns else False
    return {
        "races_bet": int(len(path)),
        "races_offered": int(n_races_total) if n_races_total else int(len(path)),
        "bets": int(path["bets"].sum()),
        "turnover_bank_multiples": float(path["staked"].sum()),
        "log10_terminal_bank": float(log_bank[-1] / np.log(10)),
        "log_growth_per_race": float(path["log_mult"].mean()),
        "mean_race_pnl_pct": float(100 * path["pnl"].mean()),
        "roi_on_turnover_pct": float(100 * path["pnl"].sum() / path["staked"].sum()) if path["staked"].sum() else np.nan,
        "max_drawdown_pct": float(100 * dd.max()),
        "pct_races_under_water": float(100 * under.mean()),
        "ruined": ruined,
        "races_to_halve_bank": _races_to_level(log_bank, np.log(0.5)),
        "races_to_1pct": _races_to_level(log_bank, np.log(0.01)),
    }


def _races_to_level(log_bank: np.ndarray, level: float) -> float:
    hit = np.flatnonzero(log_bank <= level)
    return float(hit[0] + 1) if hit.size else np.nan


def kelly_ladder(d: pd.DataFrame, fractions=(1.0, 0.5, 0.25, 0.125, 0.05),
                 race_col: str = "raceid", n_races_total: int | None = None,
                 **kw) -> pd.DataFrame:
    rows = []
    for f in fractions:
        path = bankroll_path(d, fraction=f, race_col=race_col, **kw)
        s = growth_stats(path, n_races_total)
        s["kelly_fraction"] = f
        rows.append(s)
    cols = ["kelly_fraction", "races_bet", "bets", "turnover_bank_multiples", "log10_terminal_bank",
            "log_growth_per_race", "roi_on_turnover_pct", "races_to_halve_bank", "races_to_1pct",
            "max_drawdown_pct", "ruined"]
    return pd.DataFrame(rows)[cols]


def kelly_promise_vs_reality(d: pd.DataFrame, race_col: str = "raceid") -> dict:
    """What full-Kelly growth the model promises against what it delivered.

    The promise is the model's own expected log growth summed over its bets;
    the delivery is the realised log bank. The gap is the price of believing
    probabilities that are less resolved than the market's. The mean edge is
    reported alongside the median because a handful of 100/1 shots the model
    likes dominate any average taken in edge units."""
    bets = d[d["kelly_full"] > 0]
    path = bankroll_path(d, fraction=1.0, race_col=race_col)
    w = bets["kelly_full"].values
    return {
        "runners": int(len(d)),
        "bets": int(len(bets)),
        "pct_of_field_backed": float(100 * len(bets) / len(d)),
        "mean_kelly_stake_pct": float(100 * bets["kelly_full"].mean()) if len(bets) else 0.0,
        "median_model_edge_pct": float(100 * bets["edge_model"].median()) if len(bets) else np.nan,
        "mean_model_edge_pct": float(100 * bets["edge_model"].mean()) if len(bets) else np.nan,
        "stake_weighted_model_edge_pct": float(100 * np.average(bets["edge_model"], weights=w)) if len(bets) else np.nan,
        "realised_edge_pct": float(100 * bets["ret_back"].mean()) if len(bets) else np.nan,
        "stake_weighted_realised_edge_pct": float(100 * np.average(bets["ret_back"], weights=w)) if len(bets) else np.nan,
        "promised_log_growth": float(bets["g_model"].sum()),
        "realised_log_growth": float(path["log_bank"].iloc[-1]) if len(path) else 0.0,
    }


def kelly_by_rank(d: pd.DataFrame, ranks=(1, 2, 3, 4, 5, 6), fraction: float = 0.25,
                  race_col: str = "raceid", cumulative: bool = False) -> pd.DataFrame:
    """Kelly, but only on the model's n-th choice (or its top n).

    This is the join of the two questions: sizing by the model's own
    confidence, selecting by the model's own ranking, market untouched
    except as the settlement price."""
    rows = []
    for r in ranks:
        sub = d[d["model_rank"] <= r] if cumulative else d[d["model_rank"] == r]
        if sub.empty:
            continue
        path = bankroll_path(sub, fraction=fraction, race_col=race_col)
        s = growth_stats(path)
        bets = sub[sub["kelly_full"] > 0]
        rows.append({"rank": f"<= {r}" if cumulative else str(r), "runners": len(sub),
                     "bets": int(len(bets)), "pct_backed": 100 * len(bets) / len(sub),
                     "mean_stake_pct": 100 * fraction * bets["kelly_full"].mean() if len(bets) else 0.0,
                     "roi_pct": 100 * bets["ret_back"].mean() if len(bets) else np.nan,
                     "log10_terminal_bank": s["log10_terminal_bank"],
                     "log_growth_per_race": s["log_growth_per_race"],
                     "max_drawdown_pct": s["max_drawdown_pct"], "ruined": s["ruined"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rank staking: the market never touches the selection
# ---------------------------------------------------------------------------

STAKE_MODES = ("flat", "proportional", "level_profit", "kelly_quarter")


def _stakes(g: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "flat":
        return np.ones(len(g))
    if mode == "proportional":               # stake ∝ model probability
        p = g["p_model"].values
        return p / p.mean() if p.mean() > 0 else np.zeros(len(g))
    if mode == "level_profit":               # stake to win one unit
        return np.clip(1.0 / np.maximum(g["b_net"].values, 1e-6), 0, 20)
    if mode == "kelly_quarter":
        f = 0.25 * g["kelly_full"].values
        return f / f.mean() if f.mean() > 0 else np.zeros(len(g))
    raise ValueError(mode)


def rank_staking(d: pd.DataFrame, max_rank: int = 6, modes=STAKE_MODES,
                 race_col: str = "raceid", n_boot: int = 300) -> pd.DataFrame:
    """Per model rank: win rate, ROI and staking-plan ROI, settled at BSP."""
    rows = []
    for r in range(1, max_rank + 1):
        g = d[d["model_rank"] == r]
        if g.empty:
            continue
        lo, hi = cluster_bootstrap_roi(g, "ret_back", race_col, n_boot=n_boot)
        row = {"model_rank": r, "n": len(g), "win_rate_pct": 100 * g["won"].mean(),
               "place_rate_pct": 100 * g["placed"].mean(), "avg_bsp": g["bsp"].mean(),
               "median_bsp": g["bsp"].median(),
               "roi_flat_pct": 100 * g["ret_back"].mean(),
               "roi_ci_lo_pct": 100 * lo, "roi_ci_hi_pct": 100 * hi,
               "roi_lay_flat_pct": 100 * g["ret_lay"].mean()}
        for m in modes:
            if m == "flat":
                continue
            s = _stakes(g, m)
            row[f"roi_{m}_pct"] = 100 * float(np.sum(s * g["ret_back"].values) / np.sum(s)) if s.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def topk_staking(d: pd.DataFrame, ks=(1, 2, 3, 4), race_col: str = "raceid",
                 n_boot: int = 300) -> pd.DataFrame:
    """Back the model's top k in every race, flat stakes."""
    rows = []
    for k in ks:
        g = d[d["model_rank"] <= k]
        lo, hi = cluster_bootstrap_roi(g, "ret_back", race_col, n_boot=n_boot)
        races = g[race_col].nunique()
        rows.append({"top_k": k, "bets": len(g), "races": races,
                     "strike_rate_pct": 100 * g.groupby(race_col)["won"].max().mean(),
                     "roi_pct": 100 * g["ret_back"].mean(),
                     "roi_ci_lo_pct": 100 * lo, "roi_ci_hi_pct": 100 * hi,
                     "pnl_per_race": g["ret_back"].sum() / races})
    return pd.DataFrame(rows)


def rank_segments(d: pd.DataFrame, by: str, bands=None, rank: int = 1,
                  race_col: str = "raceid", n_boot: int = 200) -> pd.DataFrame:
    """Model rank-`rank` performance split by a column (banded if numeric)."""
    g = d[d["model_rank"] == rank].copy()
    key = pd.cut(g[by], bands) if bands is not None else g[by].astype(str)
    rows = []
    for val, sub in g.groupby(key, observed=True):
        if len(sub) < 50:
            continue
        lo, hi = cluster_bootstrap_roi(sub, "ret_back", race_col, n_boot=n_boot)
        rows.append({by: str(val), "n": len(sub), "win_rate_pct": 100 * sub["won"].mean(),
                     "avg_bsp": sub["bsp"].mean(), "roi_pct": 100 * sub["ret_back"].mean(),
                     "roi_ci_lo_pct": 100 * lo, "roi_ci_hi_pct": 100 * hi})
    out = pd.DataFrame(rows)
    return out.sort_values("roi_pct", ascending=False) if len(out) else out


def time_split_table(sub: pd.DataFrame, n_splits: int = 4, race_col: str = "raceid") -> pd.DataFrame:
    """Split a selection chronologically and show ROI in each slice.

    A subset found by slicing 250,000 runners many ways will show a good ROI
    somewhere by construction. Splitting it in time is the cheapest test of
    whether the edge is a property of the horses or of the search."""
    s = sub.copy()
    s["_when"] = _race_order(s, race_col)
    s = s.sort_values("_when")
    if len(s) < n_splits * 20:
        return pd.DataFrame()
    parts = np.array_split(np.arange(len(s)), n_splits)
    rows = []
    for i, idx in enumerate(parts, 1):
        g = s.iloc[idx]
        rows.append({"slice": i, "from": str(g["_when"].min())[:10], "to": str(g["_when"].max())[:10],
                     "n": len(g), "win_rate_pct": 100 * g["won"].mean(),
                     "roi_pct": 100 * g["ret_back"].mean()})
    return pd.DataFrame(rows)


def rank_focus(d: pd.DataFrame, rank: int = 1, field_min: int | None = None,
               price_min: float | None = None, fraction: float = 0.25,
               race_col: str = "raceid", n_boot: int = 1000) -> dict:
    """One market-blind selection, examined properly: CI, bank path, stability.

    ``price_min`` filters on the model's own forecast price, not on the BSP the
    bet settles at -- the latter is unknown when the bet is struck."""
    sub = d[d["model_rank"] == rank]
    if field_min is not None:
        sub = sub[sub["field"] >= field_min]
    if price_min is not None:
        sub = sub[sub["model_price"] >= price_min]
    if sub.empty:
        return {}
    lo, hi = cluster_bootstrap_roi(sub, "ret_back", race_col, n_boot=n_boot)
    path = bankroll_path(sub, fraction=fraction, race_col=race_col)
    flat = sub["ret_back"].values
    return {
        "label": f"model #{rank}" + (f", field >= {field_min}" if field_min else "") +
                 (f", forecast price >= {price_min:g}" if price_min else ""),
        "n": int(len(sub)), "win_rate_pct": float(100 * sub["won"].mean()),
        "avg_bsp": float(sub["bsp"].mean()),
        "roi_flat_pct": float(100 * flat.mean()),
        "roi_ci_lo_pct": float(100 * lo), "roi_ci_hi_pct": float(100 * hi),
        "t_stat": float(flat.mean() / (flat.std(ddof=1) / np.sqrt(len(flat)))) if len(flat) > 1 else np.nan,
        "kelly_log10_terminal_bank": growth_stats(path)["log10_terminal_bank"] if len(path) else np.nan,
        "kelly_max_drawdown_pct": growth_stats(path)["max_drawdown_pct"] if len(path) else np.nan,
        "stability": time_split_table(sub),
    }


# ---------------------------------------------------------------------------
# How wrong are the probabilities, in staking units?
# ---------------------------------------------------------------------------

def shrinkage_scan(d: pd.DataFrame, lambdas=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
                   fraction: float = 0.25, race_col: str = "raceid",
                   commission: float = 0.05) -> pd.DataFrame:
    """Flatten the probabilities (p ∝ p^λ, race-renormalised) and re-run Kelly.

    λ=1 is the model as trained, λ=0 gives every runner 1/N. If growth only
    turns positive somewhere in between, the model's ranking is worth
    something but its confidence is not."""
    rows = []
    for lam in lambdas:
        t = d.copy()
        e = np.power(t["p_model"].clip(1e-9), lam)
        t["p_lam"] = e / t.groupby(race_col)["p_model"].transform(lambda s: np.power(s.clip(1e-9), lam).sum())
        t = add_kelly(t, p_col="p_lam", commission=commission)
        bets = t[t["kelly_full"] > 0]
        path = bankroll_path(t, fraction=fraction, race_col=race_col)
        s = growth_stats(path)
        rows.append({"lambda": lam, "bets": int(len(bets)),
                     "mean_stake_pct": 100 * bets["kelly_full"].mean() * fraction if len(bets) else 0.0,
                     "roi_pct": 100 * bets["ret_back"].mean() if len(bets) else np.nan,
                     "log10_terminal_bank": s["log10_terminal_bank"],
                     "log_growth_per_race": s["log_growth_per_race"],
                     "max_drawdown_pct": s["max_drawdown_pct"]})
    return pd.DataFrame(rows)


def price_forecast_by_rank(d: pd.DataFrame, pred_col: str = "predicted_bfsp",
                           max_rank: int = 6) -> pd.DataFrame:
    """Predicted BFSP against realised BFSP, by model rank.

    This is the quantity the strategy actually depends on: if the model says
    a horse will close at 4.0 and it closes at 5.0, money taken at the
    forecast price is money lost. ``clv_at_forecast_pct`` is the greened-up
    return from taking the forecast price and closing at BSP."""
    if pred_col not in d.columns:
        return pd.DataFrame()
    rows = []
    for r in range(1, max_rank + 1):
        g = d[d["model_rank"] == r]
        if g.empty:
            continue
        ratio = g[pred_col] / g["bsp"]
        rows.append({"model_rank": r, "n": len(g),
                     "median_pred_bsp": g[pred_col].median(), "median_bsp": g["bsp"].median(),
                     "median_pred_over_actual": ratio.median(),
                     "pct_forecast_above_bsp": 100 * (ratio > 1).mean(),
                     "clv_at_forecast_pct": 100 * (ratio - 1).median(),
                     "mean_abs_log_error": float(np.abs(np.log(ratio.clip(1e-6))).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def staking_report(df: pd.DataFrame, commission: float = 0.05, max_rank: int = 6,
                   fractions=(1.0, 0.5, 0.25, 0.125, 0.05), **kw) -> dict:
    d = prepare_bets(df, commission=commission, blend_lambda=None, **kw)
    d = add_kelly(d, commission=commission)
    n_races = int(d["raceid"].nunique())
    out = {
        "n_runners": len(d), "n_races": n_races, "commission": commission,
        "kelly_ladder": kelly_ladder(d, fractions, n_races_total=n_races, commission=commission),
        "promise_vs_reality": kelly_promise_vs_reality(d),
        "kelly_by_rank": kelly_by_rank(d),
        "kelly_top_k": kelly_by_rank(d, cumulative=True),
        "rank_staking": rank_staking(d, max_rank=max_rank),
        "topk": topk_staking(d),
        "shrinkage": shrinkage_scan(d, commission=commission),
        "price_forecast": price_forecast_by_rank(d),
        "by_field": rank_segments(d, "field", bands=[0, 5, 8, 11, 15, 100]),
        "by_price": rank_segments(d, "model_price", bands=[1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 1e9]),
        "focus": [rank_focus(d, 1), rank_focus(d, 1, field_min=12), rank_focus(d, 1, field_min=16),
                  rank_focus(d, 1, price_min=8.0), rank_focus(d, 1, field_min=12, price_min=6.0),
                  rank_focus(d, 4)],
    }
    out["_frame"] = d
    return out
