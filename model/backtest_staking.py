"""
Staking Strategy Profitability Backtest.

Tests the BFSP prediction model's profitability under 5 staking strategies:
  1. Level stakes — fixed £10 per bet
  2. Variant staking — stake proportional to perceived edge
  3. Full Kelly — full Kelly criterion
  4. Half Kelly — 50% of full Kelly
  5. Quarter Kelly — 25% of full Kelly

Uses walk-forward validation predictions so there is no lookahead bias.
Accounts for 5% Betfair commission on net winnings.

Usage:
    python -m model.backtest_staking
"""

import json
import logging
import os
import sqlite3

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.train_bfsp import (
    FEATURE_COLS,
    build_features,
    create_walk_forward_folds,
    load_data,
    train_bfsp_model,
    train_win_model,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB = os.path.join(PROJECT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

COMMISSION = 0.05
MIN_EDGE = 0.05
LAMBDA = 0.80


# ──────────────────────────────────────────────────────────────────────
# Staking strategies
# ──────────────────────────────────────────────────────────────────────

def _kelly_fraction(p: pd.Series, bfsp: pd.Series) -> pd.Series:
    """Full Kelly fraction: f* = (p*b - q) / b."""
    b = bfsp - 1
    q = 1 - p
    f = ((p * b) - q) / b
    return f.clip(lower=0)


def level_stakes(bets: pd.DataFrame, bankroll: float) -> pd.Series:
    """Fixed £10 per bet."""
    return pd.Series(10.0, index=bets.index)


def variant_stakes(bets: pd.DataFrame, bankroll: float) -> pd.Series:
    """Stake proportional to edge: base £5 + £100 * edge (capped at 2% bankroll)."""
    base = 5.0
    stake = base + 100.0 * bets["edge"].clip(lower=0)
    return stake.clip(upper=bankroll * 0.02)


def kelly_full(bets: pd.DataFrame, bankroll: float) -> pd.Series:
    """Full Kelly criterion."""
    f = _kelly_fraction(bets["p_combined"], bets["bfsp"])
    stake = f * bankroll
    return stake.clip(lower=2.0, upper=bankroll * 0.05)


def kelly_half(bets: pd.DataFrame, bankroll: float) -> pd.Series:
    """Half Kelly (50% of full)."""
    f = _kelly_fraction(bets["p_combined"], bets["bfsp"])
    stake = f * 0.5 * bankroll
    return stake.clip(lower=2.0, upper=bankroll * 0.03)


def kelly_quarter(bets: pd.DataFrame, bankroll: float) -> pd.Series:
    """Quarter Kelly (25% of full)."""
    f = _kelly_fraction(bets["p_combined"], bets["bfsp"])
    stake = f * 0.25 * bankroll
    return stake.clip(lower=2.0, upper=bankroll * 0.02)


STRATEGIES = {
    "Level Stakes": level_stakes,
    "Variant Staking": variant_stakes,
    "Full Kelly": kelly_full,
    "Half Kelly": kelly_half,
    "Quarter Kelly": kelly_quarter,
}


# ──────────────────────────────────────────────────────────────────────
# Blending and overlay detection
# ──────────────────────────────────────────────────────────────────────

def blend_and_find_overlays(df: pd.DataFrame) -> pd.DataFrame:
    """Benter blend + overlay detection on a DataFrame with p_model and bfsp."""
    df = df.copy()
    df["p_public"] = 1.0 / df["bfsp"].replace(0, np.nan)

    p_m = df["p_model"].clip(lower=1e-10)
    p_p = df["p_public"].clip(lower=1e-10)
    raw = np.power(p_m, 1 - LAMBDA) * np.power(p_p, LAMBDA)
    df["_raw"] = raw
    df["p_combined"] = df.groupby("raceid")["_raw"].transform(
        lambda x: x / x.sum()
    )
    df.drop(columns=["_raw"], inplace=True)

    df["edge"] = (df["p_combined"] / df["p_public"]) - 1
    return df


# ──────────────────────────────────────────────────────────────────────
# Simulate a single strategy
# ──────────────────────────────────────────────────────────────────────

def simulate_strategy(
    bets: pd.DataFrame,
    stake_fn,
    bankroll_start: float = 1000.0,
    dynamic_bankroll: bool = True,
) -> dict:
    """Run a staking strategy through the bet history.

    Args:
        bets: DataFrame sorted by race_date with columns:
              race_date, bfsp, won, p_combined, edge.
        stake_fn: Function(bets_df, bankroll) -> Series of stake amounts.
        bankroll_start: Starting bankroll.
        dynamic_bankroll: If True, bankroll updates after each day
                          (relevant for Kelly strategies).

    Returns:
        Dict of summary metrics plus a daily P&L DataFrame.
    """
    bets = bets.copy().sort_values("race_date").reset_index(drop=True)
    dates = bets["race_date"].unique()

    bankroll = bankroll_start
    daily_records = []

    for date in sorted(dates):
        day_bets = bets[bets["race_date"] == date].copy()
        if len(day_bets) == 0:
            continue

        # Calculate stakes
        stakes = stake_fn(day_bets, bankroll)
        # Don't bet more than we have
        stakes = stakes.clip(upper=bankroll * 0.05 if dynamic_bankroll else 500)
        # Skip if bankroll depleted
        if bankroll < 2.0:
            continue

        day_bets["stake"] = stakes

        # Remove bets below minimum
        day_bets = day_bets[day_bets["stake"] >= 2.0].copy()
        if len(day_bets) == 0:
            continue

        # Settle: winners get stake * bfsp less commission on profit
        day_bets["returns"] = np.where(
            day_bets["won"] == 1,
            day_bets["stake"] * day_bets["bfsp"] * (1 - COMMISSION),
            0.0,
        )
        day_bets["pnl"] = day_bets["returns"] - day_bets["stake"]

        day_staked = day_bets["stake"].sum()
        day_returns = day_bets["returns"].sum()
        day_pnl = day_bets["pnl"].sum()
        day_winners = int(day_bets["won"].sum())

        if dynamic_bankroll:
            bankroll += day_pnl

        daily_records.append({
            "date": date,
            "n_bets": len(day_bets),
            "n_winners": day_winners,
            "staked": round(day_staked, 2),
            "returns": round(day_returns, 2),
            "pnl": round(day_pnl, 2),
            "bankroll": round(bankroll, 2),
        })

    if not daily_records:
        return {"error": "No bets placed"}

    daily = pd.DataFrame(daily_records)
    daily["cum_pnl"] = daily["pnl"].cumsum()
    daily["cum_staked"] = daily["staked"].cumsum()
    daily["roi_pct"] = daily["cum_pnl"] / daily["cum_staked"] * 100

    # Summary metrics
    total_bets = daily["n_bets"].sum()
    total_winners = daily["n_winners"].sum()
    total_staked = daily["staked"].sum()
    total_pnl = daily["cum_pnl"].iloc[-1]

    # Max drawdown
    running_max = daily["cum_pnl"].cummax()
    drawdown = daily["cum_pnl"] - running_max
    max_dd = drawdown.min()

    # Peak profit
    peak = daily["cum_pnl"].max()

    # Sharpe ratio (annualised, daily returns)
    if len(daily) > 1 and daily["pnl"].std() > 0:
        sharpe = (daily["pnl"].mean() / daily["pnl"].std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Longest losing streak (consecutive losing days)
    losing = (daily["pnl"] < 0).astype(int)
    streak = 0
    max_streak = 0
    for v in losing:
        if v:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Win days vs lose days
    win_days = (daily["pnl"] > 0).sum()
    lose_days = (daily["pnl"] < 0).sum()

    # Profit factor
    gross_wins = daily.loc[daily["pnl"] > 0, "pnl"].sum()
    gross_losses = abs(daily.loc[daily["pnl"] < 0, "pnl"].sum())
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else float("inf")

    summary = {
        "total_bets": int(total_bets),
        "total_winners": int(total_winners),
        "strike_rate_pct": round(total_winners / total_bets * 100, 2) if total_bets > 0 else 0,
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(total_pnl / total_staked * 100, 2) if total_staked > 0 else 0,
        "final_bankroll": round(daily["bankroll"].iloc[-1], 2),
        "peak_profit": round(peak, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 4),
        "profit_factor": profit_factor,
        "win_days": int(win_days),
        "lose_days": int(lose_days),
        "longest_losing_streak_days": max_streak,
        "avg_daily_pnl": round(daily["pnl"].mean(), 2),
        "avg_stake_per_bet": round(total_staked / total_bets, 2) if total_bets > 0 else 0,
        "n_days": len(daily),
    }

    return {"summary": summary, "daily": daily}


# ──────────────────────────────────────────────────────────────────────
# Main backtest pipeline
# ──────────────────────────────────────────────────────────────────────

def run_backtest(db_path: str = DEFAULT_DB, bankroll: float = 1000.0) -> dict:
    """Run full walk-forward backtest with all staking strategies."""

    log.info("=" * 70)
    log.info("STAKING STRATEGY PROFITABILITY BACKTEST")
    log.info("=" * 70)

    # Load and build features
    log.info("Loading data and building features...")
    df = load_data(db_path)
    df = build_features(df)
    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df[df["log_bfsp"].notna()].copy()

    # Walk-forward: collect out-of-sample predictions
    log.info("Running walk-forward validation...")
    folds = create_walk_forward_folds(df, min_train_days=365, val_window_days=30, step_days=30)
    log.info(f"  {len(folds)} folds")

    all_preds = []
    for i, (train_end, val_start, val_end) in enumerate(folds):
        train = df[df["race_date"] < train_end]
        val = df[(df["race_date"] >= val_start) & (df["race_date"] < val_end)]

        if len(train) < 100 or len(val) < 10:
            continue

        # Train win model for this fold
        _, _ = train_bfsp_model(train, val, available)
        win_model, _ = train_win_model(train, val, available)

        X_val = val[available].astype(float).replace([np.inf, -np.inf], np.nan)
        val = val.copy()
        val["p_model"] = win_model.predict(X_val)
        all_preds.append(val[["raceid", "race_date", "horse_name", "bfsp", "won",
                              "p_model", "number_of_runners"]].copy())

        if (i + 1) % 5 == 0:
            log.info(f"  Completed fold {i+1}/{len(folds)}")

    log.info(f"  Completed all {len(folds)} folds")

    if not all_preds:
        log.error("No predictions generated")
        return {}

    preds = pd.concat(all_preds, ignore_index=True)
    log.info(f"  {len(preds):,} out-of-sample predictions across "
             f"{preds['raceid'].nunique():,} races")

    # Blend and find overlays
    log.info("Blending probabilities and detecting overlays...")
    preds = blend_and_find_overlays(preds)
    overlay_bets = preds[preds["edge"] >= MIN_EDGE].copy()
    log.info(f"  {len(overlay_bets):,} overlay bets (edge >= {MIN_EDGE*100:.0f}%)")

    # Test each strategy
    log.info("\n" + "=" * 70)
    log.info("RESULTS BY STAKING STRATEGY")
    log.info("=" * 70)

    all_results = {}
    for name, fn in STRATEGIES.items():
        # Kelly strategies use dynamic bankroll; level/variant use static
        dynamic = name in ("Full Kelly", "Half Kelly", "Quarter Kelly")
        result = simulate_strategy(overlay_bets, fn, bankroll, dynamic_bankroll=dynamic)

        if "error" in result:
            log.warning(f"  {name}: {result['error']}")
            continue

        s = result["summary"]
        all_results[name] = result

        log.info(f"\n  {'─' * 50}")
        log.info(f"  {name}")
        log.info(f"  {'─' * 50}")
        log.info(f"    Bets:             {s['total_bets']:,}")
        log.info(f"    Winners:          {s['total_winners']:,} ({s['strike_rate_pct']:.1f}%)")
        log.info(f"    Total Staked:     £{s['total_staked']:,.2f}")
        log.info(f"    Total P&L:        £{s['total_pnl']:+,.2f}")
        log.info(f"    ROI:              {s['roi_pct']:+.2f}%")
        log.info(f"    Final Bankroll:   £{s['final_bankroll']:,.2f} (from £{bankroll:,.2f})")
        log.info(f"    Peak Profit:      £{s['peak_profit']:+,.2f}")
        log.info(f"    Max Drawdown:     £{s['max_drawdown']:,.2f}")
        log.info(f"    Sharpe Ratio:     {s['sharpe_ratio']:.4f}")
        log.info(f"    Profit Factor:    {s['profit_factor']}")
        log.info(f"    Win/Lose Days:    {s['win_days']}/{s['lose_days']}")
        log.info(f"    Longest Losing:   {s['longest_losing_streak_days']} days")
        log.info(f"    Avg Stake:        £{s['avg_stake_per_bet']:.2f}")
        log.info(f"    Avg Daily P&L:    £{s['avg_daily_pnl']:+.2f}")

    # Comparison table
    log.info("\n\n" + "=" * 70)
    log.info("STRATEGY COMPARISON")
    log.info("=" * 70)
    header = f"  {'Strategy':<20} {'Bets':>6} {'P&L':>12} {'ROI':>8} {'Sharpe':>8} {'MaxDD':>10} {'PF':>6}"
    log.info(header)
    log.info("  " + "─" * 72)
    for name, result in all_results.items():
        s = result["summary"]
        log.info(
            f"  {name:<20} {s['total_bets']:>6,} "
            f"£{s['total_pnl']:>+10,.2f} {s['roi_pct']:>+7.2f}% "
            f"{s['sharpe_ratio']:>7.3f} £{s['max_drawdown']:>9,.2f} "
            f"{s['profit_factor']:>5.2f}"
        )

    # Save results
    output_path = os.path.join(MODEL_DIR, "backtest_results.json")
    save_data = {}
    for name, result in all_results.items():
        save_data[name] = result["summary"]
    save_data["config"] = {
        "bankroll": bankroll,
        "commission": COMMISSION,
        "min_edge": MIN_EDGE,
        "lambda": LAMBDA,
        "n_predictions": len(preds),
        "n_overlay_bets": len(overlay_bets),
        "n_races": int(preds["raceid"].nunique()),
        "date_range": f"{preds['race_date'].min()} to {preds['race_date'].max()}",
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_backtest()
