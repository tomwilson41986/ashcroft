#!/usr/bin/env python3
"""
Profitability Analysis: Overlay Detection & Staking Simulation.

Uses walk-forward out-of-sample predictions to identify overlays
(where actual BFSP > predicted BFSP, i.e. the market is offering
better odds than our model's fair price) and simulates profitability
across multiple staking strategies.

Staking Strategies:
    1. Level Stakes      — Fixed £1 per bet
    2. Variant Staking   — Stake proportional to edge (overlay %)
    3. Kelly Criterion   — Optimal growth stake
    4. Half Kelly        — 50% of Kelly (reduced variance)
    5. Quarter Kelly     — 25% of Kelly (conservative)

Usage:
    python profitability_analysis.py
    python profitability_analysis.py --start-date 2024-01-01
    python profitability_analysis.py --min-overlay 10
"""

import argparse
import json
import logging
import os
import sqlite3
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.custom_metrics import CustomMetricsEngine
from train_bfsp import build_context_features, ALL_FEATURE_COLS, load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "data", "models", "bfsp_model.lgb")
META_PATH = os.path.join(SCRIPT_DIR, "data", "models", "bfsp_model_meta.json")
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Run custom metrics and context feature engineering."""
    engine = CustomMetricsEngine()
    log.info("Calculating custom metrics...")
    df = engine.calculate_all(df)
    log.info(f"  {len(df.columns)} columns after metrics")

    log.info("Building context features...")
    df = build_context_features(df)

    # Target
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df["log_bfsp"] = np.log(df["bfsp"])

    # Filter to available features
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        log.warning(f"  Missing {len(missing)} features, filling with NaN")
        for c in missing:
            df[c] = np.nan

    return df


def simulate_staking(bets_df: pd.DataFrame) -> dict:
    """Simulate all 5 staking strategies on the bets dataframe.

    Expected columns:
        - predicted_bfsp: model's fair price
        - bfsp: actual BFSP (the odds we'd get)
        - won: bool, did the horse win?
        - overlay_pct: (bfsp - predicted_bfsp) / predicted_bfsp * 100
    """
    results = {}
    n_bets = len(bets_df)

    if n_bets == 0:
        return {"error": "no_bets"}

    # --- 1. Level Stakes (£1 per bet) ---
    stakes = np.ones(n_bets)
    returns = np.where(bets_df["won"], bets_df["bfsp"] * stakes, 0)
    results["Level Stakes"] = _staking_summary(stakes, returns, bets_df)

    # --- 2. Variant Staking (stake proportional to edge) ---
    # Stake = edge% / 10, min £0.50, max £5.00, base unit £1
    edge = bets_df["overlay_pct"].values / 100.0
    stakes = np.clip(edge * 10, 0.5, 5.0)
    returns = np.where(bets_df["won"], bets_df["bfsp"] * stakes, 0)
    results["Variant Staking"] = _staking_summary(stakes, returns, bets_df)

    # --- 3. Kelly Criterion ---
    # Kelly fraction = (p * b - 1) / (b - 1) where:
    #   p = implied probability from predicted BFSP = 1/predicted_bfsp
    #   b = decimal odds - 1 (net odds) = bfsp - 1
    p = 1.0 / bets_df["predicted_bfsp"].values
    b = bets_df["bfsp"].values - 1.0
    kelly_f = np.where(b > 0, (p * b - (1 - p)) / b, 0)
    kelly_f = np.clip(kelly_f, 0, 0.25)  # Cap at 25% of bankroll

    # Simulate with £1000 starting bankroll
    for name, fraction in [("Kelly", 1.0), ("Half Kelly", 0.5), ("Quarter Kelly", 0.25)]:
        bankroll = 1000.0
        bankroll_history = [bankroll]
        total_staked = 0
        total_returned = 0

        for i in range(n_bets):
            f = kelly_f[i] * fraction
            if f <= 0 or bankroll <= 0:
                bankroll_history.append(bankroll)
                continue

            stake = bankroll * f
            total_staked += stake

            if bets_df.iloc[i]["won"]:
                profit = stake * (bets_df.iloc[i]["bfsp"] - 1)
                bankroll += profit
                total_returned += stake + profit
            else:
                bankroll -= stake

            bankroll_history.append(bankroll)

        results[name] = {
            "n_bets": n_bets,
            "total_staked": round(total_staked, 2),
            "total_returned": round(total_returned, 2),
            "profit": round(total_returned - total_staked, 2),
            "roi_pct": round((total_returned - total_staked) / max(total_staked, 1) * 100, 2),
            "final_bankroll": round(bankroll, 2),
            "bankroll_growth_pct": round((bankroll - 1000) / 1000 * 100, 2),
            "peak_bankroll": round(max(bankroll_history), 2),
            "min_bankroll": round(min(bankroll_history), 2),
            "win_rate_pct": round(bets_df["won"].sum() / n_bets * 100, 2),
        }

    return results


def _staking_summary(stakes, returns, bets_df):
    """Summarise a flat staking strategy."""
    total_staked = stakes.sum()
    total_returned = returns.sum()
    profit = total_returned - total_staked
    n_bets = len(stakes)
    return {
        "n_bets": n_bets,
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "profit": round(profit, 2),
        "roi_pct": round(profit / max(total_staked, 1) * 100, 2),
        "win_rate_pct": round(bets_df["won"].sum() / n_bets * 100, 2),
        "avg_stake": round(total_staked / n_bets, 2),
        "avg_winner_bfsp": round(
            bets_df.loc[bets_df["won"], "bfsp"].mean()
            if bets_df["won"].any() else 0, 2
        ),
    }


def analyse_by_price_band(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Break down profitability by predicted price band."""
    bands = [
        (1.0, 3.0, "1.0-3.0 (Short)"),
        (3.0, 6.0, "3.0-6.0 (Medium)"),
        (6.0, 11.0, "6.0-11.0 (Mid-range)"),
        (11.0, 21.0, "11.0-21.0 (Long)"),
        (21.0, 51.0, "21.0-51.0 (Big)"),
        (51.0, 1000.0, "51.0+ (Outsiders)"),
    ]

    rows = []
    for lo, hi, label in bands:
        mask = (bets_df["predicted_bfsp"] >= lo) & (bets_df["predicted_bfsp"] < hi)
        band = bets_df[mask]
        if len(band) == 0:
            continue

        n = len(band)
        winners = band["won"].sum()
        # Level stakes P&L
        staked = n  # £1 each
        returned = band.loc[band["won"], "bfsp"].sum()
        profit = returned - staked
        roi = profit / staked * 100

        rows.append({
            "Price Band": label,
            "Bets": n,
            "Winners": int(winners),
            "Win%": round(winners / n * 100, 1),
            "Level P&L": round(profit, 2),
            "Level ROI%": round(roi, 1),
            "Avg Overlay%": round(band["overlay_pct"].mean(), 1),
        })

    return pd.DataFrame(rows)


def analyse_by_overlay_band(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Break down profitability by overlay percentage."""
    bands = [
        (5, 10, "5-10%"),
        (10, 15, "10-15%"),
        (15, 20, "15-20%"),
        (20, 30, "20-30%"),
        (30, 50, "30-50%"),
        (50, 100, "50-100%"),
        (100, 9999, "100%+"),
    ]

    rows = []
    for lo, hi, label in bands:
        mask = (bets_df["overlay_pct"] >= lo) & (bets_df["overlay_pct"] < hi)
        band = bets_df[mask]
        if len(band) == 0:
            continue

        n = len(band)
        winners = band["won"].sum()
        staked = n
        returned = band.loc[band["won"], "bfsp"].sum()
        profit = returned - staked
        roi = profit / staked * 100

        rows.append({
            "Overlay Band": label,
            "Bets": n,
            "Winners": int(winners),
            "Win%": round(winners / n * 100, 1),
            "Level P&L": round(profit, 2),
            "Level ROI%": round(roi, 1),
            "Avg BFSP": round(band["bfsp"].mean(), 1),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="BFSP Overlay Profitability Analysis")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Database path")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Model path")
    parser.add_argument("--meta", type=str, default=META_PATH, help="Model meta path")
    parser.add_argument(
        "--start-date", type=str, default="2025-01-01",
        help="Analyse data from this date (default: 2025-01-01)",
    )
    parser.add_argument(
        "--min-overlay", type=float, default=5.0,
        help="Minimum overlay %% to trigger a bet (default: 5)",
    )
    parser.add_argument(
        "--max-bfsp", type=float, default=100.0,
        help="Maximum BFSP to consider (default: 100)",
    )
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model):
        log.error(f"Model not found: {args.model}")
        sys.exit(1)

    model = lgb.Booster(model_file=args.model)

    with open(args.meta) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    log.info(f"Loaded model with {len(feature_cols)} features")

    # Load data — we need historical data for custom metrics context,
    # but only analyse predictions from start_date onwards.
    # Load from 2 years before start_date for metric warmup.
    warmup_date = str(int(args.start_date[:4]) - 2) + args.start_date[4:]
    log.info(f"Loading data from {warmup_date} (warmup) for analysis from {args.start_date}...")
    df = load_data(args.db, start_date=warmup_date)
    log.info(f"  {len(df):,} rows loaded")

    # Prepare features
    df = prepare_features(df, feature_cols)
    log.info(f"  {len(df):,} rows with valid BFSP")

    # Generate predictions
    log.info("Generating predictions...")
    X = df[feature_cols].astype(float)
    df["predicted_log_bfsp"] = model.predict(X)
    df["predicted_bfsp_raw"] = np.exp(df["predicted_log_bfsp"])

    # Normalise implied probabilities per race so they sum to 1
    df["raceid"] = (
        df["race_date"].dt.strftime("%Y-%m-%d")
        + "_" + df["track"].astype(str)
        + "_" + df["race_time"].astype(str)
    )
    df["_ip"] = 1.0 / df["predicted_bfsp_raw"]
    _rsum = df.groupby("raceid")["_ip"].transform("sum")
    df["predicted_win_prob_norm"] = df["_ip"] / _rsum
    df["predicted_bfsp"] = 1.0 / df["predicted_win_prob_norm"]
    df.drop(columns=["_ip"], inplace=True)

    # Filter to analysis period only
    analysis_start = pd.Timestamp(args.start_date)
    df = df[df["race_date"] >= analysis_start].copy()
    log.info(f"  {len(df):,} rows in analysis period ({args.start_date} onwards)")

    # Determine winners
    df["placing_numerical"] = pd.to_numeric(df["placing_numerical"], errors="coerce")
    df["won"] = df["placing_numerical"] == 1

    # Calculate overlay
    df["overlay_pct"] = (df["bfsp"] - df["predicted_bfsp"]) / df["predicted_bfsp"] * 100

    # Filter to bettable selections
    df = df[df["bfsp"] <= args.max_bfsp].copy()
    log.info(f"  {len(df):,} rows with BFSP <= {args.max_bfsp}")

    # =====================================================================
    # FULL MARKET ANALYSIS (all runners, not just overlays)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  BFSP MODEL PROFITABILITY ANALYSIS")
    print(f"  Period: {args.start_date} to {df['race_date'].max().date()}")
    print(f"  Total runners analysed: {len(df):,}")
    print(f"  Total races: {df['race_date'].astype(str).str.cat(df['race_time'].astype(str)).nunique():,}")
    print("=" * 70)

    # Model accuracy overview
    corr = np.corrcoef(df["log_bfsp"].values, df["predicted_log_bfsp"].values)[0, 1]
    median_ape = np.median(np.abs(df["bfsp"] - df["predicted_bfsp"]) / df["bfsp"]) * 100
    print(f"\n  Model Accuracy (out-of-sample):")
    print(f"    Correlation:    {corr:.4f}")
    print(f"    Median APE:     {median_ape:.1f}%")

    # =====================================================================
    # OVERLAY SELECTION
    # =====================================================================
    overlays = df[df["overlay_pct"] >= args.min_overlay].copy()
    log.info(f"\n  Overlays (>={args.min_overlay}%): {len(overlays):,} bets "
             f"({len(overlays)/len(df)*100:.1f}% of runners)")

    if len(overlays) == 0:
        print("\nNo overlays found with current filters.")
        sys.exit(0)

    print(f"\n  Overlay Selection (min {args.min_overlay}% edge):")
    print(f"    Qualifying bets:  {len(overlays):,}")
    print(f"    Winners:          {overlays['won'].sum():,}")
    print(f"    Win rate:         {overlays['won'].mean()*100:.1f}%")
    print(f"    Avg overlay:      {overlays['overlay_pct'].mean():.1f}%")
    print(f"    Avg pred BFSP:    {overlays['predicted_bfsp'].mean():.1f}")
    print(f"    Avg actual BFSP:  {overlays['bfsp'].mean():.1f}")

    # =====================================================================
    # STAKING STRATEGIES
    # =====================================================================
    print("\n" + "-" * 70)
    print("  STAKING STRATEGY RESULTS")
    print("-" * 70)

    staking_results = simulate_staking(overlays)

    # Level Stakes & Variant
    for name in ["Level Stakes", "Variant Staking"]:
        r = staking_results[name]
        print(f"\n  {name}:")
        print(f"    Bets:        {r['n_bets']:,}")
        print(f"    Staked:      £{r['total_staked']:,.2f}")
        print(f"    Returned:    £{r['total_returned']:,.2f}")
        print(f"    Profit:      £{r['profit']:,.2f}")
        print(f"    ROI:         {r['roi_pct']:+.2f}%")
        print(f"    Win Rate:    {r['win_rate_pct']:.1f}%")

    # Kelly variants (bankroll-based)
    for name in ["Kelly", "Half Kelly", "Quarter Kelly"]:
        r = staking_results[name]
        print(f"\n  {name} (£1,000 starting bankroll):")
        print(f"    Bets:        {r['n_bets']:,}")
        print(f"    Staked:      £{r['total_staked']:,.2f}")
        print(f"    Returned:    £{r['total_returned']:,.2f}")
        print(f"    Profit:      £{r['profit']:,.2f}")
        print(f"    ROI:         {r['roi_pct']:+.2f}%")
        print(f"    Final Bank:  £{r['final_bankroll']:,.2f}")
        print(f"    Growth:      {r['bankroll_growth_pct']:+.1f}%")
        print(f"    Peak Bank:   £{r['peak_bankroll']:,.2f}")
        print(f"    Low Point:   £{r['min_bankroll']:,.2f}")

    # =====================================================================
    # BREAKDOWN BY PRICE BAND
    # =====================================================================
    print("\n" + "-" * 70)
    print("  PROFITABILITY BY PREDICTED PRICE BAND (Level Stakes)")
    print("-" * 70)

    price_df = analyse_by_price_band(overlays)
    if not price_df.empty:
        print(price_df.to_string(index=False))

    # =====================================================================
    # BREAKDOWN BY OVERLAY BAND
    # =====================================================================
    print("\n" + "-" * 70)
    print("  PROFITABILITY BY OVERLAY BAND (Level Stakes)")
    print("-" * 70)

    overlay_df = analyse_by_overlay_band(overlays)
    if not overlay_df.empty:
        print(overlay_df.to_string(index=False))

    # =====================================================================
    # MONTHLY BREAKDOWN
    # =====================================================================
    print("\n" + "-" * 70)
    print("  MONTHLY LEVEL STAKES P&L")
    print("-" * 70)

    overlays["month"] = overlays["race_date"].dt.to_period("M")
    monthly = overlays.groupby("month").apply(
        lambda g: pd.Series({
            "Bets": len(g),
            "Winners": int(g["won"].sum()),
            "Win%": round(g["won"].mean() * 100, 1),
            "Staked": len(g),
            "Returned": round(g.loc[g["won"], "bfsp"].sum(), 2),
            "P&L": round(g.loc[g["won"], "bfsp"].sum() - len(g), 2),
            "ROI%": round((g.loc[g["won"], "bfsp"].sum() - len(g)) / len(g) * 100, 1),
        }),
        include_groups=False,
    ).reset_index()
    monthly["Month"] = monthly["month"].astype(str)
    monthly["Cumulative P&L"] = monthly["P&L"].cumsum().round(2)
    print(monthly[["Month", "Bets", "Winners", "Win%", "P&L", "ROI%", "Cumulative P&L"]].to_string(index=False))

    # Summary
    total_pl = monthly["P&L"].sum()
    total_bets = monthly["Bets"].sum()
    print(f"\n  TOTAL: {total_bets:,} bets, £{total_pl:+,.2f} P&L, "
          f"{total_pl/total_bets*100:+.1f}% ROI")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
