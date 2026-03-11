#!/usr/bin/env python3
"""
Out-of-Sample BFSP Overlay Backtest.

Runs the BFSP overlay betting strategy on truly out-of-sample predictions
from walk-forward validation. This ensures the model has NEVER seen the
data it is predicting on, giving honest profitability estimates.

Data flow:
1. Load OOS predictions from walk-forward training (bfsp_oos_predictions.csv)
   OR re-run walk-forward if the CSV doesn't exist
2. Compute overlays: (actual_bfsp - predicted_bfsp) / predicted_bfsp * 100
3. Simulate 5 staking strategies on qualifying bets

Staking Strategies:
    1. Level Stakes      — Fixed £1 per bet
    2. Variant Staking   — Stake proportional to edge (overlay %)
    3. Kelly Criterion   — Optimal growth stake (capped at 25%)
    4. Half Kelly        — 50% of Kelly
    5. Quarter Kelly     — 25% of Kelly

Usage:
    python backtest_bfsp_oos.py
    python backtest_bfsp_oos.py --start-date 2024-01-01
    python backtest_bfsp_oos.py --min-overlay 10 --max-bfsp 50
    python backtest_bfsp_oos.py --retrain   # Force re-run walk-forward
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
OOS_PATH = os.path.join(MODEL_DIR, "bfsp_oos_predictions.csv")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_oos_predictions(oos_path: str = OOS_PATH) -> pd.DataFrame:
    """Load saved out-of-sample predictions from walk-forward training."""
    if not os.path.exists(oos_path):
        return pd.DataFrame()

    df = pd.read_csv(oos_path, parse_dates=["race_date"])
    log.info(f"Loaded {len(df):,} OOS predictions from {oos_path}")
    return df


def regenerate_oos_predictions(
    db_path: str = DB_PATH,
    output_dir: str = MODEL_DIR,
) -> pd.DataFrame:
    """Re-run walk-forward training to generate OOS predictions.

    This trains a model on each fold and collects validation predictions.
    The OOS predictions CSV is saved as a side effect.
    """
    from train_bfsp import BFSPTrainer, load_data

    log.info("Re-running walk-forward training to generate OOS predictions...")
    log.info(f"Loading data from {db_path}...")

    df = load_data(db_path)
    log.info(f"  {len(df):,} rows loaded")

    trainer = BFSPTrainer()
    trainer.train(df, output_dir=output_dir)

    # The train() method now saves OOS predictions to CSV
    return load_oos_predictions(os.path.join(output_dir, "bfsp_oos_predictions.csv"))


# ---------------------------------------------------------------------------
# Staking Strategies
# ---------------------------------------------------------------------------

def simulate_level_stakes(bets: pd.DataFrame) -> dict:
    """Level Stakes: £1 per bet."""
    n = len(bets)
    stakes = np.ones(n)
    returns = np.where(bets["won"].values, bets["bfsp"].values, 0)
    pnl = returns - stakes

    return _build_summary("Level Stakes", stakes, returns, pnl, bets)


def simulate_variant_stakes(bets: pd.DataFrame) -> dict:
    """Variant Staking: stake proportional to overlay edge."""
    edge = bets["overlay_pct"].values / 100.0
    stakes = np.clip(edge * 10, 0.5, 5.0)
    returns = np.where(bets["won"].values, bets["bfsp"].values * stakes, 0)
    pnl = returns - stakes

    return _build_summary("Variant Staking", stakes, returns, pnl, bets)


def simulate_kelly(
    bets: pd.DataFrame,
    fraction: float,
    label: str,
    starting_bankroll: float = 1000.0,
) -> dict:
    """Kelly-based staking with a given fraction of full Kelly.

    Kelly formula:
        p = 1 / predicted_bfsp  (model's implied win probability)
        b = bfsp - 1             (net decimal odds on offer)
        f* = (p*b - (1-p)) / b   (full Kelly fraction)
        f  = f* * fraction        (adjusted Kelly)
    """
    n = len(bets)
    p = 1.0 / bets["predicted_bfsp"].values
    b = bets["bfsp"].values - 1.0
    kelly_f = np.where(b > 0, (p * b - (1 - p)) / b, 0)
    kelly_f = np.clip(kelly_f, 0, 0.25)  # Cap at 25%

    bankroll = starting_bankroll
    bankroll_history = [bankroll]
    total_staked = 0.0
    total_returned = 0.0
    stakes_list = []
    pnl_list = []

    for i in range(n):
        f = kelly_f[i] * fraction
        if f <= 0 or bankroll <= 0:
            stakes_list.append(0.0)
            pnl_list.append(0.0)
            bankroll_history.append(bankroll)
            continue

        stake = bankroll * f
        total_staked += stake
        stakes_list.append(stake)

        if bets.iloc[i]["won"]:
            profit = stake * (bets.iloc[i]["bfsp"] - 1)
            bankroll += profit
            total_returned += stake + profit
            pnl_list.append(profit)
        else:
            bankroll -= stake
            pnl_list.append(-stake)

        bankroll_history.append(bankroll)

    # Compute drawdown
    peak = starting_bankroll
    max_dd = 0.0
    for b_val in bankroll_history:
        if b_val > peak:
            peak = b_val
        dd = (peak - b_val) / peak * 100
        if dd > max_dd:
            max_dd = dd

    profit = total_returned - total_staked
    roi = profit / max(total_staked, 1) * 100
    n_winners = int(bets["won"].sum())

    # Sharpe ratio (daily returns)
    pnl_arr = np.array(pnl_list)
    if len(pnl_arr) > 1 and pnl_arr.std() > 0:
        # Approximate: annualise assuming ~3 bets/day, 300 racing days
        sharpe = (pnl_arr.mean() / pnl_arr.std()) * np.sqrt(min(n, 900))
    else:
        sharpe = 0.0

    return {
        "strategy": label,
        "n_bets": n,
        "n_winners": n_winners,
        "win_rate_pct": round(n_winners / max(n, 1) * 100, 2),
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "profit": round(profit, 2),
        "roi_pct": round(roi, 2),
        "starting_bankroll": starting_bankroll,
        "final_bankroll": round(bankroll, 2),
        "bankroll_growth_pct": round((bankroll - starting_bankroll) / starting_bankroll * 100, 2),
        "peak_bankroll": round(max(bankroll_history), 2),
        "min_bankroll": round(min(bankroll_history), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
    }


def _build_summary(
    label: str,
    stakes: np.ndarray,
    returns: np.ndarray,
    pnl: np.ndarray,
    bets: pd.DataFrame,
) -> dict:
    """Build summary dict for flat staking strategies."""
    total_staked = stakes.sum()
    total_returned = returns.sum()
    profit = total_returned - total_staked
    n = len(bets)
    n_winners = int(bets["won"].sum())

    # Cumulative P&L for drawdown
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    drawdowns = peak - cum_pnl
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

    return {
        "strategy": label,
        "n_bets": n,
        "n_winners": n_winners,
        "win_rate_pct": round(n_winners / max(n, 1) * 100, 2),
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "profit": round(profit, 2),
        "roi_pct": round(profit / max(total_staked, 1) * 100, 2),
        "avg_stake": round(total_staked / max(n, 1), 2),
        "max_drawdown": round(max_dd, 2),
        "avg_winner_bfsp": round(
            bets.loc[bets["won"], "bfsp"].mean() if n_winners > 0 else 0, 2
        ),
    }


# ---------------------------------------------------------------------------
# Analysis Breakdowns
# ---------------------------------------------------------------------------

def analyse_by_price_band(bets: pd.DataFrame) -> pd.DataFrame:
    """Level-stakes profitability by predicted price band."""
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
        mask = (bets["predicted_bfsp"] >= lo) & (bets["predicted_bfsp"] < hi)
        band = bets[mask]
        if len(band) == 0:
            continue
        n = len(band)
        winners = band["won"].sum()
        returned = band.loc[band["won"], "bfsp"].sum()
        profit = returned - n
        rows.append({
            "Price Band": label,
            "Bets": n,
            "Winners": int(winners),
            "Win%": round(winners / n * 100, 1),
            "Level P&L": round(profit, 2),
            "Level ROI%": round(profit / n * 100, 1),
            "Avg Overlay%": round(band["overlay_pct"].mean(), 1),
        })
    return pd.DataFrame(rows)


def analyse_by_overlay_band(bets: pd.DataFrame) -> pd.DataFrame:
    """Level-stakes profitability by overlay percentage."""
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
        mask = (bets["overlay_pct"] >= lo) & (bets["overlay_pct"] < hi)
        band = bets[mask]
        if len(band) == 0:
            continue
        n = len(band)
        winners = band["won"].sum()
        returned = band.loc[band["won"], "bfsp"].sum()
        profit = returned - n
        rows.append({
            "Overlay Band": label,
            "Bets": n,
            "Winners": int(winners),
            "Win%": round(winners / n * 100, 1),
            "Level P&L": round(profit, 2),
            "Level ROI%": round(profit / n * 100, 1),
            "Avg BFSP": round(band["bfsp"].mean(), 1),
        })
    return pd.DataFrame(rows)


def monthly_breakdown(bets: pd.DataFrame) -> pd.DataFrame:
    """Monthly level-stakes P&L."""
    bets = bets.copy()
    bets["month"] = bets["race_date"].dt.to_period("M")
    monthly = bets.groupby("month").apply(
        lambda g: pd.Series({
            "Bets": len(g),
            "Winners": int(g["won"].sum()),
            "Win%": round(g["won"].mean() * 100, 1),
            "P&L": round(g.loc[g["won"], "bfsp"].sum() - len(g), 2),
            "ROI%": round(
                (g.loc[g["won"], "bfsp"].sum() - len(g)) / len(g) * 100, 1
            ),
        }),
        include_groups=False,
    ).reset_index()
    monthly["Month"] = monthly["month"].astype(str)
    monthly["Cumulative P&L"] = monthly["P&L"].cumsum().round(2)
    return monthly[["Month", "Bets", "Winners", "Win%", "P&L", "ROI%", "Cumulative P&L"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Out-of-Sample BFSP Overlay Backtest"
    )
    parser.add_argument(
        "--oos-path", type=str, default=OOS_PATH,
        help=f"Path to OOS predictions CSV (default: {OOS_PATH})",
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help=f"Database path for retraining (default: {DB_PATH})",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force re-run walk-forward training to regenerate OOS predictions",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Analyse from this date onwards (default: all OOS data)",
    )
    parser.add_argument(
        "--min-overlay", type=float, default=5.0,
        help="Minimum overlay %% to trigger a bet (default: 5)",
    )
    parser.add_argument(
        "--max-bfsp", type=float, default=100.0,
        help="Maximum BFSP to consider (default: 100)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Starting bankroll for Kelly strategies (default: 1000)",
    )
    args = parser.parse_args()

    # --- Load or regenerate OOS predictions ---
    if args.retrain or not os.path.exists(args.oos_path):
        if not args.retrain:
            log.warning(
                f"OOS predictions not found at {args.oos_path}. "
                "Re-running walk-forward training..."
            )
        df = regenerate_oos_predictions(args.db, MODEL_DIR)
    else:
        df = load_oos_predictions(args.oos_path)

    if len(df) == 0:
        log.error("No OOS predictions available. Run 'python train_bfsp.py' first.")
        sys.exit(1)

    # --- Prepare data ---
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df["predicted_bfsp"] = pd.to_numeric(df["predicted_bfsp"], errors="coerce")
    df["placing_numerical"] = pd.to_numeric(
        df["placing_numerical"], errors="coerce"
    )

    # Drop rows without valid BFSP or predictions
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df = df[df["predicted_bfsp"].notna() & (df["predicted_bfsp"] > 1.0)].copy()

    # Determine winners
    df["won"] = df["placing_numerical"] == 1

    # Compute overlay
    df["overlay_pct"] = (
        (df["bfsp"] - df["predicted_bfsp"]) / df["predicted_bfsp"] * 100
    )

    # Apply filters
    if args.start_date:
        df = df[df["race_date"] >= pd.Timestamp(args.start_date)].copy()

    df = df[df["bfsp"] <= args.max_bfsp].copy()

    # ===================================================================
    # FULL MARKET OVERVIEW
    # ===================================================================
    date_range = f"{df['race_date'].min().date()} to {df['race_date'].max().date()}"
    n_total = len(df)

    print("\n" + "=" * 70)
    print("  OUT-OF-SAMPLE BFSP OVERLAY BACKTEST")
    print(f"  Data: Walk-forward validation predictions (NEVER seen by model)")
    print(f"  Period: {date_range}")
    print(f"  Total runners: {n_total:,}")
    print("=" * 70)

    # Model accuracy on OOS data
    if "log_bfsp" in df.columns and "predicted_log_bfsp" in df.columns:
        corr = np.corrcoef(
            df["log_bfsp"].values, df["predicted_log_bfsp"].values
        )[0, 1]
    else:
        corr = np.corrcoef(
            np.log(df["bfsp"].values), np.log(df["predicted_bfsp"].values)
        )[0, 1]
    median_ape = np.median(
        np.abs(df["bfsp"] - df["predicted_bfsp"]) / df["bfsp"]
    ) * 100

    print(f"\n  OOS Model Accuracy:")
    print(f"    Log correlation:  {corr:.4f}")
    print(f"    Median APE:       {median_ape:.1f}%")

    # ===================================================================
    # OVERLAY SELECTION
    # ===================================================================
    bets = df[df["overlay_pct"] >= args.min_overlay].copy()
    bets = bets.sort_values(["race_date", "race_time"]).reset_index(drop=True)

    print(f"\n  Overlay Selection (min {args.min_overlay}% edge, max BFSP {args.max_bfsp}):")
    print(f"    Qualifying bets:  {len(bets):,} ({len(bets)/n_total*100:.1f}% of runners)")
    print(f"    Winners:          {bets['won'].sum():,}")
    print(f"    Win rate:         {bets['won'].mean()*100:.1f}%")
    print(f"    Avg overlay:      {bets['overlay_pct'].mean():.1f}%")
    print(f"    Median overlay:   {bets['overlay_pct'].median():.1f}%")
    print(f"    Avg pred BFSP:    {bets['predicted_bfsp'].mean():.1f}")
    print(f"    Avg actual BFSP:  {bets['bfsp'].mean():.1f}")

    if len(bets) == 0:
        print("\nNo qualifying bets found. Try lowering --min-overlay.")
        sys.exit(0)

    # ===================================================================
    # STAKING STRATEGY RESULTS
    # ===================================================================
    print("\n" + "-" * 70)
    print("  STAKING STRATEGY RESULTS (OUT-OF-SAMPLE)")
    print("-" * 70)

    # Level Stakes
    level = simulate_level_stakes(bets)
    print(f"\n  {level['strategy']}:")
    print(f"    Bets:           {level['n_bets']:,}")
    print(f"    Winners:        {level['n_winners']:,}")
    print(f"    Win Rate:       {level['win_rate_pct']:.1f}%")
    print(f"    Staked:         £{level['total_staked']:,.2f}")
    print(f"    Returned:       £{level['total_returned']:,.2f}")
    print(f"    Profit:         £{level['profit']:+,.2f}")
    print(f"    ROI:            {level['roi_pct']:+.2f}%")
    print(f"    Max Drawdown:   £{level['max_drawdown']:,.2f}")
    print(f"    Avg Winner BFSP: {level['avg_winner_bfsp']:.1f}")

    # Variant Staking
    variant = simulate_variant_stakes(bets)
    print(f"\n  {variant['strategy']}:")
    print(f"    Bets:           {variant['n_bets']:,}")
    print(f"    Winners:        {variant['n_winners']:,}")
    print(f"    Win Rate:       {variant['win_rate_pct']:.1f}%")
    print(f"    Staked:         £{variant['total_staked']:,.2f}")
    print(f"    Returned:       £{variant['total_returned']:,.2f}")
    print(f"    Profit:         £{variant['profit']:+,.2f}")
    print(f"    ROI:            {variant['roi_pct']:+.2f}%")
    print(f"    Max Drawdown:   £{variant['max_drawdown']:,.2f}")
    print(f"    Avg Stake:      £{variant['avg_stake']:.2f}")

    # Kelly variants
    for frac, label in [(1.0, "Kelly"), (0.5, "Half Kelly"), (0.25, "Quarter Kelly")]:
        k = simulate_kelly(bets, frac, label, args.bankroll)
        print(f"\n  {k['strategy']} (£{args.bankroll:,.0f} starting bankroll):")
        print(f"    Bets:           {k['n_bets']:,}")
        print(f"    Winners:        {k['n_winners']:,}")
        print(f"    Win Rate:       {k['win_rate_pct']:.1f}%")
        print(f"    Staked:         £{k['total_staked']:,.2f}")
        print(f"    Returned:       £{k['total_returned']:,.2f}")
        print(f"    Profit:         £{k['profit']:+,.2f}")
        print(f"    ROI:            {k['roi_pct']:+.2f}%")
        print(f"    Final Bankroll: £{k['final_bankroll']:,.2f}")
        print(f"    Growth:         {k['bankroll_growth_pct']:+.1f}%")
        print(f"    Peak Bankroll:  £{k['peak_bankroll']:,.2f}")
        print(f"    Low Point:      £{k['min_bankroll']:,.2f}")
        print(f"    Max Drawdown:   {k['max_drawdown_pct']:.1f}%")
        print(f"    Sharpe:         {k['sharpe']:.2f}")

    # ===================================================================
    # BREAKDOWNS
    # ===================================================================
    print("\n" + "-" * 70)
    print("  PROFITABILITY BY PREDICTED PRICE BAND (Level Stakes, OOS)")
    print("-" * 70)
    price_df = analyse_by_price_band(bets)
    if not price_df.empty:
        print(price_df.to_string(index=False))

    print("\n" + "-" * 70)
    print("  PROFITABILITY BY OVERLAY BAND (Level Stakes, OOS)")
    print("-" * 70)
    overlay_df = analyse_by_overlay_band(bets)
    if not overlay_df.empty:
        print(overlay_df.to_string(index=False))

    print("\n" + "-" * 70)
    print("  MONTHLY LEVEL STAKES P&L (OOS)")
    print("-" * 70)
    month_df = monthly_breakdown(bets)
    if not month_df.empty:
        print(month_df.to_string(index=False))
        total_pl = month_df["P&L"].sum()
        total_bets = month_df["Bets"].sum()
        print(
            f"\n  TOTAL: {total_bets:,} bets, "
            f"£{total_pl:+,.2f} P&L, "
            f"{total_pl/total_bets*100:+.1f}% ROI"
        )

    print("\n" + "=" * 70)
    print("  NOTE: All predictions are OUT-OF-SAMPLE from walk-forward")
    print("  validation. The model never saw these races during training.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
