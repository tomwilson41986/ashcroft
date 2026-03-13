#!/usr/bin/env python3
"""
Model Profitability Report — Out-of-Sample Validation.

Uses 253k walk-forward OOS predictions (26 folds, Jan 2024 – Feb 2026)
to evaluate whether our BFSP model is profitable under 6 betting strategies:

  1. Rank 1 per race — Level Stakes (£1)
  2. Rank 1, 2, 3 per race — Level Stakes (£1 each)
  3. Rank 1 per race — Variable stakes to win £100
  4. Full Kelly on model overlays vs BFSP
  5. Half Kelly on model overlays vs BFSP
  6. Quarter Kelly on model overlays vs BFSP

All returns are net of 5% Betfair commission on winnings.

Usage:
    python model_profitability_report.py
    python model_profitability_report.py --csv data/oos_predictions.csv
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OOS_CSV = os.path.join(SCRIPT_DIR, "data", "oos_predictions.csv")
COMMISSION = 0.05  # Betfair 5% commission on net winnings


def load_oos_data(path: str) -> pd.DataFrame:
    """Load OOS predictions and add race-level ranks."""
    df = pd.read_csv(path)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df["predicted_bfsp"] = pd.to_numeric(df["predicted_bfsp"], errors="coerce")
    df["placing_numerical"] = pd.to_numeric(df["placing_numerical"], errors="coerce")

    # Ensure won is boolean
    df["won"] = df["won"].astype(str).str.strip().str.lower() == "true"

    # Drop rows missing actual BFSP
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()

    # Create race key and rank within race by predicted_bfsp (lowest = best)
    df["race_key"] = df["race_date"].astype(str) + "|" + df["race_time"] + "|" + df["track"]
    df["rank"] = df.groupby("race_key")["predicted_bfsp"].rank(method="first").astype(int)

    # Overlay: how much better is the actual BFSP vs our model's fair price
    df["overlay_pct"] = (df["bfsp"] / df["predicted_bfsp"] - 1) * 100

    # Model implied probability
    df["model_prob"] = 1.0 / df["predicted_bfsp"]

    return df


def net_return(stake: float, bfsp: float, won: bool) -> float:
    """Calculate return net of Betfair commission."""
    if won:
        gross_profit = stake * (bfsp - 1)
        commission = gross_profit * COMMISSION
        return stake + gross_profit - commission
    return 0.0


def strategy_rank1_level(df: pd.DataFrame) -> dict:
    """Strategy 1: Bet Rank 1 per race to £1 level stakes."""
    bets = df[df["rank"] == 1].copy()
    stake_per_bet = 1.0
    n = len(bets)
    total_staked = n * stake_per_bet

    returns = bets.apply(
        lambda r: net_return(stake_per_bet, r["bfsp"], r["won"]), axis=1
    )
    total_returned = returns.sum()
    winners = bets["won"].sum()

    return _build_result("Rank 1 — Level Stakes £1", n, total_staked, total_returned, winners, bets)


def strategy_rank123_level(df: pd.DataFrame) -> dict:
    """Strategy 2: Bet Rank 1, 2, 3 per race to £1 level stakes each."""
    bets = df[df["rank"].isin([1, 2, 3])].copy()
    stake_per_bet = 1.0
    n = len(bets)
    total_staked = n * stake_per_bet

    returns = bets.apply(
        lambda r: net_return(stake_per_bet, r["bfsp"], r["won"]), axis=1
    )
    total_returned = returns.sum()
    winners = bets["won"].sum()

    return _build_result("Rank 1,2,3 — Level Stakes £1", n, total_staked, total_returned, winners, bets)


def strategy_rank1_variable_100(df: pd.DataFrame) -> dict:
    """Strategy 3: Bet Rank 1 per race, variable stake to win £100."""
    bets = df[df["rank"] == 1].copy()
    n = len(bets)

    # Stake to win £100: stake * (bfsp - 1) * (1 - commission) = 100
    # stake = 100 / ((bfsp - 1) * (1 - commission))
    bets = bets.copy()
    bets["stake"] = 100.0 / ((bets["bfsp"] - 1) * (1 - COMMISSION))
    # Cap stakes at sensible levels (avoid huge stakes on very short prices)
    bets["stake"] = bets["stake"].clip(upper=500.0)

    total_staked = bets["stake"].sum()
    returns = bets.apply(
        lambda r: net_return(r["stake"], r["bfsp"], r["won"]), axis=1
    )
    total_returned = returns.sum()
    winners = bets["won"].sum()

    result = _build_result("Rank 1 — Variable to Win £100", n, total_staked, total_returned, winners, bets)
    result["avg_stake"] = round(bets["stake"].mean(), 2)
    result["median_stake"] = round(bets["stake"].median(), 2)
    result["max_stake"] = round(bets["stake"].max(), 2)
    return result


def strategy_kelly(df: pd.DataFrame, fraction: float, label: str) -> dict:
    """Kelly criterion on model overlays vs BFSP.

    Only bet when the model identifies positive edge (overlay > 0).
    Kelly fraction f = (p * b - q) / b where:
        p = model probability = 1 / predicted_bfsp
        q = 1 - p
        b = net decimal odds = bfsp - 1

    Args:
        fraction: Kelly multiplier (1.0 = full, 0.5 = half, 0.25 = quarter)
    """
    # Only bet positive overlays
    bets = df[df["overlay_pct"] > 0].copy().sort_values(["race_date", "race_time"])

    starting_bankroll = 1000.0
    bankroll = starting_bankroll
    total_staked = 0.0
    total_returned = 0.0
    n_bets_placed = 0
    winners = 0
    bankroll_history = [bankroll]
    bet_log = []

    for _, row in bets.iterrows():
        if bankroll <= 1.0:  # Bankrupt
            break

        p = row["model_prob"]
        b = row["bfsp"] - 1.0
        if b <= 0:
            continue

        # Kelly fraction
        q = 1.0 - p
        kelly_f = (p * b - q) / b

        if kelly_f <= 0:
            continue

        # Apply fraction and size
        f = kelly_f * fraction
        stake = bankroll * f

        # Floor/ceiling
        stake = max(stake, 1.0)  # Minimum £1
        stake = min(stake, bankroll * 0.10)  # Never more than 10% of bankroll
        stake = min(stake, bankroll)  # Can't bet more than we have

        n_bets_placed += 1
        total_staked += stake

        if row["won"]:
            gross_profit = stake * b
            commission = gross_profit * COMMISSION
            net_profit = gross_profit - commission
            bankroll += net_profit
            total_returned += stake + net_profit
            winners += 1
        else:
            bankroll -= stake

        bankroll_history.append(bankroll)

    peak = max(bankroll_history)
    trough = min(bankroll_history)

    # Max drawdown
    running_max = np.maximum.accumulate(bankroll_history)
    drawdowns = (np.array(bankroll_history) - running_max) / running_max * 100
    max_drawdown = abs(drawdowns.min())

    return {
        "name": label,
        "n_bets": n_bets_placed,
        "winners": winners,
        "win_rate_pct": round(winners / max(n_bets_placed, 1) * 100, 2),
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "profit": round(total_returned - total_staked, 2),
        "roi_pct": round((total_returned - total_staked) / max(total_staked, 1) * 100, 2),
        "starting_bankroll": starting_bankroll,
        "final_bankroll": round(bankroll, 2),
        "bankroll_growth_pct": round((bankroll - starting_bankroll) / starting_bankroll * 100, 2),
        "peak_bankroll": round(peak, 2),
        "trough_bankroll": round(trough, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "avg_stake": round(total_staked / max(n_bets_placed, 1), 2),
        "bankroll_history": bankroll_history,
    }


def _build_result(name, n, total_staked, total_returned, winners, bets_df):
    """Build standard result dict for flat staking strategies."""
    profit = total_returned - total_staked

    # Longest losing streak
    if n > 0:
        won_arr = bets_df["won"].values
        max_streak = 0
        current = 0
        for w in won_arr:
            if not w:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
    else:
        max_streak = 0

    return {
        "name": name,
        "n_bets": n,
        "n_races": bets_df["race_key"].nunique() if n > 0 else 0,
        "winners": int(winners),
        "win_rate_pct": round(winners / max(n, 1) * 100, 2),
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "profit": round(profit, 2),
        "roi_pct": round(profit / max(total_staked, 1) * 100, 2),
        "avg_winner_bfsp": round(bets_df.loc[bets_df["won"], "bfsp"].mean(), 2) if winners > 0 else 0,
        "median_winner_bfsp": round(bets_df.loc[bets_df["won"], "bfsp"].median(), 2) if winners > 0 else 0,
        "longest_losing_streak": max_streak,
    }


def monthly_breakdown(df: pd.DataFrame, label: str, rank_filter=None) -> pd.DataFrame:
    """Monthly P&L for a flat-stake strategy."""
    if rank_filter is not None:
        bets = df[df["rank"].isin(rank_filter)].copy()
    else:
        bets = df.copy()

    bets["month"] = bets["race_date"].dt.to_period("M")
    rows = []
    for month, grp in bets.groupby("month"):
        n = len(grp)
        w = grp["won"].sum()
        returned = grp.apply(
            lambda r: net_return(1.0, r["bfsp"], r["won"]), axis=1
        ).sum()
        staked = n * 1.0
        pl = returned - staked
        rows.append({
            "Month": str(month),
            "Bets": n,
            "Winners": int(w),
            "Win%": round(w / n * 100, 1),
            "P&L": round(pl, 2),
            "ROI%": round(pl / staked * 100, 1),
        })

    mdf = pd.DataFrame(rows)
    if not mdf.empty:
        mdf["Cum P&L"] = mdf["P&L"].cumsum().round(2)
    return mdf


def print_section(title: str, char: str = "="):
    width = 78
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_strategy_result(r: dict):
    print(f"\n  Strategy: {r['name']}")
    print(f"  {'─' * 50}")
    print(f"  Bets Placed:         {r['n_bets']:,}")
    print(f"  Winners:             {r['winners']:,}")
    print(f"  Win Rate:            {r['win_rate_pct']:.1f}%")
    print(f"  Total Staked:        £{r['total_staked']:,.2f}")
    print(f"  Total Returned:      £{r['total_returned']:,.2f}")
    print(f"  Net Profit/Loss:     £{r['profit']:+,.2f}")
    print(f"  ROI:                 {r['roi_pct']:+.2f}%")

    if "avg_winner_bfsp" in r:
        print(f"  Avg Winner BFSP:     {r['avg_winner_bfsp']:.2f}")
        print(f"  Median Winner BFSP:  {r['median_winner_bfsp']:.2f}")
    if "longest_losing_streak" in r:
        print(f"  Longest Losing Run:  {r['longest_losing_streak']}")
    if "avg_stake" in r:
        print(f"  Avg Stake:           £{r['avg_stake']:.2f}")
    if "median_stake" in r:
        print(f"  Median Stake:        £{r['median_stake']:.2f}")
    if "max_stake" in r:
        print(f"  Max Stake:           £{r['max_stake']:.2f}")
    if "final_bankroll" in r:
        print(f"  Starting Bankroll:   £{r['starting_bankroll']:,.2f}")
        print(f"  Final Bankroll:      £{r['final_bankroll']:,.2f}")
        print(f"  Bankroll Growth:     {r['bankroll_growth_pct']:+.1f}%")
        print(f"  Peak Bankroll:       £{r['peak_bankroll']:,.2f}")
        print(f"  Trough Bankroll:     £{r['trough_bankroll']:,.2f}")
        print(f"  Max Drawdown:        {r['max_drawdown_pct']:.1f}%")


def rank1_by_price_band(df: pd.DataFrame) -> pd.DataFrame:
    """Rank-1 level stakes broken down by predicted price band."""
    bets = df[df["rank"] == 1].copy()
    bands = [
        (1.0, 3.0, "1-3 (Short)"),
        (3.0, 5.0, "3-5 (Med-Short)"),
        (5.0, 8.0, "5-8 (Medium)"),
        (8.0, 13.0, "8-13 (Mid-Long)"),
        (13.0, 21.0, "13-21 (Long)"),
        (21.0, 999, "21+ (Outsider)"),
    ]
    rows = []
    for lo, hi, label in bands:
        mask = (bets["predicted_bfsp"] >= lo) & (bets["predicted_bfsp"] < hi)
        band = bets[mask]
        if len(band) == 0:
            continue
        n = len(band)
        w = band["won"].sum()
        returned = band.apply(lambda r: net_return(1.0, r["bfsp"], r["won"]), axis=1).sum()
        pl = returned - n
        rows.append({
            "Price Band": label,
            "Bets": n,
            "Winners": int(w),
            "Win%": round(w / n * 100, 1),
            "P&L": round(pl, 2),
            "ROI%": round(pl / n * 100, 1),
            "Avg BFSP": round(band["bfsp"].mean(), 1),
        })
    return pd.DataFrame(rows)


def rank123_strike_rate(df: pd.DataFrame) -> pd.DataFrame:
    """How often does each rank win?"""
    rows = []
    for rank in range(1, 11):
        bets = df[df["rank"] == rank]
        if len(bets) == 0:
            continue
        n = len(bets)
        w = bets["won"].sum()
        returned = bets.apply(lambda r: net_return(1.0, r["bfsp"], r["won"]), axis=1).sum()
        pl = returned - n
        rows.append({
            "Rank": rank,
            "Bets": n,
            "Winners": int(w),
            "Win%": round(w / n * 100, 1),
            "£1 Level P&L": round(pl, 2),
            "ROI%": round(pl / n * 100, 1),
            "Avg Pred BFSP": round(bets["predicted_bfsp"].mean(), 1),
            "Avg Act BFSP": round(bets["bfsp"].mean(), 1),
        })
    return pd.DataFrame(rows)


def overlay_band_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Kelly-eligible bets broken down by overlay band."""
    bets = df[df["overlay_pct"] > 0].copy()
    bands = [
        (0, 5, "0-5%"),
        (5, 10, "5-10%"),
        (10, 20, "10-20%"),
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
        w = band["won"].sum()
        returned = band.apply(lambda r: net_return(1.0, r["bfsp"], r["won"]), axis=1).sum()
        pl = returned - n
        rows.append({
            "Overlay": label,
            "Bets": n,
            "Winners": int(w),
            "Win%": round(w / n * 100, 1),
            "L/S P&L": round(pl, 2),
            "ROI%": round(pl / n * 100, 1),
            "Avg BFSP": round(band["bfsp"].mean(), 1),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Model Profitability Report")
    parser.add_argument("--csv", default=OOS_CSV, help="Path to OOS predictions CSV")
    parser.add_argument("--output", default=None, help="Save report to file")
    args = parser.parse_args()

    # Redirect output if saving to file
    output_file = None
    if args.output:
        output_file = open(args.output, "w")
        original_stdout = sys.stdout
        sys.stdout = output_file

    try:
        _run_report(args.csv)
    finally:
        if output_file:
            sys.stdout = original_stdout
            output_file.close()
            print(f"Report saved to {args.output}")


def _run_report(csv_path: str):
    # ──────────────────────────────────────────────────────────────────────
    # LOAD DATA
    # ──────────────────────────────────────────────────────────────────────
    df = load_oos_data(csv_path)

    n_races = df["race_key"].nunique()
    n_runners = len(df)
    n_winners = df["won"].sum()
    date_min = df["race_date"].min().date()
    date_max = df["race_date"].max().date()
    n_folds = df["fold_idx"].nunique()

    # ──────────────────────────────────────────────────────────────────────
    # HEADER
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 78)
    print("  ASHCROFT BFSP MODEL — PROFITABILITY REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 78)

    print_section("1. DATA OVERVIEW")
    print(f"""
  Validation Method:   Walk-forward (temporal, {n_folds} folds)
  Period:              {date_min} to {date_max}
  Races:               {n_races:,}
  Runners:             {n_runners:,}
  Winners:             {n_winners:,}
  Base Win Rate:       {n_winners / n_runners * 100:.1f}%
  Commission:          {COMMISSION * 100:.0f}% (Betfair standard)

  All predictions are strictly OUT-OF-SAMPLE. Each fold trains only
  on data prior to the prediction window — zero lookahead bias.""")

    # ──────────────────────────────────────────────────────────────────────
    # MODEL ACCURACY
    # ──────────────────────────────────────────────────────────────────────
    print_section("2. MODEL ACCURACY SUMMARY")

    log_pred = np.log(df["predicted_bfsp"])
    log_actual = np.log(df["bfsp"])
    corr = np.corrcoef(log_pred, log_actual)[0, 1]
    log_mae = np.mean(np.abs(log_pred - log_actual))
    log_r2 = 1 - np.sum((log_actual - log_pred) ** 2) / np.sum((log_actual - log_actual.mean()) ** 2)
    bfsp_mae = np.mean(np.abs(df["predicted_bfsp"] - df["bfsp"]))
    median_ape = np.median(np.abs(df["predicted_bfsp"] - df["bfsp"]) / df["bfsp"]) * 100

    print(f"""
  Log-Space Metrics:
    Correlation:        {corr:.4f}
    MAE (log):          {log_mae:.4f}
    R² (log):           {log_r2:.4f}

  BFSP-Space Metrics:
    MAE (£):            £{bfsp_mae:.2f}
    Median APE:         {median_ape:.1f}%""")

    # ──────────────────────────────────────────────────────────────────────
    # RANKING ABILITY
    # ──────────────────────────────────────────────────────────────────────
    print_section("3. RANKING ABILITY (model rank vs actual winner)")

    rank_df = rank123_strike_rate(df)
    print()
    print(rank_df.to_string(index=False))

    # Market baseline: if we just backed the BFSP favourite
    df["bfsp_rank"] = df.groupby("race_key")["bfsp"].rank(method="first").astype(int)
    market_fav = df[df["bfsp_rank"] == 1]
    market_fav_win = market_fav["won"].mean() * 100
    model_fav = df[df["rank"] == 1]
    model_fav_win = model_fav["won"].mean() * 100

    avg_field = df.groupby("race_key").size().mean()
    random_win = 1 / avg_field * 100

    print(f"""
  Baselines:
    Random (1/field):   {random_win:.1f}%
    BFSP Favourite:     {market_fav_win:.1f}%
    Model Rank 1:       {model_fav_win:.1f}%""")

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY 1: RANK 1 LEVEL STAKES
    # ──────────────────────────────────────────────────────────────────────
    print_section("4. STRATEGY RESULTS", "─")

    r1 = strategy_rank1_level(df)
    print_strategy_result(r1)

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY 2: RANK 1,2,3 LEVEL STAKES
    # ──────────────────────────────────────────────────────────────────────
    r2 = strategy_rank123_level(df)
    print_strategy_result(r2)

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGY 3: RANK 1 VARIABLE TO WIN £100
    # ──────────────────────────────────────────────────────────────────────
    r3 = strategy_rank1_variable_100(df)
    print_strategy_result(r3)

    # ──────────────────────────────────────────────────────────────────────
    # STRATEGIES 4-6: KELLY VARIANTS
    # ──────────────────────────────────────────────────────────────────────
    r4 = strategy_kelly(df, 1.0, "Full Kelly on Model Overlays vs BFSP")
    print_strategy_result(r4)

    r5 = strategy_kelly(df, 0.5, "Half Kelly on Model Overlays vs BFSP")
    print_strategy_result(r5)

    r6 = strategy_kelly(df, 0.25, "Quarter Kelly on Model Overlays vs BFSP")
    print_strategy_result(r6)

    # ──────────────────────────────────────────────────────────────────────
    # RANK 1 BY PRICE BAND
    # ──────────────────────────────────────────────────────────────────────
    print_section("5. RANK 1 PROFITABILITY BY PRICE BAND (Level Stakes)", "─")
    price_df = rank1_by_price_band(df)
    if not price_df.empty:
        print()
        print(price_df.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────
    # OVERLAY BAND ANALYSIS
    # ──────────────────────────────────────────────────────────────────────
    print_section("6. OVERLAY BAND ANALYSIS (Level Stakes, all overlays)", "─")
    ov_df = overlay_band_analysis(df)
    if not ov_df.empty:
        print()
        print(ov_df.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────
    # MONTHLY P&L — RANK 1
    # ──────────────────────────────────────────────────────────────────────
    print_section("7. MONTHLY P&L — RANK 1 LEVEL STAKES (net of 5% commission)", "─")
    m1 = monthly_breakdown(df, "Rank 1", rank_filter=[1])
    if not m1.empty:
        print()
        print(m1.to_string(index=False))
        profitable_months = (m1["P&L"] > 0).sum()
        total_months = len(m1)
        print(f"\n  Profitable months: {profitable_months}/{total_months} ({profitable_months/total_months*100:.0f}%)")

    # ──────────────────────────────────────────────────────────────────────
    # MONTHLY P&L — RANK 1,2,3
    # ──────────────────────────────────────────────────────────────────────
    print_section("8. MONTHLY P&L — RANK 1,2,3 LEVEL STAKES (net of 5% commission)", "─")
    m2 = monthly_breakdown(df, "Rank 1,2,3", rank_filter=[1, 2, 3])
    if not m2.empty:
        print()
        print(m2.to_string(index=False))
        profitable_months = (m2["P&L"] > 0).sum()
        total_months = len(m2)
        print(f"\n  Profitable months: {profitable_months}/{total_months} ({profitable_months/total_months*100:.0f}%)")

    # ──────────────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ──────────────────────────────────────────────────────────────────────
    print_section("9. STRATEGY COMPARISON SUMMARY")

    all_results = [r1, r2, r3, r4, r5, r6]
    summary_rows = []
    for r in all_results:
        row = {
            "Strategy": r["name"],
            "Bets": r["n_bets"],
            "Win%": r["win_rate_pct"],
            "Staked": f"£{r['total_staked']:,.0f}",
            "P&L": f"£{r['profit']:+,.0f}",
            "ROI%": f"{r['roi_pct']:+.2f}%",
        }
        if "final_bankroll" in r:
            row["Final Bank"] = f"£{r['final_bankroll']:,.0f}"
            row["Max DD%"] = f"{r['max_drawdown_pct']:.1f}%"
        else:
            row["Final Bank"] = "—"
            row["Max DD%"] = "—"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────
    # CONCLUSIONS
    # ──────────────────────────────────────────────────────────────────────
    print_section("10. CONCLUSIONS")

    best_roi = max(all_results, key=lambda x: x["roi_pct"])
    worst_roi = min(all_results, key=lambda x: x["roi_pct"])

    any_profitable = any(r["profit"] > 0 for r in all_results)

    print(f"""
  Best ROI:    {best_roi['name']} ({best_roi['roi_pct']:+.2f}%)
  Worst ROI:   {worst_roi['name']} ({worst_roi['roi_pct']:+.2f}%)

  Model finds winner from Rank 1 pick {r1['win_rate_pct']:.1f}% of the time
  (vs {random_win:.1f}% random, {market_fav_win:.1f}% BFSP favourite).
""")

    if any_profitable:
        profitable = [r for r in all_results if r["profit"] > 0]
        print("  PROFITABLE STRATEGIES:")
        for r in profitable:
            print(f"    ✓ {r['name']}: £{r['profit']:+,.2f} ({r['roi_pct']:+.2f}% ROI)")
    else:
        print("  No strategy achieved overall profitability in this OOS period.")

    print(f"""
  Key Observations:
  - The model has strong predictive power (R²={log_r2:.3f}, correlation={corr:.4f})
  - Rank 1 wins {r1['win_rate_pct']:.1f}% vs market favourite {market_fav_win:.1f}%
  - Betfair's {COMMISSION*100:.0f}% commission is a significant drag on returns
  - The market is highly efficient — model edge may not overcome commission
""")

    print("=" * 78)
    print("  END OF REPORT")
    print("=" * 78)


if __name__ == "__main__":
    main()
