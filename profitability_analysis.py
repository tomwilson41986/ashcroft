#!/usr/bin/env python3
"""
Profitability Analysis — BFSP Model Betting Simulation.

Loads the trained model, generates predictions on the test set, and runs
comprehensive betting simulations across multiple staking strategies and
selection filters.

Staking strategies:
    - Level stakes (£1 per bet)
    - Variant stakes (proportional to edge)
    - Kelly criterion (full, half, quarter)
    - Square root Kelly
    - Fixed fraction of edge

Selection analysis:
    - By predicted rank within race (top 1, 2, 3)
    - By edge threshold (predicted BFSP < actual BFSP)
    - By race type, class, field size, odds range
    - By confidence (gap between predicted and market)
"""

import json
import logging
import os
import sqlite3
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.custom_metrics import CustomMetricsEngine
from train_bfsp import ALL_FEATURE_COLS, build_context_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")
START_DATE = "2022-01-01"
TEST_DAYS = 60
VAL_DAYS = 60


# ---------------------------------------------------------------------------
# Data loading (reuses retrain_bfsp logic)
# ---------------------------------------------------------------------------

def load_and_prepare() -> tuple[pd.DataFrame, list[str]]:
    """Load data, compute metrics, return full dataset with features."""
    conn = sqlite3.connect(DEFAULT_DB)
    df = pd.read_sql_query(
        "SELECT * FROM race_results WHERE race_date >= ? ORDER BY race_date, race_time",
        conn,
        params=(START_DATE,),
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])

    engine = CustomMetricsEngine()
    log.info("Calculating custom metrics...")
    df = engine.calculate_all(df)
    log.info("Building context features...")
    df = build_context_features(df)

    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df["log_bfsp"] = np.log(df["bfsp"])

    seen = set()
    feature_cols = []
    for c in ALL_FEATURE_COLS:
        if c in df.columns and c not in seen:
            seen.add(c)
            feature_cols.append(c)

    df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)
    return df, feature_cols


def get_test_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into val vs test."""
    max_date = df["race_date"].max()
    test_cutoff = max_date - pd.Timedelta(days=TEST_DAYS)
    val_cutoff = test_cutoff - pd.Timedelta(days=VAL_DAYS)
    val_df = df[(df["race_date"] >= val_cutoff) & (df["race_date"] < test_cutoff)].copy()
    test_df = df[df["race_date"] >= test_cutoff].copy()
    return val_df, test_df


# ---------------------------------------------------------------------------
# Prediction + Benter normalization
# ---------------------------------------------------------------------------

def predict_and_normalize(model, subset_df, feature_cols):
    """Predict, normalize within race, return enriched DataFrame."""
    X = subset_df[feature_cols].astype(float)
    log_pred = model.predict(X)

    out = subset_df.copy()
    out["pred_log_bfsp"] = log_pred
    out["pred_bfsp_raw"] = np.exp(log_pred)
    out["implied_prob"] = 1.0 / out["pred_bfsp_raw"]

    # Benter normalization per race
    race_sum = out.groupby("raceid")["implied_prob"].transform("sum")
    out["win_prob"] = out["implied_prob"] / race_sum
    out["pred_bfsp"] = 1.0 / out["win_prob"]

    # Actual win probability from BFSP
    out["actual_implied_prob"] = 1.0 / out["bfsp"]
    actual_sum = out.groupby("raceid")["actual_implied_prob"].transform("sum")
    out["actual_win_prob"] = out["actual_implied_prob"] / actual_sum

    # Edge: our predicted prob vs market prob
    out["edge"] = out["win_prob"] - out["actual_win_prob"]
    out["edge_pct"] = (out["pred_bfsp"] / out["bfsp"] - 1) * -100  # +ve = value

    # Won flag
    out["won"] = (out["placing_numerical"] == 1).astype(int)

    # Rank within race by predicted BFSP (1 = predicted favourite)
    out["pred_rank"] = out.groupby("raceid")["pred_bfsp"].rank(method="min")

    # P&L at BFSP for a £1 win bet
    out["pnl_level"] = np.where(out["won"] == 1, out["bfsp"] - 1, -1.0)

    return out


# ---------------------------------------------------------------------------
# Staking strategies
# ---------------------------------------------------------------------------

def level_stakes_pnl(df):
    """£1 per bet."""
    return df["pnl_level"].sum(), len(df), df["pnl_level"].sum() / len(df) * 100


def variant_stakes_pnl(df, base_stake=1.0):
    """Stake proportional to edge (positive edge only), capped at 5x base."""
    stakes = np.clip(df["edge_pct"] / 10, 0.1, 5.0) * base_stake
    pnl = np.where(df["won"] == 1, stakes * (df["bfsp"] - 1), -stakes)
    return pnl.sum(), stakes.sum(), pnl.sum() / stakes.sum() * 100


def kelly_pnl(df, fraction=1.0, bankroll=1000.0):
    """Kelly criterion: stake = fraction * (p*b - q) / b."""
    results = []
    bank = bankroll
    for _, row in df.iterrows():
        p = row["win_prob"]
        b = row["bfsp"] - 1  # decimal odds minus 1
        q = 1 - p
        kelly_frac = (p * b - q) / b if b > 0 else 0
        kelly_frac = max(kelly_frac, 0) * fraction
        kelly_frac = min(kelly_frac, 0.25)  # cap at 25% of bankroll

        stake = bank * kelly_frac
        if row["won"] == 1:
            profit = stake * b
        else:
            profit = -stake
        bank += profit
        results.append({
            "stake": stake,
            "profit": profit,
            "bank": bank,
            "kelly_frac": kelly_frac,
        })
    res_df = pd.DataFrame(results)
    total_staked = res_df["stake"].sum()
    total_profit = res_df["profit"].sum()
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0
    return total_profit, total_staked, roi, bank, res_df


def sqrt_kelly_pnl(df, bankroll=1000.0):
    """Square root of Kelly fraction — less volatile than half Kelly."""
    results = []
    bank = bankroll
    for _, row in df.iterrows():
        p = row["win_prob"]
        b = row["bfsp"] - 1
        q = 1 - p
        kelly_frac = (p * b - q) / b if b > 0 else 0
        kelly_frac = max(kelly_frac, 0)
        sqrt_frac = np.sqrt(kelly_frac) * 0.1
        sqrt_frac = min(sqrt_frac, 0.15)

        stake = bank * sqrt_frac
        if row["won"] == 1:
            profit = stake * b
        else:
            profit = -stake
        bank += profit
        results.append({"stake": stake, "profit": profit, "bank": bank})
    res_df = pd.DataFrame(results)
    total_staked = res_df["stake"].sum()
    total_profit = res_df["profit"].sum()
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0
    return total_profit, total_staked, roi, bank


def fixed_edge_fraction_pnl(df, edge_multiple=0.5, base_stake=1.0):
    """Stake = base * edge_multiple * edge_pct. Simple proportional."""
    edge = df["edge_pct"].values
    stakes = np.clip(edge * edge_multiple / 100, 0, 5.0) * base_stake
    stakes = np.where(edge > 0, np.maximum(stakes, 0.1), 0)
    pnl = np.where(df["won"] == 1, stakes * (df["bfsp"].values - 1), -stakes)
    total_staked = stakes.sum()
    total_profit = pnl.sum()
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0
    return total_profit, total_staked, roi


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def analyze_subset(label, df, bankroll=1000.0):
    """Run all staking strategies on a subset and print results."""
    if len(df) == 0:
        print(f"  {label}: No selections")
        return

    n_bets = len(df)
    n_winners = df["won"].sum()
    sr = n_winners / n_bets * 100

    print(f"\n  {label}")
    print(f"  Bets: {n_bets:,}  |  Winners: {n_winners:,}  |  Strike rate: {sr:.1f}%")
    print(f"  {'Strategy':<30s} {'Profit':>10s} {'Staked':>10s} {'ROI':>8s} {'Final Bank':>12s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*12}")

    # Level stakes
    ls_profit, ls_bets, ls_roi = level_stakes_pnl(df)
    print(f"  {'Level stakes (£1)':<30s} {ls_profit:>10.2f} {ls_bets:>10.0f} {ls_roi:>7.1f}%")

    # Variant stakes
    vs_profit, vs_staked, vs_roi = variant_stakes_pnl(df)
    print(f"  {'Variant stakes':<30s} {vs_profit:>10.2f} {vs_staked:>10.2f} {vs_roi:>7.1f}%")

    # Kelly variants
    for frac, name in [(1.0, "Full Kelly"), (0.5, "Half Kelly"),
                        (0.25, "Quarter Kelly")]:
        k_profit, k_staked, k_roi, k_bank, _ = kelly_pnl(df, fraction=frac, bankroll=bankroll)
        print(f"  {name:<30s} {k_profit:>10.2f} {k_staked:>10.2f} {k_roi:>7.1f}% {k_bank:>11.2f}")

    # Sqrt Kelly
    sk_profit, sk_staked, sk_roi, sk_bank = sqrt_kelly_pnl(df, bankroll=bankroll)
    print(f"  {'Sqrt Kelly':<30s} {sk_profit:>10.2f} {sk_staked:>10.2f} {sk_roi:>7.1f}% {sk_bank:>11.2f}")

    # Fixed edge fraction
    fe_profit, fe_staked, fe_roi = fixed_edge_fraction_pnl(df)
    print(f"  {'Fixed edge fraction':<30s} {fe_profit:>10.2f} {fe_staked:>10.2f} {fe_roi:>7.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load model
    model_path = os.path.join(MODEL_DIR, "bfsp_model.lgb")
    if not os.path.exists(model_path):
        log.error(f"Model not found: {model_path}. Run retrain_bfsp.py first.")
        sys.exit(1)
    model = lgb.Booster(model_file=model_path)
    log.info(f"Loaded model from {model_path}")

    # Load and prepare data
    df, feature_cols = load_and_prepare()
    val_df, test_df = get_test_set(df)

    log.info(f"Validation: {len(val_df):,} rows, Test: {len(test_df):,} rows")

    # Generate predictions
    log.info("Generating predictions...")
    val_pred = predict_and_normalize(model, val_df, feature_cols)
    test_pred = predict_and_normalize(model, test_df, feature_cols)

    # Combine val+test for out-of-sample analysis
    oos = pd.concat([val_pred, test_pred], ignore_index=True)

    # =======================================================================
    # 1. RANK-BASED ANALYSIS
    # =======================================================================
    print_section("1. RANK-BASED ANALYSIS (by predicted BFSP rank within race)")

    for dataset_name, dataset in [("VALIDATION", val_pred), ("TEST", test_pred), ("COMBINED OOS", oos)]:
        print(f"\n  --- {dataset_name} SET ---")
        for rank in [1, 2, 3]:
            subset = dataset[dataset["pred_rank"] == rank]
            analyze_subset(f"Predicted Rank {rank}", subset)

        # Top 2 combined
        subset = dataset[dataset["pred_rank"] <= 2]
        analyze_subset("Top 2 predicted", subset)

    # =======================================================================
    # 2. EDGE-BASED ANALYSIS (value bets: pred BFSP < actual BFSP)
    # =======================================================================
    print_section("2. VALUE BET ANALYSIS (predicted BFSP < actual BFSP = overlay)")

    for dataset_name, dataset in [("VALIDATION", val_pred), ("TEST", test_pred), ("COMBINED OOS", oos)]:
        print(f"\n  --- {dataset_name} SET ---")

        # Any positive edge
        value = dataset[dataset["edge_pct"] > 0]
        analyze_subset("Any positive edge", value)

        # Edge thresholds
        for thresh in [5, 10, 15, 20, 30, 50]:
            subset = value[value["edge_pct"] >= thresh]
            analyze_subset(f"Edge >= {thresh}%", subset)

    # =======================================================================
    # 3. COMBINED RANK + EDGE
    # =======================================================================
    print_section("3. COMBINED: RANK + EDGE FILTERS")

    for dataset_name, dataset in [("TEST", test_pred), ("COMBINED OOS", oos)]:
        print(f"\n  --- {dataset_name} SET ---")

        for rank in [1, 2, 3]:
            for edge_thresh in [0, 5, 10, 20]:
                subset = dataset[(dataset["pred_rank"] == rank) & (dataset["edge_pct"] >= edge_thresh)]
                analyze_subset(f"Rank {rank}, Edge >= {edge_thresh}%", subset)

    # =======================================================================
    # 4. ODDS RANGE ANALYSIS
    # =======================================================================
    print_section("4. ODDS RANGE ANALYSIS (actual BFSP buckets)")

    for dataset_name, dataset in [("TEST", test_pred), ("COMBINED OOS", oos)]:
        print(f"\n  --- {dataset_name} SET ---")
        value = dataset[dataset["edge_pct"] > 0]

        ranges = [(1.01, 3.0, "1.01 - 3.00"), (3.0, 6.0, "3.00 - 6.00"),
                  (6.0, 10.0, "6.00 - 10.00"), (10.0, 20.0, "10.00 - 20.00"),
                  (20.0, 50.0, "20.00 - 50.00"), (50.0, 1000.0, "50.00+")]
        for lo, hi, label in ranges:
            subset = value[(value["bfsp"] >= lo) & (value["bfsp"] < hi)]
            analyze_subset(f"BFSP {label} (value bets)", subset)

    # =======================================================================
    # 5. RACE TYPE ANALYSIS
    # =======================================================================
    print_section("5. RACE TYPE ANALYSIS")

    for dataset_name, dataset in [("TEST", test_pred), ("COMBINED OOS", oos)]:
        print(f"\n  --- {dataset_name} SET ---")
        value = dataset[dataset["edge_pct"] > 0]

        if "race_type" in value.columns:
            for rt in sorted(value["race_type"].dropna().unique()):
                subset = value[value["race_type"] == rt]
                if len(subset) >= 10:
                    analyze_subset(f"Race type: {rt}", subset)

    # =======================================================================
    # 6. FIELD SIZE ANALYSIS
    # =======================================================================
    print_section("6. FIELD SIZE ANALYSIS")

    for dataset_name, dataset in [("TEST", test_pred), ("COMBINED OOS", oos)]:
        print(f"\n  --- {dataset_name} SET ---")
        value = dataset[dataset["edge_pct"] > 0]

        for lo, hi, label in [(2, 8, "Small (2-7)"), (8, 13, "Medium (8-12)"),
                                (13, 40, "Large (13+)")]:
            subset = value[(value["number_of_runners"] >= lo) & (value["number_of_runners"] < hi)]
            analyze_subset(f"Field: {label} (value bets)", subset)

    # =======================================================================
    # 7. RACE CLASS ANALYSIS
    # =======================================================================
    print_section("7. RACE CLASS ANALYSIS")

    for dataset_name, dataset in [("TEST", test_pred), ("COMBINED OOS", oos)]:
        print(f"\n  --- {dataset_name} SET ---")
        value = dataset[dataset["edge_pct"] > 0]

        if "race_class" in value.columns:
            for rc in sorted(value["race_class"].dropna().unique()):
                subset = value[value["race_class"] == rc]
                if len(subset) >= 10:
                    analyze_subset(f"Class: {rc}", subset)

    # =======================================================================
    # 8. BEST FILTER COMBINATIONS (systematic search)
    # =======================================================================
    print_section("8. SYSTEMATIC FILTER SEARCH — Best ROI combinations")

    dataset = oos
    results = []

    for rank_max in [1, 2, 3, 99]:
        for edge_min in [0, 5, 10, 15, 20, 30]:
            for bfsp_lo, bfsp_hi in [(1.01, 1000), (1.01, 10), (3, 20),
                                      (3, 50), (5, 30), (10, 1000)]:
                for runners_min in [0, 5, 8]:
                    mask = (
                        (dataset["pred_rank"] <= rank_max) &
                        (dataset["edge_pct"] >= edge_min) &
                        (dataset["bfsp"] >= bfsp_lo) &
                        (dataset["bfsp"] < bfsp_hi) &
                        (dataset["number_of_runners"] >= runners_min)
                    )
                    subset = dataset[mask]
                    if len(subset) < 20:
                        continue

                    n = len(subset)
                    wins = subset["won"].sum()
                    sr = wins / n * 100
                    ls_profit = subset["pnl_level"].sum()
                    ls_roi = ls_profit / n * 100

                    # Half Kelly
                    hk_profit, hk_staked, hk_roi, hk_bank, _ = kelly_pnl(
                        subset, fraction=0.5, bankroll=1000
                    )

                    results.append({
                        "rank_max": rank_max if rank_max < 99 else "any",
                        "edge_min": edge_min,
                        "bfsp_range": f"{bfsp_lo}-{bfsp_hi}",
                        "runners_min": runners_min,
                        "bets": n,
                        "winners": wins,
                        "strike_rate": round(sr, 1),
                        "ls_profit": round(ls_profit, 2),
                        "ls_roi": round(ls_roi, 1),
                        "hk_roi": round(hk_roi, 1),
                        "hk_final_bank": round(hk_bank, 2),
                    })

    results_df = pd.DataFrame(results)

    # Show top 20 by level stakes ROI (min 50 bets)
    profitable = results_df[(results_df["ls_roi"] > 0) & (results_df["bets"] >= 50)]
    profitable = profitable.sort_values("ls_roi", ascending=False)

    print(f"\n  Top 20 Profitable Filters (min 50 bets, level stakes ROI > 0%):")
    print(f"  {'Rank':<6s} {'Edge%':<7s} {'BFSP':<12s} {'Rnrs':<6s} "
          f"{'Bets':>5s} {'SR%':>5s} {'LS ROI':>7s} {'HK ROI':>7s} {'HK Bank':>9s}")
    print(f"  {'-'*6} {'-'*7} {'-'*12} {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*9}")

    for _, row in profitable.head(20).iterrows():
        print(f"  {str(row['rank_max']):<6s} {row['edge_min']:<7.0f} {row['bfsp_range']:<12s} "
              f"{row['runners_min']:<6.0f} {row['bets']:>5.0f} {row['strike_rate']:>5.1f} "
              f"{row['ls_roi']:>6.1f}% {row['hk_roi']:>6.1f}% {row['hk_final_bank']:>9.2f}")

    # Show top 20 by Half Kelly final bankroll
    by_bank = results_df[results_df["bets"] >= 50].sort_values("hk_final_bank", ascending=False)

    print(f"\n  Top 20 by Half Kelly Final Bankroll (min 50 bets):")
    print(f"  {'Rank':<6s} {'Edge%':<7s} {'BFSP':<12s} {'Rnrs':<6s} "
          f"{'Bets':>5s} {'SR%':>5s} {'LS ROI':>7s} {'HK ROI':>7s} {'HK Bank':>9s}")
    print(f"  {'-'*6} {'-'*7} {'-'*12} {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*9}")

    for _, row in by_bank.head(20).iterrows():
        print(f"  {str(row['rank_max']):<6s} {row['edge_min']:<7.0f} {row['bfsp_range']:<12s} "
              f"{row['runners_min']:<6.0f} {row['bets']:>5.0f} {row['strike_rate']:>5.1f} "
              f"{row['ls_roi']:>6.1f}% {row['hk_roi']:>6.1f}% {row['hk_final_bank']:>9.2f}")

    # =======================================================================
    # 9. SUMMARY
    # =======================================================================
    print_section("9. SUMMARY — RECOMMENDED STRATEGIES")

    if len(profitable) > 0:
        best = profitable.iloc[0]
        print(f"\n  Best Level Stakes ROI filter:")
        print(f"    Rank <= {best['rank_max']}, Edge >= {best['edge_min']}%, "
              f"BFSP {best['bfsp_range']}, Runners >= {best['runners_min']}")
        print(f"    {best['bets']:.0f} bets, {best['strike_rate']:.1f}% SR, "
              f"{best['ls_roi']:.1f}% LS ROI, {best['hk_roi']:.1f}% HK ROI")

    if len(by_bank) > 0:
        best_hk = by_bank.iloc[0]
        print(f"\n  Best Half Kelly bankroll growth:")
        print(f"    Rank <= {best_hk['rank_max']}, Edge >= {best_hk['edge_min']}%, "
              f"BFSP {best_hk['bfsp_range']}, Runners >= {best_hk['runners_min']}")
        print(f"    {best_hk['bets']:.0f} bets, {best_hk['strike_rate']:.1f}% SR, "
              f"£1000 -> £{best_hk['hk_final_bank']:.2f}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
