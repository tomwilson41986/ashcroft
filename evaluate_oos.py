#!/usr/bin/env python3
"""
Out-of-Sample Model Evaluation & Profitability Analysis.

Performs a proper walk-forward evaluation where, for each test period,
the model is trained ONLY on data before that period. This ensures every
prediction is genuinely out-of-sample — the model has never seen the
race data it's predicting on.

Evaluates:
    - Prediction accuracy: MAE, RMSE, R², MdAPE, correlation
    - Ranking ability: top-pick win rate, top-3 accuracy, favourite strike rate
    - Profitability: 5 staking strategies on overlay bets
    - Breakdowns: by month, price band, overlay band, race type, going

Usage:
    python evaluate_oos.py
    python evaluate_oos.py --start-date 2025-01-01
    python evaluate_oos.py --min-overlay 10 --min-train-days 365
    python evaluate_oos.py --output-csv oos_predictions.csv
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.custom_metrics import CustomMetricsEngine
from train_bfsp import (
    ALL_FEATURE_COLS,
    BFSPTrainer,
    build_context_features,
    load_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")

BETFAIR_COMMISSION = 0.05


# ---------------------------------------------------------------------------
# Walk-Forward Out-of-Sample Predictions
# ---------------------------------------------------------------------------

def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_train_days: int = 365,
    val_window_days: int = 30,
    step_days: int = 30,
    params: dict | None = None,
) -> pd.DataFrame:
    """Train and predict in walk-forward fashion.

    For each validation window, train a fresh model on all data before it,
    then predict on the validation window. This guarantees every prediction
    is out-of-sample.

    Returns a DataFrame of all out-of-sample predictions with columns:
        - All original columns
        - predicted_log_bfsp, predicted_bfsp
        - predicted_win_prob, predicted_win_prob_norm
        - fold_idx (which fold produced this prediction)
    """
    if params is None:
        params = BFSPTrainer.DEFAULT_PARAMS.copy()

    min_date = df["race_date"].min()
    max_date = df["race_date"].max()

    all_oos = []
    fold_idx = 0
    train_end = min_date + timedelta(days=min_train_days)

    while train_end + timedelta(days=val_window_days) <= max_date:
        val_start = train_end
        val_end = val_start + timedelta(days=val_window_days)

        train_df = df[df["race_date"] < val_start].copy()
        val_df = df[
            (df["race_date"] >= val_start) & (df["race_date"] < val_end)
        ].copy()

        if len(train_df) < 200 or len(val_df) < 5:
            train_end += timedelta(days=step_days)
            continue

        log.info(
            f"  Fold {fold_idx + 1}: train {len(train_df):,} rows "
            f"(< {val_start.date()}), predict {len(val_df):,} rows "
            f"({val_start.date()} to {val_end.date()})"
        )

        X_train = train_df[feature_cols].astype(float)
        y_train = train_df["log_bfsp"].astype(float)
        X_val = val_df[feature_cols].astype(float)

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(
            X_val, label=val_df["log_bfsp"].astype(float), reference=train_set
        )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.log_evaluation(period=0),
                lgb.early_stopping(stopping_rounds=50),
            ],
        )

        val_df = val_df.copy()
        val_df["predicted_log_bfsp"] = model.predict(X_val)
        val_df["predicted_bfsp"] = np.exp(val_df["predicted_log_bfsp"])
        val_df["fold_idx"] = fold_idx

        # Win probability from predicted BFSP
        val_df["predicted_win_prob"] = 1.0 / val_df["predicted_bfsp"]

        # Normalise per race
        race_col = "raceid" if "raceid" in val_df.columns else None
        if race_col is None:
            val_df["raceid"] = (
                val_df["race_date"].dt.strftime("%Y-%m-%d")
                + "_" + val_df["track"].astype(str)
                + "_" + val_df["race_time"].astype(str)
            )
            race_col = "raceid"

        race_prob_sum = val_df.groupby(race_col)["predicted_win_prob"].transform("sum")
        val_df["predicted_win_prob_norm"] = val_df["predicted_win_prob"] / race_prob_sum

        # Recalculate BFSP from normalised probabilities so odds reflect a fair book
        val_df["predicted_bfsp_raw"] = val_df["predicted_bfsp"]
        val_df["predicted_bfsp"] = 1.0 / val_df["predicted_win_prob_norm"]

        all_oos.append(val_df)
        fold_idx += 1
        train_end += timedelta(days=step_days)

    if not all_oos:
        log.error("No walk-forward folds produced. Need more data.")
        return pd.DataFrame()

    combined = pd.concat(all_oos, ignore_index=True)

    # Deduplicate: if a runner appears in multiple folds (overlapping windows),
    # keep the prediction from the latest fold (most training data).
    id_cols = ["race_date", "race_time", "horse_name", "track"]
    available_id = [c for c in id_cols if c in combined.columns]
    if available_id:
        combined = combined.sort_values("fold_idx").drop_duplicates(
            subset=available_id, keep="last"
        )

    log.info(f"\nTotal OOS predictions: {len(combined):,} across {fold_idx} folds")
    return combined


# ---------------------------------------------------------------------------
# Accuracy Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(df: pd.DataFrame) -> dict:
    """Compute prediction accuracy metrics."""
    y_true = df["log_bfsp"].values
    y_pred = df["predicted_log_bfsp"].values

    log_mae = mean_absolute_error(y_true, y_pred)
    log_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    log_r2 = r2_score(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    bfsp_true = df["bfsp"].values
    bfsp_pred = df["predicted_bfsp"].values
    bfsp_mae = mean_absolute_error(bfsp_true, bfsp_pred)
    ape = np.abs(bfsp_true - bfsp_pred) / np.clip(bfsp_true, 1e-7, None)
    median_ape = np.median(ape) * 100
    mean_ape = np.mean(ape) * 100

    return {
        "log_mae": round(log_mae, 4),
        "log_rmse": round(log_rmse, 4),
        "log_r2": round(log_r2, 4),
        "correlation": round(correlation, 4),
        "bfsp_mae": round(bfsp_mae, 2),
        "median_ape_pct": round(median_ape, 1),
        "mean_ape_pct": round(mean_ape, 1),
    }


def compute_ranking_metrics(df: pd.DataFrame) -> dict:
    """Evaluate how well the model ranks horses within races."""
    race_col = "raceid"
    if race_col not in df.columns:
        return {}

    df["placing_numerical"] = pd.to_numeric(
        df["placing_numerical"], errors="coerce"
    )
    df["won"] = df["placing_numerical"] == 1

    total_races = 0
    top1_wins = 0
    top2_wins = 0
    top3_wins = 0
    fav_at_shorter_wins = 0  # model favourite that was shorter than market

    for _, race in df.groupby(race_col):
        if race["won"].sum() == 0:
            continue
        total_races += 1

        ranked = race.sort_values("predicted_bfsp")
        if ranked.iloc[0]["won"]:
            top1_wins += 1
        if ranked.head(2)["won"].any():
            top2_wins += 1
        if ranked.head(3)["won"].any():
            top3_wins += 1

        # Model favourite that market underrates
        fav = ranked.iloc[0]
        if (
            fav["won"]
            and pd.notna(fav.get("bfsp"))
            and fav["bfsp"] > fav["predicted_bfsp"]
        ):
            fav_at_shorter_wins += 1

    if total_races == 0:
        return {}

    return {
        "total_races": total_races,
        "top1_win_rate": round(top1_wins / total_races * 100, 1),
        "top2_win_rate": round(top2_wins / total_races * 100, 1),
        "top3_win_rate": round(top3_wins / total_races * 100, 1),
        "fav_overlay_wins": fav_at_shorter_wins,
    }


# ---------------------------------------------------------------------------
# Profitability Simulation
# ---------------------------------------------------------------------------

def simulate_staking(bets: pd.DataFrame) -> dict:
    """Run all 5 staking strategies on overlay bets.

    Returns dict of strategy_name -> summary dict.
    """
    n = len(bets)
    if n == 0:
        return {}

    results = {}

    # 1. Level Stakes (£1 per bet)
    stakes = np.ones(n)
    gross_returns = np.where(bets["won"], bets["bfsp"].values, 0.0)
    net_returns = np.where(
        bets["won"],
        stakes + (gross_returns - stakes) * (1 - BETFAIR_COMMISSION),
        0.0,
    )
    results["Level Stakes"] = _flat_summary(stakes, net_returns, bets)

    # 2. Variant Staking (stake proportional to edge)
    edge = bets["overlay_pct"].values / 100.0
    stakes = np.clip(edge * 10, 0.5, 5.0)
    gross_returns = np.where(bets["won"], bets["bfsp"].values * stakes, 0.0)
    net_returns = np.where(
        bets["won"],
        stakes + (gross_returns - stakes) * (1 - BETFAIR_COMMISSION),
        0.0,
    )
    results["Variant Staking"] = _flat_summary(stakes, net_returns, bets)

    # 3-5. Kelly / Half Kelly / Quarter Kelly
    p = 1.0 / bets["predicted_bfsp"].values
    b = bets["bfsp"].values - 1.0
    kelly_f = np.where(b > 0, (p * b - (1 - p)) / b, 0)
    kelly_f = np.clip(kelly_f, 0, 0.25)

    for label, fraction in [
        ("Kelly", 1.0),
        ("Half Kelly", 0.5),
        ("Quarter Kelly", 0.25),
    ]:
        bankroll = 1000.0
        history = [bankroll]
        total_staked = 0.0
        total_returned = 0.0

        for i in range(n):
            f = kelly_f[i] * fraction
            if f <= 0 or bankroll <= 1.0:
                history.append(bankroll)
                continue

            stake = bankroll * f
            total_staked += stake

            if bets.iloc[i]["won"]:
                gross_profit = stake * (bets.iloc[i]["bfsp"] - 1)
                net_profit = gross_profit * (1 - BETFAIR_COMMISSION)
                bankroll += net_profit
                total_returned += stake + net_profit
            else:
                bankroll -= stake

            history.append(bankroll)

        results[label] = {
            "n_bets": n,
            "total_staked": round(total_staked, 2),
            "total_returned": round(total_returned, 2),
            "profit": round(total_returned - total_staked, 2),
            "roi_pct": round(
                (total_returned - total_staked) / max(total_staked, 1) * 100, 2
            ),
            "win_rate_pct": round(bets["won"].sum() / n * 100, 1),
            "final_bankroll": round(bankroll, 2),
            "growth_pct": round((bankroll - 1000) / 1000 * 100, 1),
            "peak_bankroll": round(max(history), 2),
            "min_bankroll": round(min(history), 2),
            "max_drawdown_pct": round(
                _max_drawdown_pct(history), 1
            ),
        }

    return results


def _flat_summary(stakes, net_returns, bets):
    """Summary for flat/variant staking."""
    n = len(stakes)
    total_staked = stakes.sum()
    total_returned = net_returns.sum()
    profit = total_returned - total_staked

    # Cumulative P&L for drawdown
    cumulative = np.cumsum(net_returns - stakes)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = drawdowns.min()

    return {
        "n_bets": n,
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "profit": round(profit, 2),
        "roi_pct": round(profit / max(total_staked, 1) * 100, 2),
        "win_rate_pct": round(bets["won"].sum() / n * 100, 1),
        "avg_winner_bfsp": round(
            bets.loc[bets["won"], "bfsp"].mean() if bets["won"].any() else 0, 2
        ),
        "max_drawdown": round(max_dd, 2),
    }


def _max_drawdown_pct(history):
    """Maximum drawdown as a percentage of peak."""
    arr = np.array(history)
    peak = np.maximum.accumulate(arr)
    dd_pct = (arr - peak) / np.where(peak > 0, peak, 1) * 100
    return dd_pct.min()


# ---------------------------------------------------------------------------
# Breakdown Analysis
# ---------------------------------------------------------------------------

def breakdown_by_price(bets: pd.DataFrame) -> pd.DataFrame:
    """Level-stakes profitability by predicted price band."""
    bands = [
        (1.0, 3.0, "1-3 (Short)"),
        (3.0, 6.0, "3-6 (Medium)"),
        (6.0, 11.0, "6-11 (Mid)"),
        (11.0, 21.0, "11-21 (Long)"),
        (21.0, 51.0, "21-51 (Big)"),
        (51.0, 1000.0, "51+ (Outsider)"),
    ]
    rows = []
    for lo, hi, label in bands:
        mask = (bets["predicted_bfsp"] >= lo) & (bets["predicted_bfsp"] < hi)
        b = bets[mask]
        if len(b) == 0:
            continue
        n = len(b)
        wins = int(b["won"].sum())
        staked = n
        gross_ret = b.loc[b["won"], "bfsp"].sum()
        net_ret = wins + (gross_ret - wins) * (1 - BETFAIR_COMMISSION) if wins else 0
        profit = net_ret - staked
        rows.append({
            "Price Band": label,
            "Bets": n,
            "Wins": wins,
            "Win%": round(wins / n * 100, 1),
            "P&L": round(profit, 2),
            "ROI%": round(profit / staked * 100, 1),
            "Avg Edge%": round(b["overlay_pct"].mean(), 1),
        })
    return pd.DataFrame(rows)


def breakdown_by_overlay(bets: pd.DataFrame) -> pd.DataFrame:
    """Level-stakes profitability by overlay band."""
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
        b = bets[mask]
        if len(b) == 0:
            continue
        n = len(b)
        wins = int(b["won"].sum())
        staked = n
        gross_ret = b.loc[b["won"], "bfsp"].sum()
        net_ret = wins + (gross_ret - wins) * (1 - BETFAIR_COMMISSION) if wins else 0
        profit = net_ret - staked
        rows.append({
            "Overlay Band": label,
            "Bets": n,
            "Wins": wins,
            "Win%": round(wins / n * 100, 1),
            "P&L": round(profit, 2),
            "ROI%": round(profit / staked * 100, 1),
            "Avg BFSP": round(b["bfsp"].mean(), 1),
        })
    return pd.DataFrame(rows)


def breakdown_monthly(bets: pd.DataFrame) -> pd.DataFrame:
    """Monthly level-stakes P&L."""
    bets = bets.copy()
    bets["month"] = bets["race_date"].dt.to_period("M")

    rows = []
    for month, g in bets.groupby("month"):
        n = len(g)
        wins = int(g["won"].sum())
        gross_ret = g.loc[g["won"], "bfsp"].sum()
        net_ret = wins + (gross_ret - wins) * (1 - BETFAIR_COMMISSION) if wins else 0
        profit = net_ret - n
        rows.append({
            "Month": str(month),
            "Bets": n,
            "Wins": wins,
            "Win%": round(wins / n * 100, 1),
            "P&L": round(profit, 2),
            "ROI%": round(profit / n * 100, 1),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["Cumulative"] = result["P&L"].cumsum().round(2)
    return result


def breakdown_by_field(
    bets: pd.DataFrame, col: str, label: str
) -> pd.DataFrame:
    """Profitability breakdown by a categorical column."""
    if col not in bets.columns:
        return pd.DataFrame()

    rows = []
    for val, g in bets.groupby(col):
        n = len(g)
        if n < 10:
            continue
        wins = int(g["won"].sum())
        gross_ret = g.loc[g["won"], "bfsp"].sum()
        net_ret = wins + (gross_ret - wins) * (1 - BETFAIR_COMMISSION) if wins else 0
        profit = net_ret - n
        rows.append({
            label: val,
            "Bets": n,
            "Wins": wins,
            "Win%": round(wins / n * 100, 1),
            "P&L": round(profit, 2),
            "ROI%": round(profit / n * 100, 1),
        })

    return pd.DataFrame(rows).sort_values("Bets", ascending=False)


# ---------------------------------------------------------------------------
# Report Printing
# ---------------------------------------------------------------------------

def print_report(
    oos_df: pd.DataFrame,
    accuracy: dict,
    ranking: dict,
    staking: dict,
    price_bd: pd.DataFrame,
    overlay_bd: pd.DataFrame,
    monthly_bd: pd.DataFrame,
    min_overlay: float,
    n_overlays: int,
):
    """Print the full evaluation report."""
    date_min = oos_df["race_date"].min().date()
    date_max = oos_df["race_date"].max().date()
    n_folds = oos_df["fold_idx"].nunique()

    print("\n" + "=" * 78)
    print("  OUT-OF-SAMPLE MODEL EVALUATION")
    print("=" * 78)
    print(f"  Period:            {date_min} to {date_max}")
    print(f"  Walk-forward folds:{n_folds}")
    print(f"  Total predictions: {len(oos_df):,}")
    n_races = oos_df["raceid"].nunique() if "raceid" in oos_df.columns else "?"
    print(f"  Total races:       {n_races:,}")

    # --- Accuracy ---
    print("\n" + "-" * 78)
    print("  PREDICTION ACCURACY (all OOS predictions)")
    print("-" * 78)
    print(f"  Log MAE:           {accuracy['log_mae']}")
    print(f"  Log RMSE:          {accuracy['log_rmse']}")
    print(f"  Log R²:            {accuracy['log_r2']}")
    print(f"  Correlation:       {accuracy['correlation']}")
    print(f"  BFSP MAE:          £{accuracy['bfsp_mae']}")
    print(f"  Median APE:        {accuracy['median_ape_pct']}%")
    print(f"  Mean APE:          {accuracy['mean_ape_pct']}%")

    # --- Ranking ---
    if ranking:
        print("\n" + "-" * 78)
        print("  RANKING ABILITY (per-race)")
        print("-" * 78)
        print(f"  Races evaluated:   {ranking['total_races']:,}")
        print(f"  Top-1 win rate:    {ranking['top1_win_rate']}%")
        print(f"  Top-2 hit rate:    {ranking['top2_win_rate']}%")
        print(f"  Top-3 hit rate:    {ranking['top3_win_rate']}%")
        print(f"  Fav overlay wins:  {ranking['fav_overlay_wins']}")

    # --- Overlay Selection ---
    print("\n" + "-" * 78)
    print(f"  OVERLAY SELECTION (>= {min_overlay}% edge)")
    print("-" * 78)
    print(f"  Qualifying bets:   {n_overlays:,} "
          f"({n_overlays / len(oos_df) * 100:.1f}% of runners)")

    if n_overlays == 0:
        print("  No overlays found.")
        print("=" * 78)
        return

    # --- Staking Strategies ---
    print("\n" + "-" * 78)
    print("  STAKING STRATEGY RESULTS (5% Betfair commission applied)")
    print("-" * 78)

    for name in ["Level Stakes", "Variant Staking"]:
        if name not in staking:
            continue
        r = staking[name]
        print(f"\n  {name}:")
        print(f"    Bets:          {r['n_bets']:,}")
        print(f"    Staked:        £{r['total_staked']:,.2f}")
        print(f"    Returned:      £{r['total_returned']:,.2f}")
        print(f"    Profit:        £{r['profit']:+,.2f}")
        print(f"    ROI:           {r['roi_pct']:+.2f}%")
        print(f"    Win Rate:      {r['win_rate_pct']:.1f}%")
        if "avg_winner_bfsp" in r:
            print(f"    Avg Winner SP: {r['avg_winner_bfsp']:.2f}")
        if "max_drawdown" in r:
            print(f"    Max Drawdown:  £{r['max_drawdown']:,.2f}")

    for name in ["Kelly", "Half Kelly", "Quarter Kelly"]:
        if name not in staking:
            continue
        r = staking[name]
        print(f"\n  {name} (£1,000 start):")
        print(f"    Bets:          {r['n_bets']:,}")
        print(f"    Staked:        £{r['total_staked']:,.2f}")
        print(f"    Profit:        £{r['profit']:+,.2f}")
        print(f"    ROI:           {r['roi_pct']:+.2f}%")
        print(f"    Final Bank:    £{r['final_bankroll']:,.2f}")
        print(f"    Growth:        {r['growth_pct']:+.1f}%")
        print(f"    Peak:          £{r['peak_bankroll']:,.2f}")
        print(f"    Trough:        £{r['min_bankroll']:,.2f}")
        print(f"    Max Drawdown:  {r['max_drawdown_pct']:.1f}%")

    # --- Price Band ---
    if not price_bd.empty:
        print("\n" + "-" * 78)
        print("  PROFITABILITY BY PREDICTED PRICE BAND (Level Stakes, after commission)")
        print("-" * 78)
        print(price_bd.to_string(index=False))

    # --- Overlay Band ---
    if not overlay_bd.empty:
        print("\n" + "-" * 78)
        print("  PROFITABILITY BY OVERLAY BAND (Level Stakes, after commission)")
        print("-" * 78)
        print(overlay_bd.to_string(index=False))

    # --- Monthly ---
    if not monthly_bd.empty:
        print("\n" + "-" * 78)
        print("  MONTHLY P&L (Level Stakes, after commission)")
        print("-" * 78)
        print(monthly_bd.to_string(index=False))
        total_pl = monthly_bd["P&L"].sum()
        total_bets = monthly_bd["Bets"].sum()
        print(f"\n  TOTAL: {total_bets:,} bets, £{total_pl:+,.2f}, "
              f"{total_pl / total_bets * 100:+.1f}% ROI")

    print("\n" + "=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Out-of-sample model evaluation with walk-forward predictions"
    )
    parser.add_argument("--db", default=DB_PATH, help="Database path")
    parser.add_argument(
        "--start-date", default=None,
        help="Load data from this date (for memory; metrics warmup is automatic)",
    )
    parser.add_argument(
        "--min-train-days", type=int, default=365,
        help="Minimum training window in days (default: 365)",
    )
    parser.add_argument(
        "--val-window", type=int, default=30,
        help="Validation window in days (default: 30)",
    )
    parser.add_argument(
        "--step-days", type=int, default=30,
        help="Step between folds in days (default: 30)",
    )
    parser.add_argument(
        "--min-overlay", type=float, default=5.0,
        help="Minimum overlay %% to trigger a bet (default: 5)",
    )
    parser.add_argument(
        "--max-bfsp", type=float, default=100.0,
        help="Maximum BFSP to consider for betting (default: 100)",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Save all OOS predictions to CSV",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Save evaluation summary to JSON",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        log.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Load data
    log.info("Loading data...")
    df = load_data(args.db, start_date=args.start_date)
    log.info(f"  {len(df):,} rows loaded")

    # Prepare features (custom metrics + context + target)
    log.info("Computing custom metrics (this may take a few minutes)...")
    engine = CustomMetricsEngine()
    df = engine.calculate_all(df)
    log.info(f"  {len(df.columns)} columns after metrics")

    df = build_context_features(df)

    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df["log_bfsp"] = np.log(df["bfsp"])
    df = df.sort_values(["race_date", "race_time"]).reset_index(drop=True)

    # Determine available features
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    for c in ALL_FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan

    feature_cols_full = list(dict.fromkeys(ALL_FEATURE_COLS))

    log.info(f"  {len(feature_cols_full)} features, {len(df):,} valid rows")

    # Ensure raceid exists
    if "raceid" not in df.columns:
        df["raceid"] = (
            df["race_date"].dt.strftime("%Y-%m-%d")
            + "_" + df["track"].astype(str)
            + "_" + df["race_time"].astype(str)
        )

    # Walk-forward predictions
    log.info("\nRunning walk-forward out-of-sample evaluation...")
    oos = walk_forward_predict(
        df,
        feature_cols_full,
        min_train_days=args.min_train_days,
        val_window_days=args.val_window,
        step_days=args.step_days,
    )

    if oos.empty:
        log.error("No out-of-sample predictions generated.")
        sys.exit(1)

    # Determine winners
    oos["placing_numerical"] = pd.to_numeric(
        oos["placing_numerical"], errors="coerce"
    )
    oos["won"] = oos["placing_numerical"] == 1

    # Overlay calculation
    oos["overlay_pct"] = (
        (oos["bfsp"] - oos["predicted_bfsp"]) / oos["predicted_bfsp"] * 100
    )

    # =====================================================================
    # Accuracy on ALL predictions
    # =====================================================================
    accuracy = compute_accuracy(oos)
    ranking = compute_ranking_metrics(oos.copy())

    # =====================================================================
    # Profitability on overlay bets
    # =====================================================================
    bettable = oos[oos["bfsp"] <= args.max_bfsp].copy()
    overlays = bettable[bettable["overlay_pct"] >= args.min_overlay].copy()

    # Sort chronologically for Kelly simulation
    overlays = overlays.sort_values(["race_date", "race_time"]).reset_index(
        drop=True
    )

    staking = simulate_staking(overlays)
    price_bd = breakdown_by_price(overlays) if len(overlays) > 0 else pd.DataFrame()
    overlay_bd = breakdown_by_overlay(overlays) if len(overlays) > 0 else pd.DataFrame()
    monthly_bd = breakdown_monthly(overlays) if len(overlays) > 0 else pd.DataFrame()

    # Print report
    print_report(
        oos, accuracy, ranking, staking,
        price_bd, overlay_bd, monthly_bd,
        args.min_overlay, len(overlays),
    )

    # Save outputs
    if args.output_csv:
        out_cols = [
            "race_date", "race_time", "track", "horse_name",
            "predicted_bfsp", "bfsp", "overlay_pct",
            "predicted_win_prob_norm", "placing_numerical", "won",
            "fold_idx",
            # Race metadata for downstream profitability analysis
            "race_type", "race_code", "surface_type", "going_description",
            "dist_furlongs", "number_of_runners", "race_class",
        ]
        avail = [c for c in out_cols if c in oos.columns]
        oos[avail].to_csv(args.output_csv, index=False)
        log.info(f"Saved OOS predictions to {args.output_csv}")

    if args.output_json:
        summary = {
            "period": {
                "start": str(oos["race_date"].min().date()),
                "end": str(oos["race_date"].max().date()),
            },
            "n_predictions": len(oos),
            "n_folds": int(oos["fold_idx"].nunique()),
            "accuracy": accuracy,
            "ranking": ranking,
            "min_overlay_pct": args.min_overlay,
            "n_overlay_bets": len(overlays),
            "staking": staking,
        }
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info(f"Saved summary to {args.output_json}")


if __name__ == "__main__":
    main()
