#!/usr/bin/env python3
"""
Filtered Profitability Analysis — applies recommended rules and runs
all 5 staking strategies (Level, Variant, Kelly, Half Kelly, Quarter Kelly).

Rules:
  - Max BFSP <= 30
  - Min overlay >= 15%
  - Max runners <= 16
  - Exclude All Weather races
  - Avoid Soft/Heavy going
  - Avoid Maiden Hurdle, Handicap Novices Chase race types
"""

import json
import os
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from model.custom_metrics import CustomMetricsEngine
from train_bfsp import build_context_features, load_data
from profitability_analysis import simulate_staking, analyse_by_price_band, analyse_by_overlay_band

MODEL_PATH = os.path.join(SCRIPT_DIR, "data", "models", "bfsp_model.lgb")
META_PATH = os.path.join(SCRIPT_DIR, "data", "models", "bfsp_model_meta.json")
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")

START_DATE = "2025-01-01"


def apply_filters(df):
    """Apply the recommended profitable rules."""
    n_before = len(df)
    filters_log = []

    # 1. Max BFSP <= 30
    df = df[df["bfsp"] <= 30].copy()
    filters_log.append(f"  BFSP <= 30:              {len(df):>6} rows ({n_before - len(df)} removed)")
    n = len(df)

    # 2. Min overlay >= 15%
    df = df[df["overlay_pct"] >= 15].copy()
    filters_log.append(f"  Overlay >= 15%:          {len(df):>6} rows ({n - len(df)} removed)")
    n = len(df)

    # 3. Max runners <= 16
    df["number_of_runners"] = pd.to_numeric(df["number_of_runners"], errors="coerce")
    df = df[df["number_of_runners"] <= 16].copy()
    filters_log.append(f"  Runners <= 16:           {len(df):>6} rows ({n - len(df)} removed)")
    n = len(df)

    # 4. Exclude All Weather
    df = df[df["race_code"] != "All Weather"].copy()
    filters_log.append(f"  Exclude AW:              {len(df):>6} rows ({n - len(df)} removed)")
    n = len(df)

    # 5. Avoid Soft/Heavy going
    bad_going = ["Soft", "Heavy", "Soft To Heavy"]
    df = df[~df["going_description"].isin(bad_going)].copy()
    filters_log.append(f"  Exclude Soft/Heavy:      {len(df):>6} rows ({n - len(df)} removed)")
    n = len(df)

    # 6. Avoid Maiden Hurdle, Handicap Novices Chase
    bad_types = ["Maiden Hurdle", "Handicap Novices Chase"]
    df = df[~df["race_type"].isin(bad_types)].copy()
    filters_log.append(f"  Exclude bad race types:  {len(df):>6} rows ({n - len(df)} removed)")

    return df, filters_log


def main():
    # Load model
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    # Load data
    warmup_date = str(int(START_DATE[:4]) - 2) + START_DATE[4:]
    print(f"Loading data from {warmup_date}...")
    df = load_data(DB_PATH, start_date=warmup_date)
    print(f"  {len(df):,} rows loaded")

    # Custom metrics + features
    engine = CustomMetricsEngine()
    print("Calculating custom metrics...")
    df = engine.calculate_all(df)
    print("Building context features...")
    df = build_context_features(df)

    # Target
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()
    df["log_bfsp"] = np.log(df["bfsp"])

    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Predict
    print("Generating predictions...")
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

    # Filter to analysis period
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df[df["race_date"] >= pd.Timestamp(START_DATE)].copy()

    # Core columns
    df["placing_numerical"] = pd.to_numeric(df["placing_numerical"], errors="coerce")
    df["won"] = df["placing_numerical"] == 1
    df["overlay_pct"] = (df["bfsp"] - df["predicted_bfsp"]) / df["predicted_bfsp"] * 100

    if "race_code" not in df.columns:
        df["race_code"] = df.get("race_type", pd.Series(dtype=str))

    total_runners = len(df)

    # =====================================================================
    # UNFILTERED BASELINE
    # =====================================================================
    baseline = df[(df["overlay_pct"] >= 5) & (df["bfsp"] <= 100)].copy()

    print(f"\n{'='*70}")
    print(f"  FILTERED PROFITABILITY ANALYSIS")
    print(f"  Period: {START_DATE} to {df['race_date'].max().date()}")
    print(f"  Total runners in period: {total_runners:,}")
    print(f"{'='*70}")

    # Baseline staking
    print(f"\n{'-'*70}")
    print(f"  BASELINE (overlay>=5%, BFSP<=100, no other filters)")
    print(f"{'-'*70}")

    baseline_results = simulate_staking(baseline)
    _print_staking(baseline_results)

    # =====================================================================
    # APPLY FILTERS
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  APPLYING RECOMMENDED FILTERS")
    print(f"{'='*70}")

    filtered, filters_log = apply_filters(df)
    print(f"\n  Starting runners: {total_runners:,}")
    for line in filters_log:
        print(line)
    print(f"\n  Final qualifying bets: {len(filtered):,} ({len(filtered)/total_runners*100:.1f}% of all runners)")

    # Summary stats
    print(f"\n  Winners:          {filtered['won'].sum():,}")
    print(f"  Win rate:         {filtered['won'].mean()*100:.1f}%")
    print(f"  Avg overlay:      {filtered['overlay_pct'].mean():.1f}%")
    print(f"  Avg pred BFSP:    {filtered['predicted_bfsp'].mean():.1f}")
    print(f"  Avg actual BFSP:  {filtered['bfsp'].mean():.1f}")
    print(f"  Median BFSP:      {filtered['bfsp'].median():.1f}")

    # =====================================================================
    # STAKING STRATEGIES — FILTERED
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  STAKING STRATEGY RESULTS (FILTERED)")
    print(f"{'='*70}")

    filtered_results = simulate_staking(filtered)
    _print_staking(filtered_results)

    # =====================================================================
    # IMPROVEMENT COMPARISON
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  BASELINE vs FILTERED COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Strategy':<20} {'Base ROI%':>10} {'Filt ROI%':>10} {'Improvement':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12}")

    for name in ["Level Stakes", "Variant Staking", "Kelly", "Half Kelly", "Quarter Kelly"]:
        base_roi = baseline_results[name]["roi_pct"]
        filt_roi = filtered_results[name]["roi_pct"]
        diff = filt_roi - base_roi
        print(f"  {name:<20} {base_roi:>+9.2f}% {filt_roi:>+9.2f}% {diff:>+11.2f}pp")

    # =====================================================================
    # PRICE BAND BREAKDOWN (FILTERED)
    # =====================================================================
    print(f"\n{'-'*70}")
    print(f"  PROFITABILITY BY PREDICTED PRICE BAND (Level Stakes, Filtered)")
    print(f"{'-'*70}")
    price_df = analyse_by_price_band(filtered)
    if not price_df.empty:
        print(price_df.to_string(index=False))

    # =====================================================================
    # OVERLAY BAND BREAKDOWN (FILTERED)
    # =====================================================================
    print(f"\n{'-'*70}")
    print(f"  PROFITABILITY BY OVERLAY BAND (Level Stakes, Filtered)")
    print(f"{'-'*70}")
    overlay_df = analyse_by_overlay_band(filtered)
    if not overlay_df.empty:
        print(overlay_df.to_string(index=False))

    # =====================================================================
    # MONTHLY BREAKDOWN (FILTERED)
    # =====================================================================
    print(f"\n{'-'*70}")
    print(f"  MONTHLY LEVEL STAKES P&L (FILTERED)")
    print(f"{'-'*70}")

    filtered["month"] = filtered["race_date"].dt.to_period("M")
    monthly = filtered.groupby("month").apply(
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

    total_pl = monthly["P&L"].sum()
    total_bets = monthly["Bets"].sum()
    winning_months = (monthly["P&L"] > 0).sum()
    total_months = len(monthly)
    print(f"\n  TOTAL: {total_bets:,} bets, £{total_pl:+,.2f} P&L, {total_pl/total_bets*100:+.1f}% ROI")
    print(f"  Winning months: {winning_months}/{total_months} ({winning_months/total_months*100:.0f}%)")

    # =====================================================================
    # RACE TYPE BREAKDOWN (FILTERED)
    # =====================================================================
    print(f"\n{'-'*70}")
    print(f"  ROI BY RACE TYPE (FILTERED)")
    print(f"{'-'*70}")
    print(f"  {'Race Type':<25} {'Bets':>6} {'Win':>5} {'Win%':>6} {'P&L':>10} {'ROI%':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*7}")

    for rc in sorted(filtered["race_code"].dropna().unique()):
        subset = filtered[filtered["race_code"] == rc]
        n = len(subset)
        if n < 20:
            continue
        w = subset["won"].sum()
        ret = subset.loc[subset["won"], "bfsp"].sum()
        pl = ret - n
        roi = pl / n * 100
        win_pct = w / n * 100
        marker = " ***" if roi > 0 else ""
        print(f"  {rc:<25} {n:>6} {int(w):>5} {win_pct:>5.1f}% £{pl:>+9.2f} {roi:>+6.1f}%{marker}")

    print(f"\n{'='*70}")


def _print_staking(results):
    """Print all staking strategy results."""
    # Level Stakes & Variant
    for name in ["Level Stakes", "Variant Staking"]:
        r = results[name]
        print(f"\n  {name}:")
        print(f"    Bets:        {r['n_bets']:,}")
        print(f"    Staked:      £{r['total_staked']:,.2f}")
        print(f"    Returned:    £{r['total_returned']:,.2f}")
        print(f"    Profit:      £{r['profit']:,.2f}")
        print(f"    ROI:         {r['roi_pct']:+.2f}%")
        print(f"    Win Rate:    {r['win_rate_pct']:.1f}%")

    # Kelly variants
    for name in ["Kelly", "Half Kelly", "Quarter Kelly"]:
        r = results[name]
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


if __name__ == "__main__":
    main()
