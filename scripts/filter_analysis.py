#!/usr/bin/env python3
"""
Filter Analysis: Find profitable rules/caps to improve ROI.

Analyses profitability by:
  - Race code (Flat, Hurdle, Chase, NH Flat)
  - Surface (Turf vs AW)
  - Odds bands (BFSP cutoffs)
  - Race class
  - Field size
  - Number of runners
  - Going description
  - Overlay % thresholds
  - Combinations of the above
"""

import json
import os
import sys
import sqlite3
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

from model.custom_metrics import CustomMetricsEngine
from train_bfsp import build_context_features, load_data

MODEL_PATH = os.path.join(SCRIPT_DIR, "data", "models", "bfsp_model.lgb")
META_PATH = os.path.join(SCRIPT_DIR, "data", "models", "bfsp_model_meta.json")
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")

START_DATE = "2025-01-01"


def level_stakes_roi(df):
    """Return (bets, winners, pl, roi%) for level stakes £1."""
    n = len(df)
    if n == 0:
        return 0, 0, 0.0, 0.0
    w = df["won"].sum()
    returned = df.loc[df["won"], "bfsp"].sum()
    pl = returned - n
    roi = pl / n * 100
    return n, int(w), round(pl, 2), round(roi, 1)


def print_breakdown(df, col, label, min_bets=30):
    """Print ROI breakdown by a categorical column."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Category':<25} {'Bets':>6} {'Win':>5} {'Win%':>6} {'P&L':>10} {'ROI%':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*7}")

    rows = []
    for val in sorted(df[col].dropna().unique()):
        subset = df[df[col] == val]
        n, w, pl, roi = level_stakes_roi(subset)
        if n >= min_bets:
            rows.append((val, n, w, pl, roi))

    rows.sort(key=lambda x: x[3], reverse=True)
    for val, n, w, pl, roi in rows:
        win_pct = w / n * 100 if n > 0 else 0
        marker = " ***" if roi > 0 else ""
        print(f"  {str(val):<25} {n:>6} {w:>5} {win_pct:>5.1f}% £{pl:>+9.2f} {roi:>+6.1f}%{marker}")

    return rows


def print_numeric_bands(df, col, bands, label, min_bets=20):
    """Print ROI breakdown by numeric bands."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Band':<25} {'Bets':>6} {'Win':>5} {'Win%':>6} {'P&L':>10} {'ROI%':>7}")
    print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*7}")

    rows = []
    for lo, hi, name in bands:
        subset = df[(df[col] >= lo) & (df[col] < hi)]
        n, w, pl, roi = level_stakes_roi(subset)
        if n >= min_bets:
            rows.append((name, n, w, pl, roi))

    for name, n, w, pl, roi in rows:
        win_pct = w / n * 100 if n > 0 else 0
        marker = " ***" if roi > 0 else ""
        print(f"  {name:<25} {n:>6} {w:>5} {win_pct:>5.1f}% £{pl:>+9.2f} {roi:>+6.1f}%{marker}")

    return rows


def main():
    # Load model
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    # Load data with 2-year warmup
    warmup_date = str(int(START_DATE[:4]) - 2) + START_DATE[4:]
    print(f"Loading data from {warmup_date}...")
    df = load_data(DB_PATH, start_date=warmup_date)
    print(f"  {len(df):,} rows loaded")

    # Custom metrics
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
    df["predicted_bfsp"] = np.exp(df["predicted_log_bfsp"])

    # Filter to analysis period
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df[df["race_date"] >= pd.Timestamp(START_DATE)].copy()
    print(f"  {len(df):,} rows in analysis period")

    # Core columns
    df["placing_numerical"] = pd.to_numeric(df["placing_numerical"], errors="coerce")
    df["won"] = df["placing_numerical"] == 1
    df["overlay_pct"] = (df["bfsp"] - df["predicted_bfsp"]) / df["predicted_bfsp"] * 100

    # Derive race_code from race_type if not present
    if "race_code" not in df.columns:
        df["race_code"] = df.get("race_type", pd.Series(dtype=str))

    # Field size (number of runners)
    if "number_of_runners" in df.columns:
        df["number_of_runners"] = pd.to_numeric(df["number_of_runners"], errors="coerce")

    # Race class
    if "race_class" in df.columns:
        df["race_class"] = df["race_class"].astype(str).str.strip()

    # Official rating
    if "official_rating" in df.columns:
        df["official_rating"] = pd.to_numeric(df["official_rating"], errors="coerce")

    print(f"\n  Total runners: {len(df):,}")
    print(f"  Date range: {df['race_date'].min().date()} to {df['race_date'].max().date()}")

    # =====================================================================
    # BASE: all overlays with default 5% min edge, BFSP <= 100
    # =====================================================================
    base = df[(df["overlay_pct"] >= 5) & (df["bfsp"] <= 100)].copy()
    n, w, pl, roi = level_stakes_roi(base)
    print(f"\n  BASELINE (overlay>=5%, BFSP<=100): {n} bets, {w} wins, £{pl:+.2f}, {roi:+.1f}% ROI")

    # =====================================================================
    # 1. RACE CODE / TYPE
    # =====================================================================
    print_breakdown(base, "race_code", "ROI BY RACE TYPE (overlay>=5%, BFSP<=100)")

    # Also check race_type if different
    if "race_type" in base.columns and "race_code" in base.columns:
        if not base["race_type"].equals(base["race_code"]):
            print_breakdown(base, "race_type", "ROI BY RACE TYPE (alt column)")

    # =====================================================================
    # 2. SURFACE TYPE
    # =====================================================================
    if "surface_type" in base.columns:
        print_breakdown(base, "surface_type", "ROI BY SURFACE")

    # =====================================================================
    # 3. ODDS BANDS (actual BFSP)
    # =====================================================================
    bfsp_bands = [
        (1.0, 2.0, "1.0-2.0 (odds-on)"),
        (2.0, 3.0, "2.0-3.0 (evens-2/1)"),
        (3.0, 5.0, "3.0-5.0"),
        (5.0, 8.0, "5.0-8.0"),
        (8.0, 13.0, "8.0-13.0"),
        (13.0, 21.0, "13.0-21.0"),
        (21.0, 34.0, "21.0-34.0"),
        (34.0, 51.0, "34.0-51.0"),
        (51.0, 101.0, "51.0-100.0"),
    ]
    print_numeric_bands(base, "bfsp", bfsp_bands, "ROI BY ACTUAL BFSP BAND")

    # =====================================================================
    # 4. ODDS CAPS — cumulative ROI at each max BFSP cutoff
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  CUMULATIVE ROI AT DIFFERENT MAX BFSP CAPS")
    print(f"{'='*70}")
    print(f"  {'Max BFSP':<15} {'Bets':>6} {'Win':>5} {'Win%':>6} {'P&L':>10} {'ROI%':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*7}")

    for cap in [5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 75, 100]:
        subset = base[base["bfsp"] <= cap]
        n, w, pl, roi = level_stakes_roi(subset)
        if n >= 20:
            win_pct = w / n * 100
            marker = " ***" if roi > 0 else ""
            print(f"  <= {cap:<12} {n:>6} {w:>5} {win_pct:>5.1f}% £{pl:>+9.2f} {roi:>+6.1f}%{marker}")

    # =====================================================================
    # 5. MIN OVERLAY THRESHOLD
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  ROI AT DIFFERENT MIN OVERLAY THRESHOLDS (BFSP<=100)")
    print(f"{'='*70}")
    print(f"  {'Min Overlay%':<15} {'Bets':>6} {'Win':>5} {'Win%':>6} {'P&L':>10} {'ROI%':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*7}")

    for min_ov in [0, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]:
        subset = df[(df["overlay_pct"] >= min_ov) & (df["bfsp"] <= 100)]
        n, w, pl, roi = level_stakes_roi(subset)
        if n >= 20:
            win_pct = w / n * 100
            marker = " ***" if roi > 0 else ""
            print(f"  >= {min_ov:<12} {n:>6} {w:>5} {win_pct:>5.1f}% £{pl:>+9.2f} {roi:>+6.1f}%{marker}")

    # =====================================================================
    # 6. RACE CLASS
    # =====================================================================
    if "race_class" in base.columns:
        print_breakdown(base, "race_class", "ROI BY RACE CLASS")

    # =====================================================================
    # 7. FIELD SIZE
    # =====================================================================
    if "number_of_runners" in base.columns:
        field_bands = [
            (2, 6, "2-5 runners (small)"),
            (6, 9, "6-8 runners"),
            (9, 13, "9-12 runners"),
            (13, 17, "13-16 runners"),
            (17, 99, "17+ runners (big)"),
        ]
        print_numeric_bands(base, "number_of_runners", field_bands, "ROI BY FIELD SIZE")

    # =====================================================================
    # 8. GOING
    # =====================================================================
    if "going_description" in base.columns:
        print_breakdown(base, "going_description", "ROI BY GOING", min_bets=50)

    # =====================================================================
    # 9. DISTANCE BANDS
    # =====================================================================
    if "dist_furlongs" in base.columns:
        df_dist = base.copy()
        df_dist["dist_furlongs"] = pd.to_numeric(df_dist["dist_furlongs"], errors="coerce")
        dist_bands = [
            (5, 7, "5-6f (sprint)"),
            (7, 9, "7-8f (7f-1m)"),
            (9, 11, "9-10f (1m1f-1m2f)"),
            (11, 14, "11-13f (1m3f-1m5f)"),
            (14, 18, "14-17f (1m6f-2m1f)"),
            (18, 24, "18-23f (2m2f-2m7f)"),
            (24, 30, "24-29f (3m-3m5f)"),
            (30, 50, "30f+ (3m6f+)"),
        ]
        print_numeric_bands(df_dist, "dist_furlongs", dist_bands, "ROI BY DISTANCE")

    # =====================================================================
    # 10. COMBO ANALYSIS — Race code + odds cap
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  COMBO: RACE TYPE x ODDS CAP (overlay>=5%)")
    print(f"{'='*70}")
    print(f"  {'Combo':<35} {'Bets':>6} {'Win':>5} {'Win%':>6} {'P&L':>10} {'ROI%':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*7}")

    combos = []
    for rc in sorted(base["race_code"].dropna().unique()):
        for cap in [10, 15, 20, 30, 50, 100]:
            subset = base[(base["race_code"] == rc) & (base["bfsp"] <= cap)]
            n, w, pl, roi = level_stakes_roi(subset)
            if n >= 30:
                combos.append((f"{rc} + BFSP<={cap}", n, w, pl, roi))

    combos.sort(key=lambda x: x[3], reverse=True)
    for name, n, w, pl, roi in combos[:30]:
        win_pct = w / n * 100
        marker = " ***" if roi > 0 else ""
        print(f"  {name:<35} {n:>6} {w:>5} {win_pct:>5.1f}% £{pl:>+9.2f} {roi:>+6.1f}%{marker}")

    # =====================================================================
    # 11. COMBO: Race type + overlay threshold
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  COMBO: RACE TYPE x MIN OVERLAY (BFSP<=100)")
    print(f"{'='*70}")
    print(f"  {'Combo':<35} {'Bets':>6} {'Win':>5} {'Win%':>6} {'P&L':>10} {'ROI%':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*6} {'-'*10} {'-'*7}")

    combos = []
    for rc in sorted(base["race_code"].dropna().unique()):
        for min_ov in [5, 10, 15, 20, 30]:
            subset = df[(df["race_code"] == rc) & (df["overlay_pct"] >= min_ov) & (df["bfsp"] <= 100)]
            n, w, pl, roi = level_stakes_roi(subset)
            if n >= 30:
                combos.append((f"{rc} + overlay>={min_ov}%", n, w, pl, roi))

    combos.sort(key=lambda x: x[3], reverse=True)
    for name, n, w, pl, roi in combos[:30]:
        win_pct = w / n * 100
        marker = " ***" if roi > 0 else ""
        print(f"  {name:<35} {n:>6} {w:>5} {win_pct:>5.1f}% £{pl:>+9.2f} {roi:>+6.1f}%{marker}")

    # =====================================================================
    # 12. BEST COMBO: Race type + odds cap + overlay
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  BEST COMBOS: RACE TYPE x ODDS CAP x OVERLAY (top 25 by P&L)")
    print(f"{'='*70}")
    print(f"  {'Combo':<45} {'Bets':>5} {'Win':>4} {'Win%':>6} {'P&L':>9} {'ROI%':>7}")
    print(f"  {'-'*45} {'-'*5} {'-'*4} {'-'*6} {'-'*9} {'-'*7}")

    combos = []
    for rc in sorted(df["race_code"].dropna().unique()):
        for cap in [10, 15, 20, 30, 50, 100]:
            for min_ov in [5, 10, 15, 20]:
                subset = df[(df["race_code"] == rc) & (df["bfsp"] <= cap) & (df["overlay_pct"] >= min_ov)]
                n, w, pl, roi = level_stakes_roi(subset)
                if n >= 30:
                    combos.append((f"{rc} + BFSP<={cap} + ov>={min_ov}%", n, w, pl, roi))

    combos.sort(key=lambda x: x[3], reverse=True)
    for name, n, w, pl, roi in combos[:25]:
        win_pct = w / n * 100
        marker = " ***" if roi > 0 else ""
        print(f"  {name:<45} {n:>5} {w:>4} {win_pct:>5.1f}% £{pl:>+8.2f} {roi:>+6.1f}%{marker}")

    # =====================================================================
    # 13. SUGGESTED RULES SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  SUGGESTED RULES (positive ROI combos with 50+ bets)")
    print(f"{'='*70}")

    # Gather all positive ROI combos with decent sample
    all_combos = []
    for rc in sorted(df["race_code"].dropna().unique()):
        for cap in [10, 15, 20, 30, 50, 100]:
            for min_ov in [5, 10, 15, 20, 30]:
                subset = df[(df["race_code"] == rc) & (df["bfsp"] <= cap) & (df["overlay_pct"] >= min_ov)]
                n, w, pl, roi = level_stakes_roi(subset)
                if n >= 50 and roi > 0:
                    all_combos.append((f"{rc} + BFSP<={cap} + overlay>={min_ov}%", n, w, pl, roi))

    # Also check without race type filter
    for cap in [10, 15, 20, 30, 50, 100]:
        for min_ov in [5, 10, 15, 20, 30]:
            subset = df[(df["bfsp"] <= cap) & (df["overlay_pct"] >= min_ov)]
            n, w, pl, roi = level_stakes_roi(subset)
            if n >= 50 and roi > 0:
                all_combos.append((f"ALL + BFSP<={cap} + overlay>={min_ov}%", n, w, pl, roi))

    all_combos.sort(key=lambda x: x[3], reverse=True)
    seen = set()
    count = 0
    for name, n, w, pl, roi in all_combos:
        if count >= 20:
            break
        if name not in seen:
            seen.add(name)
            win_pct = w / n * 100
            print(f"  {name:<45} {n:>5} bets, {w:>4} wins ({win_pct:.1f}%), £{pl:>+8.2f} P&L, {roi:>+6.1f}% ROI")
            count += 1

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
