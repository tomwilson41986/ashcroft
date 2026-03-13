#!/usr/bin/env python3
"""
Filtered Rank 1 Profitability Analysis.

Investigates whether applying big-theme filters to the Rank 1 level-stakes
strategy can produce a profitable edge. Uses 253k walk-forward OOS
predictions enriched with race metadata from the database.

Filters tested (big themes only):
  - Overlay only (model says horse is value)
  - Predicted BFSP price bands
  - Field size bands
  - Race code (Flat, NH, AW)
  - Race type (Handicap, Maiden, etc.)
  - Going description (ground conditions)
  - Combinations of the above

All returns net of 5% Betfair commission.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OOS_CSV = os.path.join(SCRIPT_DIR, "data", "oos_predictions_enriched.csv")
COMMISSION = 0.05


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["bfsp"] = pd.to_numeric(df["bfsp"], errors="coerce")
    df["predicted_bfsp"] = pd.to_numeric(df["predicted_bfsp"], errors="coerce")
    df["won"] = df["won"].astype(str).str.strip().str.lower() == "true"
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.0)].copy()

    # Race key and rank
    df["race_key"] = df["race_date"].astype(str) + "|" + df["race_time"] + "|" + df["track"]
    df["rank"] = df.groupby("race_key")["predicted_bfsp"].rank(method="first").astype(int)

    # Overlay
    df["overlay_pct"] = (df["bfsp"] / df["predicted_bfsp"] - 1) * 100

    # Field size (from DB or computed)
    if "number_of_runners" in df.columns:
        df["field_size"] = pd.to_numeric(df["number_of_runners"], errors="coerce")
    else:
        df["field_size"] = df.groupby("race_key")["race_key"].transform("count")

    # Simplify going descriptions into broad categories
    if "going_description" in df.columns:
        going_map = {
            "Heavy": "Soft/Heavy",
            "Soft To Heavy": "Soft/Heavy",
            "Soft": "Soft/Heavy",
            "Yielding To Soft": "Soft/Heavy",
            "Yielding": "Good/Yielding",
            "Good To Yielding": "Good/Yielding",
            "Good To Soft": "Good/Yielding",
            "Good": "Good",
            "Good To Firm": "Good/Fast",
            "Firm": "Good/Fast",
            "Standard": "Standard (AW)",
            "Standard To Slow": "Standard (AW)",
            "Standard To Fast": "Standard (AW)",
            "Slow": "Standard (AW)",
        }
        df["going_group"] = df["going_description"].map(going_map).fillna("Other")

    # Simplify race types into broad categories
    if "race_type" in df.columns:
        def broad_race_type(rt):
            if pd.isna(rt):
                return "Unknown"
            rt = rt.lower()
            if "handicap" in rt and "chase" in rt:
                return "Handicap Chase"
            if "handicap" in rt and "hurdle" in rt:
                return "Handicap Hurdle"
            if "handicap" in rt and ("nursery" in rt or "flat" not in rt):
                return "Handicap Flat"
            if "chase" in rt:
                return "Non-Hcp Chase"
            if "hurdle" in rt:
                return "Non-Hcp Hurdle"
            if "nh flat" in rt or "bumper" in rt:
                return "NH Flat"
            if "maiden" in rt:
                return "Maiden"
            if "novice" in rt:
                return "Novices"
            if "handicap" in rt:
                return "Handicap Flat"
            return "Other Flat"
        df["race_group"] = df["race_type"].apply(broad_race_type)

    return df


def roi_stats(bets: pd.DataFrame) -> dict:
    """Level stakes £1, net of commission."""
    n = len(bets)
    if n == 0:
        return {"bets": 0, "winners": 0, "win_pct": 0, "pl": 0, "roi": 0}
    w = bets["won"].sum()
    returned = bets.loc[bets["won"]].apply(
        lambda r: 1.0 + (r["bfsp"] - 1) * (1 - COMMISSION), axis=1
    ).sum()
    pl = returned - n
    return {
        "bets": n,
        "winners": int(w),
        "win_pct": round(w / n * 100, 1),
        "pl": round(pl, 2),
        "roi": round(pl / n * 100, 2),
    }


def print_table(rows: list[dict], title: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    print(df.to_string(index=False))
    if len(rows) > 1 and "roi" in df.columns:
        best = df.loc[df["roi"].idxmax()]
        print(f"\n  >> Best: {best.iloc[0]} — ROI {best['roi']:+.2f}%")


def main():
    df = load_data(OOS_CSV)
    rank1 = df[df["rank"] == 1].copy()

    total = roi_stats(rank1)

    print("=" * 78)
    print("  RANK 1 FILTERED PROFITABILITY ANALYSIS")
    print(f"  Period: {df['race_date'].min().date()} to {df['race_date'].max().date()}")
    print(f"  OOS Predictions: {len(df):,} runners, {df['race_key'].nunique():,} races")
    print(f"  Commission: {COMMISSION*100:.0f}% Betfair")
    print("=" * 78)

    print(f"\n  BASELINE — All Rank 1 (no filters):")
    print(f"    Bets: {total['bets']:,}  Winners: {total['winners']:,}  "
          f"Win%: {total['win_pct']:.1f}%  P&L: £{total['pl']:+,.2f}  ROI: {total['roi']:+.2f}%")

    # ──────────────────────────────────────────────────────────────────
    # A. OVERLAY FILTER
    # ──────────────────────────────────────────────────────────────────
    rows = []
    for label, mask in [
        ("All Rank 1", rank1.index == rank1.index),
        ("Overlays only (>0%)", rank1["overlay_pct"] > 0),
        ("Overlays ≥5%", rank1["overlay_pct"] >= 5),
        ("Overlays ≥10%", rank1["overlay_pct"] >= 10),
        ("Overlays ≥20%", rank1["overlay_pct"] >= 20),
        ("Underlays only (<0%)", rank1["overlay_pct"] < 0),
    ]:
        s = roi_stats(rank1[mask])
        rows.append({"Filter": label, **s})
    print_table(rows, "A. RANK 1 BY OVERLAY THRESHOLD")

    # ──────────────────────────────────────────────────────────────────
    # B. PREDICTED BFSP BAND
    # ──────────────────────────────────────────────────────────────────
    rows = []
    for lo, hi, label in [
        (1.0, 2.5, "1.0–2.5 (Strong Fav)"),
        (2.5, 4.0, "2.5–4.0 (Favourite)"),
        (4.0, 6.0, "4.0–6.0 (Contender)"),
        (6.0, 10.0, "6.0–10.0 (Each Way)"),
        (10.0, 999, "10.0+ (Outsider)"),
    ]:
        mask = (rank1["predicted_bfsp"] >= lo) & (rank1["predicted_bfsp"] < hi)
        s = roi_stats(rank1[mask])
        rows.append({"Price Band": label, **s})
    print_table(rows, "B. RANK 1 BY PREDICTED PRICE BAND")

    # ──────────────────────────────────────────────────────────────────
    # C. RACE CODE
    # ──────────────────────────────────────────────────────────────────
    rows = []
    for code in ["Flat", "National Hunt", "All Weather"]:
        mask = rank1["race_code"] == code
        s = roi_stats(rank1[mask])
        if s["bets"] > 0:
            rows.append({"Race Code": code, **s})
    print_table(rows, "C. RANK 1 BY RACE CODE")

    # ──────────────────────────────────────────────────────────────────
    # D. RACE TYPE (broad groups)
    # ──────────────────────────────────────────────────────────────────
    rows = []
    for rt in sorted(rank1["race_group"].unique()):
        mask = rank1["race_group"] == rt
        s = roi_stats(rank1[mask])
        if s["bets"] >= 200:
            rows.append({"Race Type": rt, **s})
    rows.sort(key=lambda x: x["roi"], reverse=True)
    print_table(rows, "D. RANK 1 BY RACE TYPE (≥200 bets)")

    # ──────────────────────────────────────────────────────────────────
    # E. FIELD SIZE
    # ──────────────────────────────────────────────────────────────────
    rows = []
    for lo, hi, label in [
        (2, 6, "Small (2-5)"),
        (6, 10, "Medium (6-9)"),
        (10, 14, "Large (10-13)"),
        (14, 99, "Very Large (14+)"),
    ]:
        mask = (rank1["field_size"] >= lo) & (rank1["field_size"] < hi)
        s = roi_stats(rank1[mask])
        rows.append({"Field Size": label, **s})
    print_table(rows, "E. RANK 1 BY FIELD SIZE")

    # ──────────────────────────────────────────────────────────────────
    # F. GOING
    # ──────────────────────────────────────────────────────────────────
    rows = []
    for going in sorted(rank1["going_group"].unique()):
        mask = rank1["going_group"] == going
        s = roi_stats(rank1[mask])
        if s["bets"] >= 200:
            rows.append({"Going": going, **s})
    rows.sort(key=lambda x: x["roi"], reverse=True)
    print_table(rows, "F. RANK 1 BY GOING (≥200 bets)")

    # ──────────────────────────────────────────────────────────────────
    # G. RACE CODE × PRICE BAND
    # ──────────────────────────────────────────────────────────────────
    rows = []
    for code in ["Flat", "National Hunt", "All Weather"]:
        code_mask = rank1["race_code"] == code
        for lo, hi, pb_label in [
            (1.0, 4.0, "Short (1-4)"),
            (4.0, 8.0, "Medium (4-8)"),
            (8.0, 999, "Long (8+)"),
        ]:
            pb_mask = (rank1["predicted_bfsp"] >= lo) & (rank1["predicted_bfsp"] < hi)
            s = roi_stats(rank1[code_mask & pb_mask])
            if s["bets"] >= 50:
                rows.append({"Code": code, "Price": pb_label, **s})
    print_table(rows, "G. RANK 1 — RACE CODE × PRICE BAND")

    # ──────────────────────────────────────────────────────────────────
    # H. RACE CODE × OVERLAY × PRICE (the key interaction)
    # ──────────────────────────────────────────────────────────────────
    rows = []
    for code in ["Flat", "National Hunt", "All Weather"]:
        code_mask = rank1["race_code"] == code
        for ov_min, ov_label in [(0, "Overlay>0%"), (10, "Overlay≥10%")]:
            ov_mask = rank1["overlay_pct"] >= ov_min
            for lo, hi, pb_label in [
                (1.0, 4.0, "Short"),
                (4.0, 8.0, "Medium"),
                (8.0, 999, "Long"),
            ]:
                pb_mask = (rank1["predicted_bfsp"] >= lo) & (rank1["predicted_bfsp"] < hi)
                s = roi_stats(rank1[code_mask & ov_mask & pb_mask])
                if s["bets"] >= 50:
                    rows.append({"Code": code, "Overlay": ov_label, "Price": pb_label, **s})
    rows.sort(key=lambda x: x["roi"], reverse=True)
    print_table(rows, "H. RANK 1 — RACE CODE × OVERLAY × PRICE (≥50 bets, by ROI)")

    # ──────────────────────────────────────────────────────────────────
    # I. RACE TYPE × OVERLAY (big types only)
    # ──────────────────────────────────────────────────────────────────
    rows = []
    for rt in rank1["race_group"].unique():
        rt_mask = rank1["race_group"] == rt
        for ov_min, ov_label in [(-999, "All"), (0, "Overlay>0%")]:
            ov_mask = rank1["overlay_pct"] >= ov_min
            s = roi_stats(rank1[rt_mask & ov_mask])
            if s["bets"] >= 200:
                rows.append({"Race Type": rt, "Filter": ov_label, **s})
    rows.sort(key=lambda x: x["roi"], reverse=True)
    print_table(rows, "I. RANK 1 — RACE TYPE × OVERLAY (≥200 bets, by ROI)")

    # ──────────────────────────────────────────────────────────────────
    # J. BEST CANDIDATE RULES (≥200 bets)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"  J. BEST CANDIDATE RULES (≥200 bets, sorted by profit)")
    print(f"{'=' * 78}")

    candidates = []

    # Overlay × Price combinations
    for ov_min in [0, 5, 10, 20]:
        for lo, hi, pb_label in [
            (4.0, 8.0, "4-8"),
            (4.0, 13.0, "4-13"),
            (5.0, 13.0, "5-13"),
            (6.0, 13.0, "6-13"),
            (6.0, 10.0, "6-10"),
            (8.0, 999, "8+"),
        ]:
            mask = (
                (rank1["overlay_pct"] >= ov_min) &
                (rank1["predicted_bfsp"] >= lo) &
                (rank1["predicted_bfsp"] < hi)
            )
            s = roi_stats(rank1[mask])
            if s["bets"] >= 200:
                candidates.append({"Rule": f"Overlay≥{ov_min}%, Price {pb_label}", **s})

    # Race code × overlay × price
    for code in ["Flat", "National Hunt", "All Weather"]:
        code_mask = rank1["race_code"] == code
        for ov_min in [0, 10]:
            for lo, hi, pb_label in [
                (1.0, 999, "All"),
                (4.0, 13.0, "4-13"),
                (5.0, 13.0, "5-13"),
                (4.0, 8.0, "4-8"),
            ]:
                mask = (
                    code_mask &
                    (rank1["overlay_pct"] >= ov_min) &
                    (rank1["predicted_bfsp"] >= lo) &
                    (rank1["predicted_bfsp"] < hi)
                )
                s = roi_stats(rank1[mask])
                if s["bets"] >= 200:
                    candidates.append({"Rule": f"{code}, Ov≥{ov_min}%, {pb_label}", **s})

    # Race type × overlay
    for rt in rank1["race_group"].unique():
        rt_mask = rank1["race_group"] == rt
        for ov_min in [0, 10]:
            for lo, hi, pb_label in [(1.0, 999, "All"), (4.0, 13.0, "4-13")]:
                mask = (
                    rt_mask &
                    (rank1["overlay_pct"] >= ov_min) &
                    (rank1["predicted_bfsp"] >= lo) &
                    (rank1["predicted_bfsp"] < hi)
                )
                s = roi_stats(rank1[mask])
                if s["bets"] >= 200:
                    candidates.append({"Rule": f"{rt}, Ov≥{ov_min}%, {pb_label}", **s})

    # Field size × overlay × price
    for fs_lo, fs_hi, fs_label in [(2, 8, "Small"), (8, 12, "Med"), (12, 99, "Large")]:
        fs_mask = (rank1["field_size"] >= fs_lo) & (rank1["field_size"] < fs_hi)
        for lo, hi, pb_label in [(4.0, 13.0, "4-13"), (4.0, 8.0, "4-8")]:
            mask = (
                fs_mask &
                (rank1["overlay_pct"] >= 0) &
                (rank1["predicted_bfsp"] >= lo) &
                (rank1["predicted_bfsp"] < hi)
            )
            s = roi_stats(rank1[mask])
            if s["bets"] >= 200:
                candidates.append({"Rule": f"Field {fs_label}, Ov>0%, {pb_label}", **s})

    # Going × overlay
    for going in rank1["going_group"].unique():
        g_mask = rank1["going_group"] == going
        ov_mask = rank1["overlay_pct"] >= 0
        for lo, hi, pb_label in [(1.0, 999, "All"), (4.0, 13.0, "4-13")]:
            pb_mask = (rank1["predicted_bfsp"] >= lo) & (rank1["predicted_bfsp"] < hi)
            s = roi_stats(rank1[g_mask & ov_mask & pb_mask])
            if s["bets"] >= 200:
                candidates.append({"Rule": f"{going}, Ov>0%, {pb_label}", **s})

    candidates.sort(key=lambda x: x["pl"], reverse=True)
    cdf = pd.DataFrame(candidates[:40])
    if not cdf.empty:
        print(cdf.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"  SUMMARY & INTERPRETATION")
    print(f"{'=' * 78}")

    profitable = [c for c in candidates if c["pl"] > 0]
    if profitable:
        print(f"\n  {len(profitable)} candidate rule(s) show positive P&L (≥200 bets):\n")
        for c in profitable[:15]:
            print(f"    {c['Rule']:<50} {c['bets']:>5} bets  "
                  f"£{c['pl']:>+9.2f}  ROI {c['roi']:>+6.2f}%")
    else:
        print("\n  No candidate rules with ≥200 bets achieved positive P&L.")

    print(f"""
  IMPORTANT CAVEATS:
  - These are in-sample filter optimisations on OOS predictions.
    Profitable filters found here may not persist going forward.
  - Filters with <500 bets have high variance and may reflect noise.
  - Only big-theme filters tested — no cherry-picking of individual
    tracks, months, or narrow thresholds.
  - The OOS predictions themselves are strictly walk-forward with
    zero lookahead bias; only the filter selection is in-sample.
""")

    print("=" * 78)
    print("  END OF FILTERED ANALYSIS")
    print("=" * 78)


if __name__ == "__main__":
    main()
