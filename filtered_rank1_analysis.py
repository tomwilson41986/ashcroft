#!/usr/bin/env python3
"""
Filtered Rank 1 Profitability Analysis.

Investigates whether applying big-theme filters to the Rank 1 level-stakes
strategy can produce a profitable edge. Uses the 253k walk-forward OOS
predictions.

Filters tested (big themes only):
  - Overlay only (model says horse is value)
  - Predicted BFSP price bands
  - Field size bands
  - Track type (NH vs Flat vs AW)
  - Combinations of the above

All returns net of 5% Betfair commission.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OOS_CSV = os.path.join(SCRIPT_DIR, "data", "oos_predictions.csv")
COMMISSION = 0.05

# ──────────────────────────────────────────────────────────────────────────
# Track classification (domain knowledge — reliable for UK/IRE racing)
# ──────────────────────────────────────────────────────────────────────────
NH_ONLY_TRACKS = {
    "Aintree", "Bangor", "Cartmel", "Cheltenham", "Exeter", "Fakenham",
    "Fontwell", "Hereford", "Hexham", "Huntingdon", "Kelso", "Ludlow",
    "Market Rasen", "Newton Abbot", "Perth", "Plumpton", "Sedgefield",
    "Stratford", "Taunton", "Uttoxeter", "Warwick", "Wetherby",
    "Wincanton", "Worcester",
    # Irish NH
    "Ballinrobe", "Clonmel", "Downpatrick", "Fairyhouse", "Galway",
    "Gowran Park", "Kilbeggan", "Killarney", "Leopardstown", "Limerick",
    "Listowel", "Naas", "Navan", "Punchestown", "Roscommon", "Sligo",
    "Thurles", "Tipperary", "Tramore", "Wexford",
    # Technically dual-purpose but mostly NH in winter
    "Down Royal", "Cork",
}

FLAT_ONLY_TRACKS = {
    "Ascot", "Bath", "Beverley", "Brighton", "Carlisle", "Catterick",
    "Chester", "Doncaster", "Epsom", "Goodwood", "Hamilton", "Haydock",
    "Leicester", "Newbury", "Newmarket (July)", "Newmarket (Rowley)",
    "Nottingham", "Pontefract", "Redcar", "Ripon", "Salisbury",
    "Sandown", "Thirsk", "Windsor", "Yarmouth", "York",
    # Irish Flat
    "Curragh", "Laytown",
}

AW_TRACKS = {
    "Chelmsford City", "Kempton", "Lingfield", "Newcastle",
    "Southwell", "Wolverhampton",
    # Irish AW
    "Dundalk",
}

# Note: Some dual-purpose tracks (Ayr, Chepstow, Ffos Las, Musselburgh)
# host both codes. We'll classify them separately.
DUAL_PURPOSE = {
    "Ayr", "Chepstow", "Ffos Las", "Musselburgh",
    "Bellewstown",
}


def classify_track(track):
    """Classify track into broad racing code."""
    if track in AW_TRACKS:
        return "All-Weather"
    if track in NH_ONLY_TRACKS:
        return "National Hunt"
    if track in FLAT_ONLY_TRACKS:
        return "Flat"
    if track in DUAL_PURPOSE:
        return "Dual Purpose"
    return "Unknown"


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

    # Field size
    df["field_size"] = df.groupby("race_key")["race_key"].transform("count")

    # Track type
    df["track_type"] = df["track"].apply(classify_track)

    return df


def roi_stats(bets: pd.DataFrame) -> dict:
    """Level stakes £1, net of commission."""
    n = len(bets)
    if n == 0:
        return {"bets": 0, "winners": 0, "win_pct": 0, "pl": 0, "roi": 0}
    w = bets["won"].sum()
    # Net returns: win pays bfsp - 1 profit, minus 5% commission on profit
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
    # Highlight best
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

    # ──────────────────────────────────────────────────────────────────────
    # A. OVERLAY FILTER
    # ──────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    # B. PREDICTED BFSP BAND
    # ──────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    # C. FIELD SIZE
    # ──────────────────────────────────────────────────────────────────────
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
    print_table(rows, "C. RANK 1 BY FIELD SIZE")

    # ──────────────────────────────────────────────────────────────────────
    # D. TRACK TYPE (proxy for race code)
    # ──────────────────────────────────────────────────────────────────────
    rows = []
    for tt in ["Flat", "National Hunt", "All-Weather", "Dual Purpose"]:
        mask = rank1["track_type"] == tt
        s = roi_stats(rank1[mask])
        if s["bets"] > 0:
            rows.append({"Track Type": tt, **s})
    print_table(rows, "D. RANK 1 BY TRACK TYPE (proxy for race code)")

    # ──────────────────────────────────────────────────────────────────────
    # E. TOP INDIVIDUAL TRACKS (min 200 bets)
    # ──────────────────────────────────────────────────────────────────────
    rows = []
    for track in sorted(rank1["track"].unique()):
        mask = rank1["track"] == track
        s = roi_stats(rank1[mask])
        if s["bets"] >= 200:
            rows.append({"Track": track, **s})
    rows.sort(key=lambda x: x["roi"], reverse=True)
    print_table(rows, "E. RANK 1 BY TRACK (≥200 bets)")

    # ──────────────────────────────────────────────────────────────────────
    # F. COMBINED FILTERS — Overlay + Price Band
    # ──────────────────────────────────────────────────────────────────────
    rows = []
    for ov_label, ov_mask in [
        ("All", rank1.index == rank1.index),
        ("Overlays >0%", rank1["overlay_pct"] > 0),
        ("Overlays ≥10%", rank1["overlay_pct"] >= 10),
    ]:
        for lo, hi, pb_label in [
            (1.0, 4.0, "Short (1-4)"),
            (4.0, 8.0, "Medium (4-8)"),
            (8.0, 999, "Long (8+)"),
        ]:
            pb_mask = (rank1["predicted_bfsp"] >= lo) & (rank1["predicted_bfsp"] < hi)
            combined = rank1[ov_mask & pb_mask]
            s = roi_stats(combined)
            if s["bets"] >= 50:
                rows.append({"Overlay": ov_label, "Price": pb_label, **s})
    print_table(rows, "F. RANK 1 — OVERLAY × PRICE BAND")

    # ──────────────────────────────────────────────────────────────────────
    # G. COMBINED FILTERS — Track Type + Price Band
    # ──────────────────────────────────────────────────────────────────────
    rows = []
    for tt in ["Flat", "National Hunt", "All-Weather"]:
        tt_mask = rank1["track_type"] == tt
        for lo, hi, pb_label in [
            (1.0, 4.0, "Short (1-4)"),
            (4.0, 8.0, "Medium (4-8)"),
            (8.0, 999, "Long (8+)"),
        ]:
            pb_mask = (rank1["predicted_bfsp"] >= lo) & (rank1["predicted_bfsp"] < hi)
            combined = rank1[tt_mask & pb_mask]
            s = roi_stats(combined)
            if s["bets"] >= 50:
                rows.append({"Code": tt, "Price": pb_label, **s})
    print_table(rows, "G. RANK 1 — TRACK TYPE × PRICE BAND")

    # ──────────────────────────────────────────────────────────────────────
    # H. COMBINED — Track Type + Overlay + Price
    # ──────────────────────────────────────────────────────────────────────
    rows = []
    for tt in ["Flat", "National Hunt", "All-Weather"]:
        tt_mask = rank1["track_type"] == tt
        for ov_min, ov_label in [(0, ">0%"), (10, "≥10%")]:
            ov_mask = rank1["overlay_pct"] >= ov_min
            for lo, hi, pb_label in [
                (1.0, 4.0, "Short"),
                (4.0, 8.0, "Medium"),
                (8.0, 999, "Long"),
            ]:
                pb_mask = (rank1["predicted_bfsp"] >= lo) & (rank1["predicted_bfsp"] < hi)
                combined = rank1[tt_mask & ov_mask & pb_mask]
                s = roi_stats(combined)
                if s["bets"] >= 50:
                    rows.append({"Code": tt, "Overlay": ov_label, "Price": pb_label, **s})
    rows.sort(key=lambda x: x["roi"], reverse=True)
    print_table(rows, "H. RANK 1 — TRACK TYPE × OVERLAY × PRICE (≥50 bets, sorted by ROI)")

    # ──────────────────────────────────────────────────────────────────────
    # I. COMBINED — Field Size + Price
    # ──────────────────────────────────────────────────────────────────────
    rows = []
    for fs_lo, fs_hi, fs_label in [
        (2, 8, "Small (2-7)"),
        (8, 12, "Medium (8-11)"),
        (12, 99, "Large (12+)"),
    ]:
        fs_mask = (rank1["field_size"] >= fs_lo) & (rank1["field_size"] < fs_hi)
        for lo, hi, pb_label in [
            (1.0, 4.0, "Short (1-4)"),
            (4.0, 8.0, "Medium (4-8)"),
            (8.0, 999, "Long (8+)"),
        ]:
            pb_mask = (rank1["predicted_bfsp"] >= lo) & (rank1["predicted_bfsp"] < hi)
            combined = rank1[fs_mask & pb_mask]
            s = roi_stats(combined)
            if s["bets"] >= 50:
                rows.append({"Field": fs_label, "Price": pb_label, **s})
    rows.sort(key=lambda x: x["roi"], reverse=True)
    print_table(rows, "I. RANK 1 — FIELD SIZE × PRICE (≥50 bets, sorted by ROI)")

    # ──────────────────────────────────────────────────────────────────────
    # J. BEST CANDIDATE RULES — sorted by profit
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"  J. BEST CANDIDATE RULES (≥200 bets, sorted by profit)")
    print(f"{'=' * 78}")

    candidates = []

    # Single filters
    for ov_min in [0, 5, 10, 20]:
        for lo, hi, pb_label in [
            (1.0, 4.0, "Short"),
            (4.0, 6.0, "Med-Short"),
            (4.0, 8.0, "Medium"),
            (6.0, 10.0, "Med-Long"),
            (8.0, 999, "Long"),
            (4.0, 13.0, "4-13"),
            (5.0, 13.0, "5-13"),
        ]:
            mask = (
                (rank1["overlay_pct"] >= ov_min) &
                (rank1["predicted_bfsp"] >= lo) &
                (rank1["predicted_bfsp"] < hi)
            )
            s = roi_stats(rank1[mask])
            if s["bets"] >= 200:
                candidates.append({
                    "Rule": f"Overlay≥{ov_min}%, Price {pb_label}",
                    **s,
                })

    # Track type combos
    for tt in ["Flat", "National Hunt", "All-Weather"]:
        tt_mask = rank1["track_type"] == tt
        for ov_min in [0, 10]:
            for lo, hi, pb_label in [
                (1.0, 999, "All"),
                (4.0, 13.0, "4-13"),
                (4.0, 8.0, "Med"),
            ]:
                mask = (
                    tt_mask &
                    (rank1["overlay_pct"] >= ov_min) &
                    (rank1["predicted_bfsp"] >= lo) &
                    (rank1["predicted_bfsp"] < hi)
                )
                s = roi_stats(rank1[mask])
                if s["bets"] >= 200:
                    candidates.append({
                        "Rule": f"{tt}, Ov≥{ov_min}%, {pb_label}",
                        **s,
                    })

    # Field size combos
    for fs_lo, fs_hi, fs_label in [(2, 8, "Small"), (8, 12, "Med"), (12, 99, "Large")]:
        fs_mask = (rank1["field_size"] >= fs_lo) & (rank1["field_size"] < fs_hi)
        for lo, hi, pb_label in [(4.0, 13.0, "4-13"), (4.0, 8.0, "Med")]:
            mask = (
                fs_mask &
                (rank1["predicted_bfsp"] >= lo) &
                (rank1["predicted_bfsp"] < hi)
            )
            s = roi_stats(rank1[mask])
            if s["bets"] >= 200:
                candidates.append({
                    "Rule": f"Field {fs_label}, Price {pb_label}",
                    **s,
                })

    candidates.sort(key=lambda x: x["pl"], reverse=True)
    cdf = pd.DataFrame(candidates[:30])
    if not cdf.empty:
        print(cdf.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"  SUMMARY & INTERPRETATION")
    print(f"{'=' * 78}")

    profitable = [c for c in candidates if c["pl"] > 0]
    if profitable:
        print(f"\n  {len(profitable)} candidate rule(s) show positive P&L (≥200 bets):")
        for c in profitable[:10]:
            print(f"    {c['Rule']:<45} {c['bets']:>5} bets  "
                  f"£{c['pl']:>+8.2f}  ROI {c['roi']:>+6.2f}%")
    else:
        print("\n  No candidate rules with ≥200 bets achieved positive P&L.")

    print(f"""
  IMPORTANT CAVEATS:
  - These are in-sample filter optimisations on OOS predictions.
    Profitable filters found here may not persist out-of-sample.
  - Filters with <500 bets have high variance and may be noise.
  - Track type is inferred from track name (no race_type column
    in current OOS data). Re-run evaluate_oos.py with the DB to
    get exact race_type/race_code for more precise analysis.
  - Only big-theme filters tested (no cherry-picking individual
    tracks, months, or narrow overlays).
""")

    print("=" * 78)
    print("  END OF FILTERED ANALYSIS")
    print("=" * 78)


if __name__ == "__main__":
    main()
