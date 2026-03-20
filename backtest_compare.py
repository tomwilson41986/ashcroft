#!/usr/bin/env python3
"""
Backtest Comparison Report Generator.

Tests multiple rule configurations (edge thresholds, staking methods,
price filters) against a CSV of historical predictions and produces
a comparison table.

Usage:
    python backtest_compare.py predictions.csv
    python backtest_compare.py predictions.csv --output comparison.csv
    python backtest_compare.py predictions.csv --bank 1000 --commission 0.05
"""

import argparse
import csv
import itertools
import sys

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Edge calculation (inline, mirrors src/ultra_betting/rules/edge.py)
# ---------------------------------------------------------------------------

def calculate_edge(predicted_bfsp: float, actual_bfsp: float) -> float:
    """Edge % for a BACK bet: positive when actual > predicted (overlay)."""
    if predicted_bfsp <= 0 or actual_bfsp <= 0:
        return 0.0
    return (actual_bfsp / predicted_bfsp - 1) * 100


# ---------------------------------------------------------------------------
# Kelly criterion (inline, mirrors src/ultra_betting/rules/staking.py)
# ---------------------------------------------------------------------------

def kelly_fraction(win_prob: float, price: float) -> float:
    """Kelly % = (bp - q) / b  where b = price-1, q = 1-p."""
    if price <= 1 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    b = price - 1
    p = win_prob
    q = 1 - p
    return max(0.0, (b * p - q) / b)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

EDGE_THRESHOLDS = [5, 10, 15, 20]

STAKING_METHODS = [
    ("Level £10", "fixed", 1.0),
    ("Kelly 25%", "kelly", 0.25),
    ("Kelly 50%", "kelly", 0.50),
]

PRICE_FILTERS = [
    ("All prices", 0.0, 9999.0),
    ("2.0-10.0", 2.0, 10.0),
    ("3.0-15.0", 3.0, 15.0),
]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(df: pd.DataFrame, edge_min: float, staking_label: str,
             staking_type: str, kelly_frac: float,
             price_lo: float, price_hi: float,
             starting_bank: float, commission: float,
             fixed_stake: float) -> dict:
    """Run a single scenario and return summary metrics."""

    # --- Filter by price band (on actual BFSP) ---
    mask = (df["actual_bfsp"] >= price_lo) & (df["actual_bfsp"] <= price_hi)
    subset = df[mask].copy()

    # --- Filter by edge threshold ---
    subset = subset[subset["edge"] >= edge_min].copy()

    n_bets = len(subset)
    if n_bets == 0:
        return {
            "edge_min": edge_min,
            "staking": staking_label,
            "price_filter": f"{price_lo}-{price_hi}" if price_hi < 9999 else "All",
            "n_bets": 0,
            "strike_rate": 0.0,
            "roi_pct": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }

    # --- Compute stakes and P&L ---
    bank = starting_bank
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    max_dd = 0.0
    total_staked = 0.0
    gross_wins = 0.0
    gross_losses = 0.0
    wins = 0

    for _, row in subset.iterrows():
        price = row["actual_bfsp"]
        won = bool(row["won"])

        # Determine stake
        if staking_type == "fixed":
            stake = fixed_stake
        else:
            # Kelly: implied prob from predicted BFSP
            win_prob = 1.0 / row["predicted_bfsp"]
            kf = kelly_fraction(win_prob, price)
            stake = round(bank * kf * kelly_frac, 2)
            if stake <= 0:
                continue

        total_staked += stake

        if won:
            wins += 1
            gross_profit = stake * (price - 1)
            net_profit = gross_profit * (1 - commission)
            cumulative_pnl += net_profit
            gross_wins += net_profit
            if staking_type == "kelly":
                bank += net_profit
        else:
            cumulative_pnl -= stake
            gross_losses += stake
            if staking_type == "kelly":
                bank -= stake

        # Track drawdown
        if cumulative_pnl > peak_pnl:
            peak_pnl = cumulative_pnl
        dd = peak_pnl - cumulative_pnl
        if dd > max_dd:
            max_dd = dd

    n_settled = wins + (n_bets - wins)  # all bets that ran
    strike_rate = (wins / n_bets * 100) if n_bets > 0 else 0.0
    roi = (cumulative_pnl / total_staked * 100) if total_staked > 0 else 0.0
    pf = (gross_wins / gross_losses) if gross_losses > 0 else (
        float("inf") if gross_wins > 0 else 0.0)

    price_label = f"{price_lo}-{price_hi}" if price_hi < 9999 else "All"

    return {
        "edge_min": edge_min,
        "staking": staking_label,
        "price_filter": price_label,
        "n_bets": n_bets,
        "strike_rate": round(strike_rate, 1),
        "roi_pct": round(roi, 2),
        "total_pnl": round(cumulative_pnl, 2),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_table(rows: list[dict], best_idx: int) -> str:
    """Format results as an aligned text table."""
    headers = ["Edge%", "Staking", "Prices", "Bets", "Strike%",
               "ROI%", "P&L", "MaxDD", "PF"]
    keys = ["edge_min", "staking", "price_filter", "n_bets", "strike_rate",
            "roi_pct", "total_pnl", "max_drawdown", "profit_factor"]

    # Try tabulate first
    try:
        from tabulate import tabulate
        table_data = []
        for i, r in enumerate(rows):
            row_vals = [r[k] for k in keys]
            if i == best_idx:
                row_vals = [f"* {v}" if j == 0 else v for j, v in enumerate(row_vals)]
            table_data.append(row_vals)
        output = tabulate(table_data, headers=headers, tablefmt="simple",
                          floatfmt=".2f", numalign="right", stralign="left")
        return output + f"\n\n* = Best configuration (row {best_idx + 1})"
    except ImportError:
        pass

    # Manual formatting fallback
    col_widths = [max(len(str(headers[j])),
                      max((len(str(r[keys[j]])) for r in rows), default=0))
                  for j in range(len(headers))]

    def fmt_row(vals):
        return "  ".join(str(v).rjust(w) for v, w in zip(vals, col_widths))

    lines = [fmt_row(headers), "-" * (sum(col_widths) + 2 * (len(headers) - 1))]
    for i, r in enumerate(rows):
        marker = " *" if i == best_idx else ""
        lines.append(fmt_row([r[k] for k in keys]) + marker)
    lines.append(f"\n* = Best configuration (row {best_idx + 1})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest comparison across multiple rule configurations.")
    parser.add_argument("csv_file", help="CSV of historical predictions")
    parser.add_argument("--output", "-o", help="Write comparison table to CSV")
    parser.add_argument("--bank", type=float, default=1000.0,
                        help="Starting bank for Kelly staking (default: 1000)")
    parser.add_argument("--commission", type=float, default=0.05,
                        help="Betfair commission rate (default: 0.05 = 5%%)")
    parser.add_argument("--fixed-stake", type=float, default=10.0,
                        help="Fixed stake amount for level staking (default: 10)")
    args = parser.parse_args()

    # --- Load CSV ---
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    # --- Normalise column names ---
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower().replace(" ", "_")
        if cl in ("result", "won", "win"):
            col_map[c] = "won"
        elif cl in ("predicted_bfsp", "pred_bfsp", "model_bfsp"):
            col_map[c] = "predicted_bfsp"
        elif cl in ("actual_bfsp", "bfsp", "actual_sp"):
            col_map[c] = "actual_bfsp"
        elif cl == "date":
            col_map[c] = "date"
        elif cl in ("runner_name", "horse", "runner"):
            col_map[c] = "runner_name"
    df = df.rename(columns=col_map)

    # --- Validate required columns ---
    required = {"predicted_bfsp", "actual_bfsp", "won"}
    missing = required - set(df.columns)
    if missing:
        print(f"Error: CSV missing required columns: {missing}", file=sys.stderr)
        print(f"Found columns: {list(df.columns)}", file=sys.stderr)
        print("Expected: date, runner_name, predicted_bfsp, actual_bfsp, result/won",
              file=sys.stderr)
        sys.exit(1)

    # --- Clean data ---
    df["predicted_bfsp"] = pd.to_numeric(df["predicted_bfsp"], errors="coerce")
    df["actual_bfsp"] = pd.to_numeric(df["actual_bfsp"], errors="coerce")
    df["won"] = df["won"].apply(
        lambda x: x in (1, True, "1", "True", "true", "yes", "Yes", "Y", "y", "Won")
        if not isinstance(x, (bool, np.bool_)) else bool(x)
    )
    df = df.dropna(subset=["predicted_bfsp", "actual_bfsp"])
    df = df[(df["predicted_bfsp"] > 1.0) & (df["actual_bfsp"] > 1.0)].copy()

    # --- Compute edge for every row ---
    df["edge"] = df.apply(
        lambda r: calculate_edge(r["predicted_bfsp"], r["actual_bfsp"]), axis=1)

    print(f"Loaded {len(df)} predictions from {args.csv_file}")
    print(f"  Winners: {df['won'].sum()}  |  "
          f"Strike rate: {df['won'].mean() * 100:.1f}%")
    print(f"  Bank: £{args.bank:.0f}  |  Commission: {args.commission * 100:.1f}%")
    print()

    # --- Run all scenario combinations ---
    results = []
    for edge_min, (stk_label, stk_type, kelly_f), (pf_label, pf_lo, pf_hi) in \
            itertools.product(EDGE_THRESHOLDS, STAKING_METHODS, PRICE_FILTERS):
        r = simulate(df, edge_min, stk_label, stk_type, kelly_f,
                     pf_lo, pf_hi, args.bank, args.commission, args.fixed_stake)
        r["price_filter"] = pf_label  # use friendly label
        results.append(r)

    # --- Find best configuration by ROI (with at least 10 bets) ---
    viable = [(i, r) for i, r in enumerate(results) if r["n_bets"] >= 10]
    if viable:
        best_idx = max(viable, key=lambda x: x[1]["roi_pct"])[0]
    else:
        best_idx = max(range(len(results)),
                       key=lambda i: results[i]["roi_pct"])

    # --- Output ---
    print("=" * 80)
    print("BACKTEST COMPARISON REPORT")
    print("=" * 80)
    print()
    print(format_table(results, best_idx))
    print()

    best = results[best_idx]
    print(f"\nBest configuration:  Edge >= {best['edge_min']}%  |  "
          f"{best['staking']}  |  Prices: {best['price_filter']}")
    print(f"  -> {best['n_bets']} bets, {best['strike_rate']}% strike rate, "
          f"ROI {best['roi_pct']}%, P&L £{best['total_pnl']:.2f}")

    # --- Optional CSV output ---
    if args.output:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output, index=False)
        print(f"\nComparison table written to {args.output}")


if __name__ == "__main__":
    main()
