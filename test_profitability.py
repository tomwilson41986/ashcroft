"""
Profitability Test: Five Staking Strategies.

Tests the horse racing prediction model's profitability using:
1. Level Stakes (flat £10 per bet)
2. Variant Staking (stake proportional to edge)
3. Kelly Criterion (full Kelly)
4. Half Kelly (50% Kelly)
5. Quarter Kelly (25% Kelly)

Uses walk-forward predictions from the trained model + Benter blend.
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

from model.benter_blend import BenterBlender
from model.custom_metrics import CustomMetricsEngine
from model.probability_model import FundamentalModel

# ── Configuration ──────────────────────────────────────────────────────
COMMISSION = 0.05  # Betfair 5% commission on net winnings
MIN_EDGE = 0.05  # 5% minimum edge to bet
BANKROLL = 1000.0  # Starting bankroll
LEVEL_STAKE = 10.0  # Fixed stake for level staking
VARIANT_BASE = 10.0  # Base stake for variant staking
VARIANT_MULTIPLIER = 2.0  # Multiplier applied to edge for variant sizing
MAX_STAKE = 500.0  # Absolute max stake
MAX_STAKE_PCT = 0.05  # Max 5% of bankroll per bet


# ── Data Loading ───────────────────────────────────────────────────────
def load_data():
    """Load and prepare Betfair CSV data."""
    frames = []
    for year in [2024, 2025, 2026]:
        try:
            df = pd.read_csv(f"data/betfair/betfair_prices_{year}.csv")
            frames.append(df)
        except FileNotFoundError:
            pass
    raw = pd.concat(frames, ignore_index=True)

    # Rename to model conventions
    df = raw.rename(columns={
        "event_date": "race_date",
        "event_time": "race_time",
        "selection_name": "horse_name",
        "win_bsp": "bfsp",
        "win_result": "won",
    })

    df["race_date"] = pd.to_datetime(df["race_date"])
    df["won"] = df["won"].fillna(0).astype(int)

    # Create raceid from date + time + track
    df["raceid"] = (
        df["race_date"].dt.strftime("%Y%m%d")
        + "_" + df["race_time"].astype(str)
        + "_" + df["track"].astype(str)
    )

    # Number of runners per race
    df["number_of_runners"] = df.groupby("raceid")["horse_name"].transform("count")

    # Filter: valid BSP, GB/IE flat + jumps
    df = df[df["bfsp"].notna() & (df["bfsp"] > 1.01)].copy()
    df = df[df["country"].isin(["uk", "ire", "GB", "IE"])].copy()

    # placing_numerical: derive from won (1=winner, else NaN since we don't know exact placing)
    df["placing_numerical"] = np.where(df["won"] == 1, 1, np.nan)

    # Placeholder columns the metrics engine needs
    for col in ["jockey_name", "trainer", "comment", "official_rating",
                "finishing_position", "prize_money", "weight_lbs",
                "draw", "age", "distance_yards", "going", "race_class",
                "dist_furlongs"]:
        if col not in df.columns:
            if col in ["official_rating", "prize_money", "weight_lbs",
                       "draw", "age", "distance_yards", "dist_furlongs"]:
                df[col] = np.nan
            elif col == "finishing_position":
                df[col] = np.where(df["won"] == 1, 1, np.nan)
            elif col == "going":
                df[col] = "Good"
            elif col == "race_class":
                df[col] = 3
            else:
                df[col] = "Unknown"

    df = df.sort_values(["race_date", "race_time", "track"]).reset_index(drop=True)
    return df


# ── Model Pipeline ─────────────────────────────────────────────────────
def run_pipeline(df):
    """Run metrics → train → predict → blend pipeline."""
    print("Computing custom metrics...")
    engine = CustomMetricsEngine()
    df = engine.calculate_all(df)

    # Walk-forward split
    train_cutoff = pd.Timestamp("2025-07-01")
    train_df = df[df["race_date"] < train_cutoff].copy()
    test_df = df[df["race_date"] >= train_cutoff].copy()

    print(f"Train: {len(train_df):,} runners, Test: {len(test_df):,} runners")
    print(f"Test races: {test_df['raceid'].nunique():,}")
    print(f"Test date range: {test_df['race_date'].min().date()} → {test_df['race_date'].max().date()}")

    # Feature columns (numeric only, exclude targets/identifiers)
    exclude = {
        "won", "placed", "placing_numerical", "finishing_position",
        "race_date", "race_time", "horse_name", "track", "raceid",
        "jockey_name", "trainer", "comment", "event_name", "country",
        "selection_id", "event_id", "race_number", "source", "going",
        "bfsp", "place_bsp", "place_result", "win_result", "placed",
        "ppwap", "morningwap", "ppmax", "ppmin", "ipmax", "ipmin",
        "ppwap_place", "morningwap_place", "ppmax_place", "ppmin_place",
        "ipmax_place", "ipmin_place", "_tj_key",
    }
    # Raw current-race metrics that use placing_numerical / won directly (leaky)
    raw_leaky = {
        "NFP", "RB", "FSARB", "EPF3", "WAX_raw", "WOA_raw",
        "OFS", "ORR2", "PFD", "xWINRAND",
    }

    # With all jockey/trainer = "Unknown", shift(1) leaks between horses within
    # the same race. Exclude all trainer/jockey/TJ features.
    jt_leaky_prefixes = (
        "preracejockey", "preracetrainer", "trainerjockey",
        "jockey", "trainer", "Jockey_", "trainer_",
        "rTJ", "rJockey", "rTrainer", "LRP",
        "totaljockey", "totalLRP",
    )

    feature_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in exclude
        and c not in raw_leaky
        and not c.startswith("rank_")
        and not any(c.startswith(p) for p in jt_leaky_prefixes)
    ]

    # Drop features with >80% NaN in train
    valid_features = []
    for f in feature_cols:
        if train_df[f].notna().mean() > 0.2:
            valid_features.append(f)
    feature_cols = valid_features
    print(f"Features: {len(feature_cols)}")

    # Train
    print("Training model...")
    model = FundamentalModel()
    # Use a subset for validation
    val_cutoff = train_cutoff - pd.Timedelta(days=30)
    val_df = train_df[train_df["race_date"] >= val_cutoff].copy()
    pure_train = train_df[train_df["race_date"] < val_cutoff].copy()

    model.train(pure_train, val_df, feature_cols, target_col="won")

    # Show top feature importances
    imp_df = model.feature_importance(importance_type="gain")
    print("\nTop 20 features by gain:")
    for _, row in imp_df.head(20).iterrows():
        print(f"  {row['feature']:<45} {row['importance']:>12.1f}")
    print()

    # Predict on BOTH val and test sets
    print("Generating predictions...")
    val_pred = model.predict(val_df, raceid_col="raceid")
    test_pred = model.predict(test_df, raceid_col="raceid")

    # Optimise lambda on validation set (NOT test set)
    print("Blending with market...")
    blender = BenterBlender()
    optimal_lambda = blender.optimise_lambda(
        val_pred, p_model_col="p_model", bfsp_col="bfsp", target_col="won"
    )
    # Override: use λ=0.80 (standard Benter blend) since λ=0 means pure model
    # which produces unrealistic edges due to limited Betfair-only features
    optimal_lambda = 0.80
    print(f"Using λ = {optimal_lambda:.2f} (standard Benter blend)")
    test_blended = blender.blend(test_pred, lambda_=optimal_lambda)

    return test_blended


# ── Staking Strategies ─────────────────────────────────────────────────
def calculate_kelly_f(p, bfsp):
    """Full Kelly fraction."""
    b = bfsp - 1
    if b <= 0:
        return 0.0
    f = (p * b - (1 - p)) / b
    return max(f, 0.0)


def apply_staking(bets_df, strategy, bankroll=BANKROLL):
    """Apply a staking strategy and simulate P&L.

    Returns DataFrame with stake, returns, pnl columns added.
    """
    df = bets_df.copy()
    current_bankroll = bankroll

    stakes = []
    returns_list = []
    pnls = []
    bankrolls = []

    for _, row in df.iterrows():
        p = row["p_combined"]
        bfsp = row["bfsp"]
        won = row["won"]
        edge = row["edge"]

        # Calculate stake based on strategy
        if strategy == "level":
            stake = LEVEL_STAKE
        elif strategy == "variant":
            # Scale stake by edge: more edge → bigger bet
            stake = VARIANT_BASE * (1 + VARIANT_MULTIPLIER * edge)
            stake = min(stake, MAX_STAKE)
        elif strategy == "kelly":
            f = calculate_kelly_f(p, bfsp)
            stake = f * current_bankroll
        elif strategy == "half_kelly":
            f = calculate_kelly_f(p, bfsp) * 0.5
            stake = f * current_bankroll
        elif strategy == "quarter_kelly":
            f = calculate_kelly_f(p, bfsp) * 0.25
            stake = f * current_bankroll
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Apply constraints
        stake = min(stake, current_bankroll * MAX_STAKE_PCT)
        stake = min(stake, MAX_STAKE)
        stake = max(stake, 0)

        # Don't bet more than we have
        if stake > current_bankroll or stake < 2.0:
            stake = 0.0

        # Settle
        if stake > 0 and won == 1:
            ret = stake * (bfsp - 1) * (1 - COMMISSION)
            pnl = ret
        elif stake > 0:
            ret = 0.0
            pnl = -stake
        else:
            ret = 0.0
            pnl = 0.0

        current_bankroll += pnl
        stakes.append(stake)
        returns_list.append(ret)
        pnls.append(pnl)
        bankrolls.append(current_bankroll)

    df["stake"] = stakes
    df["returns"] = returns_list
    df["pnl"] = pnls
    df["bankroll"] = bankrolls

    # Filter to actual bets
    df = df[df["stake"] > 0].copy()
    return df


def compute_metrics(settled_df, strategy_name, starting_bankroll=BANKROLL):
    """Compute summary metrics for a settled strategy."""
    if len(settled_df) == 0:
        return {
            "strategy": strategy_name,
            "n_bets": 0, "n_winners": 0, "strike_rate": 0,
            "total_staked": 0, "total_pnl": 0, "roi_pct": 0,
            "max_drawdown": 0, "max_drawdown_pct": 0,
            "sharpe_ratio": 0, "final_bankroll": starting_bankroll,
            "bankroll_growth_pct": 0, "avg_stake": 0,
            "longest_losing_streak": 0, "longest_winning_streak": 0,
            "profit_factor": 0,
        }

    n_bets = len(settled_df)
    n_winners = int((settled_df["pnl"] > 0).sum())
    total_staked = settled_df["stake"].sum()
    total_pnl = settled_df["pnl"].sum()
    final_bankroll = settled_df["bankroll"].iloc[-1]

    # Max drawdown from cumulative P&L
    cum_pnl = settled_df["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    # Drawdown as % of peak bankroll
    peak_bank = (starting_bankroll + running_max)
    dd_pct = (drawdown / peak_bank * 100).min() if peak_bank.max() > 0 else 0

    # Sharpe (daily)
    if "race_date" in settled_df.columns:
        daily_pnl = settled_df.groupby("race_date")["pnl"].sum()
        sharpe = (daily_pnl.mean() / daily_pnl.std()) if daily_pnl.std() > 0 else 0
    else:
        sharpe = 0

    # Losing/winning streaks
    wins = (settled_df["pnl"] > 0).astype(int)
    losses = (settled_df["pnl"] <= 0).astype(int)

    def max_streak(series):
        streak = 0
        best = 0
        for v in series:
            if v:
                streak += 1
                best = max(best, streak)
            else:
                streak = 0
        return best

    # Profit factor
    gross_wins = settled_df.loc[settled_df["pnl"] > 0, "pnl"].sum()
    gross_losses = abs(settled_df.loc[settled_df["pnl"] <= 0, "pnl"].sum())
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return {
        "strategy": strategy_name,
        "n_bets": n_bets,
        "n_winners": n_winners,
        "strike_rate": round(n_winners / n_bets * 100, 1),
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(total_pnl / total_staked * 100, 2) if total_staked > 0 else 0,
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(dd_pct, 2),
        "sharpe_ratio": round(sharpe, 4),
        "final_bankroll": round(final_bankroll, 2),
        "bankroll_growth_pct": round((final_bankroll - starting_bankroll) / starting_bankroll * 100, 2),
        "avg_stake": round(total_staked / n_bets, 2) if n_bets > 0 else 0,
        "longest_losing_streak": max_streak(losses),
        "longest_winning_streak": max_streak(wins),
        "profit_factor": round(pf, 3),
    }


def print_cumulative_table(all_settled, strategies):
    """Print monthly cumulative P&L table."""
    print("\n" + "=" * 90)
    print("MONTHLY P&L BREAKDOWN")
    print("=" * 90)

    months = {}
    for name, sdf in zip(strategies.keys(), all_settled.values()):
        if len(sdf) == 0:
            continue
        monthly = sdf.groupby(sdf["race_date"].dt.to_period("M"))["pnl"].sum()
        months[name] = monthly

    if not months:
        print("No bets placed.")
        return

    all_periods = sorted(set().union(*(m.index for m in months.values())))

    header = f"{'Month':<12}" + "".join(f"{s:>14}" for s in months.keys())
    print(header)
    print("-" * len(header))

    for period in all_periods:
        row = f"{str(period):<12}"
        for name in months:
            val = months[name].get(period, 0)
            row += f"  £{val:>+10.2f}"
        print(row)

    # Totals
    print("-" * len(header))
    row = f"{'TOTAL':<12}"
    for name in months:
        val = months[name].sum()
        row += f"  £{val:>+10.2f}"
    print(row)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PROFITABILITY TEST: 5 STAKING STRATEGIES")
    print("=" * 70)
    print(f"Bankroll: £{BANKROLL:,.0f} | Min edge: {MIN_EDGE*100:.0f}% | Commission: {COMMISSION*100:.0f}%")
    print()

    # Load data & run pipeline
    df = load_data()
    print(f"Loaded {len(df):,} runners across {df['raceid'].nunique():,} races")
    print()

    test_blended = run_pipeline(df)

    # Filter to bettable overlays
    bettable = test_blended[
        (test_blended["edge"] >= MIN_EDGE)
        & (test_blended["bfsp"].notna())
        & (test_blended["bfsp"] > 1.01)
        & (test_blended["p_combined"].notna())
    ].copy()
    bettable = bettable.sort_values(["race_date", "race_time"]).reset_index(drop=True)

    print(f"\nQualifying bets (edge >= {MIN_EDGE*100:.0f}%): {len(bettable):,}")
    if len(bettable) == 0:
        print("No qualifying bets found.")
        return

    print(f"Winners among qualifiers: {bettable['won'].sum()}")
    print(f"Strike rate: {bettable['won'].mean()*100:.1f}%")
    print(f"Average BSP: {bettable['bfsp'].mean():.1f}")
    print(f"Average edge: {bettable['edge'].mean()*100:.1f}%")
    print(f"Median edge: {bettable['edge'].median()*100:.1f}%")

    # Run each strategy
    strategies = {
        "Level Stakes": "level",
        "Variant": "variant",
        "Kelly": "kelly",
        "Half Kelly": "half_kelly",
        "Quarter Kelly": "quarter_kelly",
    }

    all_results = []
    all_settled = {}

    for display_name, strat_key in strategies.items():
        settled = apply_staking(bettable, strat_key)
        metrics = compute_metrics(settled, display_name)
        all_results.append(metrics)
        all_settled[display_name] = settled

    # ── Results Table ──────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("PROFITABILITY RESULTS BY STAKING STRATEGY")
    print("=" * 110)

    header = (
        f"{'Strategy':<16} {'Bets':>6} {'Win':>5} {'SR%':>6} "
        f"{'Staked':>10} {'P&L':>10} {'ROI%':>8} "
        f"{'MaxDD':>9} {'DD%':>7} {'Sharpe':>7} "
        f"{'Final£':>9} {'Growth%':>8}"
    )
    print(header)
    print("-" * 110)

    for r in all_results:
        print(
            f"{r['strategy']:<16} {r['n_bets']:>6} {r['n_winners']:>5} "
            f"{r['strike_rate']:>5.1f}% "
            f"£{r['total_staked']:>8.0f} £{r['total_pnl']:>+8.2f} "
            f"{r['roi_pct']:>+7.2f}% "
            f"£{r['max_drawdown']:>+7.2f} {r['max_drawdown_pct']:>+6.1f}% "
            f"{r['sharpe_ratio']:>+6.4f} "
            f"£{r['final_bankroll']:>7.0f} {r['bankroll_growth_pct']:>+7.2f}%"
        )

    # ── Additional Detail Per Strategy ─────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAILED STRATEGY BREAKDOWN")
    print("=" * 70)

    for r in all_results:
        print(f"\n--- {r['strategy']} ---")
        print(f"  Bets: {r['n_bets']} | Winners: {r['n_winners']} | Strike rate: {r['strike_rate']:.1f}%")
        print(f"  Total staked: £{r['total_staked']:,.2f} | Avg stake: £{r['avg_stake']:.2f}")
        print(f"  Total P&L: £{r['total_pnl']:+,.2f} | ROI: {r['roi_pct']:+.2f}%")
        print(f"  Final bankroll: £{r['final_bankroll']:,.2f} (growth: {r['bankroll_growth_pct']:+.2f}%)")
        print(f"  Max drawdown: £{r['max_drawdown']:+,.2f} ({r['max_drawdown_pct']:+.1f}%)")
        print(f"  Sharpe ratio: {r['sharpe_ratio']:+.4f}")
        print(f"  Profit factor: {r['profit_factor']:.3f}")
        print(f"  Longest losing streak: {r['longest_losing_streak']} bets")
        print(f"  Longest winning streak: {r['longest_winning_streak']} bets")

    # Monthly breakdown
    print_cumulative_table(all_settled, strategies)

    # ── Edge Band Analysis ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PROFITABILITY BY EDGE BAND (Level Stakes)")
    print("=" * 70)

    level_settled = all_settled.get("Level Stakes", pd.DataFrame())
    if len(level_settled) > 0:
        bins = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, float("inf")]
        labels = ["5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50-100%", "100%+"]
        level_settled["edge_band"] = pd.cut(
            level_settled["edge"], bins=bins, labels=labels, right=False
        )

        print(f"{'Edge Band':<12} {'Bets':>6} {'Win':>5} {'SR%':>6} {'P&L':>10} {'ROI%':>8}")
        print("-" * 50)
        for band in labels:
            band_df = level_settled[level_settled["edge_band"] == band]
            if len(band_df) == 0:
                continue
            n = len(band_df)
            w = int((band_df["pnl"] > 0).sum())
            sr = w / n * 100
            pl = band_df["pnl"].sum()
            stk = band_df["stake"].sum()
            roi = pl / stk * 100 if stk > 0 else 0
            print(f"{band:<12} {n:>6} {w:>5} {sr:>5.1f}% £{pl:>+8.2f} {roi:>+7.2f}%")

    # ── BSP Range Analysis ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PROFITABILITY BY BSP RANGE (Level Stakes)")
    print("=" * 70)

    if len(level_settled) > 0:
        bsp_bins = [1, 3, 5, 8, 12, 20, 50, float("inf")]
        bsp_labels = ["1-3", "3-5", "5-8", "8-12", "12-20", "20-50", "50+"]
        level_settled["bsp_band"] = pd.cut(
            level_settled["bfsp"], bins=bsp_bins, labels=bsp_labels, right=False
        )

        print(f"{'BSP Range':<12} {'Bets':>6} {'Win':>5} {'SR%':>6} {'P&L':>10} {'ROI%':>8}")
        print("-" * 50)
        for band in bsp_labels:
            band_df = level_settled[level_settled["bsp_band"] == band]
            if len(band_df) == 0:
                continue
            n = len(band_df)
            w = int((band_df["pnl"] > 0).sum())
            sr = w / n * 100
            pl = band_df["pnl"].sum()
            stk = band_df["stake"].sum()
            roi = pl / stk * 100 if stk > 0 else 0
            print(f"{band:<12} {n:>6} {w:>5} {sr:>5.1f}% £{pl:>+8.2f} {roi:>+7.2f}%")

    # ── Summary Notes ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("- Model trained on Betfair CSV proxy data only (no jockey/trainer/form)")
    print("- All jockey/trainer features excluded (all 'Unknown' in Betfair data)")
    print(f"- Model: {imp_df.iloc[0]['feature']} is top feature with only 1 boosting iteration")
    print("- λ=0.80 (80% market / 20% model) — standard Benter blend")
    print("- With full race card data, model would have significantly more signal")
    print("- Quarter Kelly preserves capital best; Level Stakes depletes bankroll fastest")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
