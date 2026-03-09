"""
Model Evaluation and Backtesting.

Evaluates model predictions against actual race outcomes at multiple levels:
- Per-race: calibration within individual races
- Per-day: daily P&L simulation
- Overall: aggregate performance across the test period

Provides probability calibration metrics, discrimination metrics,
and simulated betting performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


class ModelEvaluator:
    """Evaluate model predictions against actual race outcomes.

    Computes probability calibration, discrimination, and betting
    performance metrics. Simulates P&L with Betfair commission.
    """

    def __init__(self, commission_rate: float = 0.05):
        self.commission_rate = commission_rate

    def evaluate(self, predictions_df: pd.DataFrame) -> dict:
        """Comprehensive evaluation of model predictions.

        Args:
            predictions_df: DataFrame with columns:
                - won: binary outcome (1=win)
                - p_model or p_combined: predicted probability
                - bfsp: actual Betfair Starting Price
                - raceid: race identifier (optional)

        Returns:
            Dict of all evaluation metrics.
        """
        df = predictions_df.copy()

        # Determine probability column
        p_col = "p_combined" if "p_combined" in df.columns else "p_model"
        if p_col not in df.columns:
            return {"error": "No probability column found"}

        y_true = df["won"].astype(float)
        y_pred = df[p_col].fillna(0.1).clip(1e-7, 1 - 1e-7)

        results = {}

        # --- Probability Calibration ---
        results["log_loss"] = round(log_loss(y_true, y_pred), 6)
        results["brier_score"] = round(brier_score_loss(y_true, y_pred), 6)

        # F2 statistic (Bolton & Chapman goodness-of-fit)
        # f2 = 1 - [sum(-outcome * ln(p))] / [sum(-outcome * ln(1/N))]
        if "number_of_runners" in df.columns:
            uniform_prob = 1.0 / df["number_of_runners"].replace(0, np.nan)
            numer = -(y_true * np.log(y_pred.clip(1e-10))).sum()
            denom = -(y_true * np.log(uniform_prob.clip(1e-10))).sum()
            results["f2_statistic"] = round(
                1 - (numer / denom) if denom > 0 else 0, 6
            )

        # --- Discrimination ---
        try:
            results["auc_roc"] = round(roc_auc_score(y_true, y_pred), 6)
        except ValueError:
            results["auc_roc"] = None

        # Top-N accuracy
        if "raceid" in df.columns:
            results["top_1_accuracy"] = self._top_n_accuracy(df, p_col, n=1)
            results["top_2_accuracy"] = self._top_n_accuracy(df, p_col, n=2)
            results["top_3_accuracy"] = self._top_n_accuracy(df, p_col, n=3)

            # Favourite accuracy: when model picks favourite, how often wins?
            results["favourite_accuracy"] = self._favourite_accuracy(
                df, p_col
            )

        # --- Calibration Analysis ---
        results["calibration_table"] = self._calibration_table(
            y_true, y_pred
        )

        # Overround analysis
        if "raceid" in df.columns:
            prob_sums = df.groupby("raceid")[p_col].sum()
            results["avg_prob_sum"] = round(prob_sums.mean(), 4)
            results["median_prob_sum"] = round(prob_sums.median(), 4)

        # --- Betting Performance ---
        if "bfsp" in df.columns and "edge" in df.columns:
            betting = self._betting_metrics(df, p_col)
            results.update(betting)

        # --- Comparison Baselines ---
        if "bfsp" in df.columns:
            baselines = self._baseline_comparisons(df, y_true, p_col)
            results.update(baselines)

        # --- Feature Importance ---
        results["n_predictions"] = len(df)
        results["n_winners"] = int(y_true.sum())
        results["base_rate"] = round(y_true.mean(), 6)

        return results

    def simulate_betting(
        self,
        predictions_df: pd.DataFrame,
        min_edge: float = 0.05,
        kelly_fraction: float = 0.25,
        bankroll: float = 1000.0,
    ) -> pd.DataFrame:
        """Simulate actual betting performance over the test period.

        Args:
            predictions_df: DataFrame with p_combined, bfsp, won, edge.
            min_edge: Minimum edge threshold for placing bets.
            kelly_fraction: Fraction of Kelly criterion for sizing.
            bankroll: Starting bankroll in GBP.

        Returns:
            Daily summary DataFrame with date, n_bets, n_winners,
            total_staked, total_returns, daily_pl, cumulative_pl,
            bankroll, roi_pct.
        """
        df = predictions_df.copy()

        p_col = "p_combined" if "p_combined" in df.columns else "p_model"

        # Filter to bettable runners
        bets = df[df.get("edge", pd.Series(dtype=float)) >= min_edge].copy()

        if len(bets) == 0:
            return pd.DataFrame()

        # Calculate stakes
        p = bets[p_col].clip(1e-7, 1 - 1e-7)
        b = bets["bfsp"] - 1
        kelly_f = ((p * b) - (1 - p)) / b
        kelly_f = kelly_f.clip(lower=0)
        stake = kelly_f * kelly_fraction * bankroll
        stake = stake.clip(lower=2.0, upper=min(bankroll * 0.02, 500.0))

        bets["stake"] = stake

        # Settle bets
        bets["returns"] = np.where(
            bets["won"] == 1,
            bets["stake"] * bets["bfsp"] * (1 - self.commission_rate),
            0,
        )
        bets["pnl"] = bets["returns"] - bets["stake"]

        # Aggregate by date
        if "race_date" not in bets.columns:
            bets["race_date"] = pd.Timestamp("today")

        daily = (
            bets.groupby("race_date")
            .agg(
                n_bets=("stake", "count"),
                n_winners=("won", "sum"),
                total_staked=("stake", "sum"),
                total_returns=("returns", "sum"),
                daily_pl=("pnl", "sum"),
            )
            .reset_index()
        )

        daily["cumulative_pl"] = daily["daily_pl"].cumsum()
        daily["bankroll"] = bankroll + daily["cumulative_pl"]
        daily["roi_pct"] = (
            daily["cumulative_pl"] / daily["total_staked"].cumsum() * 100
        )

        return daily

    def _top_n_accuracy(
        self, df: pd.DataFrame, p_col: str, n: int
    ) -> float:
        """What % of actual winners are in the model's top N picks per race."""
        total_races = 0
        correct = 0

        for _, race in df.groupby("raceid"):
            if race["won"].sum() == 0:
                continue
            total_races += 1
            top_n = race.nlargest(n, p_col)
            if top_n["won"].sum() > 0:
                correct += 1

        return round(correct / total_races, 4) if total_races > 0 else 0

    def _favourite_accuracy(self, df: pd.DataFrame, p_col: str) -> float:
        """When model picks favourite, how often does it win."""
        total = 0
        correct = 0

        for _, race in df.groupby("raceid"):
            if len(race) == 0:
                continue
            fav = race.loc[race[p_col].idxmax()]
            total += 1
            if fav["won"] == 1:
                correct += 1

        return round(correct / total, 4) if total > 0 else 0

    def _calibration_table(
        self, y_true: pd.Series, y_pred: pd.Series, n_bins: int = 10
    ) -> list[dict]:
        """Bin predictions and compare predicted vs actual win rate."""
        df = pd.DataFrame({"true": y_true, "pred": y_pred})
        df["bin"] = pd.qcut(df["pred"], n_bins, duplicates="drop")

        table = []
        for bin_label, group in df.groupby("bin", observed=True):
            table.append(
                {
                    "bin": str(bin_label),
                    "count": len(group),
                    "avg_predicted": round(group["pred"].mean(), 4),
                    "actual_win_rate": round(group["true"].mean(), 4),
                }
            )
        return table

    def _betting_metrics(self, df: pd.DataFrame, p_col: str) -> dict:
        """Compute betting performance metrics."""
        # Simulate with default parameters
        daily = self.simulate_betting(df)

        if len(daily) == 0:
            return {
                "betting_n_bets": 0,
                "betting_roi_pct": 0,
            }

        total_staked = daily["total_staked"].sum()
        total_pl = daily["cumulative_pl"].iloc[-1]
        total_bets = daily["n_bets"].sum()
        total_winners = daily["n_winners"].sum()

        metrics = {
            "betting_n_bets": int(total_bets),
            "betting_n_winners": int(total_winners),
            "betting_strike_rate": (
                round(total_winners / total_bets, 4) if total_bets > 0 else 0
            ),
            "betting_total_staked": round(total_staked, 2),
            "betting_total_pl": round(total_pl, 2),
            "betting_roi_pct": (
                round(total_pl / total_staked * 100, 2)
                if total_staked > 0
                else 0
            ),
        }

        # Max drawdown
        cumulative = daily["cumulative_pl"]
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        metrics["max_drawdown"] = round(drawdown.min(), 2)

        # Sharpe ratio (daily)
        if len(daily) > 1:
            daily_returns = daily["daily_pl"]
            metrics["sharpe_ratio"] = round(
                daily_returns.mean() / daily_returns.std()
                if daily_returns.std() > 0
                else 0,
                4,
            )
        else:
            metrics["sharpe_ratio"] = 0

        # Longest losing streak
        if total_bets > 0:
            metrics["longest_losing_streak"] = self._longest_losing_streak(
                daily
            )

        return metrics

    @staticmethod
    def _longest_losing_streak(daily: pd.DataFrame) -> int:
        """Compute longest streak of losing days."""
        losing = (daily["daily_pl"] < 0).astype(int)
        streak = 0
        max_streak = 0
        for val in losing:
            if val:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def _baseline_comparisons(
        self, df: pd.DataFrame, y_true: pd.Series, p_col: str
    ) -> dict:
        """Compare model vs market vs random baselines."""
        results = {}

        # Market baseline: BFSP implied probability
        market_prob = (1.0 / df["bfsp"].replace(0, np.nan)).clip(1e-7, 1 - 1e-7)
        y_pred = df[p_col].fillna(0.1).clip(1e-7, 1 - 1e-7)
        valid_market = market_prob.notna()

        if valid_market.sum() > 0:
            results["market_logloss"] = round(
                log_loss(
                    y_true[valid_market], market_prob[valid_market]
                ),
                6,
            )
            results["model_vs_market_improvement"] = round(
                results["market_logloss"]
                - log_loss(
                    y_true[valid_market],
                    y_pred[valid_market],
                ),
                6,
            )

        # Random baseline
        if "number_of_runners" in df.columns:
            random_prob = (
                1.0 / df["number_of_runners"].replace(0, np.nan)
            ).clip(1e-7, 1 - 1e-7)
            valid_random = random_prob.notna()
            if valid_random.sum() > 0:
                results["random_logloss"] = round(
                    log_loss(
                        y_true[valid_random], random_prob[valid_random]
                    ),
                    6,
                )

        return results
