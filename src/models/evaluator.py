"""Backtesting and performance metrics for the prediction model.

Evaluates model predictions against actual race outcomes at multiple levels:
- Per-race calibration
- Daily P&L simulation
- Aggregate performance across the test period
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class ModelEvaluator:
    """Evaluate model predictions against actual race outcomes."""

    def __init__(self, commission_rate: float = 0.05):
        self.commission_rate = commission_rate

    def evaluate(self, predictions_df: pd.DataFrame,
                 target_col: str = "won",
                 race_id_col: str = "raceid") -> dict:
        """Comprehensive evaluation of model predictions.

        Args:
            predictions_df: DataFrame with columns p_model, p_combined (optional),
                            bfsp, and the target column.
            target_col: Binary outcome column (1 = win).
            race_id_col: Race identifier column.

        Returns:
            Dict of evaluation metrics.
        """
        df = predictions_df.copy()
        y_true = df[target_col].values.astype(int)

        metrics = {}

        # --- Probability calibration ---
        prob_col = "p_combined" if "p_combined" in df.columns else "p_model"
        p = df[prob_col].clip(1e-10, 1 - 1e-10).values

        metrics["log_loss"] = round(log_loss(y_true, p), 6)
        metrics["brier_score"] = round(brier_score_loss(y_true, p), 6)

        # Bolton & Chapman f2 statistic
        # f2 = 1 - [sum(-y * ln(p))] / [sum(-y * ln(1/N))]
        if "number_of_runners" in df.columns:
            uniform_p = 1.0 / df["number_of_runners"].values
            actual_entropy = -np.sum(y_true * np.log(np.clip(p, 1e-10, 1)))
            uniform_entropy = -np.sum(y_true * np.log(np.clip(uniform_p, 1e-10, 1)))
            if uniform_entropy > 0:
                metrics["f2_statistic"] = round(1 - actual_entropy / uniform_entropy, 6)
            else:
                metrics["f2_statistic"] = None
        else:
            metrics["f2_statistic"] = None

        # --- Discrimination ---
        try:
            metrics["auc_roc"] = round(roc_auc_score(y_true, p), 6)
        except ValueError:
            metrics["auc_roc"] = None

        # Top-N accuracy
        if race_id_col in df.columns:
            metrics["top_1_accuracy"] = self._top_n_accuracy(df, prob_col, target_col, race_id_col, n=1)
            metrics["top_2_accuracy"] = self._top_n_accuracy(df, prob_col, target_col, race_id_col, n=2)
            metrics["top_3_accuracy"] = self._top_n_accuracy(df, prob_col, target_col, race_id_col, n=3)

        # --- Market comparison ---
        if "bfsp" in df.columns:
            market_p = (1.0 / df["bfsp"].clip(lower=1.01)).clip(1e-10, 1 - 1e-10).values
            metrics["market_log_loss"] = round(log_loss(y_true, market_p), 6)
            metrics["model_vs_market_improvement"] = round(
                metrics["market_log_loss"] - metrics["log_loss"], 6
            )

        # --- Calibration curve data ---
        metrics["calibration"] = self._calibration_data(p, y_true, n_bins=20)

        return metrics

    def simulate_betting(self, predictions_df: pd.DataFrame,
                         min_edge: float = 0.05,
                         kelly_fraction: float = 0.25,
                         bankroll: float = 1000.0,
                         max_stake_pct: float = 0.02,
                         target_col: str = "won",
                         date_col: str = "date") -> pd.DataFrame:
        """Simulate actual betting performance over the test period.

        For each predicted race:
        1. Identify overlays (edge > min_threshold)
        2. Calculate Kelly stakes
        3. Settle bets using actual outcomes
        4. Track running P&L, bankroll, ROI

        Args:
            predictions_df: Blended predictions with edge, bfsp, etc.
            min_edge: Minimum edge threshold.
            kelly_fraction: Kelly fraction multiplier.
            bankroll: Starting bankroll.
            max_stake_pct: Max bet as fraction of bankroll.
            target_col: Binary outcome column.
            date_col: Date column.

        Returns:
            Daily summary DataFrame.
        """
        df = predictions_df.copy()
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])

        # Filter to bets with edge
        if "edge" not in df.columns:
            return pd.DataFrame()

        bets = df[df["edge"].notna() & (df["edge"] >= min_edge)].copy()
        if bets.empty:
            return pd.DataFrame()

        current_bankroll = bankroll
        results = []

        # Process day by day
        if date_col in bets.columns:
            dates = sorted(bets[date_col].unique())
        else:
            dates = [pd.Timestamp("2000-01-01")]
            bets[date_col] = dates[0]

        total_staked = 0.0
        total_returns = 0.0
        cumulative_pl = 0.0

        for dt in dates:
            day_bets = bets[bets[date_col] == dt]
            day_staked = 0.0
            day_returns = 0.0
            day_winners = 0

            for _, bet in day_bets.iterrows():
                p = bet["p_combined"]
                bfsp = bet["bfsp"]

                if not np.isfinite(bfsp) or bfsp <= 1.0:
                    continue

                # Kelly stake
                b = bfsp - 1.0
                q = 1.0 - p
                f_star = (p * b - q) / b
                if f_star <= 0:
                    continue

                stake = kelly_fraction * f_star * current_bankroll
                max_s = max_stake_pct * current_bankroll
                stake = min(stake, max_s, 500.0)
                if stake < 2.0:
                    continue

                stake = round(stake, 2)
                day_staked += stake

                won = bet[target_col] == 1
                if won:
                    profit = stake * (bfsp - 1.0) * (1 - self.commission_rate)
                    day_returns += stake + profit
                    day_winners += 1
                    current_bankroll += profit
                else:
                    current_bankroll -= stake

            day_pl = day_returns - day_staked
            total_staked += day_staked
            total_returns += day_returns
            cumulative_pl += day_pl

            if day_staked > 0:
                results.append({
                    "date": dt,
                    "n_bets": len(day_bets),
                    "n_winners": day_winners,
                    "total_staked": round(day_staked, 2),
                    "total_returns": round(day_returns, 2),
                    "daily_pl": round(day_pl, 2),
                    "cumulative_pl": round(cumulative_pl, 2),
                    "bankroll": round(current_bankroll, 2),
                    "roi_pct": round((total_returns / total_staked - 1) * 100, 2) if total_staked > 0 else 0.0,
                })

        if not results:
            return pd.DataFrame()

        daily_df = pd.DataFrame(results)

        return daily_df

    def betting_summary(self, daily_df: pd.DataFrame, initial_bankroll: float = 1000.0) -> dict:
        """Compute summary betting metrics from daily simulation results.

        Args:
            daily_df: Output from simulate_betting().
            initial_bankroll: Starting bankroll.

        Returns:
            Dict of summary metrics.
        """
        if daily_df.empty:
            return {"error": "No betting data"}

        total_bets = daily_df["n_bets"].sum()
        total_winners = daily_df["n_winners"].sum()
        total_staked = daily_df["total_staked"].sum()
        total_returns = daily_df["total_returns"].sum()
        final_pl = daily_df["cumulative_pl"].iloc[-1]

        # Drawdown
        cumpl = daily_df["cumulative_pl"].values
        peak = np.maximum.accumulate(cumpl)
        drawdown = peak - cumpl
        max_drawdown = drawdown.max()

        # Sharpe ratio (daily)
        daily_pl = daily_df["daily_pl"].values
        if daily_pl.std() > 0:
            sharpe = daily_pl.mean() / daily_pl.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        # Longest losing streak
        streak = 0
        max_streak = 0
        for pl in daily_pl:
            if pl < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        roi = (total_returns / total_staked - 1) * 100 if total_staked > 0 else 0.0
        strike_rate = total_winners / total_bets * 100 if total_bets > 0 else 0.0

        return {
            "total_bets": int(total_bets),
            "total_winners": int(total_winners),
            "strike_rate_pct": round(strike_rate, 2),
            "total_staked": round(total_staked, 2),
            "total_returns": round(total_returns, 2),
            "net_pl": round(final_pl, 2),
            "roi_pct": round(roi, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 4),
            "longest_losing_streak_days": max_streak,
            "final_bankroll": round(daily_df["bankroll"].iloc[-1], 2),
        }

    def _top_n_accuracy(self, df: pd.DataFrame, prob_col: str,
                        target_col: str, race_id_col: str, n: int) -> float:
        """Fraction of races where the winner is in the model's top-N picks."""
        correct = 0
        total = 0
        for _, race in df.groupby(race_id_col):
            winners = race[race[target_col] == 1]
            if winners.empty:
                continue
            top_n = race.nlargest(n, prob_col)
            if winners.index.isin(top_n.index).any():
                correct += 1
            total += 1
        return round(correct / total, 4) if total > 0 else 0.0

    def _calibration_data(self, predicted: np.ndarray, actual: np.ndarray,
                          n_bins: int = 20) -> list[dict]:
        """Compute calibration curve data: predicted vs actual win rate by bin."""
        order = np.argsort(predicted)
        predicted_sorted = predicted[order]
        actual_sorted = actual[order]

        bin_size = len(predicted_sorted) // n_bins
        if bin_size < 1:
            return []

        calibration = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(predicted_sorted)
            bin_pred = predicted_sorted[start:end]
            bin_actual = actual_sorted[start:end]
            calibration.append({
                "bin": i,
                "mean_predicted": round(float(bin_pred.mean()), 6),
                "mean_actual": round(float(bin_actual.mean()), 6),
                "count": int(len(bin_pred)),
            })
        return calibration

    def plot_calibration(self, predictions_df: pd.DataFrame, output_path: str,
                         target_col: str = "won") -> None:
        """Generate calibration plot: predicted probability vs actual win rate.

        Also plots market calibration for comparison. Saves as PNG.
        """
        if not HAS_MPL:
            return

        prob_col = "p_combined" if "p_combined" in predictions_df.columns else "p_model"
        p = predictions_df[prob_col].clip(1e-10, 1 - 1e-10).values
        y = predictions_df[target_col].values.astype(int)

        cal_data = self._calibration_data(p, y, n_bins=20)
        if not cal_data:
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        pred_vals = [d["mean_predicted"] for d in cal_data]
        actual_vals = [d["mean_actual"] for d in cal_data]
        ax.plot(pred_vals, actual_vals, "o-", label="Model", color="blue")

        # Market calibration if available
        if "bfsp" in predictions_df.columns:
            market_p = (1.0 / predictions_df["bfsp"].clip(lower=1.01)).clip(1e-10, 1 - 1e-10).values
            market_cal = self._calibration_data(market_p, y, n_bins=20)
            if market_cal:
                m_pred = [d["mean_predicted"] for d in market_cal]
                m_actual = [d["mean_actual"] for d in market_cal]
                ax.plot(m_pred, m_actual, "s-", label="Market (BFSP)", color="orange")

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title("Calibration Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def plot_monthly_pnl(self, daily_df: pd.DataFrame, output_path: str) -> None:
        """Generate monthly P&L bar chart and cumulative P&L line chart."""
        if not HAS_MPL or daily_df.empty:
            return

        daily_df = daily_df.copy()
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df["month"] = daily_df["date"].dt.to_period("M")

        monthly = daily_df.groupby("month").agg(
            monthly_pl=("daily_pl", "sum"),
            n_bets=("n_bets", "sum"),
        ).reset_index()
        monthly["month_str"] = monthly["month"].astype(str)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Monthly P&L bars
        colors = ["green" if x >= 0 else "red" for x in monthly["monthly_pl"]]
        ax1.bar(monthly["month_str"], monthly["monthly_pl"], color=colors)
        ax1.set_title("Monthly P&L")
        ax1.set_ylabel("P&L (GBP)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.axhline(y=0, color="black", linewidth=0.5)
        ax1.grid(True, alpha=0.3)

        # Cumulative P&L line
        ax2.plot(daily_df["date"], daily_df["cumulative_pl"], color="blue")
        ax2.set_title("Cumulative P&L")
        ax2.set_ylabel("Cumulative P&L (GBP)")
        ax2.set_xlabel("Date")
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
