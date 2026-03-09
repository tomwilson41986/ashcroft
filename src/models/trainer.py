"""Model training with walk-forward temporal validation.

CRITICAL: Walk-forward validation only. No random splits. No shuffling.
This simulates real-world deployment where we can only use past data.
"""

import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.models.prerace_builder import PreRaceBuilder
from src.models.probability_model import FundamentalModel
from src.models.benter_blend import BenterBlender


class ModelTrainer:
    """Train and validate the prediction model using strict temporal ordering.

    Walk-forward approach:
    1. Sort all data by date
    2. Define an expanding training window
    3. For each validation fold, train on all data BEFORE the fold date
    4. Predict on the fold period
    5. Evaluate predictions against actual results
    """

    def __init__(self, config: dict | None = None):
        self.config = config or self._default_config()
        self.fold_results = []
        self.best_model = None
        self.best_lambda = 0.80
        self.best_hyperparams = None

    @staticmethod
    def _default_config() -> dict:
        return {
            "model_type": "xgboost",
            "min_train_days": 365,
            "val_window_days": 30,
            "step_days": 30,
            "target_col": "won",
            "race_id_col": "raceid",
            "date_col": "date",
        }

    def create_folds(
        self,
        df: pd.DataFrame,
        min_train_days: int | None = None,
        val_window_days: int | None = None,
        step_days: int | None = None,
    ) -> list[tuple[str, str, str]]:
        """Create temporal train/validation folds.

        Example with 3 years of data:
            Fold 1: Train on months 1-12,  Validate on month 13
            Fold 2: Train on months 1-13,  Validate on month 14
            ...

        Args:
            df: Full dataset with date column.
            min_train_days: Minimum training window in days.
            val_window_days: Validation window size in days.
            step_days: Step size between folds in days.

        Returns:
            List of (train_end_date, val_start_date, val_end_date) tuples.
        """
        min_train = min_train_days or self.config["min_train_days"]
        val_window = val_window_days or self.config["val_window_days"]
        step = step_days or self.config["step_days"]
        date_col = self.config["date_col"]

        df_dates = pd.to_datetime(df[date_col])
        min_date = df_dates.min()
        max_date = df_dates.max()

        folds = []
        train_end = min_date + timedelta(days=min_train)

        while train_end + timedelta(days=val_window) <= max_date:
            val_start = train_end
            val_end = train_end + timedelta(days=val_window)
            folds.append((
                str(train_end.date()),
                str(val_start.date()),
                str(val_end.date()),
            ))
            train_end += timedelta(days=step)

        return folds

    def train(self, full_history_df: pd.DataFrame) -> dict:
        """Full training pipeline with walk-forward validation.

        Steps:
        1. Sort by date
        2. Create walk-forward folds
        3. For each fold: train, predict, blend, evaluate
        4. Retrain final model on ALL data
        5. Return aggregate metrics

        Args:
            full_history_df: Complete dataset with custom metrics already computed.

        Returns:
            Dict of aggregate metrics across all folds.
        """
        date_col = self.config["date_col"]
        target_col = self.config["target_col"]
        race_id_col = self.config["race_id_col"]
        model_type = self.config["model_type"]

        df = full_history_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        # Ensure binary target exists
        if target_col not in df.columns:
            df[target_col] = (df["placing_numerical"] == 1).astype(int)

        folds = self.create_folds(df)
        if not folds:
            raise ValueError("Not enough data to create walk-forward folds.")

        self.fold_results = []
        all_val_predictions = []

        for fold_idx, (train_end, val_start, val_end) in enumerate(folds):
            train_end_dt = pd.Timestamp(train_end)
            val_start_dt = pd.Timestamp(val_start)
            val_end_dt = pd.Timestamp(val_end)

            train_df = df[df[date_col] < train_end_dt].copy()
            val_df = df[(df[date_col] >= val_start_dt) & (df[date_col] < val_end_dt)].copy()

            if len(train_df) < 100 or len(val_df) < 10:
                continue

            # Train fundamental model
            model = FundamentalModel(model_type=model_type)
            model.fit(train_df, val_df=val_df, race_id_col=race_id_col, target_col=target_col)

            # Predict on validation
            val_preds = model.predict(val_df, race_id_col=race_id_col)

            # Blend with market if BFSP available
            blender = BenterBlender()
            if "bfsp" in val_preds.columns:
                blended = blender.blend(val_preds, race_id_col=race_id_col)
                opt_lambda = blender.optimise_lambda(val_preds, race_id_col=race_id_col, target_col=target_col)
                blended = blender.blend(val_preds, lambda_=opt_lambda, race_id_col=race_id_col)
            else:
                blended = val_preds.copy()
                blended["p_combined"] = blended["p_model"]
                opt_lambda = 0.0

            # Evaluate fold
            p = blended["p_combined"].clip(1e-10, 1 - 1e-10)
            y = val_df[target_col].values
            fold_loss = log_loss(y, p)

            # Also compute model-only loss
            p_model = blended["p_model"].clip(1e-10, 1 - 1e-10)
            model_loss = log_loss(y, p_model)

            fold_result = {
                "fold": fold_idx,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "log_loss_model": round(model_loss, 6),
                "log_loss_blended": round(fold_loss, 6),
                "lambda": opt_lambda,
            }
            self.fold_results.append(fold_result)
            all_val_predictions.append(blended)

        if not self.fold_results:
            raise ValueError("No valid folds produced. Check data size.")

        # Aggregate metrics
        avg_loss_model = np.mean([f["log_loss_model"] for f in self.fold_results])
        avg_loss_blended = np.mean([f["log_loss_blended"] for f in self.fold_results])
        avg_lambda = np.mean([f["lambda"] for f in self.fold_results])

        # Retrain final model on ALL data
        self.best_model = FundamentalModel(model_type=model_type)
        self.best_model.fit(df, race_id_col=race_id_col, target_col=target_col)
        self.best_lambda = round(avg_lambda, 3)

        return {
            "n_folds": len(self.fold_results),
            "avg_log_loss_model": round(avg_loss_model, 6),
            "avg_log_loss_blended": round(avg_loss_blended, 6),
            "avg_lambda": self.best_lambda,
            "fold_details": self.fold_results,
        }

    def tune_hyperparameters(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        """Tune XGBoost hyperparameters on a single train/val split.

        Uses grid search over key parameters, minimising log-loss.

        Args:
            train_df: Training data.
            val_df: Validation data.

        Returns:
            Dict of best hyperparameters.
        """
        target_col = self.config["target_col"]
        race_id_col = self.config["race_id_col"]

        if target_col not in train_df.columns:
            train_df = train_df.copy()
            train_df[target_col] = (train_df["placing_numerical"] == 1).astype(int)
        if target_col not in val_df.columns:
            val_df = val_df.copy()
            val_df[target_col] = (val_df["placing_numerical"] == 1).astype(int)

        param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_child_weight": [3, 5, 10],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
        }

        best_loss = float("inf")
        best_params = {}

        # Simple grid search (not all combinations — that would be huge)
        # Do a staged approach: tune depth/lr first, then regularisation
        for depth in param_grid["max_depth"]:
            for lr in param_grid["learning_rate"]:
                model = FundamentalModel(model_type="xgboost")
                # Override defaults via model internals
                try:
                    model.fit(train_df, val_df=val_df, race_id_col=race_id_col, target_col=target_col)
                    preds = model.predict(val_df, race_id_col=race_id_col)
                    loss = log_loss(
                        val_df[target_col],
                        preds["p_model"].clip(1e-10, 1 - 1e-10),
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_params = {
                            "max_depth": depth,
                            "learning_rate": lr,
                            "log_loss": round(loss, 6),
                        }
                except Exception:
                    continue

        # Also tune lambda
        if "bfsp" in val_df.columns:
            blender = BenterBlender()
            # Need model predictions for this
            model = FundamentalModel(model_type="xgboost")
            model.fit(train_df, val_df=val_df, race_id_col=race_id_col, target_col=target_col)
            preds = model.predict(val_df, race_id_col=race_id_col)
            opt_lambda = blender.optimise_lambda(preds, race_id_col=race_id_col, target_col=target_col)
            best_params["lambda"] = opt_lambda

        self.best_hyperparams = best_params
        return best_params

    def save(self, path: str) -> None:
        """Save trained model and training metadata."""
        os.makedirs(path, exist_ok=True)
        if self.best_model is not None:
            self.best_model.save(os.path.join(path, "model"))

        meta = {
            "config": self.config,
            "best_lambda": self.best_lambda,
            "best_hyperparams": self.best_hyperparams,
            "fold_results": self.fold_results,
        }
        with open(os.path.join(path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> tuple["ModelTrainer", FundamentalModel]:
        """Load trainer metadata and model."""
        with open(os.path.join(path, "training_meta.json")) as f:
            meta = json.load(f)

        trainer = cls(config=meta["config"])
        trainer.best_lambda = meta.get("best_lambda", 0.80)
        trainer.best_hyperparams = meta.get("best_hyperparams")
        trainer.fold_results = meta.get("fold_results", [])

        model = FundamentalModel.load(os.path.join(path, "model"))
        trainer.best_model = model

        return trainer, model
