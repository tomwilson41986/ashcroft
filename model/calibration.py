"""
Probability calibration for BFSP predictions.

Applies isotonic regression to map raw model probabilities (1/predicted_bfsp)
to well-calibrated probabilities that match observed win rates. This is
critical for accurate overlay detection — miscalibrated probabilities lead
to false overlays that lose money.

Usage:
    calibrator = IsotonicCalibrator()
    calibrator.fit(predicted_probs, actual_outcomes)
    calibrated = calibrator.calibrate(new_probs)
"""

import json
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Isotonic regression calibration for predicted probabilities.

    Maps raw model probabilities to calibrated probabilities using
    monotonic (isotonic) regression. This preserves ranking while
    fixing systematic over/under-confidence at different price ranges.
    """

    def __init__(self):
        self.ir = IsotonicRegression(
            y_min=0.001, y_max=0.999, out_of_bounds="clip"
        )
        self.is_fitted = False

    def fit(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> "IsotonicCalibrator":
        """Fit calibration mapping from OOS predictions.

        Args:
            predicted_probs: Raw model P(win) estimates.
            actual_outcomes: Binary outcomes (1=win, 0=lose).

        Returns:
            self, for chaining.
        """
        mask = np.isfinite(predicted_probs) & np.isfinite(actual_outcomes)
        self.ir.fit(predicted_probs[mask], actual_outcomes[mask])
        self.is_fitted = True
        return self

    def calibrate(self, predicted_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to new predictions.

        Args:
            predicted_probs: Raw model P(win) estimates.

        Returns:
            Calibrated probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        result = np.full_like(predicted_probs, np.nan, dtype=float)
        mask = np.isfinite(predicted_probs)
        if mask.any():
            result[mask] = self.ir.predict(predicted_probs[mask])
        return result

    def calibrate_bfsp(
        self,
        predicted_bfsp: np.ndarray,
    ) -> np.ndarray:
        """Calibrate BFSP predictions via probability space.

        Converts BFSP -> prob, calibrates, converts back to BFSP.

        Args:
            predicted_bfsp: Raw predicted BFSP values.

        Returns:
            Calibrated BFSP values.
        """
        raw_prob = 1.0 / np.clip(predicted_bfsp, 1.01, None)
        cal_prob = self.calibrate(raw_prob)
        # Clip to avoid division by zero
        cal_prob = np.clip(cal_prob, 0.001, 0.999)
        return 1.0 / cal_prob

    def save(self, filepath: str):
        """Save calibrator to JSON (stores the isotonic mapping)."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted.")
        data = {
            "X_thresholds": self.ir.X_thresholds_.tolist(),
            "y_thresholds": self.ir.y_thresholds_.tolist(),
            "X_min": float(self.ir.X_min_),
            "X_max": float(self.ir.X_max_),
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "IsotonicCalibrator":
        """Load a saved calibrator."""
        with open(filepath) as f:
            data = json.load(f)
        instance = cls()
        instance.ir.X_thresholds_ = np.array(data["X_thresholds"])
        instance.ir.y_thresholds_ = np.array(data["y_thresholds"])
        instance.ir.X_min_ = data["X_min"]
        instance.ir.X_max_ = data["X_max"]
        instance.ir.increasing_ = True
        instance.is_fitted = True
        return instance
