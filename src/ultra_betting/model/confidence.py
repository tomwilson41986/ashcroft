"""Model confidence scoring for BFSP predictions.

Assigns grades (A/B/C/D) to predictions based on edge magnitude, price range
reliability, field size, and feature completeness.
"""

from __future__ import annotations

from dataclasses import dataclass

# Total model features (19 custom metric families -> 187 individual features)
TOTAL_FEATURES = 187

# Price range where the model is most reliable
RELIABLE_PRICE_LOW = 3.0
RELIABLE_PRICE_HIGH = 15.0

# Grade thresholds (inclusive lower bound)
GRADE_THRESHOLDS = {
    "A": 70,
    "B": 50,
    "C": 30,
    "D": 0,
}


@dataclass
class ConfidenceResult:
    """Result of grading a single prediction."""

    grade: str
    confidence_score: float
    factors: dict[str, float]


class ConfidenceGrader:
    """Assigns confidence grades to BFSP predictions.

    Evaluates four factors:
      - Edge magnitude: larger predicted edge -> higher confidence
      - Price range reliability: mid-range prices (3-15) are more reliable
      - Field size: larger fields (15+) are harder to predict
      - Feature completeness: fraction of 187 features that are non-null/non-zero
    """

    def __init__(
        self,
        total_features: int = TOTAL_FEATURES,
        reliable_price_low: float = RELIABLE_PRICE_LOW,
        reliable_price_high: float = RELIABLE_PRICE_HIGH,
    ) -> None:
        self.total_features = total_features
        self.reliable_price_low = reliable_price_low
        self.reliable_price_high = reliable_price_high

    def _score_edge(self, predicted_bfsp: float, live_price: float) -> float:
        """Score based on edge magnitude (0-100).

        Edge = |live_price / predicted_bfsp - 1| * 100 (percentage).
        Thresholds:
          >20%  -> 100 (strong)
          10-20% -> 70  (moderate)
          5-10%  -> 40  (marginal)
          <5%    -> 15  (weak)
        """
        if predicted_bfsp <= 0 or live_price <= 0:
            return 0.0

        edge_pct = abs(live_price / predicted_bfsp - 1) * 100

        if edge_pct > 20:
            return 100.0
        elif edge_pct > 10:
            # Linear interpolation 70-100 across 10-20%
            return 70.0 + (edge_pct - 10) / 10 * 30
        elif edge_pct > 5:
            # Linear interpolation 40-70 across 5-10%
            return 40.0 + (edge_pct - 5) / 5 * 30
        else:
            # Linear interpolation 0-40 across 0-5%
            return edge_pct / 5 * 40

    def _score_price_range(self, live_price: float) -> float:
        """Score based on price range reliability (0-100).

        Mid-range (3.0-15.0) = 100.
        Scores decay as prices move away from the reliable range.
        """
        if live_price <= 0:
            return 0.0

        if self.reliable_price_low <= live_price <= self.reliable_price_high:
            return 100.0

        if live_price < self.reliable_price_low:
            # Very short prices: less reliable. 1.0 -> ~50, approaching 0 -> 0
            ratio = live_price / self.reliable_price_low
            return max(0.0, ratio * 100)
        else:
            # Long prices: less reliable. Decay from 100 at 15 to ~30 at 50+
            overshoot = live_price - self.reliable_price_high
            decay = min(overshoot / 50.0, 1.0)  # fully decayed at 65.0
            return max(0.0, 100.0 - decay * 70)

    def _score_field_size(self, num_runners: int) -> float:
        """Score based on field size (0-100).

        Fewer runners = easier to predict.
          1-8   -> 100
          9-14  -> 80
          15-19 -> 50
          20+   -> 30
        """
        if num_runners <= 0:
            return 0.0

        if num_runners <= 8:
            return 100.0
        elif num_runners <= 14:
            # Linear interpolation 80-100 across 9-14
            return 100.0 - (num_runners - 8) / 6 * 20
        elif num_runners <= 19:
            # Linear interpolation 50-80 across 15-19
            return 80.0 - (num_runners - 14) / 5 * 30
        else:
            # 20+ runners: cap at 30
            return 30.0

    def _score_feature_completeness(
        self, feature_count: int, non_null_features: int
    ) -> float:
        """Score based on fraction of features that are non-null/non-zero (0-100).

        Uses the smaller of feature_count and total_features as the denominator.
        """
        if feature_count <= 0:
            return 0.0

        denominator = min(feature_count, self.total_features)
        if denominator <= 0:
            return 0.0

        ratio = min(non_null_features / denominator, 1.0)
        return ratio * 100

    def _assign_grade(self, score: float) -> str:
        """Map a 0-100 confidence score to a letter grade."""
        for grade, threshold in GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return "D"

    def grade(self, prediction: dict) -> ConfidenceResult:
        """Grade a single prediction dict.

        Expected keys:
            predicted_bfsp: float
            live_price: float
            num_runners: int
            feature_count: int       (total features available)
            non_null_features: int   (features that are non-null/non-zero)

        Optional keys:
            predicted_win_prob: float (not currently used in scoring but
                                       reserved for future factors)

        Returns:
            ConfidenceResult with grade, confidence_score, and factors dict.
        """
        predicted_bfsp = prediction.get("predicted_bfsp", 0.0)
        live_price = prediction.get("live_price", 0.0)
        num_runners = prediction.get("num_runners", 0)
        feature_count = prediction.get("feature_count", 0)
        non_null_features = prediction.get("non_null_features", 0)

        edge_score = self._score_edge(predicted_bfsp, live_price)
        price_score = self._score_price_range(live_price)
        field_score = self._score_field_size(num_runners)
        completeness_score = self._score_feature_completeness(
            feature_count, non_null_features
        )

        # Weighted average — edge is the most important signal
        weights = {
            "edge": 0.40,
            "price_range": 0.20,
            "field_size": 0.15,
            "feature_completeness": 0.25,
        }
        confidence_score = (
            edge_score * weights["edge"]
            + price_score * weights["price_range"]
            + field_score * weights["field_size"]
            + completeness_score * weights["feature_completeness"]
        )

        factors = {
            "edge": round(edge_score, 2),
            "price_range": round(price_score, 2),
            "field_size": round(field_score, 2),
            "feature_completeness": round(completeness_score, 2),
        }

        grade = self._assign_grade(confidence_score)

        return ConfidenceResult(
            grade=grade,
            confidence_score=round(confidence_score, 2),
            factors=factors,
        )

    def grade_batch(self, predictions: list[dict]) -> list[ConfidenceResult]:
        """Grade a list of prediction dicts.

        Returns a list of ConfidenceResult in the same order.
        """
        return [self.grade(p) for p in predictions]


# ── Module-level convenience functions ──────────────────────────────────────


_default_grader = ConfidenceGrader()


def grade_prediction(prediction: dict) -> ConfidenceResult:
    """Grade a single prediction using the default ConfidenceGrader."""
    return _default_grader.grade(prediction)


def grade_batch(predictions: list[dict]) -> list[ConfidenceResult]:
    """Grade a list of predictions using the default ConfidenceGrader."""
    return _default_grader.grade_batch(predictions)
