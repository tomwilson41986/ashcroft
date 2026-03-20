"""Tests for the model confidence scoring module."""

import pytest

from ultra_betting.model.confidence import (
    ConfidenceGrader,
    ConfidenceResult,
    grade_batch,
    grade_prediction,
)


@pytest.fixture
def grader():
    return ConfidenceGrader()


# ── A-grade prediction ──────────────────────────────────────────────────────


class TestAGradePrediction:
    """Strong edge, mid-range price, reasonable field, full features."""

    def test_a_grade(self, grader):
        pred = {
            "predicted_bfsp": 5.0,
            "live_price": 8.0,       # 60% edge -> strong
            "num_runners": 8,        # small field
            "feature_count": 187,
            "non_null_features": 185,
        }
        result = grader.grade(pred)

        assert result.grade == "A"
        assert result.confidence_score >= 70
        assert result.factors["edge"] == 100.0  # >20% edge
        assert result.factors["price_range"] == 100.0  # mid-range
        assert result.factors["field_size"] == 100.0  # <=8 runners

    def test_a_grade_moderate_edge(self, grader):
        """Moderate edge but everything else perfect can still be A."""
        pred = {
            "predicted_bfsp": 6.0,
            "live_price": 7.5,       # 25% edge -> strong
            "num_runners": 6,
            "feature_count": 187,
            "non_null_features": 187,
        }
        result = grader.grade(pred)

        assert result.grade == "A"
        assert result.confidence_score >= 70


# ── D-grade prediction ──────────────────────────────────────────────────────


class TestDGradePrediction:
    """Weak edge, extreme price, large field, sparse features."""

    def test_d_grade(self, grader):
        pred = {
            "predicted_bfsp": 50.0,
            "live_price": 51.0,      # ~2% edge -> weak
            "num_runners": 22,       # large field
            "feature_count": 187,
            "non_null_features": 30, # very sparse
        }
        result = grader.grade(pred)

        assert result.grade == "D"
        assert result.confidence_score < 30

    def test_d_grade_extreme_short_price(self, grader):
        pred = {
            "predicted_bfsp": 1.2,
            "live_price": 1.25,      # ~4% edge -> weak
            "num_runners": 22,
            "feature_count": 187,
            "non_null_features": 20,  # very sparse features
        }
        result = grader.grade(pred)

        assert result.grade == "D"
        assert result.confidence_score < 30


# ── Batch grading ───────────────────────────────────────────────────────────


class TestBatchGrading:

    def test_batch_returns_correct_count(self, grader):
        preds = [
            {
                "predicted_bfsp": 5.0,
                "live_price": 8.0,
                "num_runners": 6,
                "feature_count": 187,
                "non_null_features": 180,
            },
            {
                "predicted_bfsp": 50.0,
                "live_price": 51.0,
                "num_runners": 22,
                "feature_count": 187,
                "non_null_features": 30,
            },
        ]
        results = grader.grade_batch(preds)

        assert len(results) == 2
        assert all(isinstance(r, ConfidenceResult) for r in results)

    def test_batch_preserves_order(self, grader):
        strong = {
            "predicted_bfsp": 5.0,
            "live_price": 8.0,
            "num_runners": 6,
            "feature_count": 187,
            "non_null_features": 180,
        }
        weak = {
            "predicted_bfsp": 50.0,
            "live_price": 51.0,
            "num_runners": 22,
            "feature_count": 187,
            "non_null_features": 30,
        }
        results = grader.grade_batch([strong, weak])

        assert results[0].confidence_score > results[1].confidence_score

    def test_batch_empty_list(self, grader):
        assert grader.grade_batch([]) == []

    def test_module_level_grade_batch(self):
        preds = [
            {
                "predicted_bfsp": 5.0,
                "live_price": 8.0,
                "num_runners": 6,
                "feature_count": 187,
                "non_null_features": 180,
            },
        ]
        results = grade_batch(preds)
        assert len(results) == 1


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_zero_live_price(self, grader):
        pred = {
            "predicted_bfsp": 5.0,
            "live_price": 0.0,
            "num_runners": 8,
            "feature_count": 187,
            "non_null_features": 150,
        }
        result = grader.grade(pred)

        assert result.factors["edge"] == 0.0
        assert result.factors["price_range"] == 0.0
        assert result.grade in ("C", "D")

    def test_zero_predicted_bfsp(self, grader):
        pred = {
            "predicted_bfsp": 0.0,
            "live_price": 5.0,
            "num_runners": 8,
            "feature_count": 187,
            "non_null_features": 150,
        }
        result = grader.grade(pred)

        assert result.factors["edge"] == 0.0

    def test_missing_features_all_zero(self, grader):
        pred = {
            "predicted_bfsp": 5.0,
            "live_price": 8.0,
            "num_runners": 8,
            "feature_count": 0,
            "non_null_features": 0,
        }
        result = grader.grade(pred)

        assert result.factors["feature_completeness"] == 0.0

    def test_missing_keys_default_to_zero(self, grader):
        """Empty dict should not raise — all values default to 0."""
        result = grader.grade({})

        assert result.grade == "D"
        assert result.confidence_score == 0.0

    def test_zero_runners(self, grader):
        pred = {
            "predicted_bfsp": 5.0,
            "live_price": 8.0,
            "num_runners": 0,
            "feature_count": 187,
            "non_null_features": 150,
        }
        result = grader.grade(pred)

        assert result.factors["field_size"] == 0.0

    def test_module_level_grade_prediction(self):
        pred = {
            "predicted_bfsp": 5.0,
            "live_price": 8.0,
            "num_runners": 8,
            "feature_count": 187,
            "non_null_features": 180,
        }
        result = grade_prediction(pred)

        assert isinstance(result, ConfidenceResult)
        assert result.grade in ("A", "B", "C", "D")


# ── Factor scoring unit tests ──────────────────────────────────────────────


class TestFactorScoring:

    def test_edge_strong(self, grader):
        assert grader._score_edge(5.0, 8.0) == 100.0  # 60% edge

    def test_edge_moderate(self, grader):
        score = grader._score_edge(10.0, 11.5)  # 15% edge
        assert 70 <= score <= 100

    def test_edge_marginal(self, grader):
        score = grader._score_edge(10.0, 10.7)  # 7% edge
        assert 40 <= score < 70

    def test_edge_weak(self, grader):
        score = grader._score_edge(10.0, 10.3)  # 3% edge
        assert 0 < score < 40

    def test_price_mid_range(self, grader):
        assert grader._score_price_range(8.0) == 100.0

    def test_price_extreme_low(self, grader):
        score = grader._score_price_range(1.5)
        assert score < 100

    def test_price_extreme_high(self, grader):
        score = grader._score_price_range(60.0)
        assert score < 100

    def test_field_small(self, grader):
        assert grader._score_field_size(6) == 100.0

    def test_field_large(self, grader):
        score = grader._score_field_size(20)
        assert score == 30.0

    def test_completeness_full(self, grader):
        score = grader._score_feature_completeness(187, 187)
        assert score == 100.0

    def test_completeness_half(self, grader):
        score = grader._score_feature_completeness(187, 93)
        assert 49 <= score <= 51
