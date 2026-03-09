"""
BFSP Prediction & Race Probability Model.

A 3-stage prediction pipeline for thoroughbred horse racing:

  Stage 1: Fundamental Model — Win probability from pre-race features
  Stage 2: Benter Blending — Combine model with market probabilities
  Stage 3: Overlay Detection — Identify value bets with Kelly staking

Modules:
  custom_metrics    — 19 proprietary performance metrics (CustomMetricsEngine)
  prerace_builder   — Pre-race feature vector construction (PreRaceBuilder)
  probability_model — Win probability prediction (FundamentalModel)
  benter_blend      — Log-linear probability blending (BenterBlender)
  overlay_detector  — Value detection & Kelly staking (OverlayDetector)
  trainer           — Walk-forward validation training (ModelTrainer)
  evaluator         — Backtesting & performance metrics (ModelEvaluator)

Usage:
    # Training
    python -m model.trainer --db horse_racing.db

    # Prediction (legacy BFSP regression)
    python -m model.predict --date 2025-03-15
"""

from model.benter_blend import BenterBlender
from model.custom_metrics import CustomMetricsEngine
from model.evaluator import ModelEvaluator
from model.overlay_detector import OverlayDetector
from model.prerace_builder import PreRaceBuilder
from model.probability_model import FundamentalModel
from model.trainer import ModelTrainer

__all__ = [
    "CustomMetricsEngine",
    "PreRaceBuilder",
    "FundamentalModel",
    "BenterBlender",
    "OverlayDetector",
    "ModelTrainer",
    "ModelEvaluator",
]
