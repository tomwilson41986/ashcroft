"""Feature engineering — delegates to existing model/custom_metrics.py."""

# This module exists as a thin wrapper. The actual feature engineering
# is done by model.custom_metrics.CustomMetricsEngine and train_bfsp.build_context_features,
# which are called by predict_bfsp_today.prepare_and_predict().
#
# The ultra_betting.model.predict module handles the full pipeline.
