"""Ashcroft model prediction runner — wraps predict_bfsp_today.py."""

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from ultra_betting.config import DB_PATH, MODEL_DIR
from ultra_betting.data.schemas import Prediction

log = logging.getLogger(__name__)

# Ensure project root is importable
_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


def run_predictions(
    target_date: date | None = None,
    from_db: bool = False,
    start_date: str = "2020-01-01",
) -> list[Prediction]:
    """Run the Ashcroft BFSP prediction model for a given date.

    Args:
        target_date: Date to predict for. Defaults to today.
        from_db: If True, use runners from the database (no HRB login).
        start_date: Earliest historical date to load (for memory efficiency).

    Returns:
        List of Prediction objects.
    """
    # Import from the existing pipeline
    from predict_bfsp_today import (
        load_bfsp_model,
        load_historical,
        get_runners_from_db,
        fetch_racecard_from_hrb,
        prepare_and_predict,
    )

    if target_date is None:
        target_date = date.today()

    db_path = str(DB_PATH)
    model_dir = str(MODEL_DIR)

    # Load model
    model, feature_cols = load_bfsp_model(model_dir)

    # Load historical data
    log.info(f"Loading historical data from {start_date}...")
    historical = load_historical(db_path, start_date=start_date)
    log.info(f"Loaded {len(historical):,} historical rows")

    # Get target runners
    if from_db:
        target_runners = get_runners_from_db(db_path, str(target_date))
    else:
        import os
        if os.getenv("HRB_USERNAME"):
            target_runners = fetch_racecard_from_hrb(target_date)
        else:
            log.warning("No HRB credentials, using database")
            target_runners = get_runners_from_db(db_path, str(target_date))

    if len(target_runners) == 0:
        log.warning(f"No runners found for {target_date}")
        return []

    # Separate history
    history_before = historical[
        historical["race_date"].dt.date < target_date
    ].copy()

    if len(history_before) < 100:
        history_before = historical.copy()

    # Run predictions
    preds_df = prepare_and_predict(
        history_before, target_runners, model, feature_cols, target_date
    )

    if len(preds_df) == 0:
        return []

    # Convert to Prediction objects
    predictions = []
    for _, row in preds_df.iterrows():
        predictions.append(Prediction(
            date=str(target_date),
            venue=str(row.get("track", "")),
            race_time=str(row.get("race_time", "")),
            runner_name=str(row.get("horse_name", "")),
            predicted_bfsp=float(row.get("predicted_bfsp", 0)),
            predicted_win_prob=float(row.get("predicted_win_prob_norm", 0)),
        ))

    log.info(f"Generated {len(predictions)} predictions for {target_date}")
    return predictions


def predictions_to_dataframe(predictions: list[Prediction]) -> pd.DataFrame:
    """Convert Prediction objects to a DataFrame for S3 storage."""
    if not predictions:
        return pd.DataFrame()
    return pd.DataFrame([p.model_dump() for p in predictions])
