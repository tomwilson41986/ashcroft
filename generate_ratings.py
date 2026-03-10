"""
Generate Ratings for Today.

Finds the latest trained model and generates win probability ratings
for today's declared runners. Optionally trains a new model first
if no model exists or --train-first is specified.

Usage:
    python generate_ratings.py                          # Rate today, latest model
    python generate_ratings.py --date 2026-03-15        # Rate a specific date
    python generate_ratings.py --train-first            # Train then rate
    python generate_ratings.py --model-dir data/models  # Use specific model
    python generate_ratings.py --dry-run                # Print only, no email

Environment variables (in .env):
    HRB_USERNAME   - horseracebase.com login
    HRB_PASSWORD   - horseracebase.com password
    SMTP_HOST      - SMTP server (default: smtp.gmail.com)
    SMTP_PORT      - SMTP port (default: 587)
    SMTP_USERNAME  - Email sender address
    SMTP_PASSWORD  - Email password / app password
"""

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "generate_ratings.log")
        ),
    ],
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(SCRIPT_DIR, "horse_racing.db")
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")


def find_latest_model(base_dir: str = DEFAULT_MODEL_DIR) -> str | None:
    """Find the most recently trained model directory.

    Searches for training_summary.json files and returns the directory
    containing the most recently modified one. Also checks the base
    directory itself for a flat model layout.

    Returns:
        Path to model directory, or None if no model found.
    """
    base = Path(base_dir)

    candidates = []

    # Check flat layout: model files directly in base_dir
    if (base / "probability_model.lgb").exists():
        summary = base / "training_summary.json"
        mtime = summary.stat().st_mtime if summary.exists() else 0
        candidates.append((str(base), mtime))

    # Check subdirectories (versioned model layout)
    if base.exists():
        for summary_file in base.glob("*/training_summary.json"):
            model_dir = summary_file.parent
            if (model_dir / "probability_model.lgb").exists():
                candidates.append(
                    (str(model_dir), summary_file.stat().st_mtime)
                )

    if not candidates:
        return None

    # Return the most recently modified
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def print_model_info(model_dir: str) -> None:
    """Print summary of the model being used."""
    summary_path = os.path.join(model_dir, "training_summary.json")
    meta_path = os.path.join(model_dir, "probability_model_meta.json")

    log.info(f"Model directory: {model_dir}")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

        data_range = summary.get("data_range", {})
        log.info(f"  Training data: {data_range.get('min_date', '?')} "
                 f"to {data_range.get('max_date', '?')} "
                 f"({data_range.get('total_rows', '?'):,} rows)")
        log.info(f"  Walk-forward folds: {summary.get('n_folds', '?')}")
        log.info(f"  Optimal lambda: {summary.get('optimal_lambda', '?')}")

        final_metrics = summary.get("final_model_metrics", {})
        if final_metrics:
            log.info(f"  Final log-loss: "
                     f"{final_metrics.get('val_logloss_normalised', '?')}")

        eval_results = summary.get("evaluation", {})
        if eval_results:
            log.info(f"  Walk-forward log-loss: "
                     f"{eval_results.get('log_loss', '?')}")
            log.info(f"  AUC-ROC: {eval_results.get('auc_roc', '?')}")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_features = len(meta.get("feature_cols", []))
        log.info(f"  Features: {n_features}")


def train_model(db_path: str, output_dir: str) -> str:
    """Train a new model and return the output directory.

    Uses the existing ModelTrainer with walk-forward validation.
    """
    from model.trainer import ModelTrainer, load_data

    if not os.path.exists(db_path):
        log.error(f"Database not found: {db_path}")
        log.error("Run 'python scraper.py' first to collect historical data.")
        sys.exit(1)

    log.info("Loading historical data for training...")
    df = load_data(db_path)
    log.info(f"  Loaded {len(df):,} rows")

    trainer = ModelTrainer(
        min_train_days=365,
        val_window_days=30,
        step_days=30,
    )

    summary = trainer.train(df, output_dir=output_dir)

    log.info("Training complete!")
    log.info(f"  Folds: {summary.get('n_folds', 0)}")
    final = summary.get("final_model_metrics", {})
    if final:
        log.info(f"  Final log-loss: "
                 f"{final.get('val_logloss_normalised', 'N/A')}")

    return output_dir


def generate_ratings(
    target_date: date,
    model_dir: str,
    db_path: str,
    dry_run: bool = True,
    recipient: str = "racingsquared@gmail.com",
) -> None:
    """Generate ratings for the given date using the specified model.

    This wraps the daily_predictions pipeline:
    1. Fetch today's racecards from horseracebase.com
    2. Build historical features for each runner
    3. Run predictions through the trained model
    4. Output/email the ratings
    """
    from daily_predictions import run_pipeline

    log.info(f"Generating ratings for {target_date}...")
    print_model_info(model_dir)

    predictions = run_pipeline(
        target_date=target_date,
        db_path=db_path,
        model_dir=model_dir,
        recipient=recipient,
        dry_run=dry_run,
    )

    if predictions is None or len(predictions) == 0:
        log.warning("No predictions generated.")
        return

    # Print summary stats
    n_races = (
        predictions["raceid"].nunique()
        if "raceid" in predictions.columns else "?"
    )
    log.info(f"Generated ratings for {len(predictions)} runners "
             f"across {n_races} races")

    # Print top picks per race
    if "raceid" in predictions.columns:
        print("\n" + "=" * 60)
        print(f"TOP PICKS - {target_date.strftime('%A %d %B %Y')}")
        print("=" * 60)

        for race_id, race_df in predictions.groupby("raceid"):
            top = race_df.sort_values("p_model", ascending=False).iloc[0]
            track = top.get("track", "?")
            rtime = top.get("race_time", "?")
            horse = top.get("horse_name", "?")
            p_win = top.get("p_model", 0)
            pred_bfsp = top.get("predicted_bfsp", 0)
            print(f"  {track} {rtime}: {horse} "
                  f"(P={p_win:.1%}, BFSP={pred_bfsp:.1f})")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ratings for today based on the latest trained model"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB,
        help=f"Path to SQLite database (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to model directory. Default: auto-detect latest",
    )
    parser.add_argument(
        "--train-first",
        action="store_true",
        help="Train a new model before generating ratings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Where to save trained model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="racingsquared@gmail.com",
        help="Recipient email address",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print predictions to stdout instead of emailing (default: True)",
    )
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Actually send the email (overrides --dry-run)",
    )
    args = parser.parse_args()

    target_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )

    dry_run = not args.send_email

    log.info(f"{'=' * 60}")
    log.info(f"GENERATE RATINGS - {target_date}")
    log.info(f"{'=' * 60}")

    # Step 1: Optionally train a new model
    if args.train_first:
        log.info("Training new model first...")
        train_model(args.db, args.output_dir)

    # Step 2: Find the model to use
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = find_latest_model(DEFAULT_MODEL_DIR)

    if model_dir is None:
        log.error("No trained model found.")
        log.error(
            "Either train a model first with:\n"
            "  python generate_ratings.py --train-first\n"
            "  python -m model.trainer\n"
            "Or specify a model directory with --model-dir"
        )
        sys.exit(1)

    model_path = os.path.join(model_dir, "probability_model.lgb")
    if not os.path.exists(model_path):
        log.error(f"Model file not found: {model_path}")
        sys.exit(1)

    # Step 3: Generate ratings
    generate_ratings(
        target_date=target_date,
        model_dir=model_dir,
        db_path=args.db,
        dry_run=dry_run,
        recipient=args.email,
    )


if __name__ == "__main__":
    main()
