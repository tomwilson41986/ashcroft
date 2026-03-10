#!/usr/bin/env python3
"""
Generate Race Ratings.

End-to-end script that:
  1. Validates prerequisites (database, racecard)
  2. Auto-trains the model if no trained model exists
  3. Builds historical feature profiles for today's declared runners
  4. Generates win probability ratings via the 3-stage pipeline
  5. Outputs a formatted ratings report

Usage:
    python generate_ratings.py                          # Today's ratings
    python generate_ratings.py --date 2026-03-10        # Specific date
    python generate_ratings.py --racecard path/to.csv   # Use a specific racecard CSV
    python generate_ratings.py --retrain                # Force model retrain before rating

Prerequisites:
    - horse_racing.db   — historical race results (from scraper.py)
    - csv/racecards/racecard_YYYY-MM-DD.csv — today's declared runners
      (downloaded via daily_predictions.py or placed manually)
    - data/models/probability_model.lgb — trained model
      (auto-trained from DB if missing)
"""

import argparse
import io
import json
import logging
import os
import sqlite3
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd

from model.benter_blend import BenterBlender
from model.custom_metrics import CustomMetricsEngine
from model.overlay_detector import OverlayDetector
from model.prerace_builder import PreRaceBuilder
from model.probability_model import FundamentalModel
from model.trainer import ModelTrainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")
RACECARD_DIR = os.path.join(SCRIPT_DIR, "csv", "racecards")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 1. Prerequisite checks
# ------------------------------------------------------------------

def check_database(db_path: str) -> bool:
    """Verify the historical database exists and has data."""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM race_results")
        count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def get_db_stats(db_path: str) -> dict:
    """Return summary statistics about the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM race_results")
    total_rows = cur.fetchone()[0]
    cur.execute("SELECT MIN(race_date), MAX(race_date) FROM race_results")
    min_date, max_date = cur.fetchone()
    cur.execute("SELECT COUNT(DISTINCT horse_name) FROM race_results")
    n_horses = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT track) FROM race_results")
    n_tracks = cur.fetchone()[0]
    conn.close()
    return {
        "total_rows": total_rows,
        "min_date": min_date,
        "max_date": max_date,
        "n_horses": n_horses,
        "n_tracks": n_tracks,
    }


def check_model(model_dir: str) -> bool:
    """Check if a trained model exists."""
    model_path = os.path.join(model_dir, "probability_model.lgb")
    return os.path.exists(model_path)


def find_racecard(target_date: date, racecard_path: str | None = None) -> str | None:
    """Locate the racecard CSV for the target date."""
    if racecard_path and os.path.exists(racecard_path):
        return racecard_path
    default_path = os.path.join(
        RACECARD_DIR, f"racecard_{target_date.isoformat()}.csv"
    )
    if os.path.exists(default_path):
        return default_path
    return None


# ------------------------------------------------------------------
# 2. Data loading
# ------------------------------------------------------------------

def load_historical_data(db_path: str) -> pd.DataFrame:
    """Load all historical race results from the SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results ORDER BY race_date, race_time", conn
    )
    conn.close()
    df["race_date"] = pd.to_datetime(df["race_date"])
    log.info(f"Loaded {len(df):,} historical rows")
    return df


def load_racecard(csv_path: str) -> pd.DataFrame:
    """Load and normalise the racecard CSV."""
    df = pd.read_csv(csv_path)

    # Normalise column names to match DB schema
    col_map = {
        "racedate": "race_date",
        "racetime": "race_time",
        "race_distance": "race_distance",
        "going_description": "going_description",
        "horse_name": "horse_name",
        "jockey_name": "jockey_name",
        "trainer": "trainer",
        "horse_age": "horse_age",
        "pounds": "pounds",
        "stall": "stall",
        "official_rating": "official_rating",
        "number_of_runners": "number_of_runners",
        "race_class": "race_class",
        "prize_money": "prize_money",
        "Dist_Furlongs": "dist_furlongs",
        "BFSP": "bfsp",
        "odds": "odds",
        "track": "track",
        "Headgear": "headgear",
        "dayssincelr": "days_since_lr",
        "careerruns": "career_runs",
        "stallion": "stallion",
        "surfacetype": "surface_type",
        "HorseSex": "horse_sex",
        "RaceType": "race_type",
        "CardNo": "card_no",
        "StallPositioning": "stall_positioning",
        "TrackDirection": "track_direction",
        "RailMove": "rail_move",
        "MaxORinRace": "max_or_in_race",
    }
    renamed = {}
    for old, new in col_map.items():
        if old in df.columns:
            renamed[old] = new
    df = df.rename(columns=renamed)

    # Ensure numeric columns
    for col in [
        "number_of_runners", "horse_age", "pounds", "stall",
        "official_rating", "dist_furlongs", "odds", "bfsp",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Loaded racecard: {len(df)} runners from {csv_path}")
    return df


# ------------------------------------------------------------------
# 3. Model training (auto-train if needed)
# ------------------------------------------------------------------

def auto_train_model(db_path: str, model_dir: str) -> None:
    """Train the model from the historical database."""
    log.info("=" * 60)
    log.info("AUTO-TRAINING MODEL (no trained model found)")
    log.info("=" * 60)

    historical = load_historical_data(db_path)

    trainer = ModelTrainer(
        min_train_days=365,
        val_window_days=30,
        step_days=30,
    )
    summary = trainer.train(historical, output_dir=model_dir)

    log.info("Model training complete!")
    if "final_model_metrics" in summary:
        fm = summary["final_model_metrics"]
        log.info(
            f"  Final log-loss: "
            f"{fm.get('val_logloss_normalised', 'N/A')}"
        )
    if "optimal_lambda" in summary:
        log.info(f"  Optimal lambda: {summary['optimal_lambda']}")
    log.info(f"  Model saved to: {model_dir}")


# ------------------------------------------------------------------
# 4. Feature building & prediction
# ------------------------------------------------------------------

def _parse_distance_to_furlongs(distance_series: pd.Series) -> pd.Series:
    """Convert distance strings like '2m 4f' or '7f' to furlongs."""
    import re

    def convert(val):
        if pd.isna(val):
            return np.nan
        s = str(val).lower().strip()
        try:
            return float(s)
        except ValueError:
            pass
        miles = furlongs = yards = 0
        m = re.search(r"(\d+)\s*m", s)
        if m:
            miles = int(m.group(1))
        f = re.search(r"(\d+)\s*f", s)
        if f:
            furlongs = int(f.group(1))
        y = re.search(r"(\d+)\s*y", s)
        if y:
            yards = int(y.group(1))
        total = miles * 8 + furlongs + yards / 220
        return total if total > 0 else np.nan

    return distance_series.apply(convert)


def build_features(
    today_runners: pd.DataFrame,
    historical_df: pd.DataFrame,
    target_date: date,
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate custom metrics on history and build features for today's runners."""
    log.info("Calculating custom metrics on historical data...")
    engine = CustomMetricsEngine()
    enriched = engine.calculate_all(historical_df)
    log.info(f"  Calculated metrics for {len(enriched):,} rows")

    log.info("Building pre-race feature lookups...")
    builder = PreRaceBuilder(enriched)
    feature_cols = builder.get_feature_columns()

    runners = today_runners.copy()

    if "race_class" in runners.columns:
        runners["race_class"] = pd.to_numeric(
            runners["race_class"], errors="coerce"
        )
    if "dist_furlongs" not in runners.columns and "race_distance" in runners.columns:
        runners["dist_furlongs"] = _parse_distance_to_furlongs(
            runners["race_distance"]
        )
    if "going_description" not in runners.columns:
        runners["going_description"] = "Good"

    n_tracks = runners["track"].nunique() if "track" in runners.columns else "?"
    log.info(f"Building features for {len(runners)} runners across {n_tracks} tracks...")
    features_df = builder.build_features(runners, str(target_date))

    # Carry forward identification columns
    id_cols = ["race_date", "race_time", "track", "horse_name"]
    for col in id_cols:
        if col in runners.columns and col not in features_df.columns:
            features_df[col] = runners[col].values

    # Add raceid for per-race normalisation
    if "raceid" not in features_df.columns:
        if "track" in features_df.columns and "race_time" in features_df.columns:
            features_df["raceid"] = (
                str(target_date) + "_"
                + features_df["track"].astype(str) + "_"
                + features_df["race_time"].astype(str)
            )

    # Carry forward odds/bfsp if available
    for col in ["bfsp", "odds"]:
        if col in runners.columns and col not in features_df.columns:
            features_df[col] = runners[col].values

    return features_df, feature_cols


def run_predictions(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    model_dir: str,
) -> pd.DataFrame:
    """Load the trained model and generate predictions."""
    log.info("Loading trained model...")
    model = FundamentalModel.load(model_dir)

    model_features = model.feature_cols if model.feature_cols else feature_cols

    # Ensure all model features exist, fill missing with NaN
    for col in model_features:
        if col not in features_df.columns:
            features_df[col] = np.nan

    log.info(f"Running predictions on {len(features_df)} runners...")
    predictions = model.predict(
        features_df, raceid_col="raceid", normalise=True
    )

    # Load blend config
    blend_path = os.path.join(model_dir, "blend_config.json")
    if os.path.exists(blend_path):
        with open(blend_path) as f:
            blend_config = json.load(f)
        optimal_lambda = blend_config.get("optimal_lambda", 0.80)
    else:
        optimal_lambda = 0.80

    # Blend with market prices if available
    blender = BenterBlender()
    if "bfsp" in predictions.columns and predictions["bfsp"].notna().any():
        log.info(f"Blending with market prices (lambda={optimal_lambda})...")
        predictions = blender.blend(predictions, lambda_=optimal_lambda)
    else:
        log.info("No market prices available - using pure model predictions")
        predictions["p_combined"] = predictions["p_model"]
        predictions["model_fair_bfsp"] = predictions["predicted_bfsp"]
        predictions["edge"] = 0.0
        predictions["edge_pct"] = 0.0

    return predictions


# ------------------------------------------------------------------
# 5. Output formatting
# ------------------------------------------------------------------

def format_ratings_report(
    predictions: pd.DataFrame, target_date: date, db_stats: dict
) -> str:
    """Format the full ratings report as plain text."""
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  RACE RATINGS — {target_date.strftime('%A %d %B %Y')}")
    lines.append("=" * 72)
    lines.append("")

    # Database profile summary
    lines.append(f"  Historic DB: {db_stats['total_rows']:,} results | "
                 f"{db_stats['n_horses']:,} horses | "
                 f"{db_stats['n_tracks']} tracks | "
                 f"{db_stats['min_date']} to {db_stats['max_date']}")
    lines.append("")

    if len(predictions) == 0:
        lines.append("  No runners found for today.")
        return "\n".join(lines)

    # Determine grouping
    if "raceid" in predictions.columns:
        group_col = "raceid"
    elif "track" in predictions.columns and "race_time" in predictions.columns:
        predictions = predictions.copy()
        predictions["_group"] = (
            predictions["track"].astype(str) + " "
            + predictions["race_time"].astype(str)
        )
        group_col = "_group"
    else:
        group_col = None

    total_runners = len(predictions)
    n_races = predictions[group_col].nunique() if group_col else 1
    lines.append(f"  {total_runners} runners across {n_races} races")
    lines.append("")

    # ---- Top picks per race ----
    lines.append("  TOP PICKS (Highest rated per race)")
    lines.append("  " + "-" * 68)

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            top = race_df.sort_values("p_model", ascending=False).iloc[0]
            track = top.get("track", "?")
            rtime = top.get("race_time", "?")
            horse = top.get("horse_name", "?")
            p_win = top.get("p_model", 0)
            pred_bfsp = top.get("predicted_bfsp", 0)
            star = " ***" if p_win >= 0.25 else " **" if p_win >= 0.15 else ""
            lines.append(
                f"    {track:>15} {rtime:>5}:  {horse:<25} "
                f"P={p_win:>6.1%}  BFSP={pred_bfsp:>5.1f}{star}"
            )
    lines.append("")

    # ---- Full race-by-race breakdown ----
    lines.append("  FULL RACE BREAKDOWN")
    lines.append("  " + "=" * 68)

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            race_df = race_df.sort_values("p_model", ascending=False)
            track = race_df["track"].iloc[0] if "track" in race_df.columns else "?"
            rtime = race_df["race_time"].iloc[0] if "race_time" in race_df.columns else "?"
            n_run = len(race_df)

            lines.append("")
            lines.append(f"    {track} — {rtime} ({n_run} runners)")
            lines.append("    " + "-" * 64)
            lines.append(
                f"    {'#':<4} {'Horse':<25} {'Rating':<8} "
                f"{'Pred BFSP':<10} {'Fair BFSP':<10} {'Edge':<8}"
            )
            lines.append("    " + "-" * 64)

            for rank, (_, row) in enumerate(race_df.iterrows(), 1):
                horse = str(row.get("horse_name", "?"))[:24]
                p_win = row.get("p_model", 0)
                pred_bfsp = row.get("predicted_bfsp", 0)
                fair_bfsp = row.get("model_fair_bfsp", pred_bfsp)
                edge = row.get("edge_pct", 0)
                edge_str = f"{edge:+.1f}%" if edge != 0 else "—"

                lines.append(
                    f"    {rank:<4} {horse:<25} {p_win:<8.1%} "
                    f"{pred_bfsp:<10.1f} {fair_bfsp:<10.1f} {edge_str:<8}"
                )
    else:
        predictions = predictions.sort_values("p_model", ascending=False)
        for rank, (_, row) in enumerate(predictions.iterrows(), 1):
            horse = str(row.get("horse_name", "?"))[:24]
            p_win = row.get("p_model", 0)
            pred_bfsp = row.get("predicted_bfsp", 0)
            lines.append(
                f"    {rank}. {horse} — Rating: {p_win:.1%}, "
                f"Predicted BFSP: {pred_bfsp:.1f}"
            )

    # ---- Value bets ----
    if "edge" in predictions.columns:
        value_bets = predictions[predictions["edge"] > 0.05].sort_values(
            "edge", ascending=False
        )
        if len(value_bets) > 0:
            lines.append("")
            lines.append("  VALUE BETS (Edge > 5%)")
            lines.append("  " + "-" * 68)
            for _, row in value_bets.iterrows():
                horse = row.get("horse_name", "?")
                track = row.get("track", "?")
                rtime = row.get("race_time", "?")
                edge = row.get("edge_pct", 0)
                p_win = row.get("p_combined", row.get("p_model", 0))
                fair = row.get("model_fair_bfsp", 0)
                lines.append(
                    f"    {track:>15} {rtime:>5}:  {horse:<25} "
                    f"Edge={edge:>+5.1f}%  P={p_win:.1%}  Fair={fair:.1f}"
                )

    # ---- Kelly bet card ----
    if "bfsp" in predictions.columns and predictions["bfsp"].notna().any():
        detector = OverlayDetector()
        bet_card = detector.generate_bet_card(predictions)
        if len(bet_card) > 0:
            lines.append("")
            lines.append("  KELLY BET CARD (Fractional Kelly @ 25%)")
            lines.append("  " + "-" * 68)
            total_stake = 0
            total_ev = 0
            for _, row in bet_card.iterrows():
                horse = row.get("horse_name", "?")
                track = row.get("track", "?")
                rtime = row.get("race_time", "?")
                stake = row.get("stake_gbp", 0)
                ev = row.get("expected_value", 0)
                edge = row.get("edge_pct", 0)
                total_stake += stake
                total_ev += ev
                lines.append(
                    f"    {track:>15} {rtime:>5}:  {horse:<22} "
                    f"Stake=£{stake:>6.2f}  EV=£{ev:>+6.2f}  Edge={edge:>+5.1f}%"
                )
            lines.append("    " + "-" * 64)
            lines.append(
                f"    {'TOTAL':>22}  "
                f"Stake=£{total_stake:>6.2f}  EV=£{total_ev:>+6.2f}"
            )

    lines.append("")
    lines.append("  " + "-" * 68)
    lines.append("  Generated by Ultra Betting prediction pipeline")
    lines.append("  Model: 3-stage LightGBM + Benter blend + Kelly staking")
    lines.append(
        "  Disclaimer: Predictions are for informational purposes only. "
        "Bet responsibly."
    )
    lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def generate_ratings(
    target_date: date,
    db_path: str = DB_PATH,
    model_dir: str = MODEL_DIR,
    racecard_path: str | None = None,
    retrain: bool = False,
) -> pd.DataFrame:
    """Generate race ratings for the target date.

    Steps:
        1. Validate database exists
        2. Auto-train model if needed
        3. Load racecard for today
        4. Build features from historical profiles
        5. Run 3-stage prediction pipeline
        6. Output formatted ratings report

    Returns:
        DataFrame of predictions.
    """
    log.info("=" * 60)
    log.info(f"GENERATING RACE RATINGS — {target_date}")
    log.info("=" * 60)

    # Step 1: Check database
    log.info("Step 1: Checking database...")
    if not check_database(db_path):
        log.error(f"Database not found or empty: {db_path}")
        log.error("Run 'python scraper.py' first to collect historical data.")
        sys.exit(1)

    db_stats = get_db_stats(db_path)
    log.info(
        f"  Database: {db_stats['total_rows']:,} results, "
        f"{db_stats['n_horses']:,} horses, "
        f"{db_stats['min_date']} to {db_stats['max_date']}"
    )

    # Step 2: Check/train model
    log.info("Step 2: Checking trained model...")
    if retrain or not check_model(model_dir):
        if retrain:
            log.info("  Retrain requested — training new model...")
        else:
            log.info("  No trained model found — auto-training...")
        auto_train_model(db_path, model_dir)
    else:
        log.info(f"  Trained model found at {model_dir}")
        meta_path = os.path.join(model_dir, "probability_model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            n_features = meta.get("metrics", {}).get("n_features", "?")
            log.info(f"  Model features: {n_features}")

    # Step 3: Load racecard
    log.info("Step 3: Loading racecard...")
    csv_path = find_racecard(target_date, racecard_path)
    if csv_path is None:
        log.error(
            f"No racecard found for {target_date}. "
            f"Expected at: {RACECARD_DIR}/racecard_{target_date.isoformat()}.csv"
        )
        log.error(
            "Either:\n"
            "  1. Run 'python daily_predictions.py --date {0} --dry-run' "
            "to download it\n"
            "  2. Place a racecard CSV manually in csv/racecards/\n"
            "  3. Use --racecard path/to/file.csv".format(target_date)
        )
        sys.exit(1)

    today_runners = load_racecard(csv_path)
    if len(today_runners) == 0:
        log.warning(f"Racecard is empty for {target_date}. No racing today?")
        return pd.DataFrame()

    n_tracks = today_runners["track"].nunique() if "track" in today_runners.columns else 0
    log.info(f"  {len(today_runners)} runners across {n_tracks} tracks")

    # Step 4: Build features from historical profiles
    log.info("Step 4: Building historical feature profiles...")
    historical = load_historical_data(db_path)
    features_df, feature_cols = build_features(
        today_runners, historical, target_date
    )
    log.info(f"  Built {len(feature_cols)} features for {len(features_df)} runners")

    # Step 5: Run 3-stage predictions
    log.info("Step 5: Running 3-stage prediction pipeline...")
    predictions = run_predictions(features_df, feature_cols, model_dir)
    log.info(f"  Generated ratings for {len(predictions)} runners")

    # Step 6: Output report
    report = format_ratings_report(predictions, target_date, db_stats)
    print(report)

    # Save predictions to CSV
    os.makedirs(RACECARD_DIR, exist_ok=True)
    output_path = os.path.join(
        RACECARD_DIR, f"ratings_{target_date.isoformat()}.csv"
    )
    output_cols = [
        "race_date", "race_time", "track", "horse_name",
        "p_model", "predicted_bfsp", "p_combined", "model_fair_bfsp",
        "edge", "edge_pct",
    ]
    available_cols = [c for c in output_cols if c in predictions.columns]
    predictions[available_cols].to_csv(output_path, index=False)
    log.info(f"Ratings saved to {output_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate race ratings from historical profiles and today's racecard"
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
        default=DB_PATH,
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help=f"Path to model artifacts (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--racecard",
        type=str,
        default=None,
        help="Path to racecard CSV (overrides default location)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force model retrain before generating ratings",
    )
    args = parser.parse_args()

    target_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )

    generate_ratings(
        target_date=target_date,
        db_path=args.db,
        model_dir=args.model_dir,
        racecard_path=args.racecard,
        retrain=args.retrain,
    )


if __name__ == "__main__":
    main()
