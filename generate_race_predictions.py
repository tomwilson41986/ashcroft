#!/usr/bin/env python3
"""
Generate Race Predictions for Today.

Scrapes today's race cards from Sporting Life (free, public),
maps runner data to model features, and predicts BFSP using the
trained LightGBM regression model.

When horse_racing.db is available, calculates full custom metrics
on historical data for maximum accuracy. Without the database,
uses race-card-only features (LightGBM handles NaN natively).

Usage:
    python generate_race_predictions.py                    # Today
    python generate_race_predictions.py --date 2026-03-23  # Specific date
    python generate_race_predictions.py --output-csv preds.csv
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sporting Life Race Card Scraper
# ---------------------------------------------------------------------------

def create_session() -> requests.Session:
    """Create an HTTP session with browser-like headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    })
    return session


def fetch_meetings(session: requests.Session, target_date: date) -> list[dict]:
    """Fetch meeting list from Sporting Life for a given date."""
    url = f"https://www.sportinglife.com/racing/racecards/{target_date.isoformat()}"
    log.info(f"Fetching meetings from {url}")
    for attempt in range(4):
        resp = session.get(url, timeout=20)
        if resp.status_code == 200:
            break
        log.warning(f"Attempt {attempt+1} got HTTP {resp.status_code}, retrying...")
        time.sleep(2 ** (attempt + 1))
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    next_data = soup.find("script", id="__NEXT_DATA__")
    if not next_data:
        log.error("Could not find __NEXT_DATA__ in page")
        return []

    data = json.loads(next_data.get_text())
    page_props = data.get("props", {}).get("pageProps", {})
    meetings = page_props.get("meetings", [])

    # Filter to UK/IRE meetings only
    uk_meetings = []
    for m in meetings:
        course = m.get("meeting_summary", {}).get("course", {})
        country = course.get("country", {}).get("short_name", "")
        if country in ("ENG", "IRE", "SCO", "WAL"):
            uk_meetings.append(m)

    log.info(f"Found {len(uk_meetings)} UK/IRE meetings ({len(meetings)} total)")
    return uk_meetings


def fetch_race_card(
    session: requests.Session, race_url: str
) -> dict | None:
    """Fetch full race card with runners from Sporting Life."""
    for attempt in range(3):
        resp = session.get(race_url, timeout=20)
        if resp.status_code == 200:
            break
        time.sleep(2 ** attempt)
    if resp.status_code != 200:
        log.warning(f"Failed to fetch {race_url}: HTTP {resp.status_code}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    next_data = soup.find("script", id="__NEXT_DATA__")
    if not next_data:
        return None

    data = json.loads(next_data.get_text())
    page_props = data.get("props", {}).get("pageProps", {})
    return page_props.get("race")


def parse_distance_to_furlongs(distance_str: str) -> float | None:
    """Convert distance string like '2m 3f 179y' to furlongs."""
    if not distance_str:
        return None

    miles = 0
    furlongs = 0
    yards = 0

    m_match = re.search(r"(\d+)m", distance_str)
    f_match = re.search(r"(\d+)f", distance_str)
    y_match = re.search(r"(\d+)y", distance_str)

    if m_match:
        miles = int(m_match.group(1))
    if f_match:
        furlongs = int(f_match.group(1))
    if y_match:
        yards = int(y_match.group(1))

    total_furlongs = miles * 8 + furlongs + yards / 220.0
    return round(total_furlongs, 2) if total_furlongs > 0 else None


def parse_weight(weight_str: str) -> int | None:
    """Convert weight like '11-10' (stones-pounds) to total pounds."""
    if not weight_str:
        return None
    match = re.match(r"(\d+)-(\d+)", str(weight_str))
    if match:
        return int(match.group(1)) * 14 + int(match.group(2))
    return None


def parse_odds_to_decimal(odds_str: str) -> float | None:
    """Convert fractional odds like '9/2' to decimal."""
    if not odds_str:
        return None
    match = re.match(r"(\d+)/(\d+)", str(odds_str))
    if match:
        return int(match.group(1)) / int(match.group(2)) + 1
    if odds_str.lower() == "evs":
        return 2.0
    return None


def scrape_todays_runners(
    session: requests.Session, target_date: date
) -> pd.DataFrame:
    """Scrape all UK/IRE runners for a given date from Sporting Life."""
    meetings = fetch_meetings(session, target_date)
    if not meetings:
        return pd.DataFrame()

    all_runners = []
    base_url = "https://www.sportinglife.com"

    for meeting in meetings:
        meeting_summary = meeting.get("meeting_summary", {})
        course = meeting_summary.get("course", {})
        course_name = course.get("name", "Unknown")
        meeting_going = meeting_summary.get("going", "")
        surface = meeting_summary.get("surface_summary", "")

        for race in meeting.get("races", []):
            race_name = race.get("name", "")
            race_time = race.get("time", "")
            race_class = race.get("race_class", "")
            distance = race.get("distance", "")
            going = race.get("going", meeting_going)
            race_id = race.get("race_summary_reference", {}).get("id", "")
            race_stage = race.get("race_stage", "")
            age_info = race.get("age", "")
            is_handicap = race.get("has_handicap", False)
            ride_count = race.get("ride_count", 0)
            course_surface = race.get("course_surface", {}).get("surface", surface)

            # Determine race type from name and surface
            race_type = "Flat"
            race_name_lower = race_name.lower()
            if "hurdle" in race_name_lower:
                race_type = "Hurdle"
            elif "chase" in race_name_lower:
                race_type = "Chase"
            elif "bumper" in race_name_lower or "nhf" in race_name_lower:
                race_type = "NH Flat"

            # Build race card URL from the links on the page
            # We'll construct it from the race info
            course_slug = course_name.lower().replace(" ", "-")
            race_name_slug = re.sub(r"[^a-z0-9]+", "-", race_name.lower()).strip("-")
            race_url = (
                f"{base_url}/racing/racecards/{target_date.isoformat()}"
                f"/{course_slug}/racecard/{race_id}/{race_name_slug}"
            )

            # Fetch individual race card with full runner data
            time.sleep(0.5)  # Rate limiting
            race_data = fetch_race_card(session, race_url)

            if not race_data:
                log.warning(f"Could not fetch race card for {course_name} {race_time}")
                continue

            rides = race_data.get("rides", [])
            n_runners = len([r for r in rides if r.get("ride_status") == "RUNNER"])
            betting = race_data.get("betting_forecast", [])

            log.info(
                f"  {course_name} {race_time} - {n_runners} runners "
                f"({race_type}, Class {race_class})"
            )

            for ride in rides:
                if ride.get("ride_status") != "RUNNER":
                    continue

                horse = ride.get("horse", {})
                trainer = ride.get("trainer", {})
                jockey = ride.get("jockey", {})
                handicap = ride.get("handicap", "")

                # Get betting odds
                odds_str = ""
                bet_info = ride.get("betting", {})
                if bet_info:
                    odds_str = bet_info.get("current_odds", "")

                runner = {
                    "race_date": target_date.isoformat(),
                    "race_time": race_time,
                    "track": course_name,
                    "horse_name": horse.get("name", "Unknown"),
                    "horse_age": horse.get("age"),
                    "horse_sex": horse.get("sex", {}).get("type", ""),
                    "jockey_name": jockey.get("name", ""),
                    "trainer": trainer.get("name", ""),
                    "official_rating": ride.get("bha_rating", None),
                    "pounds": parse_weight(handicap),
                    "stall": ride.get("draw_number"),
                    "headgear": ", ".join(
                        h.get("type", "") for h in ride.get("headgear", [])
                    ) if ride.get("headgear") else "",
                    "days_since_lr": horse.get("last_ran_days"),
                    "number_of_runners": n_runners,
                    "race_class": race_class,
                    "race_distance": distance,
                    "dist_furlongs": parse_distance_to_furlongs(distance),
                    "going_description": going,
                    "surface_type": course_surface,
                    "race_type": race_type,
                    "race_name": race_name,
                    "odds": parse_odds_to_decimal(odds_str),
                    "cloth_number": ride.get("cloth_number"),
                    "form": horse.get("formsummary", {}).get("display_text", ""),
                }

                # Extract OR from previous results if not in ride directly
                if runner["official_rating"] is None:
                    prev = horse.get("previous_results", [])
                    if prev:
                        for p in prev:
                            bha = p.get("bha")
                            if bha:
                                runner["official_rating"] = bha
                                break

                all_runners.append(runner)

    if not all_runners:
        return pd.DataFrame()

    df = pd.DataFrame(all_runners)
    log.info(
        f"Scraped {len(df)} runners across "
        f"{df['track'].nunique()} venues, "
        f"{df.groupby(['track', 'race_time']).ngroups} races"
    )
    return df


# ---------------------------------------------------------------------------
# Feature Engineering & Prediction
# ---------------------------------------------------------------------------

def build_prediction_features(runners_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature vectors from race card data.

    Uses build_context_features from train_bfsp where possible,
    and fills in additional derived features.
    """
    df = runners_df.copy()

    # Ensure required columns exist
    for col in ["official_rating", "race_class", "going_description",
                "horse_age", "pounds", "stall", "days_since_lr",
                "career_runs", "surface_type", "race_type", "horse_sex",
                "headgear", "track", "dist_furlongs", "number_of_runners"]:
        if col not in df.columns:
            df[col] = np.nan

    # Use the standard context feature builder
    from train_bfsp import build_context_features
    df = build_context_features(df)

    # Additional derived features we can compute from race card
    # Max/median OR per race
    race_key = df["race_date"].astype(str) + "_" + df["track"].astype(str) + "_" + df["race_time"].astype(str)
    df["raceid"] = race_key

    or_vals = pd.to_numeric(df["official_rating"], errors="coerce")
    df["or_num"] = or_vals
    df["max_or_race"] = df.groupby("raceid")["or_num"].transform("max")
    df["median_or_num"] = df.groupby("raceid")["or_num"].transform("median")
    df["or_vs_max"] = df["or_num"] - df["max_or_race"]
    df["or_vs_median"] = df["or_num"] - df["median_or_num"]

    # Weight vs race average
    df["pounds_num"] = pd.to_numeric(df["pounds"], errors="coerce")
    df["weight_vs_avg"] = df["pounds_num"] - df.groupby("raceid")["pounds_num"].transform("mean")

    # Stall relative position
    df["stall_num"] = pd.to_numeric(df["stall"], errors="coerce")
    max_stall = df.groupby("raceid")["stall_num"].transform("max")
    df["draw_relative"] = df["stall_num"] / max_stall.replace(0, np.nan)

    return df


def load_model(model_dir: str) -> tuple[lgb.Booster, list[str]]:
    """Load trained BFSP model and feature columns."""
    model_path = os.path.join(model_dir, "bfsp_model.lgb")
    meta_path = os.path.join(model_dir, "bfsp_model_meta.json")

    if not os.path.exists(model_path):
        log.error(f"Model not found at {model_path}")
        sys.exit(1)

    model = lgb.Booster(model_file=model_path)

    feature_cols = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        feature_cols = meta.get("feature_cols", [])

    if not feature_cols:
        from train_bfsp import ALL_FEATURE_COLS
        feature_cols = list(ALL_FEATURE_COLS)

    log.info(f"Loaded BFSP model ({len(feature_cols)} features)")
    return model, feature_cols


def predict_bfsp(
    runners_df: pd.DataFrame,
    model: lgb.Booster,
    feature_cols: list[str],
    historical_df: pd.DataFrame | None = None,
    target_date: date | None = None,
) -> pd.DataFrame:
    """Generate BFSP predictions for runners.

    If historical_df is provided, runs the full custom metrics pipeline.
    Otherwise, uses only race-card-derived features (NaN for the rest).
    """
    if historical_df is not None and len(historical_df) > 0 and target_date:
        # Full pipeline with historical metrics
        log.info("Using full historical metrics pipeline")
        from predict_bfsp_today import prepare_and_predict
        return prepare_and_predict(
            historical_df, runners_df, model, feature_cols, target_date
        )

    # Race-card-only prediction (no historical DB)
    log.info("Using race-card-only features (no historical database)")
    df = build_prediction_features(runners_df)

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Predict
    X = df[feature_cols].astype(float)
    log_bfsp_pred = model.predict(X)

    df["predicted_log_bfsp"] = log_bfsp_pred
    df["predicted_bfsp"] = np.exp(log_bfsp_pred)
    df["predicted_win_prob"] = 1.0 / df["predicted_bfsp"]

    # Normalise probabilities per race
    race_prob_sum = df.groupby("raceid")["predicted_win_prob"].transform("sum")
    df["predicted_win_prob_norm"] = df["predicted_win_prob"] / race_prob_sum
    df["predicted_bfsp_raw"] = df["predicted_bfsp"]
    df["predicted_bfsp"] = 1.0 / df["predicted_win_prob_norm"]

    return df


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

def format_predictions(predictions: pd.DataFrame, target_date: date) -> str:
    """Format predictions into a readable report."""
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(
        f"  BFSP PREDICTIONS — {target_date.strftime('%A %d %B %Y')}"
    )
    lines.append("=" * 78)

    if len(predictions) == 0:
        lines.append("  No runners found.")
        return "\n".join(lines)

    has_db = "preracehorsecareerNFP" in predictions.columns and predictions["preracehorsecareerNFP"].notna().any()
    if not has_db:
        lines.append("  NOTE: Predictions based on race-card features only (no historical DB)")
        lines.append("  Accuracy is reduced without full custom metrics.")
        lines.append("")

    n_races = predictions["raceid"].nunique()
    lines.append(f"  {len(predictions)} runners across {n_races} races")
    lines.append("")

    for race_id, race_df in predictions.groupby("raceid"):
        race_df = race_df.sort_values("predicted_bfsp")

        track = race_df["track"].iloc[0] if "track" in race_df.columns else "?"
        rtime = race_df["race_time"].iloc[0] if "race_time" in race_df.columns else "?"
        n_run = len(race_df)
        race_name = race_df["race_name"].iloc[0][:40] if "race_name" in race_df.columns else ""
        race_class = race_df["race_class"].iloc[0] if "race_class" in race_df.columns else ""

        lines.append(f"  {track} — {rtime} ({n_run} runners) Class {race_class}")
        if race_name:
            lines.append(f"  {race_name}")
        lines.append("  " + "-" * 74)
        lines.append(
            f"  {'#':<3} {'Horse':<24} {'Pred BFSP':>10} "
            f"{'P(Win)':>7} {'OR':>4} {'Odds':>6} {'Form':>10}"
        )
        lines.append("  " + "-" * 74)

        for rank, (_, row) in enumerate(race_df.iterrows(), 1):
            horse = str(row.get("horse_name", "?"))[:23]
            pred_bfsp = row.get("predicted_bfsp", 0)
            p_win = row.get("predicted_win_prob_norm", 0)
            or_val = row.get("official_rating", np.nan)
            odds = row.get("odds", np.nan)
            form = str(row.get("form", ""))[:10]

            or_str = f"{int(or_val)}" if pd.notna(or_val) else "-"
            odds_str = f"{odds:.1f}" if pd.notna(odds) else "-"

            lines.append(
                f"  {rank:<3} {horse:<24} {pred_bfsp:>10.2f} "
                f"{p_win:>6.1%} {or_str:>4} {odds_str:>6} {form:>10}"
            )

        lines.append("")

    # Top selections
    lines.append("  TOP SELECTIONS (predicted lowest BFSP per race)")
    lines.append("  " + "=" * 74)

    for race_id, race_df in predictions.groupby("raceid"):
        top = race_df.sort_values("predicted_bfsp").iloc[0]
        track = top.get("track", "?")
        rtime = top.get("race_time", "?")
        horse = top.get("horse_name", "?")
        pred_bfsp = top.get("predicted_bfsp", 0)
        p_win = top.get("predicted_win_prob_norm", 0)

        lines.append(
            f"  {track} {rtime}: {horse} "
            f"— Pred BFSP {pred_bfsp:.2f}, P(Win) {p_win:.1%}"
        )

    lines.append("")
    lines.append("  " + "-" * 74)
    lines.append("  Generated by Ashcroft BFSP Prediction Model")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate race predictions for today"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--model-dir", type=str, default=MODEL_DIR,
        help="Path to model artifacts",
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help="Path to SQLite database (optional, improves accuracy)",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Save predictions to CSV file",
    )
    parser.add_argument(
        "--start-date", type=str, default="2020-01-01",
        help="Earliest historical date to load (default: 2020-01-01)",
    )
    args = parser.parse_args()

    target_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )

    log.info(f"Generating predictions for {target_date}")

    # Load model
    model, feature_cols = load_model(args.model_dir)

    # Scrape today's race cards
    session = create_session()
    runners = scrape_todays_runners(session, target_date)

    if len(runners) == 0:
        log.error("No runners found. Exiting.")
        sys.exit(1)

    # Try to load historical data if database exists
    historical_df = None
    if os.path.exists(args.db):
        log.info(f"Loading historical data from {args.db}...")
        from predict_bfsp_today import load_historical
        historical_df = load_historical(args.db, start_date=args.start_date)
        log.info(f"  {len(historical_df):,} historical rows loaded")

        # Use only history before target date
        historical_df = historical_df[
            historical_df["race_date"].dt.date < target_date
        ].copy()
    else:
        log.warning(
            f"Database not found at {args.db}. "
            "Predictions will use race-card features only."
        )

    # Generate predictions
    predictions = predict_bfsp(
        runners, model, feature_cols, historical_df, target_date
    )

    if len(predictions) == 0:
        log.error("No predictions generated")
        sys.exit(1)

    # Format and display
    report = format_predictions(predictions, target_date)
    print(report)

    # Save to CSV
    output_csv = args.output_csv or os.path.join(
        SCRIPT_DIR, "data", "predictions",
        f"predictions_{target_date.isoformat()}.csv"
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    output_cols = [
        "race_date", "race_time", "track", "horse_name",
        "predicted_bfsp", "predicted_win_prob_norm",
        "official_rating", "odds", "form",
        "race_class", "race_distance", "going_description",
        "jockey_name", "trainer",
    ]
    avail = [c for c in output_cols if c in predictions.columns]
    predictions[avail].to_csv(output_csv, index=False)
    log.info(f"Saved predictions to {output_csv}")

    return predictions


if __name__ == "__main__":
    main()
