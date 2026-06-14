"""
Daily Predictions Pipeline.

Pulls today's racecards from horseracebase.com, builds historical features
for each declared runner, runs predictions through the trained model,
and emails the results.

Usage:
    python daily_predictions.py                    # Run for today
    python daily_predictions.py --date 2026-03-10  # Run for a specific date
    python daily_predictions.py --dry-run          # Skip email, print to stdout

Environment variables (in .env):
    HRB_USERNAME        - horseracebase.com login
    HRB_PASSWORD        - horseracebase.com password
    SMTP_HOST           - SMTP server (default: smtp.gmail.com)
    SMTP_PORT           - SMTP port (default: 587)
    SMTP_USERNAME       - Email sender address
    SMTP_PASSWORD       - Email password / app password
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import smtplib
import sqlite3
import sys
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from betfair_client import BetfairClient, BetfairAPIError, match_runner_name
from model.benter_blend import BenterBlender
from model.custom_metrics import CustomMetricsEngine
from model.overlay_detector import OverlayDetector
from model.prerace_builder import PreRaceBuilder
from model.probability_model import FundamentalModel

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
MODEL_DIR = os.path.join(SCRIPT_DIR, "data", "models")
RACECARD_DIR = os.path.join(SCRIPT_DIR, "csv", "racecards")

BASE_URL = "https://www.horseracebase.com"
LOGIN_URL = f"{BASE_URL}/horseracebase_login.php"
RESULTS_URL = f"{BASE_URL}/horse-racing-results.php"
TODAY_URL = f"{BASE_URL}/horse-racing-today.php"
CSV_URL = f"{BASE_URL}/excelresults.php"

RECIPIENT_EMAIL = "racingsquared@gmail.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(SCRIPT_DIR, "daily_predictions.log")
        ),
    ],
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Step 1 & 2: Pull racecards from horseracebase.com
# ------------------------------------------------------------------

def create_session() -> requests.Session:
    """Create an HTTP session with browser-like headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return session


def login(session: requests.Session) -> str | None:
    """Log in to horseracebase.com. Returns user ID on success."""
    username = os.getenv("HRB_USERNAME")
    password = os.getenv("HRB_PASSWORD")

    if not username or not password:
        log.error(
            "HRB_USERNAME and HRB_PASSWORD environment variables must be set"
        )
        return None

    # Get CSRF token from the login page. The login form posts back to the
    # same URL and carries a hidden "csrf_token" field.
    resp = session.get(LOGIN_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    csrf_input = (
        soup.find("input", {"name": "csrf_token"})
        or soup.find("input", {"name": "CSRFtoken"})
    )
    if not csrf_input:
        log.error("Could not find CSRF token on login page")
        return None

    # Post login to the login page (form has no action, so it self-submits).
    resp2 = session.post(LOGIN_URL, data={
        "login": username,
        "password": password,
        "csrf_token": csrf_input.get("value"),
    })
    resp2.raise_for_status()

    if "not recognised" in resp2.text.lower():
        log.error("Login failed: credentials not recognised")
        return None

    # Get user ID from results page
    resp3 = session.get(RESULTS_URL)
    soup3 = BeautifulSoup(resp3.text, "lxml")
    user_input = soup3.find("input", {"name": "user"})
    if not user_input:
        log.error("Login failed: could not find user ID")
        return None

    user_id = user_input.get("value")
    log.info(f"Logged in successfully (user_id={user_id})")
    return user_id


def download_racecard_csv(
    session: requests.Session, user_id: str, target_date: date
) -> str | None:
    """Download racecard CSV for a given date via the CSV export endpoint.

    Tries the same excelresults.php endpoint used for historical results,
    as HRB uses the same format for today's declared runners.
    """
    racedate = f"{target_date.year}-{target_date.month}-{target_date.day}"

    resp = session.post(CSV_URL, data={
        "csv": "1",
        "user": user_id,
        "racedate": racedate,
    })
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if not resp.text.strip():
        return None

    # Check if we got valid CSV data (header row should contain 'horse_name')
    first_line = resp.text.strip().split("\n")[0].lower()
    if "horse_name" in first_line or "racedate" in first_line:
        log.info(f"Downloaded racecard CSV ({len(resp.text)} bytes)")
        return resp.text

    return None


def scrape_racecard_html(
    session: requests.Session, target_date: date
) -> pd.DataFrame | None:
    """Scrape today's racecards from the onedayracecards.php page.

    This page contains all races for the day in structured tables,
    each preceded by span elements with race metadata (time, track,
    class, distance, going, prize money).
    """
    log.info("Attempting HTML racecard scrape from onedayracecards.php...")
    racecard_url = f"{BASE_URL}/onedayracecards.php"
    resp = session.get(racecard_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    all_runners = []
    race_count = 0

    # Find race tables: those whose first row starts with 'No.'
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header_cells = rows[0].find_all(["td", "th"])
        header_texts = [c.get_text(strip=True) for c in header_cells]
        if not header_texts or header_texts[0] != "No." or len(header_texts) < 8:
            continue

        # Build column index from header
        col_idx = {}
        for i, h in enumerate(header_texts):
            col_idx[h] = i

        # Extract race metadata from preceding span elements
        track = ""
        race_time = ""
        going = ""
        race_class = ""
        race_distance = ""
        prize_money = ""
        num_runners_declared = 0
        race_name = ""

        prev_spans = []
        prev = table.find_previous_sibling()
        while prev and len(prev_spans) < 5:
            if prev.name == "span":
                prev_spans.insert(0, prev.get_text(strip=True))
            elif prev.name == "table":
                break
            prev = prev.find_previous_sibling()

        for span_text in prev_spans:
            # Pattern: "1.20 Cheltenham(11 runners)"
            time_track = re.match(
                r"(\d{1,2}\.\d{2})\s+(.+?)(?:\((\d+)\s*runners?\))?$",
                span_text,
            )
            if time_track:
                race_time = time_track.group(1).replace(".", ":")
                track = time_track.group(2).strip()
                if time_track.group(3):
                    num_runners_declared = int(time_track.group(3))
                continue

            # Pattern: "Class 1, 2m½f , Good To Soft, 4yo+, Win: £84405"
            info_match = re.match(
                r"(?:(?:Grade\s*\d|Class\s*\d)[^,]*)?,?\s*(.*)", span_text
            )
            class_match = re.search(r"(Grade\s*\d|Class\s*\d)", span_text, re.I)
            if class_match:
                race_class = class_match.group(1)

            going_match = re.search(
                r"(Heavy|Soft|Good To Soft|Good To Firm|Good|Firm|Hard|"
                r"Standard To Slow|Standard|Slow|Yielding)",
                span_text, re.I,
            )
            if going_match:
                going = going_match.group(1)

            dist_match = re.search(
                r"(\d+m\S*f?\s*\S*|\d+f\S*)", span_text
            )
            if dist_match:
                race_distance = dist_match.group(1).strip().rstrip(",")

            prize_match = re.search(r"Win:\s*£([\d,]+)", span_text)
            if prize_match:
                prize_money = prize_match.group(1)

            # If no time/track matched, it might be the race name
            if not time_track and not class_match and not going_match:
                race_name = span_text

        # Determine if this race has a 'Stall' column (flat races)
        has_stall = "Stall" in col_idx

        # Parse runner rows
        race_runners = []
        for row in rows[1:]:  # Skip header
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(strip=True) for c in cells]

            if len(cell_texts) < 8:
                continue

            # Skip rows where first cell is not a number (e.g. stallion notes)
            if not cell_texts[0].isdigit():
                continue

            runner = {
                "race_date": target_date.isoformat(),
                "race_time": race_time,
                "track": track,
                "going_description": going,
                "race_class": race_class,
                "race_distance": race_distance,
                "prize_money": prize_money,
                "race_name": race_name,
            }

            # Parse by known column positions
            if "No." in col_idx:
                runner["stall"] = _safe_int(cell_texts[col_idx["No."]])
            if "Horse" in col_idx:
                runner["horse_name"] = cell_texts[col_idx["Horse"]]
            if "Age" in col_idx:
                runner["horse_age"] = _safe_int(cell_texts[col_idx["Age"]])
            if "Weight" in col_idx:
                weight_match = re.match(
                    r"^(\d{1,2})-(\d{1,2})$",
                    cell_texts[col_idx["Weight"]],
                )
                if weight_match:
                    stones = int(weight_match.group(1))
                    lbs = int(weight_match.group(2))
                    runner["pounds"] = stones * 14 + lbs
            if "Jockey" in col_idx:
                runner["jockey_name"] = cell_texts[col_idx["Jockey"]]
            if "Trainer" in col_idx:
                runner["trainer"] = cell_texts[col_idx["Trainer"]]
            if "OR" in col_idx:
                runner["official_rating"] = _safe_int(cell_texts[col_idx["OR"]])
            if "Headgear" in col_idx:
                runner["headgear"] = cell_texts[col_idx["Headgear"]]
            if "Days" in col_idx:
                days_text = cell_texts[col_idx["Days"]]
                if days_text.isdigit():
                    runner["days_since_lr"] = int(days_text)
            if has_stall and "Stall" in col_idx:
                runner["stall"] = _safe_int(cell_texts[col_idx["Stall"]])
            if "Odds" in col_idx:
                odds_text = cell_texts[col_idx["Odds"]]
                odds_match = re.match(r"^(\d+)/(\d+)$", odds_text)
                if odds_match:
                    num = int(odds_match.group(1))
                    den = int(odds_match.group(2))
                    runner["odds"] = num / den + 1  # Convert to decimal

            if runner.get("horse_name"):
                race_runners.append(runner)

        # Set number of runners for this race
        num_runners = len(race_runners)
        for r in race_runners:
            r["number_of_runners"] = num_runners

        all_runners.extend(race_runners)
        race_count += 1

    if not all_runners:
        return None

    df = pd.DataFrame(all_runners)
    log.info(f"Scraped {len(df)} runners from {race_count} races")
    return df


def _safe_int(text: str) -> int | None:
    """Convert text to int, returning None if not a valid integer."""
    try:
        return int(text)
    except (ValueError, TypeError):
        return None


def save_racecard_csv(csv_text: str, target_date: date) -> str:
    """Save racecard CSV to disk and return the file path."""
    os.makedirs(RACECARD_DIR, exist_ok=True)
    filepath = os.path.join(
        RACECARD_DIR, f"racecard_{target_date.isoformat()}.csv"
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(csv_text)
    log.info(f"Saved racecard CSV to {filepath}")
    return filepath


def parse_racecard_csv(csv_text: str) -> pd.DataFrame:
    """Parse the racecard CSV into a DataFrame."""
    df = pd.read_csv(io.StringIO(csv_text))

    # Normalise column names to match our DB schema
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

    return df


def get_todays_runners(
    session: requests.Session, user_id: str, target_date: date
) -> pd.DataFrame:
    """Get today's declared runners from HRB, trying CSV first then HTML."""

    # Attempt 1: CSV download via the export endpoint
    csv_text = download_racecard_csv(session, user_id, target_date)

    if csv_text:
        save_racecard_csv(csv_text, target_date)
        df = parse_racecard_csv(csv_text)
        if len(df) > 0:
            log.info(
                f"Got {len(df)} runners from CSV for {target_date}"
            )
            return df

    # Attempt 2: HTML scrape of the racecard page
    log.info("CSV download returned no data, trying HTML scrape...")
    df = scrape_racecard_html(session, target_date)

    if df is not None and len(df) > 0:
        # Save as CSV for records
        csv_out = df.to_csv(index=False)
        save_racecard_csv(csv_out, target_date)
        return df

    log.error(f"Could not get racecard data for {target_date}")
    return pd.DataFrame()


# ------------------------------------------------------------------
# Step 3: Look up runners and generate historic data
# ------------------------------------------------------------------

def load_historical_data(db_path: str) -> pd.DataFrame:
    """Load all historical race results from the SQLite database."""
    if not os.path.exists(db_path):
        log.error(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM race_results ORDER BY race_date, race_time", conn
    )
    conn.close()

    df["race_date"] = pd.to_datetime(df["race_date"])
    log.info(f"Loaded {len(df):,} historical rows")
    return df


def build_features_for_runners(
    today_runners: pd.DataFrame,
    historical_df: pd.DataFrame,
    target_date: date,
) -> pd.DataFrame:
    """Calculate custom metrics on history and build features for today's runners.

    Steps:
        1. Run CustomMetricsEngine on the full historical dataset
        2. Build PreRaceBuilder lookups from the enriched history
        3. Build feature vectors for today's declared runners
    """
    # Limit history to last 4 years — sufficient for expanding metrics
    # and dramatically faster than processing the full 15+ year database
    cutoff = pd.Timestamp(str(target_date)) - pd.DateOffset(years=4)
    recent_df = historical_df[historical_df["race_date"] >= cutoff].copy()
    log.info(
        f"  Trimmed history to {len(recent_df):,} rows "
        f"(from {cutoff.date()} onwards, was {len(historical_df):,})"
    )

    log.info("Calculating custom metrics on historical data...")
    engine = CustomMetricsEngine()
    enriched = engine.calculate_all(recent_df)
    log.info(f"  Calculated metrics for {len(enriched):,} rows")

    log.info("Building pre-race feature lookups...")
    builder = PreRaceBuilder(enriched)
    feature_cols = builder.get_feature_columns()

    # Prepare today's runners DataFrame for the builder
    runners = today_runners.copy()

    # Ensure required columns exist with correct names
    if "race_class" in runners.columns:
        # Extract numeric class from strings like "Class 5" or "Grade 1"
        runners["race_class"] = (
            runners["race_class"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
        )
        runners["race_class"] = pd.to_numeric(
            runners["race_class"], errors="coerce"
        )
    if "dist_furlongs" not in runners.columns and "race_distance" in runners.columns:
        runners["dist_furlongs"] = _parse_distance_to_furlongs(
            runners["race_distance"]
        )
    if "going_description" not in runners.columns:
        runners["going_description"] = "Good"

    log.info(
        f"Building features for {len(runners)} runners "
        f"across {runners['track'].nunique() if 'track' in runners.columns else '?'} "
        f"tracks..."
    )
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

    return features_df, feature_cols


def _parse_distance_to_furlongs(distance_series: pd.Series) -> pd.Series:
    """Convert distance strings like '2m½f', '2m 4f', or '7f' to furlongs."""
    def convert(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()

        # Replace Unicode fractions
        s = s.replace("½", ".5").replace("¼", ".25").replace("¾", ".75")

        s = s.lower()

        # Try direct numeric
        try:
            return float(s)
        except ValueError:
            pass

        miles = 0
        furlongs = 0.0
        yards = 0

        m_match = re.search(r"(\d+(?:\.\d+)?)\s*m", s)
        if m_match:
            miles = float(m_match.group(1))
        f_match = re.search(r"(\d+(?:\.\d+)?)\s*f", s)
        if f_match:
            furlongs = float(f_match.group(1))
        y_match = re.search(r"(\d+)\s*y", s)
        if y_match:
            yards = int(y_match.group(1))

        total = miles * 8 + furlongs + yards / 220
        return total if total > 0 else np.nan

    return distance_series.apply(convert)


# ------------------------------------------------------------------
# Step 3b: Enrich runners with live Betfair exchange odds
# ------------------------------------------------------------------

def enrich_with_betfair_odds(
    runners: pd.DataFrame,
    target_date: date,
) -> pd.DataFrame:
    """Fetch live Betfair exchange odds and merge into runner data.

    Matches runners by venue + horse name. Adds columns:
    - betfair_back: best available back price
    - betfair_lay: best available lay price
    - betfair_sp_near: projected SP (near side)
    - betfair_last_traded: last matched price
    - betfair_market_id: Betfair market ID for reference

    If BFSP column is missing or empty, fills it with the best back
    price as a live market proxy.

    Args:
        runners: DataFrame of today's runners.
        target_date: Date to query Betfair markets.

    Returns:
        runners DataFrame enriched with Betfair columns.
    """
    # Check credentials
    if not os.getenv("BETFAIR_APP_KEY"):
        log.info("BETFAIR_APP_KEY not set - skipping Betfair odds enrichment")
        return runners

    try:
        client = BetfairClient()
        client.login()
    except BetfairAPIError as e:
        log.warning(f"Betfair login failed, skipping odds enrichment: {e}")
        return runners

    try:
        live_data = client.get_live_odds_for_date(target_date)
    except Exception as e:
        log.warning(f"Failed to fetch Betfair odds: {e}")
        client.logout()
        return runners
    finally:
        client.logout()

    if not live_data:
        log.info("No Betfair market data returned")
        return runners

    log.info(f"  Fetched {len(live_data)} Betfair runner prices")

    # Build Betfair lookup: (venue_lower, horse_name_lower) -> odds data
    bf_lookup = {}
    for row in live_data:
        venue = (row.get("venue") or "").strip().lower()
        name = (row.get("runner_name") or "").strip()
        # Strip country suffix for matching
        name_clean = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name).strip().lower()
        key = (venue, name_clean)
        bf_lookup[key] = row

    # Match and merge
    runners = runners.copy()
    runners["betfair_back"] = np.nan
    runners["betfair_lay"] = np.nan
    runners["betfair_sp_near"] = np.nan
    runners["betfair_last_traded"] = np.nan
    runners["betfair_market_id"] = ""

    matched = 0
    for idx, runner in runners.iterrows():
        track = str(runner.get("track", "")).strip().lower()
        horse = str(runner.get("horse_name", "")).strip().lower()
        horse_clean = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", horse).strip()

        # Try direct match
        bf = bf_lookup.get((track, horse_clean))

        # Try partial venue match if direct fails
        if bf is None:
            for (venue, name), data in bf_lookup.items():
                if name == horse_clean and (venue in track or track in venue):
                    bf = data
                    break

        if bf is not None:
            runners.at[idx, "betfair_back"] = bf.get("best_back_price")
            runners.at[idx, "betfair_lay"] = bf.get("best_lay_price")
            runners.at[idx, "betfair_sp_near"] = bf.get("sp_near_price")
            runners.at[idx, "betfair_last_traded"] = bf.get("last_traded_price")
            runners.at[idx, "betfair_market_id"] = bf.get("market_id", "")
            matched += 1

    log.info(f"  Matched {matched}/{len(runners)} runners to Betfair odds")

    # If BFSP column is missing/empty, use best back price as proxy
    if "bfsp" not in runners.columns:
        runners["bfsp"] = np.nan
    mask = runners["bfsp"].isna() & runners["betfair_back"].notna()
    if mask.any():
        runners.loc[mask, "bfsp"] = runners.loc[mask, "betfair_back"]
        log.info(
            f"  Filled {mask.sum()} missing BFSP values with Betfair back prices"
        )

    return runners


# ------------------------------------------------------------------
# Step 4: Run predictions
# ------------------------------------------------------------------

def run_predictions(
    features_df: pd.DataFrame, feature_cols: list[str], model_dir: str
) -> pd.DataFrame:
    """Load the trained model and generate predictions.

    Returns DataFrame with predicted probabilities and BFSP.
    """
    # Load trained model
    model_path = os.path.join(model_dir, "probability_model.lgb")
    meta_path = os.path.join(model_dir, "probability_model_meta.json")

    if not os.path.exists(model_path):
        log.error(f"Trained model not found at {model_path}")
        log.error("Run 'python -m model.trainer' first to train a model.")
        sys.exit(1)

    log.info("Loading trained model...")
    model = FundamentalModel.load(model_dir)

    # Use model's own feature columns if available (ensures consistency)
    if model.feature_cols:
        model_features = model.feature_cols
    else:
        model_features = feature_cols

    # Ensure all model features exist in the data, fill missing with NaN
    for col in model_features:
        if col not in features_df.columns:
            features_df[col] = np.nan

    log.info(f"Running predictions on {len(features_df)} runners...")
    predictions = model.predict(
        features_df, raceid_col="raceid", normalise=True
    )

    # Load blend config and apply blending
    blend_path = os.path.join(model_dir, "blend_config.json")
    if os.path.exists(blend_path):
        with open(blend_path) as f:
            blend_config = json.load(f)
        optimal_lambda = blend_config.get("optimal_lambda", 0.80)
    else:
        optimal_lambda = 0.80

    # For pre-race predictions, we typically don't have actual BFSP yet.
    # If early market prices are available (from morning odds), use them.
    blender = BenterBlender()
    if "bfsp" in predictions.columns and predictions["bfsp"].notna().any():
        log.info(
            f"Blending with market prices (lambda={optimal_lambda})..."
        )
        predictions = blender.blend(
            predictions, lambda_=optimal_lambda
        )
    else:
        # No market prices - use pure model predictions
        log.info("No market prices available - using pure model predictions")
        predictions["p_combined"] = predictions["p_model"]
        predictions["model_fair_bfsp"] = predictions["predicted_bfsp"]
        predictions["edge"] = 0.0
        predictions["edge_pct"] = 0.0

    return predictions


# ------------------------------------------------------------------
# Step 5: Email predictions
# ------------------------------------------------------------------

def format_predictions_text(predictions: pd.DataFrame, target_date: date) -> str:
    """Format predictions into a readable plain-text report."""
    lines = []
    lines.append(f"DAILY RACING PREDICTIONS - {target_date.strftime('%A %d %B %Y')}")
    lines.append("=" * 70)
    lines.append("")

    if len(predictions) == 0:
        lines.append("No runners found for today.")
        return "\n".join(lines)

    # Group by race
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
    n_races = (
        predictions[group_col].nunique() if group_col else 1
    )
    lines.append(f"Total: {total_runners} runners across {n_races} races")
    lines.append("")

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            race_df = race_df.sort_values("p_model", ascending=False)

            track = race_df["track"].iloc[0] if "track" in race_df.columns else "?"
            rtime = race_df["race_time"].iloc[0] if "race_time" in race_df.columns else "?"
            n_run = len(race_df)

            lines.append(f"  {track} - {rtime} ({n_run} runners)")
            lines.append("  " + "-" * 66)
            has_bf = "betfair_back" in race_df.columns and race_df["betfair_back"].notna().any()
            if has_bf:
                lines.append(
                    f"  {'Rank':<5} {'Horse':<25} {'P(Win)':<8} "
                    f"{'Pred BFSP':<10} {'Fair BFSP':<10} {'BF Back':<9}"
                )
            else:
                lines.append(
                    f"  {'Rank':<5} {'Horse':<25} {'P(Win)':<8} "
                    f"{'Pred BFSP':<10} {'Fair BFSP':<10}"
                )
            lines.append("  " + "-" * 66)

            for rank, (_, row) in enumerate(race_df.iterrows(), 1):
                horse = str(row.get("horse_name", "?"))[:24]
                p_win = row.get("p_model", 0)
                pred_bfsp = row.get("predicted_bfsp", 0)
                fair_bfsp = row.get("model_fair_bfsp", pred_bfsp)

                line = (
                    f"  {rank:<5} {horse:<25} {p_win:<8.1%} "
                    f"{pred_bfsp:<10.1f} {fair_bfsp:<10.1f}"
                )
                if has_bf:
                    bf_back = row.get("betfair_back")
                    bf_s = f"{bf_back:.2f}" if pd.notna(bf_back) else "-"
                    line += f" {bf_s:<9}"
                lines.append(line)
            lines.append("")
    else:
        predictions = predictions.sort_values("p_model", ascending=False)
        for rank, (_, row) in enumerate(predictions.iterrows(), 1):
            horse = str(row.get("horse_name", "?"))[:24]
            p_win = row.get("p_model", 0)
            pred_bfsp = row.get("predicted_bfsp", 0)
            lines.append(f"  {rank}. {horse} - P(Win): {p_win:.1%}, "
                         f"Predicted BFSP: {pred_bfsp:.1f}")

    # Top picks summary
    lines.append("")
    lines.append("TOP PICKS (Highest P(Win) per race)")
    lines.append("=" * 70)

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            top = race_df.sort_values("p_model", ascending=False).iloc[0]
            track = top.get("track", "?")
            rtime = top.get("race_time", "?")
            horse = top.get("horse_name", "?")
            p_win = top.get("p_model", 0)
            pred_bfsp = top.get("predicted_bfsp", 0)
            lines.append(
                f"  {track} {rtime}: {horse} "
                f"(P={p_win:.1%}, BFSP={pred_bfsp:.1f})"
            )

    # Value bets (if edge data available)
    if "edge" in predictions.columns:
        value_bets = predictions[predictions["edge"] > 0.05].sort_values(
            "edge", ascending=False
        )
        if len(value_bets) > 0:
            lines.append("")
            lines.append("VALUE BETS (Edge > 5%)")
            lines.append("=" * 70)
            for _, row in value_bets.iterrows():
                horse = row.get("horse_name", "?")
                track = row.get("track", "?")
                rtime = row.get("race_time", "?")
                edge = row.get("edge_pct", 0)
                lines.append(
                    f"  {track} {rtime}: {horse} (Edge: {edge:.1f}%)"
                )

    lines.append("")
    lines.append("-" * 70)
    lines.append("Generated by Ultra Betting prediction pipeline")
    lines.append(f"Model: {MODEL_DIR}")
    lines.append(
        "Disclaimer: Predictions are for informational purposes only. "
        "Bet responsibly."
    )

    return "\n".join(lines)


def format_predictions_html(predictions: pd.DataFrame, target_date: date) -> str:
    """Format predictions as an HTML email body."""
    html = []
    html.append("<html><body>")
    html.append(
        f"<h2>Daily Racing Predictions - "
        f"{target_date.strftime('%A %d %B %Y')}</h2>"
    )

    if len(predictions) == 0:
        html.append("<p>No runners found for today.</p>")
        html.append("</body></html>")
        return "\n".join(html)

    # Determine grouping column
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
    html.append(
        f"<p><strong>{total_runners} runners</strong> across "
        f"<strong>{n_races} races</strong></p>"
    )

    # Top picks summary table
    html.append("<h3>Top Picks (Highest P(Win) per race)</h3>")
    html.append(
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse; font-family:monospace;'>"
    )
    html.append(
        "<tr style='background:#333; color:white;'>"
        "<th>Track</th><th>Time</th><th>Horse</th>"
        "<th>P(Win)</th><th>Pred BFSP</th></tr>"
    )

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            top = race_df.sort_values("p_model", ascending=False).iloc[0]
            track = top.get("track", "?")
            rtime = top.get("race_time", "?")
            horse = top.get("horse_name", "?")
            p_win = top.get("p_model", 0)
            pred_bfsp = top.get("predicted_bfsp", 0)
            html.append(
                f"<tr><td>{track}</td><td>{rtime}</td>"
                f"<td><strong>{horse}</strong></td>"
                f"<td>{p_win:.1%}</td><td>{pred_bfsp:.1f}</td></tr>"
            )
    html.append("</table>")

    # Full race-by-race breakdown
    html.append("<h3>Full Race Breakdown</h3>")

    if group_col:
        for race_id, race_df in predictions.groupby(group_col):
            race_df = race_df.sort_values("p_model", ascending=False)
            track = race_df["track"].iloc[0] if "track" in race_df.columns else "?"
            rtime = race_df["race_time"].iloc[0] if "race_time" in race_df.columns else "?"

            html.append(
                f"<h4>{track} - {rtime} ({len(race_df)} runners)</h4>"
            )
            html.append(
                "<table border='1' cellpadding='4' cellspacing='0' "
                "style='border-collapse:collapse; font-size:13px;'>"
            )
            has_bf = "betfair_back" in race_df.columns and race_df["betfair_back"].notna().any()
            bf_header = "<th>BF Back</th><th>BF Lay</th>" if has_bf else ""
            html.append(
                "<tr style='background:#eee;'>"
                "<th>#</th><th>Horse</th><th>P(Win)</th>"
                f"<th>Predicted BFSP</th><th>Fair BFSP</th>{bf_header}</tr>"
            )

            for rank, (_, row) in enumerate(race_df.iterrows(), 1):
                horse = row.get("horse_name", "?")
                p_win = row.get("p_model", 0)
                pred_bfsp = row.get("predicted_bfsp", 0)
                fair_bfsp = row.get("model_fair_bfsp", pred_bfsp)
                bg = "#ffffcc" if rank == 1 else ""
                style = f" style='background:{bg}'" if bg else ""
                bf_cells = ""
                if has_bf:
                    bf_back = row.get("betfair_back")
                    bf_lay = row.get("betfair_lay")
                    bf_back_s = f"{bf_back:.2f}" if pd.notna(bf_back) else "-"
                    bf_lay_s = f"{bf_lay:.2f}" if pd.notna(bf_lay) else "-"
                    bf_cells = f"<td>{bf_back_s}</td><td>{bf_lay_s}</td>"
                html.append(
                    f"<tr{style}><td>{rank}</td>"
                    f"<td>{horse}</td>"
                    f"<td>{p_win:.1%}</td>"
                    f"<td>{pred_bfsp:.1f}</td>"
                    f"<td>{fair_bfsp:.1f}</td>{bf_cells}</tr>"
                )
            html.append("</table>")

    # Value bets
    if "edge" in predictions.columns:
        value_bets = predictions[predictions["edge"] > 0.05].sort_values(
            "edge", ascending=False
        )
        if len(value_bets) > 0:
            html.append("<h3>Value Bets (Edge &gt; 5%)</h3>")
            html.append(
                "<table border='1' cellpadding='4' cellspacing='0' "
                "style='border-collapse:collapse;'>"
            )
            html.append(
                "<tr style='background:#4CAF50; color:white;'>"
                "<th>Track</th><th>Time</th><th>Horse</th>"
                "<th>Edge</th></tr>"
            )
            for _, row in value_bets.iterrows():
                html.append(
                    f"<tr><td>{row.get('track', '?')}</td>"
                    f"<td>{row.get('race_time', '?')}</td>"
                    f"<td>{row.get('horse_name', '?')}</td>"
                    f"<td>{row.get('edge_pct', 0):.1f}%</td></tr>"
                )
            html.append("</table>")

    html.append("<hr>")
    html.append(
        "<p style='font-size:11px; color:#888;'>"
        "Generated by Ultra Betting prediction pipeline. "
        "Predictions are for informational purposes only. "
        "Bet responsibly.</p>"
    )
    html.append("</body></html>")

    return "\n".join(html)


def send_email(
    recipient: str,
    subject: str,
    text_body: str,
    html_body: str,
) -> bool:
    """Send predictions email via SMTP.

    Uses environment variables for SMTP configuration:
        SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_pass:
        log.error(
            "SMTP_USERNAME and SMTP_PASSWORD environment variables must be set "
            "to send email. Add them to your .env file."
        )
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
        log.info(f"Email sent to {recipient}")
        return True
    except Exception as e:
        log.error(f"Failed to send email: {e}")
        return False


# ------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------

def run_pipeline(
    target_date: date,
    db_path: str = DB_PATH,
    model_dir: str = MODEL_DIR,
    recipient: str = RECIPIENT_EMAIL,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Execute the full daily predictions pipeline.

    1. Pull today's racecards from horseracebase.com
    2. Download and store racecard CSV
    3. Look up runners and generate historic performance data
    4. Run through predictions model to predict BFSP
    5. Send predictions via email

    Args:
        target_date: The date to predict for.
        db_path: Path to the historical SQLite database.
        model_dir: Directory containing trained model artifacts.
        recipient: Email address to send predictions to.
        dry_run: If True, print predictions but don't send email.

    Returns:
        DataFrame of predictions.
    """
    log.info(f"{'='*60}")
    log.info(f"DAILY PREDICTIONS PIPELINE - {target_date}")
    log.info(f"{'='*60}")

    # Step 1 & 2: Pull racecards
    log.info("Step 1: Logging in to horseracebase.com...")
    session = create_session()
    user_id = login(session)
    if not user_id:
        log.error("Failed to login. Check HRB_USERNAME and HRB_PASSWORD.")
        sys.exit(1)

    log.info("Step 2: Downloading today's racecard...")
    today_runners = get_todays_runners(session, user_id, target_date)

    if len(today_runners) == 0:
        log.warning(f"No runners found for {target_date}. No racing today?")
        return pd.DataFrame()

    log.info(
        f"  Found {len(today_runners)} runners for {target_date}"
    )

    # Step 3: Generate historic features
    log.info("Step 3: Loading historical data and building features...")
    historical = load_historical_data(db_path)

    # Step 3b: Enrich with live Betfair exchange odds
    log.info("Step 3b: Fetching live Betfair exchange odds...")
    today_runners = enrich_with_betfair_odds(today_runners, target_date)

    features_df, feature_cols = build_features_for_runners(
        today_runners, historical, target_date
    )
    log.info(f"  Built features for {len(features_df)} runners")

    # Step 4: Run predictions
    log.info("Step 4: Running predictions...")
    predictions = run_predictions(features_df, feature_cols, model_dir)
    log.info(f"  Generated predictions for {len(predictions)} runners")

    # Step 5: Email results
    subject = (
        f"Racing Predictions - "
        f"{target_date.strftime('%A %d %b %Y')}"
    )
    text_body = format_predictions_text(predictions, target_date)
    html_body = format_predictions_html(predictions, target_date)

    if dry_run:
        log.info("DRY RUN - printing predictions to stdout:")
        print("\n" + text_body)
    else:
        log.info(f"Step 5: Sending predictions to {recipient}...")
        sent = send_email(recipient, subject, text_body, html_body)
        if sent:
            log.info("Pipeline complete - email sent successfully!")
        else:
            log.warning(
                "Pipeline complete but email failed. "
                "Printing predictions to stdout:"
            )
            print("\n" + text_body)

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Daily racing predictions pipeline"
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
        "--email",
        type=str,
        default=RECIPIENT_EMAIL,
        help=f"Recipient email (default: {RECIPIENT_EMAIL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print predictions to stdout instead of sending email",
    )
    args = parser.parse_args()

    target_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )

    run_pipeline(
        target_date=target_date,
        db_path=args.db,
        model_dir=args.model_dir,
        recipient=args.email,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
