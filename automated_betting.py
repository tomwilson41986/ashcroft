"""Automated Betting Pipeline.

End-to-end workflow that:
1. Fetches today's racecards from HRB
2. Generates BFSP predictions using the trained regression model
3. Fetches live Betfair exchange prices
4. Matches HRB runners to Betfair selections
5. Identifies overlays by comparing predicted BFSP vs live price
6. Stakes using fractional Kelly criterion
7. Places orders on qualifying bets
8. Logs everything to an audit trail

The overlay detection uses the BFSP regression model directly (as validated
in profitability_analysis.py with 37.55% ROI), NOT the Benter-blended
probability path (which degenerates at lambda=1.0).

Usage:
    python automated_betting.py                      # Run for today
    python automated_betting.py --date 2026-03-11    # Specific date
    python automated_betting.py --dry-run            # Log bets, don't place

Environment variables:
    HRB_USERNAME / HRB_PASSWORD     - horseracebase.com credentials
    BETFAIR_USERNAME / PASSWORD      - Betfair account
    BETFAIR_APP_KEY                  - Betfair API application key
    BETFAIR_CERTS_DIR                - Path to SSL certs directory
    BETTING_ENABLED                  - Kill switch (true/false)
    BETTING_DRY_RUN                  - Dry-run mode (true/false)
    BETTING_BANKROLL                 - Bankroll in GBP
    BETTING_*                        - See betfair/config.py for all options
    SMTP_USERNAME / SMTP_PASSWORD    - Email credentials (optional)
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from betfair.audit import BetAuditLog
from betfair.betting import BettingEngine
from betfair.client import BetfairClient
from betfair.config import BettingConfig
from betfair.matcher import HorseNameMatcher
from daily_predictions import (
    RECIPIENT_EMAIL,
    create_session,
    get_todays_runners,
    login,
    send_email,
)
from predict_bfsp_today import (
    DB_PATH,
    MODEL_DIR,
    load_historical,
    prepare_and_predict,
)

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(SCRIPT_DIR, "automated_betting.log")
        ),
    ],
)
log = logging.getLogger(__name__)


def _compute_overlays(
    predictions: pd.DataFrame,
    config: BettingConfig,
) -> pd.DataFrame:
    """Compute overlays and Kelly stakes using direct BFSP comparison.

    This replicates the validated logic from profitability_analysis.py:
    - overlay_pct = (live_back_price - predicted_bfsp) / predicted_bfsp * 100
    - Kelly: p = 1/predicted_bfsp, b = live_back_price - 1

    Args:
        predictions: DataFrame with ``predicted_bfsp`` and ``live_back_price``.
        config: Betting configuration with staking parameters.

    Returns:
        DataFrame of qualifying bets with stakes.
    """
    df = predictions.copy()

    # Overlay: live price exceeds model's fair price
    df["overlay_pct"] = (
        (df["live_back_price"] - df["predicted_bfsp"])
        / df["predicted_bfsp"]
        * 100
    )

    # Filter to overlays meeting minimum edge
    min_edge_pct = config.min_edge * 100  # config stores as fraction
    qualifying = df[df["overlay_pct"] >= min_edge_pct].copy()

    if len(qualifying) == 0:
        return pd.DataFrame()

    # Kelly criterion: p = model implied prob, b = net decimal odds on offer
    p = 1.0 / qualifying["predicted_bfsp"].values
    b = qualifying["live_back_price"].values - 1.0
    kelly_f = np.where(b > 0, (p * b - (1 - p)) / b, 0)
    kelly_f = np.clip(kelly_f, 0, 0.25)  # Cap at 25% of bankroll

    qualifying["kelly_f"] = kelly_f
    qualifying["adjusted_kelly_f"] = kelly_f * config.kelly_fraction

    # Compute stakes with constraints
    stakes = qualifying["adjusted_kelly_f"] * config.bankroll
    stakes = np.minimum(stakes, config.bankroll * config.max_stake_pct)
    stakes = np.minimum(stakes, config.max_stake)
    stakes = np.where(stakes >= config.min_stake, stakes, 0)

    qualifying["stake_gbp"] = np.round(stakes, 2)

    # Drop zero stakes
    qualifying = qualifying[qualifying["stake_gbp"] > 0].copy()

    if len(qualifying) == 0:
        return pd.DataFrame()

    # Expected value (accounting for commission)
    qualifying["expected_value"] = qualifying["stake_gbp"] * (
        (1.0 / qualifying["predicted_bfsp"])
        * (qualifying["live_back_price"] - 1)
        * (1 - config.commission_rate)
        - (1 - 1.0 / qualifying["predicted_bfsp"])
    )

    # Rename for compatibility with BettingEngine
    qualifying["edge_pct"] = qualifying["overlay_pct"]
    qualifying["p_model"] = 1.0 / qualifying["predicted_bfsp"]

    return qualifying.sort_values("expected_value", ascending=False).reset_index(
        drop=True
    )


def run_pipeline(
    target_date: date,
    db_path: str = DB_PATH,
    model_dir: str = MODEL_DIR,
    dry_run: bool = False,
    recipient: str = RECIPIENT_EMAIL,
) -> pd.DataFrame:
    """Execute the full automated betting pipeline.

    Uses the BFSP regression model to predict fair prices, then detects
    overlays by comparing predicted BFSP against live Betfair exchange
    prices. This matches the approach validated in profitability_analysis.py.

    Args:
        target_date: The racing date to predict and bet on.
        db_path: Path to the historical SQLite database.
        model_dir: Directory containing trained model artifacts.
        dry_run: If True, log bets but don't place real orders.
        recipient: Email address for bet confirmation.

    Returns:
        DataFrame of the placed bet card.
    """
    log.info("=" * 60)
    log.info(f"AUTOMATED BETTING PIPELINE - {target_date}")
    log.info("=" * 60)

    # --- Load config ---
    config = BettingConfig.from_env()
    if dry_run:
        config.dry_run = True
    mode = "DRY RUN" if config.dry_run else "LIVE"
    log.info(f"Mode: {mode} | Bankroll: GBP {config.bankroll:.0f}")

    if not config.enabled:
        log.warning("Betting is DISABLED (BETTING_ENABLED=false). Exiting.")
        return pd.DataFrame()

    # --- Step 1: HRB login & racecard fetch ---
    log.info("Step 1: Fetching racecards from HRB...")
    session = create_session()
    user_id = login(session)
    if not user_id:
        log.error("Failed to login to HRB. Exiting.")
        sys.exit(1)

    today_runners = get_todays_runners(session, user_id, target_date)
    if len(today_runners) == 0:
        log.warning(f"No runners found for {target_date}. Exiting.")
        return pd.DataFrame()
    log.info(f"  Found {len(today_runners)} runners")

    # --- Step 2: Generate BFSP predictions (regression model) ---
    log.info("Step 2: Loading BFSP regression model and generating predictions...")

    # Load the BFSP regression model
    model_path = os.path.join(model_dir, "bfsp_model.lgb")
    meta_path = os.path.join(model_dir, "bfsp_model_meta.json")

    if not os.path.exists(model_path):
        log.error(f"BFSP model not found at {model_path}")
        log.error("Run 'python train_bfsp.py' first to train the model.")
        sys.exit(1)

    model = lgb.Booster(model_file=model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    historical = load_historical(db_path)
    predictions = prepare_and_predict(
        historical, today_runners, model, feature_cols, target_date
    )

    if len(predictions) == 0:
        log.warning("No predictions generated. Exiting.")
        return pd.DataFrame()

    log.info(f"  Generated BFSP predictions for {len(predictions)} runners")

    # --- Step 3: Betfair login & market fetch ---
    log.info("Step 3: Connecting to Betfair Exchange...")
    bf_client = BetfairClient()
    bf_client.login()

    try:
        target_dt = datetime(
            target_date.year, target_date.month, target_date.day,
            tzinfo=timezone.utc,
        )
        catalogues = bf_client.list_market_catalogue(target_date=target_dt)
        if not catalogues:
            log.warning("No Betfair markets found. Exiting.")
            bf_client.logout()
            return pd.DataFrame()
        log.info(f"  Found {len(catalogues)} Betfair WIN markets")

        # --- Step 4: Match HRB runners to Betfair selections ---
        log.info("Step 4: Matching runners to Betfair selections...")
        matcher = HorseNameMatcher()
        matched = matcher.match_runners(predictions, catalogues)

        matched_count = matched["selection_id"].notna().sum()
        log.info(f"  Matched {matched_count}/{len(matched)} runners")

        if matched_count == 0:
            log.warning("No runners matched to Betfair. Exiting.")
            bf_client.logout()
            return pd.DataFrame()

        # --- Step 5: Fetch live prices ---
        log.info("Step 5: Fetching live Betfair prices...")
        market_ids = matched["market_id"].dropna().unique().tolist()
        books = bf_client.get_market_books(market_ids)

        # Build price lookup: (market_id, selection_id) -> best back price
        price_lookup = {}
        for book in books:
            for runner in book["runners"]:
                key = (book["market_id"], runner["selection_id"])
                price_lookup[key] = runner.get("best_back_price")

        # Attach live prices to matched runners
        matched["live_back_price"] = matched.apply(
            lambda row: price_lookup.get(
                (row["market_id"], row["selection_id"])
            ),
            axis=1,
        )

        priced = matched["live_back_price"].notna().sum()
        log.info(f"  Got live prices for {priced}/{matched_count} matched runners")

        # --- Step 6: Detect overlays (BFSP model vs live price) ---
        log.info("Step 6: Detecting overlays (predicted BFSP vs live price)...")

        priced_df = matched[matched["live_back_price"].notna()].copy()
        if len(priced_df) == 0:
            log.warning("No runners with live prices. Exiting.")
            bf_client.logout()
            return pd.DataFrame()

        bet_card = _compute_overlays(priced_df, config)

        if len(bet_card) == 0:
            log.info("No overlays found meeting minimum edge threshold. Done.")
            bf_client.logout()
            return pd.DataFrame()

        log.info(f"  Found {len(bet_card)} qualifying bets")
        for _, row in bet_card.iterrows():
            log.info(
                f"    {row.get('horse_name', '?')} @ {row.get('track', '?')} "
                f"{row.get('race_time', '?')}: "
                f"pred={row['predicted_bfsp']:.1f} live={row['live_back_price']:.1f} "
                f"overlay={row['overlay_pct']:.1f}% "
                f"stake=GBP {row['stake_gbp']:.2f}"
            )

        # --- Step 7: Place bets ---
        log.info(f"Step 7: Placing bets [{mode}]...")
        audit = BetAuditLog(target_date=target_date)
        engine = BettingEngine(bf_client, config, audit)
        placed_card = engine.place_bet_card(bet_card)

        # --- Step 8: Audit summary ---
        summary = audit.log_daily_summary()
        log.info(f"  Summary: {summary}")

        # --- Step 9: Email confirmation ---
        if recipient:
            _send_bet_confirmation(placed_card, target_date, config, recipient)

        bf_client.logout()

    except Exception as exc:
        log.error(f"Pipeline error: {exc}", exc_info=True)
        bf_client.logout()
        raise

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)

    return placed_card


def _send_bet_confirmation(
    bet_card: pd.DataFrame,
    target_date: date,
    config: BettingConfig,
    recipient: str,
) -> None:
    """Send a bet confirmation email."""
    mode = "DRY RUN" if config.dry_run else "LIVE"
    subject = f"Betting {mode} - {target_date.strftime('%A %d %b %Y')}"

    lines = [
        f"AUTOMATED BETTING {mode} - {target_date.strftime('%A %d %B %Y')}",
        "=" * 60,
        "",
    ]

    if len(bet_card) == 0:
        lines.append("No bets placed today.")
    else:
        lines.append(f"Total bets: {len(bet_card)}")
        total_staked = bet_card["stake_gbp"].sum()
        lines.append(f"Total staked: GBP {total_staked:.2f}")
        lines.append("")

        for _, row in bet_card.iterrows():
            status = row.get("bet_status", "?")
            horse = row.get("horse_name", "?")
            track = row.get("track", "?")
            rtime = row.get("race_time", "?")
            price = row.get("live_back_price", 0)
            stake = row.get("stake_gbp", 0)
            edge = row.get("edge_pct", 0)

            lines.append(
                f"  {track} {rtime}: BACK {horse} "
                f"@ {price:.2f} for GBP {stake:.2f} "
                f"(edge={edge:.1f}%) [{status}]"
            )

    lines.append("")
    lines.append("-" * 60)
    lines.append(f"Bankroll: GBP {config.bankroll:.0f}")
    lines.append(f"Kelly fraction: {config.kelly_fraction}")
    lines.append(f"Min edge: {config.min_edge:.0%}")

    text_body = "\n".join(lines)
    html_body = "<html><body><pre>" + text_body + "</pre></body></html>"

    try:
        send_email(recipient, subject, text_body, html_body)
    except Exception as exc:
        log.warning(f"Failed to send bet confirmation email: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated Betfair betting pipeline"
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
        "--dry-run",
        action="store_true",
        help="Log bets but don't place real orders on Betfair",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=RECIPIENT_EMAIL,
        help=f"Recipient email for confirmation (default: {RECIPIENT_EMAIL})",
    )
    args = parser.parse_args()

    target_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )

    run_pipeline(
        target_date=target_date,
        db_path=args.db,
        model_dir=args.model_dir,
        dry_run=args.dry_run,
        recipient=args.email,
    )


if __name__ == "__main__":
    main()
