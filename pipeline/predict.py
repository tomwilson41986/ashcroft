"""Morning prediction job.

1. Ensure database is available
2. Run Ashcroft BFSP model for today
3. Match predictions to Betfair markets (market_id + selection_id)
4. Write predictions CSV to S3

Entry point: python -m pipeline.predict
"""

from datetime import date

from pipeline.shared import setup_logging, ensure_database

log = setup_logging("pipeline.predict")


def match_predictions_to_markets(predictions, target_date):
    """Match model predictions to Betfair market/selection IDs."""
    from ultra_betting.betfair.auth import ensure_session
    from ultra_betting.betfair.client import get_client, BetfairAPIError
    from betfair_client import match_runner_name

    try:
        ensure_session()
        client = get_client()
        markets = client.list_horse_racing_markets(target_date)
    except BetfairAPIError as e:
        log.warning(f"Could not fetch Betfair markets: {e}")
        return predictions

    # Build lookup: (venue_lower, runner_name_lower) -> (market_id, selection_id, start_time)
    bf_lookup = {}
    for market in markets:
        event = market.get("event", {})
        venue = event.get("venue", event.get("name", "")).strip().lower()
        from betfair_client import _normalise_venue
        venue_norm = _normalise_venue(event.get("venue", "")).strip().lower()
        start_time = market.get("marketStartTime", "")

        for runner in market.get("runners", []):
            sel_id = runner["selectionId"]
            name = runner["runnerName"]
            bf_lookup[(venue_norm, name.strip().lower())] = {
                "market_id": market["marketId"],
                "selection_id": sel_id,
                "market_start_time": start_time,
            }

    matched = 0
    for pred in predictions:
        venue = pred.venue.strip().lower()
        name = pred.runner_name.strip().lower()

        # Try direct match
        key = (venue, name)
        if key in bf_lookup:
            info = bf_lookup[key]
            pred.market_id = info["market_id"]
            pred.selection_id = info["selection_id"]
            pred.prediction_id = f"{pred.date}_{pred.market_id}_{pred.selection_id}"
            matched += 1
            continue

        # Try fuzzy match (strip country suffix)
        import re
        name_clean = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", name)
        for (v, n), info in bf_lookup.items():
            n_clean = re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", n)
            if venue in v or v in venue:
                if name_clean == n_clean:
                    pred.market_id = info["market_id"]
                    pred.selection_id = info["selection_id"]
                    pred.prediction_id = f"{pred.date}_{pred.market_id}_{pred.selection_id}"
                    matched += 1
                    break

    log.info(f"Matched {matched}/{len(predictions)} predictions to Betfair markets")
    return predictions


def main():
    target_date = date.today()
    log.info(f"=== PREDICT JOB — {target_date} ===")

    # Step 1: Ensure database
    ensure_database()

    # Step 2: Run predictions
    from ultra_betting.model.predict import run_predictions, predictions_to_dataframe
    predictions = run_predictions(target_date=target_date)

    if not predictions:
        log.warning("No predictions generated, exiting")
        return

    log.info(f"Generated {len(predictions)} predictions")

    # Step 3: Match to Betfair markets
    predictions = match_predictions_to_markets(predictions, target_date)

    # Step 4: Write to S3
    from ultra_betting.data.s3 import write_csv
    df = predictions_to_dataframe(predictions)
    write_csv("predictions", target_date, df)

    log.info(f"=== PREDICT JOB COMPLETE — {len(predictions)} predictions written ===")


if __name__ == "__main__":
    main()
