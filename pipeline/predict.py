"""Morning prediction job.

1. Ensure database is available
2. Run Ashcroft BFSP model for today, keeping the card as fetched (S3 racecards/)
3. Match predictions to Betfair markets (market_id + selection_id)
4. Snapshot the exchange price alongside each prediction
5. Write predictions CSV to S3

Entry point: python -m pipeline.predict
"""

from datetime import date, datetime, timezone

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


def attach_exchange_prices(predictions, target_date):
    """Write the exchange price beside each prediction, as it stands now.

    This is the other half of forward closing-line value. The racecard gives a
    bookmaker's early price; this gives the exchange's -- best back, best lay,
    the projected SP, and what has been matched so far -- at the moment the
    forecast was made. Without it the only price on record is the one the
    market closed at, and a forecast of the close cannot be scored against the
    close for value.

    This is not cheap: `get_live_odds_for_date` re-lists the day's markets and
    then issues one `listMarketBook` per market, so a UK/IE card costs thirty
    to forty serial calls on top of the catalogue call step 3 already made.
    That is why it runs after the predictions have been written rather than
    before -- failure or delay here costs the prices, never the card.
    """
    from ultra_betting.betfair.auth import ensure_session
    from ultra_betting.betfair.client import get_client

    matched = [p for p in predictions if p.market_id and p.selection_id]
    if not matched:
        log.warning("No predictions carry a market id; skipping the price snapshot")
        return predictions

    try:
        ensure_session()
        runners = get_client().get_live_odds_for_date(target_date)
    except Exception as e:      # BetfairAPIError, a dead session, a timeout --
        log.warning(f"Could not snapshot exchange prices: {e}")   # none of them
        return predictions                                        # worth losing
                                                                  # the card over

    book = {(r.get("market_id"), r.get("selection_id")): r for r in runners}
    now = datetime.utcnow()
    filled = 0
    for pred in matched:
        r = book.get((pred.market_id, pred.selection_id))
        if not r:
            continue
        pred.bf_best_back = _pos(r.get("best_back_price"))
        pred.bf_best_lay = _pos(r.get("best_lay_price"))
        pred.bf_sp_near = _pos(r.get("sp_near_price"))
        pred.bf_sp_far = _pos(r.get("sp_far_price"))
        pred.bf_last_traded = _pos(r.get("last_traded_price"))
        # The runner's own matched volume, not the market's. `total_matched`
        # is market-level and identical across the field, so storing it here
        # would make every runner's share of the race's money exactly 1/n.
        pred.bf_total_matched = _num(r.get("runner_matched"))
        pred.price_snapshot_at = now
        filled += 1

    log.info(f"Exchange price on {filled}/{len(predictions)} predictions "
             f"({sum(p.bf_sp_near is not None for p in predictions)} with a projected SP)")
    return predictions


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pos(v):
    """A price, or None. Betfair returns 0 for "nothing on offer"."""
    f = _num(v)
    return f if f is not None and f > 1.0 else None


def main():
    target_date = date.today()
    log.info(f"=== PREDICT JOB — {target_date} ===")

    # Step 1: Ensure database
    ensure_database()

    # Step 2: Run predictions, keeping the card as fetched. The result rows later
    # overwrite the going, the jockeys and the field, so this is the only record
    # of what was known at 06:00. Keyed by the fetch time: a re-run adds a file
    # and never replaces the morning's.
    from ultra_betting.data.s3 import write_csv
    from ultra_betting.model.predict import run_predictions, predictions_to_dataframe
    fetched_at = datetime.now(timezone.utc).strftime("%H%M")
    predictions = run_predictions(
        target_date=target_date,
        card_sink=lambda card: write_csv("racecards", f"{target_date}_{fetched_at}", card),
    )

    if not predictions:
        log.warning("No predictions generated, exiting")
        return

    log.info(f"Generated {len(predictions)} predictions")

    # Step 3: Match to Betfair markets
    predictions = match_predictions_to_markets(predictions, target_date)

    # Step 4: Write to S3 BEFORE reaching for prices.
    #
    # The price snapshot is a live Betfair call and the client sets no socket
    # timeout, so a hung connection would otherwise stall here and the day's
    # predictions would never be written at all. The snapshot is worth having
    # and the predictions are worth more, so they go out first and the file is
    # rewritten once the prices are in. A failure past this point costs eight
    # columns, not the card.
    write_csv("predictions", target_date, predictions_to_dataframe(predictions))

    # Step 5: Snapshot the prices that exist right now, for forward CLV
    predictions = attach_exchange_prices(predictions, target_date)
    if any(p.price_snapshot_at for p in predictions):
        write_csv("predictions", target_date, predictions_to_dataframe(predictions))

    log.info(f"=== PREDICT JOB COMPLETE — {len(predictions)} predictions written ===")


if __name__ == "__main__":
    main()
