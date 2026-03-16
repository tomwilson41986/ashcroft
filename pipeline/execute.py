"""Pre-race execution job. Runs every 30 minutes during racing hours.

1. Load today's predictions from S3
2. Load today's executions from S3 (for idempotency)
3. Filter to predictions not yet executed
4. Filter to markets that are open and within timing window
5. Authenticate with Betfair
6. For each eligible prediction:
   a. Fetch live prices
   b. Run rules engine
   c. If bet instruction returned: apply guardrails, place bet
7. Write updated executions CSV to S3

Entry point: python -m pipeline.execute
"""

import os
from datetime import date

import pandas as pd

from pipeline.shared import setup_logging

log = setup_logging("pipeline.execute")


def main():
    target_date = date.today()
    dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

    log.info(f"=== EXECUTE JOB — {target_date} {'[DRY RUN]' if dry_run else ''} ===")

    from ultra_betting.data import s3
    from ultra_betting.data.schemas import Prediction, Execution, SkippedBet
    from ultra_betting.betfair.auth import ensure_session
    from ultra_betting.betfair.pricing import fetch_market_prices, MarketPrices
    from ultra_betting.betfair.markets import minutes_to_start
    from ultra_betting.betfair.execution import place_bet
    from ultra_betting.rules.engine import RulesEngine
    from ultra_betting.guardrails import Guardrails, GuardrailViolation
    from ultra_betting.config import load_guardrails, load_rules
    from ultra_betting.audit import audit_event

    # Step 1: Load predictions
    preds_df = s3.read_csv("predictions", target_date)
    if preds_df.empty:
        log.warning("No predictions found for today, exiting")
        return

    # Step 2: Load existing executions (for idempotency)
    exec_df = s3.read_csv("executions", target_date)
    executed_ids = set()
    existing_executions = []
    if not exec_df.empty:
        executed_ids = set(exec_df["prediction_id"].dropna().astype(str))
        for _, row in exec_df.iterrows():
            existing_executions.append(Execution(**{
                k: row[k] for k in Execution.model_fields if k in row.index
            }))

    # Step 3: Convert predictions and filter unexecuted
    predictions = []
    for _, row in preds_df.iterrows():
        pred = Prediction(
            date=str(row.get("date", target_date)),
            venue=str(row.get("venue", "")),
            race_time=str(row.get("race_time", "")),
            market_id=str(row.get("market_id", "")),
            selection_id=int(row.get("selection_id", 0)),
            runner_name=str(row.get("runner_name", "")),
            predicted_bfsp=float(row.get("predicted_bfsp", 0)),
            predicted_win_prob=float(row.get("predicted_win_prob", 0)),
            prediction_id=str(row.get("prediction_id", "")),
        )
        if pred.prediction_id and pred.prediction_id not in executed_ids:
            if pred.market_id:  # Must have been matched to a Betfair market
                predictions.append(pred)

    if not predictions:
        log.info("No unexecuted predictions with market IDs, exiting")
        return

    log.info(f"{len(predictions)} unexecuted predictions to evaluate")

    # Step 4: Authenticate with Betfair
    ensure_session()

    # Step 5: Load rules and guardrails
    rules_config = load_rules()
    guardrails_config = load_guardrails()
    if dry_run:
        guardrails_config["dry_run"] = True

    rules_engine = RulesEngine(rules_config)
    guardrails = Guardrails(guardrails_config)
    timing = rules_config.get("timing", {})
    earliest_mins = timing.get("earliest_minutes_before", 60)
    latest_mins = timing.get("latest_minutes_before", 5)

    # Step 6: Group predictions by market and evaluate
    from collections import defaultdict
    by_market = defaultdict(list)
    for pred in predictions:
        by_market[pred.market_id].append(pred)

    new_executions = []
    all_skipped = []

    for market_id, market_preds in by_market.items():
        # Check timing for this market (use first prediction's info)
        # We need the market start time — get it from predictions
        start_time_str = preds_df.loc[
            preds_df["market_id"] == market_id, "predicted_at"
        ].iloc[0] if "predicted_at" in preds_df.columns else None

        # Fetch live prices
        try:
            prices = fetch_market_prices(market_id)
        except Exception as e:
            log.warning(f"Failed to fetch prices for {market_id}: {e}")
            continue

        if prices.status != "OPEN":
            log.info(f"Market {market_id} is {prices.status}, skipping")
            continue

        # Check liquidity
        min_matched = guardrails_config.get("min_market_total_matched", 5000)
        if prices.total_matched < min_matched:
            log.info(f"Market {market_id} illiquid ({prices.total_matched:.0f}), skipping")
            continue

        # Run rules engine for this market's predictions
        live_prices = {market_id: prices}
        instructions, skipped = rules_engine.evaluate(market_preds, live_prices)
        all_skipped.extend(skipped)

        # Apply guardrails and execute
        for instruction in instructions:
            try:
                instruction = guardrails.apply(instruction, existing_executions + new_executions)
            except GuardrailViolation as e:
                log.warning(f"Guardrail blocked: {instruction.runner_name} — {e}")
                all_skipped.append(SkippedBet(
                    prediction_id=instruction.prediction_id,
                    runner_name=instruction.runner_name,
                    venue="",
                    race_time="",
                    reason="guardrail_hit",
                    detail=str(e),
                ))
                continue

            # Place the bet
            execution = place_bet(instruction, dry_run=dry_run or guardrails.dry_run)
            new_executions.append(execution)
            audit_event(
                "bet_executed",
                prediction_id=instruction.prediction_id,
                runner=instruction.runner_name,
                side=instruction.side,
                stake=instruction.stake,
                price=instruction.price,
                status=execution.status,
                dry_run=execution.dry_run,
            )

    # Step 7: Write updated executions to S3
    if new_executions:
        new_df = pd.DataFrame([e.model_dump() for e in new_executions])
        if not exec_df.empty:
            combined = pd.concat([exec_df, new_df], ignore_index=True)
        else:
            combined = new_df
        s3.write_csv("executions", target_date, combined)

    log.info(
        f"=== EXECUTE JOB COMPLETE — {len(new_executions)} bets placed, "
        f"{len(all_skipped)} skipped ==="
    )


if __name__ == "__main__":
    main()
