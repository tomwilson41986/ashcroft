"""Evening settlement job.

1. Load today's executions from S3
2. Authenticate with Betfair
3. Fetch settled bets from Betfair API
4. Match settlements to executions
5. Calculate per-bet P&L
6. Calculate daily totals
7. Update cumulative P&L tracker
8. Generate HTML report with chart
9. Email report
10. Write settlements + daily P&L to S3

Entry point: python -m pipeline.settle
"""

from datetime import date

import pandas as pd

from pipeline.shared import setup_logging

log = setup_logging("pipeline.settle")


def main():
    target_date = date.today()
    log.info(f"=== SETTLE JOB — {target_date} ===")

    from ultra_betting.data import s3
    from ultra_betting.data.schemas import Execution, Settlement, DailyPnL
    from ultra_betting.betfair.auth import ensure_session
    from ultra_betting.betfair.settlement import get_settled_bets, match_settlements_to_executions
    from ultra_betting.reporting.pnl import (
        calculate_settlement_pnl,
        calculate_daily_pnl,
        update_cumulative_pnl,
        get_current_bank,
    )
    from ultra_betting.reporting.daily_report import build_html_report, send_email_report
    from ultra_betting.audit import audit_event

    # Step 1: Load executions
    exec_df = s3.read_csv("executions", target_date)
    if exec_df.empty:
        log.info("No executions found for today, exiting")
        return

    # Filter to actual bets (not dry runs, not failed)
    matched_df = exec_df[exec_df["status"].isin(["MATCHED", "DRY_RUN"])]
    if matched_df.empty:
        log.info("No matched/dry-run executions to settle")
        return

    executions = []
    for _, row in matched_df.iterrows():
        executions.append(Execution(**{
            k: row[k] for k in Execution.model_fields if k in row.index
        }))

    bet_ids = {e.bet_id for e in executions if e.bet_id and not e.dry_run}

    # Step 2: Authenticate with Betfair
    ensure_session()

    # Step 3: Fetch settled bets
    bf_settlements = get_settled_bets(target_date)
    bf_matched = match_settlements_to_executions(bf_settlements, bet_ids)

    # Build lookup: bet_id -> settlement data
    bf_lookup = {}
    for s in bf_matched:
        bf_lookup[str(s.get("betId", ""))] = s

    # Step 4 & 5: Create settlement records with P&L
    settlements = []
    for exec_rec in executions:
        if exec_rec.dry_run:
            # For dry runs, simulate the result
            result = "PENDING"
            gross, comm, net = 0, 0, 0
        elif exec_rec.bet_id in bf_lookup:
            bf_data = bf_lookup[exec_rec.bet_id]
            profit = bf_data.get("profit", 0)
            if profit > 0:
                result = "WON"
            elif profit < 0:
                result = "LOST"
            else:
                result = "VOID"
            gross, comm, net = calculate_settlement_pnl(exec_rec, result)
        else:
            log.warning(f"No settlement found for bet_id={exec_rec.bet_id}")
            result = "PENDING"
            gross, comm, net = 0, 0, 0

        settlements.append(Settlement(
            date=str(target_date),
            venue=exec_rec.venue,
            race_time=exec_rec.race_time,
            market_id=exec_rec.market_id,
            selection_id=exec_rec.selection_id,
            runner_name=exec_rec.runner_name,
            side=exec_rec.side,
            stake=exec_rec.stake,
            price_matched=exec_rec.price_matched,
            result=result,
            pnl=round(gross, 2),
            commission=round(comm, 2),
            net_pnl=round(net, 2),
            bet_id=exec_rec.bet_id,
        ))

    # Step 6: Calculate daily P&L
    settled_only = [s for s in settlements if s.result in ("WON", "LOST")]
    bank_start = get_current_bank()
    daily_pnl = calculate_daily_pnl(settled_only, bank_start)
    daily_pnl.date = str(target_date)

    log.info(
        f"Daily P&L: {daily_pnl.bets_placed} bets, "
        f"{daily_pnl.bets_won}W/{daily_pnl.bets_lost}L, "
        f"Net: £{daily_pnl.net_pnl:.2f}, ROI: {daily_pnl.roi_percent:.1f}%"
    )

    # Step 7: Update cumulative P&L
    cumulative = update_cumulative_pnl(daily_pnl)

    # Step 8: Generate HTML report
    html = build_html_report(
        daily_pnl=daily_pnl,
        settlements=settled_only,
        cumulative_data=cumulative,
    )

    # Step 9: Email report
    subject = (
        f"Ultra Betting — {target_date} — "
        f"{'+'if daily_pnl.net_pnl >= 0 else ''}£{daily_pnl.net_pnl:.2f} "
        f"({daily_pnl.roi_percent:+.1f}%)"
    )
    send_email_report(html, subject)

    # Step 10: Write to S3
    settle_df = pd.DataFrame([s.model_dump() for s in settlements])
    s3.write_csv("settlements", target_date, settle_df)
    s3.write_json(f"daily_pnl/{target_date}.json", daily_pnl.model_dump())
    s3.write_html("reports", target_date, html)

    audit_event(
        "settlement_complete",
        date=str(target_date),
        bets=daily_pnl.bets_placed,
        net_pnl=daily_pnl.net_pnl,
        bank_end=daily_pnl.bank_end,
    )

    log.info(f"=== SETTLE JOB COMPLETE ===")


if __name__ == "__main__":
    main()
