"""Sync pipeline data to the web dashboard S3 keys.

Reads settlements and daily P&L from the pipeline's S3 paths and writes
them to the dashboard keys that the Netlify frontend reads:

    dashboard/bets.json      — all settled + pending bets
    dashboard/daily_pnl.json — daily P&L summaries

Called automatically after predict (adds PENDING bets) and settle (updates
results and P&L).

Entry point: python -m pipeline.sync_dashboard
"""

import json
import logging
from datetime import date, timedelta

from pipeline.shared import setup_logging

log = setup_logging("pipeline.sync_dashboard")

# S3 keys the dashboard frontend reads
DASHBOARD_BETS_KEY = "dashboard/bets.json"
DASHBOARD_PNL_KEY = "dashboard/daily_pnl.json"


def sync_dashboard(lookback_days: int = 90):
    """Read pipeline artifacts from S3 and write dashboard JSON files.

    Scans the last `lookback_days` for predictions, executions, and
    settlements, merges them into the dashboard format, and writes
    the combined data to the dashboard S3 keys.
    """
    from ultra_betting.data import s3 as pipe_s3
    from web.s3_store import save_bets, save_daily_pnl

    today = date.today()
    all_bets = []
    all_daily_pnl = []
    cumulative_pnl = 0.0
    bank = 1000.0

    # Read existing dashboard data to preserve history beyond lookback
    existing_bets = _read_dashboard_json(DASHBOARD_BETS_KEY) or []
    existing_pnl = _read_dashboard_json(DASHBOARD_PNL_KEY) or []

    lookback_start = today - timedelta(days=lookback_days)

    # Dates covered by fresh pipeline data
    fresh_dates = set()

    for day_offset in range(lookback_days, -1, -1):
        dt = today - timedelta(days=day_offset)
        dt_str = str(dt)

        # Try to read settlements first (settled bets)
        settle_df = pipe_s3.read_csv("settlements", dt)
        if not settle_df.empty:
            fresh_dates.add(dt_str)
            day_bets = 0
            day_winners = 0
            day_pnl = 0.0
            day_staked = 0.0

            for _, row in settle_df.iterrows():
                result = row.get("result", "PENDING")
                if result not in ("WON", "LOST"):
                    continue

                stake = float(row.get("stake", 10))
                net = float(row.get("net_pnl", 0))
                gross = float(row.get("pnl", 0))
                commission = float(row.get("commission", 0))

                all_bets.append({
                    "race_date": dt_str,
                    "race_time": str(row.get("race_time", "")),
                    "track": str(row.get("venue", "")),
                    "horse_name": str(row.get("runner_name", "")),
                    "predicted_bfsp": None,
                    "predicted_win_prob": None,
                    "betfair_back": float(row.get("price_matched", 0)) if row.get("price_matched") else None,
                    "edge_pct": None,
                    "bet_type": str(row.get("side", "BACK")),
                    "stake": stake,
                    "price": float(row.get("price_matched", 0)) if row.get("price_matched") else None,
                    "status": result,
                    "actual_bsp": None,
                    "placing": None,
                    "gross_pnl": round(gross, 2),
                    "commission": round(commission, 2),
                    "net_pnl": round(net, 2),
                })

                day_bets += 1
                day_pnl += net
                day_staked += stake
                if result == "WON":
                    day_winners += 1

            if day_bets > 0:
                cumulative_pnl += day_pnl
                bank_start = bank
                bank += day_pnl
                roi = (day_pnl / day_staked * 100) if day_staked > 0 else 0

                all_daily_pnl.append({
                    "date": dt_str,
                    "num_bets": day_bets,
                    "winners": day_winners,
                    "losers": day_bets - day_winners,
                    "voids": 0,
                    "total_staked": round(day_staked, 2),
                    "gross_pnl": round(day_pnl, 2),
                    "commission": 0,
                    "net_pnl": round(day_pnl, 2),
                    "roi_pct": round(roi, 2),
                    "cumulative_pnl": round(cumulative_pnl, 2),
                    "bank_start": round(bank_start, 2),
                    "bank_end": round(bank, 2),
                })
            continue

        # No settlements — check for predictions (pending bets for today)
        pred_df = pipe_s3.read_csv("predictions", dt)
        if not pred_df.empty and dt == today:
            fresh_dates.add(dt_str)
            for _, row in pred_df.iterrows():
                all_bets.append({
                    "race_date": dt_str,
                    "race_time": str(row.get("race_time", "")),
                    "track": str(row.get("venue", row.get("track", ""))),
                    "horse_name": str(row.get("runner_name", row.get("horse_name", ""))),
                    "predicted_bfsp": round(float(row["predicted_bfsp"]), 2) if "predicted_bfsp" in row and row.get("predicted_bfsp") else None,
                    "predicted_win_prob": round(float(row["win_prob"]), 4) if "win_prob" in row and row.get("win_prob") else None,
                    "betfair_back": None,
                    "edge_pct": round(float(row["edge_pct"]), 1) if "edge_pct" in row and row.get("edge_pct") else None,
                    "bet_type": "BACK",
                    "stake": 10,
                    "price": None,
                    "status": "PENDING",
                    "actual_bsp": None,
                    "placing": None,
                    "gross_pnl": 0,
                    "commission": 0,
                    "net_pnl": 0,
                })

    # Merge: keep existing bets for dates we didn't refresh
    if existing_bets:
        kept = [b for b in existing_bets if b.get("race_date") not in fresh_dates]
        all_bets = kept + all_bets

    if existing_pnl:
        kept_pnl = [p for p in existing_pnl if p.get("date") not in fresh_dates]
        all_daily_pnl = kept_pnl + all_daily_pnl

    # Sort
    all_bets.sort(key=lambda b: (b.get("race_date", ""), b.get("race_time", "")))
    all_daily_pnl.sort(key=lambda p: p.get("date", ""))

    # Recompute cumulative P&L across all days
    cumulative = 0.0
    bank = 1000.0
    for day in all_daily_pnl:
        bank_start = bank
        cumulative += day.get("net_pnl", 0)
        bank += day.get("net_pnl", 0)
        day["cumulative_pnl"] = round(cumulative, 2)
        day["bank_start"] = round(bank_start, 2)
        day["bank_end"] = round(bank, 2)

    # Write to dashboard S3 keys
    log.info(f"Writing {len(all_bets)} bets to {DASHBOARD_BETS_KEY}")
    save_bets(all_bets)

    log.info(f"Writing {len(all_daily_pnl)} daily P&L records to {DASHBOARD_PNL_KEY}")
    save_daily_pnl(all_daily_pnl)

    log.info("Dashboard sync complete")


def _read_dashboard_json(key: str) -> list | None:
    """Read a JSON list from the dashboard S3 bucket."""
    try:
        from web.s3_store import _read_json
        return _read_json(key)
    except Exception:
        return None


def main():
    sync_dashboard()


if __name__ == "__main__":
    main()
