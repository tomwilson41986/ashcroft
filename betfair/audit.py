"""Audit trail and daily P&L logging for automated betting.

Writes one JSONL (JSON Lines) file per day under ``betting_logs/``.
Each line is a self-contained JSON record of a placed (or attempted)
bet.  The audit log is the source of truth for daily exposure, bet
counts, and running P&L.
"""

import json
import logging
import os
from datetime import date, datetime, timezone

import pandas as pd

log = logging.getLogger(__name__)


class BetAuditLog:
    """Append-only audit log for bet placement.

    Args:
        log_dir: Directory to write JSONL files into.
        target_date: The racing date (defaults to today).
    """

    def __init__(
        self,
        log_dir: str = "betting_logs",
        target_date: date | None = None,
    ):
        self.log_dir = log_dir
        self.target_date = target_date or date.today()
        os.makedirs(self.log_dir, exist_ok=True)
        self._filepath = os.path.join(
            self.log_dir, f"{self.target_date.isoformat()}.jsonl"
        )
        # In-memory cache of today's bets
        self._bets: list[dict] = self._load_existing()

    def _load_existing(self) -> list[dict]:
        """Load any bets already logged today (e.g. from a re-run)."""
        if not os.path.exists(self._filepath):
            return []
        bets = []
        with open(self._filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    bets.append(json.loads(line))
        return bets

    def log_bet(
        self,
        row: pd.Series | dict,
        price: float,
        stake: float,
        bet_id: str | None,
        status: str,
    ) -> None:
        """Record a single bet to the JSONL audit log.

        Args:
            row: Bet card row (Series or dict) with horse/race info.
            price: The Betfair tick price the order was placed at.
            stake: Stake in GBP.
            bet_id: Betfair bet ID (``None`` for dry runs / errors).
            status: Placement status string.
        """
        if isinstance(row, pd.Series):
            row = row.to_dict()

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "race_date": str(self.target_date),
            "track": str(row.get("track", "")),
            "race_time": str(row.get("race_time", "")),
            "horse_name": str(row.get("horse_name", "")),
            "market_id": str(row.get("market_id", "")),
            "selection_id": _safe_int(row.get("selection_id")),
            "side": "BACK",
            "price": price,
            "stake_gbp": stake,
            "bet_id": bet_id,
            "status": status,
            "p_model": _safe_float(row.get("p_model")),
            "p_combined": _safe_float(row.get("p_combined")),
            "edge_pct": _safe_float(row.get("edge_pct")),
            "kelly_f": _safe_float(row.get("kelly_f")),
        }

        self._bets.append(record)

        with open(self._filepath, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_daily_summary(self) -> dict:
        """Write and return a summary record for the day."""
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "daily_summary",
            "race_date": str(self.target_date),
            "total_bets": len(self._bets),
            "total_staked": sum(b.get("stake_gbp", 0) for b in self._bets),
            "statuses": _count_statuses(self._bets),
        }

        with open(self._filepath, "a") as f:
            f.write(json.dumps(summary) + "\n")

        log.info(
            f"Daily summary: {summary['total_bets']} bets, "
            f"GBP {summary['total_staked']:.2f} staked"
        )
        return summary

    # ------------------------------------------------------------------
    # Query helpers (used by BettingEngine safety checks)
    # ------------------------------------------------------------------

    def get_daily_exposure(self) -> float:
        """Total stakes placed/attempted today."""
        return sum(
            b.get("stake_gbp", 0)
            for b in self._bets
            if b.get("status") in ("SUCCESS", "DRY_RUN")
        )

    def get_daily_bet_count(self) -> int:
        """Number of bets placed/attempted today."""
        return sum(
            1 for b in self._bets
            if b.get("status") in ("SUCCESS", "DRY_RUN")
        )

    def get_daily_pnl(self) -> float:
        """Running P&L for today (0 until bets are settled)."""
        # P&L is only known after settlement. For safety checks during
        # placement we return 0 (worst case: loss limit only triggers
        # when we have settlement data from a previous run).
        return sum(b.get("pnl", 0) for b in self._bets)

    def get_race_bet_count(self, market_id: str) -> int:
        """Number of bets placed on a specific market today."""
        return sum(
            1 for b in self._bets
            if b.get("market_id") == market_id
            and b.get("status") in ("SUCCESS", "DRY_RUN")
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None and not pd.isna(val) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    try:
        return int(val) if val is not None and not pd.isna(val) else None
    except (TypeError, ValueError):
        return None


def _count_statuses(bets: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in bets:
        s = b.get("status", "UNKNOWN")
        counts[s] = counts.get(s, 0) + 1
    return counts
