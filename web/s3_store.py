"""S3-backed data store for the Ashcroft web dashboard.

All dashboard data (bets, daily P&L, odds snapshots) is stored as JSON
in S3 so the API backend is stateless and needs no local database.

S3 keys:
    dashboard/bets.json         — array of all bet records
    dashboard/daily_pnl.json    — array of daily P&L summaries
    dashboard/odds/{date}.json  — odds snapshots per date
"""

import io
import json
import logging
import os
from datetime import date
from functools import lru_cache
from time import time

import boto3
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

S3_BUCKET = os.getenv("ULTRA_BETTING_S3_BUCKET", "ashcroft")

_client = None
_cache = {}
_cache_ttl = 60  # seconds


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("s3")
    return _client


def _read_json(key: str) -> list | dict | None:
    """Read JSON from S3 with simple TTL cache."""
    now = time()
    if key in _cache and now - _cache[key]["ts"] < _cache_ttl:
        return _cache[key]["data"]

    try:
        obj = _get_client().get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read().decode())
        _cache[key] = {"data": data, "ts": now}
        return data
    except _get_client().exceptions.NoSuchKey:
        log.debug(f"No file at s3://{S3_BUCKET}/{key}")
        return None
    except Exception as e:
        log.warning(f"Failed to read s3://{S3_BUCKET}/{key}: {e}")
        return None


def _write_json(key: str, data: list | dict) -> None:
    """Write JSON to S3 and update cache."""
    body = json.dumps(data, indent=2, default=str)
    _get_client().put_object(Bucket=S3_BUCKET, Key=key, Body=body)
    _cache[key] = {"data": data, "ts": time()}
    log.info(f"Wrote s3://{S3_BUCKET}/{key}")


def invalidate_cache(key: str = None):
    """Clear cache for a specific key, or all keys."""
    if key:
        _cache.pop(key, None)
    else:
        _cache.clear()


# ---------------------------------------------------------------------------
# Bets
# ---------------------------------------------------------------------------

BETS_KEY = "dashboard/bets.json"


def get_all_bets() -> list[dict]:
    return _read_json(BETS_KEY) or []


def get_bets_for_date(target_date: str) -> list[dict]:
    return [b for b in get_all_bets() if b.get("race_date") == target_date]


def get_bets_by_status(status: str) -> list[dict]:
    return [b for b in get_all_bets() if b.get("status") == status]


def get_bet_dates() -> list[str]:
    return sorted(set(b["race_date"] for b in get_all_bets()), reverse=True)


def save_bets(bets: list[dict]) -> None:
    _write_json(BETS_KEY, bets)


def upsert_bet(bet: dict) -> None:
    """Add or update a bet (matched by date+time+track+horse)."""
    bets = get_all_bets()
    key = (bet["race_date"], bet.get("race_time"), bet.get("track"), bet.get("horse_name"))

    for i, existing in enumerate(bets):
        ekey = (existing["race_date"], existing.get("race_time"), existing.get("track"), existing.get("horse_name"))
        if ekey == key:
            bets[i] = bet
            save_bets(bets)
            return

    bets.append(bet)
    save_bets(bets)


# ---------------------------------------------------------------------------
# Daily P&L
# ---------------------------------------------------------------------------

PNL_KEY = "dashboard/daily_pnl.json"


def get_daily_pnl() -> list[dict]:
    data = _read_json(PNL_KEY) or []
    return sorted(data, key=lambda x: x.get("date", ""))


def save_daily_pnl(pnl: list[dict]) -> None:
    _write_json(PNL_KEY, pnl)


# ---------------------------------------------------------------------------
# Odds snapshots
# ---------------------------------------------------------------------------

def get_odds_for_date(target_date: str) -> list[dict]:
    key = f"dashboard/odds/{target_date}.json"
    return _read_json(key) or []


def save_odds_for_date(target_date: str, odds: list[dict]) -> None:
    key = f"dashboard/odds/{target_date}.json"
    _write_json(key, odds)


# ---------------------------------------------------------------------------
# Aggregated queries (computed from the JSON data)
# ---------------------------------------------------------------------------

def get_pnl_summary() -> dict:
    """Compute overall P&L summary from all settled bets."""
    bets = get_all_bets()
    settled = [b for b in bets if b.get("status") in ("WON", "LOST")]

    if not settled:
        return {"total_bets": 0, "wins": 0, "net_pnl": 0, "roi": 0, "bank": 1000, "avg_edge": 0}

    wins = sum(1 for b in settled if b["status"] == "WON")
    net_pnl = sum(b.get("net_pnl", 0) or 0 for b in settled)
    staked = sum(b.get("stake", 0) or 0 for b in settled)
    edges = [b.get("edge_pct", 0) or 0 for b in settled]

    daily = get_daily_pnl()
    bank = daily[-1]["bank_end"] if daily else 1000

    return {
        "total_bets": len(settled),
        "wins": wins,
        "strike_rate": round(wins / len(settled) * 100, 1) if settled else 0,
        "net_pnl": round(net_pnl, 2),
        "roi": round(net_pnl / staked * 100, 2) if staked else 0,
        "avg_edge": round(sum(edges) / len(edges), 1) if edges else 0,
        "bank": round(bank, 2),
    }


def get_edge_stats() -> list[dict]:
    """Strike rate broken down by edge bucket."""
    bets = [b for b in get_all_bets() if b.get("status") in ("WON", "LOST")]

    buckets = {"20%+": [], "15-20%": [], "10-15%": [], "5-10%": [], "<5%": []}
    for b in bets:
        edge = b.get("edge_pct", 0) or 0
        if edge >= 20:
            buckets["20%+"].append(b)
        elif edge >= 15:
            buckets["15-20%"].append(b)
        elif edge >= 10:
            buckets["10-15%"].append(b)
        elif edge >= 5:
            buckets["5-10%"].append(b)
        else:
            buckets["<5%"].append(b)

    result = []
    for bucket_name in ["20%+", "15-20%", "10-15%", "5-10%", "<5%"]:
        items = buckets[bucket_name]
        if not items:
            continue
        result.append({
            "bucket": bucket_name,
            "total": len(items),
            "won": sum(1 for b in items if b["status"] == "WON"),
            "pnl": round(sum(b.get("net_pnl", 0) or 0 for b in items), 2),
        })

    return result


# ---------------------------------------------------------------------------
# Paper trades
# ---------------------------------------------------------------------------

PAPER_TRADES_KEY = "dashboard/paper_trades.json"


def get_paper_trades() -> list[dict]:
    return _read_json(PAPER_TRADES_KEY) or []


def save_paper_trades(trades: list[dict]) -> None:
    _write_json(PAPER_TRADES_KEY, trades)


def upsert_paper_trade(trade: dict) -> None:
    """Add or update a paper trade (matched by date+time+track+horse)."""
    trades = get_paper_trades()
    key = (trade["race_date"], trade.get("race_time"), trade.get("track"), trade.get("horse_name"))

    for i, existing in enumerate(trades):
        ekey = (existing["race_date"], existing.get("race_time"), existing.get("track"), existing.get("horse_name"))
        if ekey == key:
            trades[i] = trade
            save_paper_trades(trades)
            return

    trades.append(trade)
    save_paper_trades(trades)
