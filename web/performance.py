"""Model performance metrics computation.

Computes detailed performance breakdowns from settled bet data,
including ROI by edge band, price range, venue, and rolling windows.
"""

from collections import defaultdict
from datetime import datetime, timedelta


def compute_performance_metrics(bets: list[dict]) -> dict:
    """Compute full model performance metrics from bet/settlement data.

    Args:
        bets: List of bet dicts. Each should have at minimum:
            - status: "WON" | "LOST" (others are filtered out)
            - net_pnl: float (profit/loss for the bet)
            - stake: float
            - edge_pct: float (model edge percentage)
            - price: float (the odds taken)
            - race_date: str (YYYY-MM-DD)
            - track: str (venue name)

    Returns:
        Dict with full performance breakdown.
    """
    settled = [b for b in bets if b.get("status") in ("WON", "LOST")]

    if not settled:
        return _empty_metrics()

    wins = [b for b in settled if b["status"] == "WON"]
    losses = [b for b in settled if b["status"] == "LOST"]

    total_staked = sum(b.get("stake", 0) or 0 for b in settled)
    total_pnl = sum(b.get("net_pnl", 0) or 0 for b in settled)

    return {
        "total_bets": len(settled),
        "wins": len(wins),
        "losses": len(losses),
        "hit_rate": _pct(len(wins), len(settled)),
        "total_staked": _r(total_staked),
        "net_pnl": _r(total_pnl),
        "roi_pct": _pct(total_pnl, total_staked),
        "roi_by_edge_band": _roi_by_edge_band(settled),
        "roi_by_price_range": _roi_by_price_range(settled),
        "win_rate_by_venue": _win_rate_by_venue(settled),
        "avg_edge_winners": _avg(b.get("edge_pct", 0) or 0 for b in wins),
        "avg_edge_losers": _avg(b.get("edge_pct", 0) or 0 for b in losses),
        "profit_factor": _profit_factor(wins, losses),
        "best_day": _best_day(settled),
        "worst_day": _worst_day(settled),
        "rolling_7d_roi": _rolling_roi(settled, 7),
        "rolling_30d_roi": _rolling_roi(settled, 30),
    }


def compute_performance_summary(bets: list[dict]) -> dict:
    """Return condensed key metrics only."""
    settled = [b for b in bets if b.get("status") in ("WON", "LOST")]

    if not settled:
        return _empty_summary()

    wins = [b for b in settled if b["status"] == "WON"]
    losses = [b for b in settled if b["status"] == "LOST"]

    total_staked = sum(b.get("stake", 0) or 0 for b in settled)
    total_pnl = sum(b.get("net_pnl", 0) or 0 for b in settled)

    return {
        "total_bets": len(settled),
        "wins": len(wins),
        "hit_rate": _pct(len(wins), len(settled)),
        "roi_pct": _pct(total_pnl, total_staked),
        "net_pnl": _r(total_pnl),
        "profit_factor": _profit_factor(wins, losses),
        "avg_edge_winners": _avg(b.get("edge_pct", 0) or 0 for b in wins),
        "avg_edge_losers": _avg(b.get("edge_pct", 0) or 0 for b in losses),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(v: float) -> float:
    return round(v, 2)


def _pct(num: float, denom: float) -> float:
    return round(num / denom * 100, 2) if denom else 0.0


def _avg(values) -> float:
    vals = list(values)
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _empty_metrics() -> dict:
    return {
        "total_bets": 0, "wins": 0, "losses": 0, "hit_rate": 0.0,
        "total_staked": 0.0, "net_pnl": 0.0, "roi_pct": 0.0,
        "roi_by_edge_band": [], "roi_by_price_range": [],
        "win_rate_by_venue": [], "avg_edge_winners": 0.0,
        "avg_edge_losers": 0.0, "profit_factor": 0.0,
        "best_day": None, "worst_day": None,
        "rolling_7d_roi": [], "rolling_30d_roi": [],
    }


def _empty_summary() -> dict:
    return {
        "total_bets": 0, "wins": 0, "hit_rate": 0.0, "roi_pct": 0.0,
        "net_pnl": 0.0, "profit_factor": 0.0,
        "avg_edge_winners": 0.0, "avg_edge_losers": 0.0,
    }


def _roi_by_edge_band(settled: list[dict]) -> list[dict]:
    """ROI broken down by edge percentage bands: 0-10%, 10-20%, 20%+."""
    bands = {"0-10%": [], "10-20%": [], "20%+": []}
    for b in settled:
        edge = b.get("edge_pct", 0) or 0
        if edge >= 20:
            bands["20%+"].append(b)
        elif edge >= 10:
            bands["10-20%"].append(b)
        else:
            bands["0-10%"].append(b)

    result = []
    for name in ["0-10%", "10-20%", "20%+"]:
        items = bands[name]
        if not items:
            continue
        staked = sum(b.get("stake", 0) or 0 for b in items)
        pnl = sum(b.get("net_pnl", 0) or 0 for b in items)
        won = sum(1 for b in items if b["status"] == "WON")
        result.append({
            "band": name,
            "bets": len(items),
            "won": won,
            "staked": _r(staked),
            "pnl": _r(pnl),
            "roi_pct": _pct(pnl, staked),
        })
    return result


def _roi_by_price_range(settled: list[dict]) -> list[dict]:
    """ROI broken down by price range: short <3.0, mid 3.0-8.0, long 8.0+."""
    ranges = {"short (<3.0)": [], "mid (3.0-8.0)": [], "long (8.0+)": []}
    for b in settled:
        price = b.get("price", 0) or 0
        if price < 3.0:
            ranges["short (<3.0)"].append(b)
        elif price < 8.0:
            ranges["mid (3.0-8.0)"].append(b)
        else:
            ranges["long (8.0+)"].append(b)

    result = []
    for name in ["short (<3.0)", "mid (3.0-8.0)", "long (8.0+)"]:
        items = ranges[name]
        if not items:
            continue
        staked = sum(b.get("stake", 0) or 0 for b in items)
        pnl = sum(b.get("net_pnl", 0) or 0 for b in items)
        won = sum(1 for b in items if b["status"] == "WON")
        result.append({
            "range": name,
            "bets": len(items),
            "won": won,
            "staked": _r(staked),
            "pnl": _r(pnl),
            "roi_pct": _pct(pnl, staked),
        })
    return result


def _win_rate_by_venue(settled: list[dict]) -> list[dict]:
    """Win rate by venue, top 10 by number of bets."""
    by_venue: dict[str, list[dict]] = defaultdict(list)
    for b in settled:
        venue = b.get("track", "Unknown") or "Unknown"
        by_venue[venue].append(b)

    ranked = sorted(by_venue.items(), key=lambda x: len(x[1]), reverse=True)[:10]

    result = []
    for venue, items in ranked:
        won = sum(1 for b in items if b["status"] == "WON")
        pnl = sum(b.get("net_pnl", 0) or 0 for b in items)
        result.append({
            "venue": venue,
            "bets": len(items),
            "won": won,
            "win_rate": _pct(won, len(items)),
            "pnl": _r(pnl),
        })
    return result


def _profit_factor(wins: list[dict], losses: list[dict]) -> float:
    """Gross wins / gross losses. Returns 0.0 if no losses."""
    gross_wins = sum(b.get("net_pnl", 0) or 0 for b in wins)
    gross_losses = abs(sum(b.get("net_pnl", 0) or 0 for b in losses))
    return round(gross_wins / gross_losses, 2) if gross_losses else 0.0


def _daily_pnl(settled: list[dict]) -> dict[str, float]:
    """Aggregate net P&L by date."""
    by_date: dict[str, float] = defaultdict(float)
    for b in settled:
        d = b.get("race_date", "")
        by_date[d] += b.get("net_pnl", 0) or 0
    return by_date


def _best_day(settled: list[dict]) -> dict | None:
    by_date = _daily_pnl(settled)
    if not by_date:
        return None
    best = max(by_date, key=by_date.get)
    return {"date": best, "pnl": _r(by_date[best])}


def _worst_day(settled: list[dict]) -> dict | None:
    by_date = _daily_pnl(settled)
    if not by_date:
        return None
    worst = min(by_date, key=by_date.get)
    return {"date": worst, "pnl": _r(by_date[worst])}


def _rolling_roi(settled: list[dict], window_days: int) -> list[dict]:
    """Compute rolling ROI over a sliding window of N days.

    Returns a list of {date, roi_pct} entries, one per date that has bets.
    """
    by_date: dict[str, list[dict]] = defaultdict(list)
    for b in settled:
        d = b.get("race_date", "")
        if d:
            by_date[d].append(b)

    if not by_date:
        return []

    all_dates = sorted(by_date.keys())
    result = []

    for target in all_dates:
        try:
            target_dt = datetime.strptime(target, "%Y-%m-%d")
        except ValueError:
            continue

        window_start = (target_dt - timedelta(days=window_days - 1)).strftime("%Y-%m-%d")
        window_bets = []
        for d in all_dates:
            if window_start <= d <= target:
                window_bets.extend(by_date[d])

        staked = sum(b.get("stake", 0) or 0 for b in window_bets)
        pnl = sum(b.get("net_pnl", 0) or 0 for b in window_bets)
        result.append({
            "date": target,
            "roi_pct": _pct(pnl, staked),
        })

    return result
