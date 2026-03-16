"""Cumulative P&L chart generation."""

import base64
import io
import logging

log = logging.getLogger(__name__)


def generate_cumulative_pnl_chart(cumulative_data: list[dict]) -> str:
    """Generate a cumulative P&L line chart as a base64-encoded PNG.

    Args:
        cumulative_data: List of daily P&L dicts with 'date' and 'net_pnl' keys.

    Returns:
        Base64-encoded PNG string for embedding in HTML.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        log.warning("matplotlib not installed, skipping chart generation")
        return ""

    if not cumulative_data:
        return ""

    from datetime import datetime

    dates = []
    cum_pnl = []
    running_total = 0.0

    for entry in sorted(cumulative_data, key=lambda x: x["date"]):
        try:
            dates.append(datetime.strptime(entry["date"], "%Y-%m-%d"))
        except (ValueError, KeyError):
            continue
        running_total += entry.get("net_pnl", 0)
        cum_pnl.append(running_total)

    if not dates:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, cum_pnl, linewidth=2, color="#2563eb", marker="o", markersize=4)
    ax.axhline(y=0, color="#94a3b8", linewidth=0.5, linestyle="--")
    ax.fill_between(
        dates, cum_pnl, 0,
        where=[p >= 0 for p in cum_pnl], alpha=0.1, color="#22c55e"
    )
    ax.fill_between(
        dates, cum_pnl, 0,
        where=[p < 0 for p in cum_pnl], alpha=0.1, color="#ef4444"
    )

    ax.set_title("Cumulative Net P&L", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (£)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
