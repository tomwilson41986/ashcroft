"""
Daily Results Report — matches predictions to actual results, calculates P&L,
and emails a summary.

Runs after results have been scraped and BSPs synced for the day.

Usage:
    python results_report.py                        # Report for today
    python results_report.py --date 2026-03-23      # Specific date
    python results_report.py --dry-run               # Print to stdout, skip email

Environment variables:
    SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD
"""

import argparse
import logging
import os
import smtplib
import sqlite3
import sys
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "horse_racing.db")
RECIPIENT_EMAIL = "racingsquared@gmail.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SCRIPT_DIR, "results_report.log")),
    ],
)
log = logging.getLogger(__name__)


def get_predictions_for_date(conn, target_date):
    """Load predictions (bets) recorded for this date."""
    rows = conn.execute(
        """SELECT id, race_date, race_time, track, horse_name,
                  predicted_bfsp, predicted_win_prob, betfair_back, betfair_lay,
                  edge_pct, bet_type, stake, price,
                  status, actual_bsp, placing, gross_pnl, commission, net_pnl
           FROM bets WHERE race_date = ?
           ORDER BY race_time, track""",
        (str(target_date),),
    ).fetchall()
    return [dict(r) for r in rows]


def get_results_for_date(conn, target_date):
    """Load actual race results for the date."""
    rows = conn.execute(
        """SELECT horse_name, track, racetime, placing_numerical, BFSP,
                  odds, race_name, race_class, number_of_runners
           FROM race_results WHERE racedate = ?
           ORDER BY racetime, track""",
        (str(target_date),),
    ).fetchall()
    return [dict(r) for r in rows]


def get_daily_pnl(conn, target_date):
    """Load daily P&L summary if it exists."""
    row = conn.execute(
        "SELECT * FROM daily_pnl WHERE date = ?", (str(target_date),)
    ).fetchone()
    return dict(row) if row else None


def get_cumulative_pnl(conn):
    """Get the latest cumulative P&L record."""
    row = conn.execute(
        "SELECT * FROM daily_pnl ORDER BY date DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def build_html_report(target_date, bets, daily_pnl, cumulative, num_races, num_runners):
    """Build an HTML email report for the day's results."""
    settled = [b for b in bets if b["status"] in ("WON", "LOST", "VOID")]
    pending = [b for b in bets if b["status"] == "PENDING"]
    winners = [b for b in settled if b["status"] == "WON"]
    losers = [b for b in settled if b["status"] == "LOST"]

    total_pnl = sum(b["net_pnl"] or 0 for b in settled)
    total_staked = sum(b["stake"] or 0 for b in settled)
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    pnl_color = "#27ae60" if total_pnl >= 0 else "#e74c3c"

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
             max-width: 700px; margin: 0 auto; padding: 20px; color: #333;">

<h1 style="border-bottom: 3px solid #2c3e50; padding-bottom: 10px;">
    Racing Results — {target_date.strftime('%A %d %B %Y')}
</h1>

<div style="display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px;">
    <div style="background: #ecf0f1; padding: 15px 20px; border-radius: 8px; flex: 1; min-width: 120px;">
        <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase;">Races</div>
        <div style="font-size: 24px; font-weight: bold;">{num_races}</div>
    </div>
    <div style="background: #ecf0f1; padding: 15px 20px; border-radius: 8px; flex: 1; min-width: 120px;">
        <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase;">Bets Tracked</div>
        <div style="font-size: 24px; font-weight: bold;">{len(bets)}</div>
    </div>
    <div style="background: #ecf0f1; padding: 15px 20px; border-radius: 8px; flex: 1; min-width: 120px;">
        <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase;">Record</div>
        <div style="font-size: 24px; font-weight: bold;">{len(winners)}W / {len(losers)}L</div>
    </div>
    <div style="background: {pnl_color}; color: white; padding: 15px 20px; border-radius: 8px; flex: 1; min-width: 120px;">
        <div style="font-size: 12px; text-transform: uppercase; opacity: 0.9;">Net P&amp;L</div>
        <div style="font-size: 24px; font-weight: bold;">{'+' if total_pnl >= 0 else ''}{total_pnl:.2f}</div>
        <div style="font-size: 12px; opacity: 0.8;">ROI: {roi:+.1f}%</div>
    </div>
</div>
"""

    # Settled bets table
    if settled:
        html += """
<h2 style="color: #2c3e50;">Settled Bets</h2>
<table style="width: 100%; border-collapse: collapse; font-size: 13px;">
<thead>
<tr style="background: #2c3e50; color: white;">
    <th style="padding: 8px; text-align: left;">Time</th>
    <th style="padding: 8px; text-align: left;">Track</th>
    <th style="padding: 8px; text-align: left;">Horse</th>
    <th style="padding: 8px; text-align: center;">Type</th>
    <th style="padding: 8px; text-align: right;">Stake</th>
    <th style="padding: 8px; text-align: right;">Pred BFSP</th>
    <th style="padding: 8px; text-align: right;">Actual BSP</th>
    <th style="padding: 8px; text-align: center;">Place</th>
    <th style="padding: 8px; text-align: center;">Result</th>
    <th style="padding: 8px; text-align: right;">P&amp;L</th>
</tr>
</thead>
<tbody>
"""
        for b in settled:
            bg = "#eafaf1" if b["status"] == "WON" else "#fdedec" if b["status"] == "LOST" else "#fef9e7"
            net = b["net_pnl"] or 0
            html += f"""<tr style="background: {bg};">
    <td style="padding: 6px 8px;">{b['race_time'] or ''}</td>
    <td style="padding: 6px 8px;">{b['track'] or ''}</td>
    <td style="padding: 6px 8px; font-weight: 500;">{b['horse_name']}</td>
    <td style="padding: 6px 8px; text-align: center;">{b['bet_type'] or 'BACK'}</td>
    <td style="padding: 6px 8px; text-align: right;">{b['stake']:.2f}</td>
    <td style="padding: 6px 8px; text-align: right;">{b['predicted_bfsp']:.2f}</td>
    <td style="padding: 6px 8px; text-align: right;">{b['actual_bsp']:.2f if b['actual_bsp'] else '—'}</td>
    <td style="padding: 6px 8px; text-align: center;">{b['placing'] or '—'}</td>
    <td style="padding: 6px 8px; text-align: center; font-weight: bold;">{b['status']}</td>
    <td style="padding: 6px 8px; text-align: right; color: {'#27ae60' if net >= 0 else '#e74c3c'}; font-weight: bold;">
        {'+' if net >= 0 else ''}{net:.2f}
    </td>
</tr>"""

        html += f"""
</tbody>
<tfoot>
<tr style="background: #2c3e50; color: white; font-weight: bold;">
    <td colspan="4" style="padding: 8px;">TOTAL</td>
    <td style="padding: 8px; text-align: right;">{total_staked:.2f}</td>
    <td colspan="3"></td>
    <td style="padding: 8px; text-align: center;">{len(winners)}W/{len(losers)}L</td>
    <td style="padding: 8px; text-align: right;">{'+' if total_pnl >= 0 else ''}{total_pnl:.2f}</td>
</tr>
</tfoot>
</table>
"""

    if pending:
        html += f"""
<p style="color: #7f8c8d; margin-top: 15px;">
    {len(pending)} bet(s) still pending settlement (results not yet available).
</p>
"""

    # Cumulative section
    if cumulative:
        cum_pnl = cumulative.get("cumulative_pnl", 0)
        bank = cumulative.get("bank_end", 1000)
        cum_color = "#27ae60" if cum_pnl >= 0 else "#e74c3c"
        html += f"""
<h2 style="color: #2c3e50; margin-top: 30px;">Cumulative Performance</h2>
<div style="display: flex; gap: 15px; flex-wrap: wrap;">
    <div style="background: #ecf0f1; padding: 15px 20px; border-radius: 8px; flex: 1;">
        <div style="font-size: 12px; color: #7f8c8d; text-transform: uppercase;">Bank</div>
        <div style="font-size: 24px; font-weight: bold;">{bank:.2f}</div>
    </div>
    <div style="background: {cum_color}; color: white; padding: 15px 20px; border-radius: 8px; flex: 1;">
        <div style="font-size: 12px; text-transform: uppercase; opacity: 0.9;">Cumulative P&amp;L</div>
        <div style="font-size: 24px; font-weight: bold;">{'+' if cum_pnl >= 0 else ''}{cum_pnl:.2f}</div>
    </div>
</div>
"""

    html += """
<hr style="margin-top: 30px; border: none; border-top: 1px solid #ecf0f1;">
<p style="font-size: 11px; color: #bdc3c7; text-align: center;">
    Ashcroft Racing — Automated Results Report
</p>
</body></html>
"""
    return html


def build_text_report(target_date, bets, daily_pnl, cumulative, num_races, num_runners):
    """Build a plain-text fallback report."""
    settled = [b for b in bets if b["status"] in ("WON", "LOST", "VOID")]
    winners = [b for b in settled if b["status"] == "WON"]
    losers = [b for b in settled if b["status"] == "LOST"]
    total_pnl = sum(b["net_pnl"] or 0 for b in settled)
    total_staked = sum(b["stake"] or 0 for b in settled)
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    lines = [
        f"RACING RESULTS — {target_date.strftime('%A %d %B %Y')}",
        "=" * 50,
        f"Races: {num_races}  |  Bets: {len(bets)}  |  Record: {len(winners)}W / {len(losers)}L",
        f"Net P&L: {total_pnl:+.2f}  |  ROI: {roi:+.1f}%",
        "",
    ]

    if settled:
        lines.append("SETTLED BETS:")
        lines.append("-" * 50)
        for b in settled:
            net = b["net_pnl"] or 0
            lines.append(
                f"  {b['race_time'] or '':>5}  {b['track'] or '':<15} "
                f"{b['horse_name']:<20} {b['status']:>4}  P&L: {net:+.2f}"
            )
        lines.append("-" * 50)
        lines.append(f"  TOTAL: {total_pnl:+.2f}")

    if cumulative:
        lines.append("")
        lines.append(f"Bank: {cumulative.get('bank_end', 1000):.2f}  |  "
                      f"Cumulative P&L: {cumulative.get('cumulative_pnl', 0):+.2f}")

    return "\n".join(lines)


def send_email(recipient, subject, text_body, html_body):
    """Send results email via SMTP."""
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_pass:
        log.error("SMTP_USERNAME and SMTP_PASSWORD must be set to send email.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = recipient

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())
        log.info(f"Results email sent to {recipient}")
        return True
    except Exception as e:
        log.error(f"Failed to send email: {e}")
        return False


def run_report(target_date, db_path=DB_PATH, recipient=RECIPIENT_EMAIL, dry_run=False):
    """Generate and send the daily results report."""
    log.info(f"{'='*50}")
    log.info(f"DAILY RESULTS REPORT — {target_date}")
    log.info(f"{'='*50}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get actual results for the date
    results = get_results_for_date(conn, target_date)
    num_runners = len(results)
    race_times = set((r.get("racetime", ""), r.get("track", "")) for r in results)
    num_races = len(race_times)
    log.info(f"Results: {num_races} races, {num_runners} runners")

    # Get predictions/bets for the date
    bets = get_predictions_for_date(conn, target_date)
    log.info(f"Bets tracked: {len(bets)}")

    if not bets and not results:
        log.warning(f"No bets or results found for {target_date}")
        conn.close()
        return

    # Get P&L data
    daily_pnl = get_daily_pnl(conn, target_date)
    cumulative = get_cumulative_pnl(conn)

    conn.close()

    # Build reports
    html = build_html_report(target_date, bets, daily_pnl, cumulative, num_races, num_runners)
    text = build_text_report(target_date, bets, daily_pnl, cumulative, num_races, num_runners)

    settled = [b for b in bets if b["status"] in ("WON", "LOST", "VOID")]
    total_pnl = sum(b["net_pnl"] or 0 for b in settled)
    pnl_str = f"{'+' if total_pnl >= 0 else ''}{total_pnl:.2f}"

    subject = f"Racing Results {target_date.strftime('%d %b')} — P&L: {pnl_str}"

    if dry_run:
        print(text)
        print(f"\n[Dry run — email would be sent to {recipient}]")
        print(f"[Subject: {subject}]")
    else:
        send_email(recipient, subject, text, html)


def main():
    parser = argparse.ArgumentParser(description="Daily racing results report")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: today")
    parser.add_argument("--db", type=str, default=DB_PATH)
    parser.add_argument("--email", type=str, default=RECIPIENT_EMAIL)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print report instead of emailing")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    run_report(target, db_path=args.db, recipient=args.email, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
