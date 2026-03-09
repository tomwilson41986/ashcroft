"""
Betfair SP Price Data Downloader

Downloads Betfair Starting Price data for UK and Irish horse racing from
two sources:

1. Betfair Automation Hub (betfair-datascientists.github.io)
   - Available: 2024-present
   - Fields: date, track, event_id, selection_id, runner_name, win_bsp, place_bsp
   - Accessible from any network

2. Betfair Promo (promo.betfair.com/betfairsp/prices/)
   - Available: 2019-present
   - Fields: BSP + PPWAP, MORNINGWAP, PPMAX, PPMIN, IPMAX, IPMIN
   - Requires residential IP (Cloudflare-protected)

Files are saved to data/betfair_raw/.

Usage:
    python -m betfair_prices.download                           # Download all available
    python -m betfair_prices.download --source hub              # Automation Hub only
    python -m betfair_prices.download --source promo            # Promo site only
    python -m betfair_prices.download --from 2020-01-01         # Custom start date
    python -m betfair_prices.download --workers 10              # Parallel promo downloads
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_DIR, "data", "betfair_raw")

# Betfair Promo config
PROMO_BASE_URL = "https://promo.betfair.com/betfairsp/prices/"
PROMO_MARKETS = [
    ("uk", "win", "dwbfpricesukwin{date}.csv"),
    ("uk", "place", "dwbfpricesukplace{date}.csv"),
    ("ire", "win", "dwbfpricesirewin{date}.csv"),
    ("ire", "place", "dwbfpricesireplace{date}.csv"),
]

# Betfair Automation Hub config
HUB_BASE_URL = "https://betfair-datascientists.github.io/data/assets/"
HUB_FILES = [
    "UK_IE_Thoroughbred_Racing_Model_2024.csv",
    "UK_IE_Thoroughbred_Racing_Model_2025.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-01.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-02.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-03.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-04.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-05.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-06.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-07.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-08.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-09.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-10.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-11.csv",
    "UK_IE_Thoroughbred_Racing_Model_2026-12.csv",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "betfair_download.log")),
    ],
)
log = logging.getLogger(__name__)


# ── Automation Hub downloads ──────────────────────────────────────────────

def download_hub_files(output_dir: str = RAW_DIR) -> dict:
    """Download UK/IE data from Betfair Automation Hub (GitHub-hosted, no restrictions)."""
    os.makedirs(output_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    downloaded = 0
    skipped = 0
    not_found = 0

    for filename in HUB_FILES:
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            log.info(f"Skipped (exists): {filename}")
            skipped += 1
            continue

        url = HUB_BASE_URL + filename
        try:
            resp = session.get(url, timeout=30)

            if resp.status_code == 404:
                log.debug(f"Not found: {filename}")
                not_found += 1
                continue

            resp.raise_for_status()

            # Verify CSV content (not an HTML error page)
            if resp.text.strip().startswith("<!"):
                log.warning(f"Got HTML instead of CSV: {filename}")
                not_found += 1
                continue

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(resp.text)

            rows = resp.text.count("\n")
            log.info(f"Downloaded: {filename} ({rows} rows)")
            downloaded += 1

        except requests.RequestException as e:
            log.warning(f"Error downloading {filename}: {e}")

    log.info(f"Hub downloads: {downloaded} new, {skipped} skipped, {not_found} not available")
    return {"downloaded": downloaded, "skipped": skipped, "not_found": not_found}


# ── Betfair Promo downloads ──────────────────────────────────────────────

def make_promo_filename(template: str, d: date) -> str:
    """Generate filename for a given date using DDMMYYYY format."""
    return template.format(date=d.strftime("%d%m%Y"))


def download_promo_file(session: requests.Session, filename: str, output_dir: str,
                        max_retries: int = 3) -> tuple[str, bool, str]:
    """Download a single promo CSV file. Returns (filename, success, message)."""
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        return (filename, True, "skipped (exists)")

    url = PROMO_BASE_URL + filename

    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30)

            if resp.status_code == 404:
                return (filename, False, "not found (no racing)")

            if resp.status_code == 403:
                return (filename, False, "forbidden (Cloudflare block - use residential IP)")

            resp.raise_for_status()

            text = resp.text.strip()
            if not text or len(text) < 50:
                return (filename, False, "empty response")

            # Verify it's CSV, not HTML
            if text.startswith("<!"):
                return (filename, False, "got HTML (Cloudflare challenge)")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(resp.text)

            return (filename, True, "downloaded")

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return (filename, False, f"error: {e}")

    return (filename, False, "max retries exceeded")


def download_promo_all(start_date: date, end_date: date, output_dir: str = RAW_DIR,
                       max_workers: int = 5, markets: list | None = None) -> dict:
    """Download daily Betfair promo price files for the given date range.

    NOTE: promo.betfair.com is Cloudflare-protected. This only works from
    residential IPs, not cloud servers. If you get 403 errors, run this
    script from your local machine instead.
    """
    os.makedirs(output_dir, exist_ok=True)

    if markets is None:
        markets = PROMO_MARKETS

    # Generate all filenames
    filenames = []
    current = start_date
    while current <= end_date:
        for _, _, template in markets:
            filenames.append(make_promo_filename(template, current))
        current += timedelta(days=1)

    total = len(filenames)
    log.info(f"Promo download: {total} files ({start_date} to {end_date})")

    # Quick test - try first file to detect Cloudflare block
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    test_fn, test_ok, test_msg = download_promo_file(session, filenames[0], output_dir)
    if "forbidden" in test_msg or "Cloudflare" in test_msg:
        log.error("Cloudflare block detected. promo.betfair.com requires a residential IP.")
        log.error("Run this script from your local machine, not a cloud server.")
        log.error("Skipping promo downloads. Use --source hub for Automation Hub data.")
        return {"downloaded": 0, "skipped": 0, "not_found": 0, "failed": total,
                "blocked": True}

    downloaded = 0
    skipped = 0
    failed = 0
    not_found = 0

    # Count the test result
    if test_ok:
        if "skipped" in test_msg:
            skipped += 1
        else:
            downloaded += 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_promo_file, session, fn, output_dir): fn
            for fn in filenames[1:]  # Skip first (already tested)
        }

        for i, future in enumerate(as_completed(futures), 2):
            filename, success, message = future.result()

            if success:
                if "skipped" in message:
                    skipped += 1
                else:
                    downloaded += 1
            else:
                if "not found" in message:
                    not_found += 1
                else:
                    failed += 1
                    if failed <= 5:
                        log.warning(f"Failed: {filename} - {message}")

            if i % 200 == 0:
                log.info(f"Progress: {i}/{total} "
                         f"(downloaded={downloaded}, skipped={skipped}, "
                         f"not_found={not_found}, failed={failed})")

    log.info(f"Promo complete: {downloaded} downloaded, {skipped} skipped, "
             f"{not_found} not found, {failed} failed")

    return {"downloaded": downloaded, "skipped": skipped,
            "not_found": not_found, "failed": failed, "blocked": False}


def main():
    parser = argparse.ArgumentParser(
        description="Download Betfair SP price data for UK and Irish horse racing"
    )
    parser.add_argument("--source", type=str, choices=["all", "hub", "promo"],
                        default="all",
                        help="Data source: hub (Automation Hub), promo (promo.betfair.com), "
                             "or all (default: all)")
    parser.add_argument("--from", dest="from_date", type=str, default="2019-01-01",
                        help="Start date for promo downloads (YYYY-MM-DD). Default: 2019-01-01")
    parser.add_argument("--to", dest="to_date", type=str,
                        help="End date for promo downloads (YYYY-MM-DD). Default: yesterday")
    parser.add_argument("--workers", type=int, default=5,
                        help="Parallel download workers for promo (default: 5)")
    parser.add_argument("--market", type=str, choices=["win", "place", "all"],
                        default="all", help="Market type for promo (default: all)")
    parser.add_argument("--country", type=str, choices=["uk", "ire", "all"],
                        default="all", help="Country for promo (default: all)")
    parser.add_argument("--output", type=str, default=RAW_DIR,
                        help=f"Output directory (default: {RAW_DIR})")
    args = parser.parse_args()

    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date) if args.to_date else date.today() - timedelta(days=1)

    if args.source in ("all", "hub"):
        log.info("=== Downloading from Betfair Automation Hub ===")
        download_hub_files(args.output)

    if args.source in ("all", "promo"):
        log.info("=== Downloading from Betfair Promo ===")
        markets = PROMO_MARKETS
        if args.market != "all":
            markets = [m for m in markets if m[1] == args.market]
        if args.country != "all":
            markets = [m for m in markets if m[0] == args.country]

        download_promo_all(start, end, args.output, args.workers, markets)


if __name__ == "__main__":
    main()
