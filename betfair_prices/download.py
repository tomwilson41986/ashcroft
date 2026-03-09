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
    python -m betfair_prices.download --proxy http://host:port  # Use proxy for promo
    python -m betfair_prices.download --proxy-file proxies.txt  # Rotate through proxy list
"""

import argparse
import logging
import os
import random
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


# ── Proxy support ─────────────────────────────────────────────────────────

def load_proxies(proxy: str | None = None, proxy_file: str | None = None) -> list[str]:
    """Load proxy list from a single URL or a file with one proxy per line."""
    proxies = []
    if proxy:
        proxies.append(proxy)
    if proxy_file and os.path.exists(proxy_file):
        with open(proxy_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Normalise: add http:// if no scheme
                    if not line.startswith(("http://", "https://", "socks")):
                        line = "http://" + line
                    proxies.append(line)
    return proxies


def make_session(proxy: str | None = None) -> requests.Session:
    """Create a requests session, optionally with a proxy."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


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
                       max_workers: int = 5, markets: list | None = None,
                       proxies: list[str] | None = None) -> dict:
    """Download daily Betfair promo price files for the given date range.

    NOTE: promo.betfair.com is Cloudflare-protected. This only works from
    residential IPs or via a proxy. If you get 403 errors, use --proxy or
    run this script from your local machine.
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

    if proxies:
        log.info(f"Using {len(proxies)} proxy/proxies for downloads")

    # Quick test - try first file to detect Cloudflare block
    # If we have proxies, try each until one works
    test_proxy = None
    working_proxies = []

    if proxies:
        for p in proxies:
            session = make_session(p)
            test_fn, test_ok, test_msg = download_promo_file(session, filenames[0], output_dir)
            if "forbidden" not in test_msg and "Cloudflare" not in test_msg:
                log.info(f"Proxy working: {p}")
                test_proxy = p
                working_proxies.append(p)
                break
            else:
                log.warning(f"Proxy blocked: {p}")
        if not working_proxies:
            log.error("All proxies were blocked by Cloudflare.")
            return {"downloaded": 0, "skipped": 0, "not_found": 0, "failed": total,
                    "blocked": True}
        # Test remaining proxies in background
        for p in proxies:
            if p != test_proxy:
                try:
                    s = make_session(p)
                    # Use a different test file
                    _, ok, msg = download_promo_file(s, filenames[min(1, len(filenames)-1)], output_dir)
                    if "forbidden" not in msg and "Cloudflare" not in msg:
                        working_proxies.append(p)
                        log.info(f"Proxy working: {p}")
                except Exception:
                    pass
    else:
        session = make_session()
        test_fn, test_ok, test_msg = download_promo_file(session, filenames[0], output_dir)
        if "forbidden" in test_msg or "Cloudflare" in test_msg:
            log.error("Cloudflare block detected. promo.betfair.com requires a residential IP.")
            log.error("Use --proxy or --proxy-file, or run from your local machine.")
            return {"downloaded": 0, "skipped": 0, "not_found": 0, "failed": total,
                    "blocked": True}

    downloaded = 0
    skipped = 0
    failed = 0
    not_found = 0

    # Count the test result
    if not proxies or test_proxy:
        if test_ok:
            if "skipped" in test_msg:
                skipped += 1
            else:
                downloaded += 1

    def _download_with_proxy(fn):
        """Pick a random working proxy (or none) and download."""
        if working_proxies:
            proxy = random.choice(working_proxies)
            s = make_session(proxy)
        else:
            s = make_session()
        return download_promo_file(s, fn, output_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_with_proxy, fn): fn
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
    parser.add_argument("--proxy", type=str, default=None,
                        help="Proxy URL for promo downloads (e.g. http://host:port "
                             "or socks5://host:port)")
    parser.add_argument("--proxy-file", type=str, default=None,
                        help="File with proxy URLs, one per line. Proxies are "
                             "tested and rotated automatically.")
    args = parser.parse_args()

    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date) if args.to_date else date.today() - timedelta(days=1)

    proxy_list = load_proxies(args.proxy, args.proxy_file)

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

        download_promo_all(start, end, args.output, args.workers, markets,
                           proxies=proxy_list if proxy_list else None)


if __name__ == "__main__":
    main()
