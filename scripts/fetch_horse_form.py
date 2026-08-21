"""
Fetch horse form pages from horseracebase.com (requires HRB login).

Logs in with HRB_USERNAME / HRB_PASSWORD and saves the raw logged-in
HTML of each horse's form page (horses.php?id=N) so the run history
can be parsed offline.

Usage:
    python scripts/fetch_horse_form.py --ids 397142,399032 --out-dir data/h2h
"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from scraper import BASE_URL, REQUEST_DELAY, create_session, log, login


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True,
                        help="Comma-separated horseracebase horse IDs")
    parser.add_argument("--out-dir", default="data/h2h",
                        help="Directory to save raw HTML pages (default: data/h2h)")
    args = parser.parse_args()

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    if not ids:
        log.error("No horse IDs given")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    session = create_session()
    user_id = login(session)
    if not user_id:
        log.error("HRB login failed")
        sys.exit(1)

    failures = 0
    for hid in ids:
        url = f"{BASE_URL}/horses.php?id={hid}"
        log.info(f"Fetching {url}")
        resp = session.get(url, timeout=60)
        resp.raise_for_status()

        if "must hold a valid" in resp.text.lower():
            log.error(f"Horse {hid}: page still behind membership wall — "
                      "login did not carry over")
            failures += 1
        else:
            log.info(f"Horse {hid}: got {len(resp.text)} bytes of logged-in form page")

        path = os.path.join(args.out_dir, f"horse_{hid}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        log.info(f"Saved {path}")
        time.sleep(REQUEST_DELAY)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
