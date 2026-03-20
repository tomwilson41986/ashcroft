#!/usr/bin/env python3
"""Race-day health check for the Ashcroft betting system.

Verifies that the database, models, config, and credentials are all in
order before running predictions or placing bets.

Usage:
    python health_check.py                # Quick checks only
    python health_check.py --test-s3      # Also test S3 connection
    python health_check.py --test-betfair # Also test Betfair login
    python health_check.py --all          # All checks including connectivity
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Colour helpers (ANSI, degrades gracefully on dumb terminals)
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _pass(msg: str) -> str:
    return f"  {GREEN}PASS{RESET}  {msg}"


def _fail(msg: str) -> str:
    return f"  {RED}FAIL{RESET}  {msg}"


def _warn(msg: str) -> str:
    return f"  {YELLOW}WARN{RESET}  {msg}"


# ---------------------------------------------------------------------------
# Import DB_PATH from the project's own schema module
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from db_schema import DB_PATH  # noqa: E402

MODELS_DIR = os.path.join(SCRIPT_DIR, "data", "models")
RULES_YAML = os.path.join(SCRIPT_DIR, "config", "rules.yaml")
GUARDRAILS_YAML = os.path.join(SCRIPT_DIR, "config", "guardrails.yaml")

REQUIRED_MODEL_FILES = ["bfsp_model.lgb", "bfsp_model_meta.json"]

REQUIRED_ENV_VARS = {
    "betfair": ["BETFAIR_USERNAME", "BETFAIR_PASSWORD", "BETFAIR_APP_KEY"],
    "aws": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
}


# ---------------------------------------------------------------------------
# Load .env.local if present (same approach the rest of the project uses)
# ---------------------------------------------------------------------------
def _load_env_local() -> None:
    env_file = os.path.join(SCRIPT_DIR, ".env.local")
    if not os.path.isfile(env_file):
        return
    with open(env_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_database() -> bool:
    """Check DB exists and has recent data (within last 3 days)."""
    ok = True

    if not os.path.isfile(DB_PATH):
        print(_fail(f"Database not found: {DB_PATH}"))
        return False

    size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(_pass(f"Database exists ({size_mb:.1f} MB)"))

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check for recent data in race_results
        cursor.execute(
            "SELECT MAX(race_date) FROM race_results"
        )
        row = cursor.fetchone()
        conn.close()

        if row and row[0]:
            latest = row[0]
            # Parse the date — handle common formats
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"):
                try:
                    latest_dt = datetime.strptime(latest, fmt)
                    break
                except ValueError:
                    continue
            else:
                print(_warn(f"Could not parse latest race_date: {latest}"))
                return ok

            days_old = (datetime.now() - latest_dt).days
            if days_old <= 3:
                print(_pass(f"Latest race data: {latest} ({days_old}d ago)"))
            else:
                print(_warn(f"Latest race data: {latest} ({days_old}d ago — stale)"))
        else:
            print(_fail("race_results table is empty"))
            ok = False

    except Exception as exc:
        print(_fail(f"Database query error: {exc}"))
        ok = False

    return ok


def check_models() -> bool:
    """Check that required model files exist."""
    ok = True

    if not os.path.isdir(MODELS_DIR):
        print(_fail(f"Models directory missing: {MODELS_DIR}"))
        return False

    for fname in REQUIRED_MODEL_FILES:
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(_pass(f"Model file: {fname} ({size_kb:.0f} KB)"))
        else:
            print(_fail(f"Model file missing: {fname}"))
            ok = False

    return ok


def check_env_vars() -> bool:
    """Check that required environment variables are set."""
    ok = True

    for group, var_names in REQUIRED_ENV_VARS.items():
        for var in var_names:
            val = os.environ.get(var, "")
            if val:
                # Mask the value for display
                print(_pass(f"Env var {var} is set"))
            else:
                print(_fail(f"Env var {var} is NOT set"))
                ok = False

    return ok


def check_configs() -> bool:
    """Validate that YAML config files are parseable."""
    ok = True

    try:
        import yaml
    except ImportError:
        print(_warn("PyYAML not installed — skipping config parse check"))
        # Still check the files exist
        for path in (RULES_YAML, GUARDRAILS_YAML):
            if os.path.isfile(path):
                print(_pass(f"Config exists: {os.path.basename(path)}"))
            else:
                print(_fail(f"Config missing: {path}"))
                ok = False
        return ok

    for path in (RULES_YAML, GUARDRAILS_YAML):
        name = os.path.basename(path)
        if not os.path.isfile(path):
            print(_fail(f"Config missing: {path}"))
            ok = False
            continue
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh)
            if data is None:
                print(_warn(f"Config is empty: {name}"))
            else:
                print(_pass(f"Config valid: {name}"))
        except yaml.YAMLError as exc:
            print(_fail(f"Config parse error in {name}: {exc}"))
            ok = False

    return ok


def check_s3() -> bool:
    """Test S3 connectivity by listing the bucket."""
    try:
        import boto3
    except ImportError:
        print(_fail("boto3 not installed — cannot test S3"))
        return False

    bucket = os.environ.get("S3_BUCKET", "horseracingresults")
    try:
        s3 = boto3.client("s3")
        resp = s3.head_bucket(Bucket=bucket)
        print(_pass(f"S3 bucket '{bucket}' is accessible"))
        return True
    except Exception as exc:
        print(_fail(f"S3 connectivity failed: {exc}"))
        return False


def check_betfair() -> bool:
    """Test Betfair authentication."""
    try:
        from betfair_client import BetfairClient
    except ImportError:
        print(_fail("Cannot import BetfairClient"))
        return False

    try:
        client = BetfairClient()
        token = client.login()
        if token:
            print(_pass("Betfair login successful"))
            return True
        else:
            print(_fail("Betfair login returned no session token"))
            return False
    except Exception as exc:
        print(_fail(f"Betfair login failed: {exc}"))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ashcroft race-day health check"
    )
    parser.add_argument(
        "--test-s3", action="store_true", help="Test S3 connectivity"
    )
    parser.add_argument(
        "--test-betfair", action="store_true", help="Test Betfair authentication"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all checks including connectivity"
    )
    args = parser.parse_args()

    if args.all:
        args.test_s3 = True
        args.test_betfair = True

    _load_env_local()

    print(f"\n{BOLD}Ashcroft Health Check{RESET}")
    print(f"{'=' * 40}\n")

    all_ok = True

    # 1. Database
    print(f"{BOLD}Database{RESET}")
    all_ok &= check_database()
    print()

    # 2. Models
    print(f"{BOLD}Models{RESET}")
    all_ok &= check_models()
    print()

    # 3. Environment variables
    print(f"{BOLD}Environment Variables{RESET}")
    all_ok &= check_env_vars()
    print()

    # 4. Config files
    print(f"{BOLD}Configuration{RESET}")
    all_ok &= check_configs()
    print()

    # 5. Optional: S3
    if args.test_s3:
        print(f"{BOLD}S3 Connectivity{RESET}")
        all_ok &= check_s3()
        print()

    # 6. Optional: Betfair
    if args.test_betfair:
        print(f"{BOLD}Betfair Authentication{RESET}")
        all_ok &= check_betfair()
        print()

    # Summary
    print("=" * 40)
    if all_ok:
        print(f"{GREEN}{BOLD}All checks passed.{RESET}\n")
    else:
        print(f"{RED}{BOLD}Some checks failed.{RESET}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
